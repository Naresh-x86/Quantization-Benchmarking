import os
# Disable flashinfer because its JIT compiler fails on RTX 5090 (compute_120a) with older nvcc
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_MODELSCOPE"]    = "False"

import sys
import json
import argparse
from inference import LLMEngine
from agent import ReActAgent
from metrics import GPUTracker


def format_trace(agent_result: dict, must_call_raw: list, task_description: str) -> str:
    history      = agent_result.get("final_history", [])
    success      = agent_result.get("success", False)
    steps        = agent_result.get("steps", 0)
    parse_errors = agent_result.get("parse_error_count", 0)

    lines = []
    lines.append("=== METRICS ===")
    lines.append(f"Success: {'✅' if success else '❌'}")
    lines.append(f"Total Steps: {steps}")
    lines.append(f"Parse Errors: {parse_errors}")
    lines.append(f"Total Tokens: {agent_result.get('total_generated_tokens', 0)}")

    expected_formatted = []
    expected_names     = []
    for mc in must_call_raw:
        if isinstance(mc, dict):
            expected_names.append(mc["tool"])
            params_str = json.dumps(mc.get("params", {}))
            expected_formatted.append(f"{mc['tool']}({params_str})")
        else:
            expected_names.append(mc)
            expected_formatted.append(mc)

    lines.append("\n=== SUMMARY ===")
    lines.append("Expected tool call sequence:")
    lines.append(" -> ".join(expected_formatted) if expected_formatted else "None")

    actual_seq = []
    for msg in history:
        if msg.get("role") == "user" and "action" in msg:
            actual_seq.append(f"{msg['action']}({msg.get('action_input', '{}')})")

    lines.append("\nActual tool call sequence:")
    lines.append(" -> ".join(actual_seq) if actual_seq else "None")

    lines.append("\n=== TASK ===")
    lines.append(task_description)

    lines.append("\n=== INTERACTION ===")

    expected_idx = 0
    for msg in history[1:]:
        if msg["role"] == "assistant":
            lines.append(f"\n[Agent]:\n{msg['content']}")
        elif msg["role"] == "user":
            if "action" in msg:
                action       = msg["action"]
                action_input = msg.get("action_input", "{}")

                expected_tool_str  = expected_formatted[expected_idx] if expected_idx < len(expected_formatted) else "None (task should be complete)"
                expected_tool_name = expected_names[expected_idx]      if expected_idx < len(expected_names)     else "None"

                lines.append("\nAgent has called a tool")
                lines.append(f"Expected next call: {expected_tool_str}")
                lines.append(f"Actual call: {action}({action_input})")

                if action == expected_tool_name:
                    expected_idx += 1

                content = msg["content"]
                if content.startswith("Tool Result: "):
                    content = content[13:]
                lines.append(f"Result: {content}")
            else:
                lines.append(f"\n[System/Environment]:\n{msg['content']}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",     type=str,   required=True)
    parser.add_argument("--model_name",     type=str,   required=True)
    parser.add_argument("--dataset",        type=str,   required=True)
    parser.add_argument("--agent_id",       type=str,   required=True)
    parser.add_argument("--num_trials",     type=int,   default=1)
    parser.add_argument("--use_vllm",       action="store_true")
    parser.add_argument("--save_traces",    action="store_true")
    parser.add_argument("--output_json",    type=str,   required=True)
    parser.add_argument("--output_dir",     type=str,   required=True)
    # Inference knobs forwarded from config.ini
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.2)
    parser.add_argument("--do_sample",      action="store_true")
    return parser.parse_args()


def get_params_billion_from_name(model_name: str) -> float:
    import re
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    return float(match.group(1)) if match else 7.0


def get_weights_size_gb(model_path: str, model_name: str = "") -> float:
    """Sum .safetensors / .bin file sizes as a proxy for weight VRAM footprint.
    Checks model_path first; if not found on disk, checks the Hugging Face cache snapshot.
    """
    total_bytes = 0
    search_dir = None
    if os.path.exists(model_path):
        search_dir = model_path
    elif model_name:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_folder = f"models--{model_name.replace('/', '--')}"
        snapshots_dir = os.path.join(hf_home, "hub", hub_folder, "snapshots")
        if os.path.isdir(snapshots_dir):
            snaps = os.listdir(snapshots_dir)
            if snaps:
                search_dir = os.path.join(snapshots_dir, snaps[0])

    if search_dir and os.path.isdir(search_dir):
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if fname.endswith((".safetensors", ".bin")):
                    total_bytes += os.path.getsize(os.path.join(root, fname))
    return total_bytes / (1024 ** 3)


def compute_tool_quality_metrics(agent_result: dict, strict_success_criteria: dict) -> dict:
    """Compute tool-call quality metrics from the agent trace."""
    must_call_raw  = list(strict_success_criteria.get("must_call", []))
    must_call      = [mc["tool"] if isinstance(mc, dict) else mc for mc in must_call_raw]
    must_not_call  = set(strict_success_criteria.get("must_not_call", []))

    tool_seq       = agent_result.get("tool_call_sequence", [])
    parse_errors   = agent_result.get("parse_error_count", 0)
    success        = agent_result.get("success", False)

    expected_steps    = len(must_call)
    actual_tool_calls = len(tool_seq)
    unique_tools      = list(dict.fromkeys(tool_seq))
    unique_tools_count = len(unique_tools)
    redundant_calls   = max(0, actual_tool_calls - expected_steps)

    seen = set()
    repeat_count = 0
    for t in tool_seq:
        if t in seen:
            repeat_count += 1
        seen.add(t)

    must_call_in_seq = []
    for t in tool_seq:
        if t in must_call and t not in must_call_in_seq:
            must_call_in_seq.append(t)
    correct_order = (must_call_in_seq == must_call)

    forbidden_called = sorted(must_not_call.intersection(set(tool_seq)))
    skipped          = [t for t in must_call if t not in set(tool_seq)]

    final_tool_without_prior = False
    if must_call and len(must_call) > 1:
        final_tool   = must_call[-1]
        prior_tools  = set(must_call[:-1])
        if final_tool in set(tool_seq) and not prior_tools.issubset(set(tool_seq)):
            final_tool_without_prior = True

    efficiency             = expected_steps / max(actual_tool_calls, 1)
    total_tokens           = agent_result.get("total_generated_tokens", 0)
    tokens_per_useful_step = total_tokens / max(expected_steps, 1)

    if not success:
        exec_path = "FAILED"
        detail    = "Task was not completed successfully."
    elif final_tool_without_prior:
        exec_path = "SHORTCUT"
        detail    = f"Called final tool '{must_call[-1]}' without completing prior tools: {skipped}"
    elif correct_order and actual_tool_calls == expected_steps and parse_errors == 0:
        exec_path = "PERFECT"
        detail    = "All required tools called in correct order with no extras or errors."
    elif set(must_call).issubset(set(tool_seq)):
        exec_path = "WANDERING"
        reasons   = []
        if redundant_calls > 0: reasons.append(f"{redundant_calls} redundant tool call(s)")
        if repeat_count   > 0: reasons.append(f"{repeat_count} repeated tool call(s)")
        if parse_errors   > 0: reasons.append(f"{parse_errors} parse error(s)")
        if not correct_order:  reasons.append("tools called out of expected order")
        detail = "Succeeded but: " + "; ".join(reasons) if reasons else "Succeeded with minor issues."
    else:
        exec_path = "WANDERING"
        detail    = "Succeeded via alternate path."

    return {
        "Exec_Path":                exec_path,
        "Exec_Path_Detail":         detail,
        "Expected_Steps":           expected_steps,
        "Actual_Tool_Calls":        actual_tool_calls,
        "Redundant_Tool_Calls":     redundant_calls,
        "Parse_Errors":             parse_errors,
        "Tool_Call_Efficiency":     round(efficiency, 4),
        "Tokens_per_Useful_Step":   round(tokens_per_useful_step, 2),
        "Correct_Order":            correct_order,
        "Tool_Repeat_Count":        repeat_count,
        "Unique_Tools_Called":      unique_tools_count,
        "Forbidden_Tools_Called":   ", ".join(forbidden_called) if forbidden_called else "",
        "Skipped_Tools":            ", ".join(skipped)          if skipped          else "",
        "Final_Tool_Without_Prior": final_tool_without_prior,
        "Tool_Call_Sequence":       " -> ".join(tool_seq)       if tool_seq         else "",
    }


def main():
    args = parse_args()

    with open(args.dataset, "r") as f:
        tasks = json.load(f)

    task = next((t for t in tasks if t["agent_id"] == args.agent_id), None)
    if not task:
        print(f"Error: No task found for agent {args.agent_id}")
        sys.exit(1)

    model_name      = args.model_name
    model_path      = args.model_path
    params_billion  = get_params_billion_from_name(model_name)
    weights_size_gb = get_weights_size_gb(model_path, model_name)

    quant_type  = "FP16"
    model_upper = model_name.upper()
    if   "AWQ"      in model_upper: quant_type = "AWQ"
    elif "GPTQ-8BIT" in model_upper: quant_type = "GPTQ-8BIT"
    elif "GPTQ"     in model_upper: quant_type = "GPTQ"
    elif "BNB-8BIT" in model_upper: quant_type = "BNB-8BIT"
    elif "BNB-4BIT" in model_upper: quant_type = "BNB-4BIT"
    elif "BNB"      in model_upper: quant_type = "BNB-4BIT"

    if quant_type == "FP16" and not os.path.exists(model_path):
        model_path = model_name  # fallback to HF Hub ID

    # vLLM base URL: use env var (set by run_benchmark.sh) or default
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    all_trials_results = []

    if args.save_traces:
        traces_dir = os.path.join(args.output_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)

    try:
        engine = LLMEngine(
            model_path,
            quant_type=quant_type,
            use_vllm=args.use_vllm,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            vllm_base_url=vllm_base_url,
        )
        agent = ReActAgent(engine, agent_id=args.agent_id)

        for trial in range(args.num_trials):
            print(f"  Trial {trial + 1}/{args.num_trials}")

            tracker = GPUTracker(poll_interval=0.05)
            tracker.start()

            agent_result = agent.run(
                task_description=task["description"],
                strict_success_criteria=task.get("strict_success_criteria"),
            )

            gpu_metrics = tracker.stop()

            tflops = engine.calculate_tflops(
                params_billion,
                agent_result["total_generated_tokens"],
                agent_result["total_inference_time_sec"],
            )

            result_row = {
                "Model":              model_name,
                "Quantization":       quant_type,
                "Params (B)":         params_billion,
                "Weights Size (GB)":  weights_size_gb,
                "Trial":              trial + 1,
                "Success":            agent_result["success"],
                "Steps":              agent_result["steps"],
                "Total Tokens":       agent_result["total_generated_tokens"],
                "Inference Time (s)": agent_result["total_inference_time_sec"],
                "Tokens / Sec":       (
                    agent_result["total_generated_tokens"] / agent_result["total_inference_time_sec"]
                    if agent_result["total_inference_time_sec"] > 0 else 0
                ),
                "Achieved TFLOPS":    tflops,
                "Tools Called":       ", ".join(agent_result.get("tools_called", [])),
            }

            if gpu_metrics:
                result_row.update({
                    "Avg Power (W)":    gpu_metrics["avg_power_W"],
                    "Peak Power (W)":   gpu_metrics["peak_power_W"],
                    "Total Energy (J)": gpu_metrics["total_energy_J"],
                    "Avg VRAM (GB)":    gpu_metrics["avg_vram_GB"],
                    "Peak VRAM (GB)":   gpu_metrics["peak_vram_GB"],
                    "Avg GPU Util (%)": gpu_metrics["avg_gpu_util_%"],
                })

            quality_metrics = compute_tool_quality_metrics(
                agent_result, task.get("strict_success_criteria", {})
            )
            result_row.update(quality_metrics)
            all_trials_results.append(result_row)

            if args.save_traces:
                safe_name      = model_name.replace("/", "__")
                trace_filename = f"{args.agent_id}_{safe_name}_Trial_{trial + 1}.txt"
                trace_path     = os.path.join(traces_dir, trace_filename)
                must_call      = task.get("strict_success_criteria", {}).get("must_call", [])
                formatted_trace = format_trace(agent_result, must_call, task["description"])
                with open(trace_path, "w") as tf:
                    tf.write(formatted_trace)

    except Exception as e:
        import traceback
        print(f"Fatal error during execution: {e}")
        traceback.print_exc()
        all_trials_results.append({
            "Model":           model_name,
            "Quantization":    quant_type,
            "Params (B)":      params_billion,
            "Weights Size (GB)": weights_size_gb,
            "Success":         False,
            "Error":           str(e),
        })

    with open(args.output_json, "w") as f:
        json.dump(all_trials_results, f)


if __name__ == "__main__":
    main()
