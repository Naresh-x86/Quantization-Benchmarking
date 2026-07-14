import os
# Disable flashinfer because its JIT compiler fails on RTX 5090 (compute_120a) with older nvcc
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_MODELSCOPE"] = "False"
import sys
import json
import argparse
from inference import LLMEngine
from agent import ReActAgent
from metrics import GPUTracker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--agent_id", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()

def get_params_billion_from_name(model_name: str) -> float:
    import re
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    if match:
        return float(match.group(1))
    return 7.0

def main():
    args = parse_args()
    
    with open(args.dataset, "r") as f:
        tasks = json.load(f)
        
    # Find the task for this agent
    task = next((t for t in tasks if t["agent_id"] == args.agent_id), None)
    if not task:
        print(f"Error: No task found for agent {args.agent_id}")
        return
        
    model_name = args.model_name
    model_path = args.model_path
    params_billion = get_params_billion_from_name(model_name)
    
    quant_type = "baseline"
    if "INT8" in model_name: quant_type = "INT8"
    elif "INT4" in model_name: quant_type = "INT4"
    elif "FP8" in model_name: quant_type = "FP8"
    elif "FP16" in model_name: quant_type = "FP16"
    
    all_trials_results = []
    
    try:
        # Initialize engine once per model
        engine = LLMEngine(model_path, use_vllm=args.use_vllm, quant_type=quant_type)
        agent = ReActAgent(engine, agent_id=args.agent_id)
        
        for trial in range(args.num_trials):
            print(f"--- Running Trial {trial+1}/{args.num_trials} ---")
            
            tracker = GPUTracker(poll_interval=0.05)
            tracker.start()
            
            agent_result = agent.run(
                task_description=task['description'],
                strict_success_criteria=task.get('strict_success_criteria')
            )
            
            gpu_metrics = tracker.stop()
            
            tflops = engine.calculate_tflops(
                params_billion, 
                agent_result["total_generated_tokens"], 
                agent_result["total_inference_time_sec"]
            )
            
            result_row = {
                "Model": model_name,
                "Quantization": quant_type,
                "Params (B)": params_billion,
                "Trial": trial + 1,
                "Success": agent_result["success"],
                "Steps": agent_result["steps"],
                "Total Tokens": agent_result["total_generated_tokens"],
                "Inference Time (s)": agent_result["total_inference_time_sec"],
                "Tokens / Sec": agent_result["total_generated_tokens"] / agent_result["total_inference_time_sec"] if agent_result["total_inference_time_sec"] > 0 else 0,
                "Achieved TFLOPS": tflops,
                "Tools Called": ", ".join(agent_result.get("tools_called", []))
            }
            
            if gpu_metrics:
                result_row.update({
                    "Avg Power (W)": gpu_metrics["avg_power_W"],
                    "Peak Power (W)": gpu_metrics["peak_power_W"],
                    "Total Energy (J)": gpu_metrics["total_energy_J"],
                    "Avg VRAM (GB)": gpu_metrics["avg_vram_GB"],
                    "Peak VRAM (GB)": gpu_metrics["peak_vram_GB"],
                    "Avg GPU Util (%)": gpu_metrics["avg_gpu_util_%"]
                })
            
            all_trials_results.append(result_row)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        # If initialization or something else fatally fails
        all_trials_results.append({
            "Model": model_name,
            "Quantization": quant_type,
            "Params (B)": params_billion,
            "Success": False,
            "Error": str(e)
        })
        
    with open(args.output_json, "w") as f:
        json.dump(all_trials_results, f)

if __name__ == "__main__":
    main()
