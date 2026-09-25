#!/usr/bin/env python3
"""Benchmark entry-point.

Reads config.ini, iterates over enabled models × enabled agents, launches
the vLLM server container per model on demand, runs run_single.py, and
shuts down the vLLM container cleanly.

Output (stdout + stderr) of the entire run is tee'd to
  <output_dir>/<prefix>_<timestamp>/output.log
automatically — no manual piping needed.
"""

import os
import sys
import json
import time
import datetime
import subprocess
import configparser
import urllib.request
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_config(config_file: str = "config.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.optionxform = str  # preserve key case (model names)
    config.read(config_file)
    return config


def _load_models_from_section(config, section_name: str, dir_key: str) -> list:
    """Return [(model_name, model_path), …] for every True entry in *section_name*."""
    if section_name not in config:
        return []
    section = config[section_name]
    base_dir = section.get(dir_key, "").strip()
    models = []
    for key, val in section.items():
        if key == dir_key:
            continue
        if val.strip().lower() == "true":
            full_path = os.path.join(base_dir, key) if base_dir else key
            models.append((key, full_path))
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Result aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(df: pd.DataFrame, agent_id: str, output_dir: str, prefix: str):
    if df.empty:
        return

    grouped = df.groupby(["Model", "Quantization", "Params (B)"])

    agg_funcs = {
        "Success": [("sum", "sum"), ("mean", "mean")],
    }
    if "Trial" in df.columns:
        agg_funcs["Trial"] = "count"

    numeric_cols = [
        "Steps", "Total Tokens", "Inference Time (s)", "Tokens / Sec",
        "Achieved TFLOPS", "Weights Size (GB)", "Avg Power (W)", "Peak Power (W)",
        "Total Energy (J)", "Avg VRAM (GB)", "Peak VRAM (GB)", "Avg GPU Util (%)",
        "Expected_Steps", "Actual_Tool_Calls", "Redundant_Tool_Calls",
        "Parse_Errors", "Tool_Call_Efficiency", "Tokens_per_Useful_Step",
        "Tool_Repeat_Count", "Unique_Tools_Called",
    ]
    for col in numeric_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"

    bool_cols = ["Correct_Order", "Final_Tool_Without_Prior"]
    for col in bool_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"

    summary_df = grouped.agg(agg_funcs).reset_index()

    # Flatten multi-level columns
    summary_df.columns = [
        " ".join(col).strip() if isinstance(col, tuple) else col
        for col in summary_df.columns.values
    ]
    summary_df = summary_df.rename(columns={
        "Trial count": "Number of Trials",
        "Success sum": "Success Times",
        "Success mean": "Success Rate",
        "Correct_Order mean": "Correct_Order_Rate",
        "Final_Tool_Without_Prior mean": "Shortcut_Rate",
    })

    # Exec_Path distribution
    if "Exec_Path" in df.columns:
        path_counts = (
            df.groupby(["Model", "Quantization", "Params (B)"])["Exec_Path"]
            .value_counts()
            .unstack(fill_value=0)
        )
        path_counts.columns = [f"Path_{c}_Count" for c in path_counts.columns]
        for expected in [
            "Path_FAILED_Count", "Path_PERFECT_Count",
            "Path_WANDERING_Count", "Path_SHORTCUT_Count",
        ]:
            if expected not in path_counts.columns:
                path_counts[expected] = 0
        path_counts = path_counts.reset_index()
        summary_df = summary_df.merge(
            path_counts, on=["Model", "Quantization", "Params (B)"], how="left"
        )

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{prefix}_{agent_id}_raw_trials.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"{prefix}_{agent_id}_all_summary.csv"), index=False)

    df_success = df[df["Success"] == True]
    df_failure = df[df["Success"] == False]

    if not df_success.empty:
        df_success.groupby(["Model", "Quantization", "Params (B)"]).mean(numeric_only=True).reset_index().to_csv(
            os.path.join(output_dir, f"{prefix}_{agent_id}_successes_only.csv"), index=False
        )
    if not df_failure.empty:
        df_failure.groupby(["Model", "Quantization", "Params (B)"]).mean(numeric_only=True).reset_index().to_csv(
            os.path.join(output_dir, f"{prefix}_{agent_id}_failures_only.csv"), index=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tee helper — mirrors stdout/stderr to a log file simultaneously
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Wraps a stream so that every write goes both to the console and to log_file."""

    def __init__(self, original, log_file):
        self._orig = original
        self._log = log_file

    def write(self, data):
        self._orig.write(data)
        self._orig.flush()
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._orig.flush()
        self._log.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# vLLM container orchestration
# ─────────────────────────────────────────────────────────────────────────────

def start_vllm_server(
    model_to_serve: str,
    gpu_mem_util: float,
    max_model_len: int,
    port: int = 8000,
    container_name: str = "vllm-server",
) -> bool:
    """Launch the official vllm/vllm-openai container and wait for /health."""
    print(f"\n  [vLLM] Launching server for model: {model_to_serve} ...")
    subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    host_qwen  = os.environ.get("HOST_MODELS_QWEN", "/home/ror-technologies/Quantization/models")
    host_llama = os.environ.get("HOST_MODELS_LLAMA", "/home/ror-technologies/IT-helpdesk-agent/quantized_models")
    host_hf    = os.environ.get("HOST_HF_HOME", "/home/ror-technologies/.cache/huggingface")

    docker_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--gpus", "all",
        "--network", "host",
        "--ipc", "host",
        "-v", f"{host_qwen}:/models/qwen:ro",
        "-v", f"{host_llama}:/models/llama:ro",
        "-v", f"{host_hf}:/root/.cache/huggingface:ro",
        "-e", "HF_HOME=/root/.cache/huggingface",
        "-e", "VLLM_ATTENTION_BACKEND=FLASH_ATTN",
        "-e", "VLLM_USE_MODELSCOPE=False",
        "vllm/vllm-openai:latest",
        "--model", model_to_serve,
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--port", str(port),
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    res = subprocess.run(docker_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [vLLM] ✗ Failed to launch container: {res.stderr.strip()}")
        return False

    print(f"  [vLLM] Waiting for server readiness at http://localhost:{port}/health ...")
    max_wait = 300
    start_t = time.time()
    while time.time() - start_t < max_wait:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    elapsed = round(time.time() - start_t, 1)
                    print(f"  [vLLM] ✓ Server ready in {elapsed}s.")
                    return True
        except Exception:
            time.sleep(3)

    print(f"  [vLLM] ✗ Timed out waiting for server ({max_wait}s). Dumping logs:")
    logs = subprocess.run(["docker", "logs", "--tail", "40", container_name], capture_output=True, text=True)
    print(logs.stdout or logs.stderr)
    subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return False


def stop_vllm_server(container_name: str = "vllm-server"):
    """Stop the vLLM server container and wait for GPU memory release."""
    print(f"  [vLLM] Stopping container '{container_name}'...")
    subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config_path = os.environ.get("BENCHMARK_CONFIG", "config.ini")
    if not os.path.exists(config_path):
        print(f"Error: config file '{config_path}' not found.")
        sys.exit(1)

    config = parse_config(config_path)

    # ── Agents ──────────────────────────────────────────────────────────────
    agents_enabled = [
        key.upper()
        for key, val in config["AGENTS"].items()
        if val.lower() == "true"
    ]

    # ── Models ──────────────────────────────────────────────────────────────
    qwen_models  = _load_models_from_section(config, "QWEN_MODELS",  "qwen_models_dir")
    llama_models = _load_models_from_section(config, "LLAMA_MODELS", "llama_models_dir")
    all_models   = qwen_models + llama_models

    # ── Benchmark settings ──────────────────────────────────────────────────
    bench = config["BENCHMARK"]
    use_vllm        = bench.getboolean("use_vllm",          fallback=True)
    repeated_trials = bench.getboolean("repeated_trials",   fallback=True)
    num_trials      = bench.getint("number_of_trials",      fallback=10) if repeated_trials else 1
    save_traces     = bench.getboolean("save_traces",        fallback=False)
    output_dir_base = bench.get("output_dir",               fallback="./results")
    output_prefix   = bench.get("output_prefix",            fallback="benchmark")

    # ── Inference settings ───────────────────────────────────────────────────
    inf = config["INFERENCE"] if "INFERENCE" in config else {}
    max_new_tokens         = int(inf.get("max_new_tokens", 512))
    temperature            = float(inf.get("temperature",    0.2))
    do_sample              = inf.get("do_sample", "True").strip().lower() == "true"
    gpu_memory_utilization = float(inf.get("gpu_memory_utilization", 0.05))
    max_model_len          = int(inf.get("max_model_len", 4096))

    # ── Output directory for this run ────────────────────────────────────────
    timestamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir_base, f"{output_prefix}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # ── Redirect stdout/stderr through Tee so output.log is written automatically ──
    log_path = os.path.join(run_output_dir, "output.log")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    # ── Banner ───────────────────────────────────────────────────────────────
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║           Quantization Benchmark Suite                      ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"Timestamp           : {timestamp}")
    print(f"Config              : {config_path}")
    print(f"Agents              : {agents_enabled}")
    print(f"Qwen models         : {len(qwen_models)}")
    print(f"Llama models        : {len(llama_models)}")
    print(f"Total models        : {len(all_models)}")
    print(f"Trials/model        : {num_trials}")
    print(f"vLLM Engine         : {use_vllm}")
    print(f"GPU Memory Util     : {gpu_memory_utilization} (KV-cache minimized for accurate VRAM)")
    print(f"Output Directory    : {run_output_dir}")
    print(f"Log File            : {log_path}")
    print()

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")
    results_by_agent = {agent_id: [] for agent_id in agents_enabled}

    # ── Model Loop ───────────────────────────────────────────────────────────
    for model_name, model_path in all_models:
        print(f"\n{'━'*64}")
        print(f"  MODEL: {model_name}")
        print(f"{'━'*64}")

        # Resolve model path: if not a local folder on disk, it's an HF repo ID in HF cache
        if os.path.exists(model_path):
            model_to_serve = model_path
            print(f"  Source: Local directory ({model_path})")
        else:
            model_to_serve = model_name
            print(f"  Source: Hugging Face cache ({model_name})")

        vllm_started = False
        if use_vllm:
            vllm_started = start_vllm_server(
                model_to_serve=model_to_serve,
                gpu_mem_util=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            if not vllm_started:
                print(f"  ✗ Skipping model {model_name} due to vLLM server startup failure.")
                for agent_id in agents_enabled:
                    results_by_agent[agent_id].append({
                        "Model": model_name, "Success": False,
                        "Error": "vLLM server startup failed"
                    })
                continue

        try:
            # Benchmark each enabled agent with the currently loaded model
            for agent_id in agents_enabled:
                print(f"\n  ── Agent: {agent_id} ──")
                agent_output_dir = os.path.join(run_output_dir, agent_id)
                os.makedirs(agent_output_dir, exist_ok=True)

                safe_name   = model_name.replace("/", "__")
                output_json = os.path.join(run_output_dir, f"temp_{safe_name}_{agent_id}.json")

                cmd = [
                    sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single.py"),
                    "--model_path",     model_path,
                    "--model_name",     model_name,
                    "--dataset",        dataset_path,
                    "--agent_id",       agent_id,
                    "--num_trials",     str(num_trials),
                    "--output_json",    output_json,
                    "--output_dir",     agent_output_dir,
                    "--max_new_tokens", str(max_new_tokens),
                    "--temperature",    str(temperature),
                ]
                if use_vllm:
                    cmd.append("--use_vllm")
                if do_sample:
                    cmd.append("--do_sample")
                if save_traces:
                    cmd.append("--save_traces")

                try:
                    subprocess.run(cmd, check=True)
                    if os.path.exists(output_json):
                        with open(output_json, "r") as f:
                            results_by_agent[agent_id].extend(json.load(f))
                        os.remove(output_json)
                    else:
                        results_by_agent[agent_id].append({
                            "Model": model_name, "Success": False,
                            "Error": "Subprocess failed to write output.",
                        })
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error running {model_name} on {agent_id}. Exit code: {e.returncode}")
                    results_by_agent[agent_id].append({
                        "Model": model_name, "Success": False,
                        "Error": f"Subprocess crashed with code {e.returncode}",
                    })
        finally:
            if use_vllm and vllm_started:
                stop_vllm_server()

    # ── Aggregate and Save Results ────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Aggregating final results...")
    print(f"{'='*64}\n")

    for agent_id, agent_results in results_by_agent.items():
        if agent_results:
            df = pd.DataFrame(agent_results)
            agent_output_dir = os.path.join(run_output_dir, agent_id)
            aggregate_results(df, agent_id, agent_output_dir, output_prefix)
            print(f"  ✓ {agent_id}: Results saved to {agent_output_dir}")

    print(f"\n{'='*64}")
    print(f"  All benchmarks complete.")
    print(f"  Full directory : {run_output_dir}")
    print(f"  Output log     : {log_path}")
    print(f"{'='*64}\n")

    log_file.close()


if __name__ == "__main__":
    main()
