#!/usr/bin/env python3
"""Benchmark entry-point.

Reads config.ini, iterates over enabled agents × enabled models, and for
each (agent, model) pair spawns run_single.py as a subprocess.

Output (stdout + stderr) of the entire run is tee'd to
  <output_dir>/<prefix>_<timestamp>/output.log
automatically — no need to pipe manually.
"""

import os
import sys
import json
import datetime
import subprocess
import configparser
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
    """Wraps a file-like object so that every write goes both to the original
    stream and to *log_file*."""

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
        # subprocess.run needs a real fd; delegate to the original stream
        return self._orig.fileno()

    def isatty(self):
        return False


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
    max_new_tokens = int(inf.get("max_new_tokens", 512))
    temperature    = float(inf.get("temperature",    0.2))
    do_sample      = inf.get("do_sample", "True").strip().lower() == "true"

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
    print(f"Timestamp      : {timestamp}")
    print(f"Config         : {config_path}")
    print(f"Agents         : {agents_enabled}")
    print(f"Qwen models    : {len(qwen_models)}")
    print(f"Llama models   : {len(llama_models)}")
    print(f"Total models   : {len(all_models)}")
    print(f"Trials/model   : {num_trials}")
    print(f"vLLM           : {use_vllm}")
    print(f"Output dir     : {run_output_dir}")
    print(f"Log file       : {log_path}")
    print()

    # ── Run benchmarks ───────────────────────────────────────────────────────
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

    for agent_id in agents_enabled:
        print(f"\n{'='*64}")
        print(f"  TESTING AGENT: {agent_id}")
        print(f"{'='*64}\n")

        agent_results   = []
        agent_output_dir = os.path.join(run_output_dir, agent_id)
        os.makedirs(agent_output_dir, exist_ok=True)

        for model_name, model_path in all_models:
            print(f"--- Benchmarking: {model_name} ---")
            safe_name   = model_name.replace("/", "__")
            output_json = os.path.join(run_output_dir, f"temp_{safe_name}_{agent_id}.json")

            cmd = [
                sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single.py"),
                "--model_path",    model_path,
                "--model_name",    model_name,
                "--dataset",       dataset_path,
                "--agent_id",      agent_id,
                "--num_trials",    str(num_trials),
                "--output_json",   output_json,
                "--output_dir",    agent_output_dir,
                "--max_new_tokens", str(max_new_tokens),
                "--temperature",   str(temperature),
            ]
            if use_vllm:
                cmd.append("--use_vllm")
            if do_sample:
                cmd.append("--do_sample")
            if save_traces:
                cmd.append("--save_traces")

            try:
                # Inherit the tee'd stdout/stderr so subprocess output also goes to the log
                subprocess.run(cmd, check=True)

                if os.path.exists(output_json):
                    with open(output_json, "r") as f:
                        agent_results.extend(json.load(f))
                    os.remove(output_json)
                else:
                    agent_results.append({
                        "Model": model_name, "Success": False,
                        "Error": "Subprocess failed to write output.",
                    })

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error running {model_name}. Exit code: {e.returncode}")
                agent_results.append({
                    "Model": model_name, "Success": False,
                    "Error": f"Subprocess crashed with code {e.returncode}",
                })

        if agent_results:
            df = pd.DataFrame(agent_results)
            aggregate_results(df, agent_id, agent_output_dir, output_prefix)
            print(f"\n✓ {agent_id} complete — results saved to {agent_output_dir}")

    print(f"\n{'='*64}")
    print(f"  All benchmarks finished.")
    print(f"  Results + log: {run_output_dir}")
    print(f"{'='*64}\n")

    log_file.close()


if __name__ == "__main__":
    main()
