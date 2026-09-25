import os
import sys
import json
import subprocess
import configparser
import datetime
import threading
import atexit
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# OS-level Tee: mirrors all stdout + stderr (process & subprocesses) to output.log
# ─────────────────────────────────────────────────────────────────────────────

class OSTee:
    """Redirects OS-level stdout and stderr (file descriptors 1 and 2) through
    a pipe so that all output from Python, C extensions, and subprocesses is
    simultaneously printed to the console in real-time and written to a log file.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8")

        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)

        self.r, self.w = os.pipe()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        # Redirect stdout and stderr to the write end of our pipe
        os.dup2(self.w, self.stdout_fd)
        os.dup2(self.w, self.stderr_fd)
        os.close(self.w)

        self._closed = False
        atexit.register(self.close)

    def _worker(self):
        with os.fdopen(self.r, "rb", buffering=0) as pipe_reader:
            while True:
                chunk = pipe_reader.read(4096)
                if not chunk:
                    break
                # Echo to real console
                try:
                    os.write(self.saved_stdout_fd, chunk)
                except OSError:
                    pass
                # Mirror to log file
                try:
                    self.log_file.write(chunk.decode("utf-8", errors="replace"))
                    self.log_file.flush()
                except Exception:
                    pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            # Restore original descriptors
            os.dup2(self.saved_stdout_fd, self.stdout_fd)
            os.dup2(self.saved_stderr_fd, self.stderr_fd)
            os.close(self.saved_stdout_fd)
            os.close(self.saved_stderr_fd)
            self.thread.join(timeout=1.0)
            self.log_file.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity for model names
    config.read(config_file)
    return config


def _load_models_from_section(config, section_name: str, dir_key: str) -> list:
    """Read a model section and return a list of (model_name, model_path) tuples
    for every entry whose value is 'true' (case-insensitive).
    """
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

def aggregate_results(df, agent_id, output_dir, prefix):
    if df.empty:
        return

    grouped = df.groupby(["Model", "Quantization", "Params (B)"])

    agg_funcs = {
        "Success": [("sum", "sum"), ("mean", "mean")],
    }
    if "Trial" in df.columns:
        agg_funcs["Trial"] = "count"

    numeric_cols = ["Steps", "Total Tokens", "Inference Time (s)", "Tokens / Sec",
                    "Achieved TFLOPS", "Weights Size (GB)", "Avg Power (W)", "Peak Power (W)",
                    "Total Energy (J)", "Avg VRAM (GB)", "Peak VRAM (GB)", "Avg GPU Util (%)",
                    "Expected_Steps", "Actual_Tool_Calls", "Redundant_Tool_Calls",
                    "Parse_Errors", "Tool_Call_Efficiency", "Tokens_per_Useful_Step",
                    "Tool_Repeat_Count", "Unique_Tools_Called"]

    for col in numeric_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"

    bool_cols = ["Correct_Order", "Final_Tool_Without_Prior"]
    for col in bool_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"

    summary_df = grouped.agg(agg_funcs).reset_index()

    summary_df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col
                          for col in summary_df.columns.values]

    summary_df = summary_df.rename(columns={
        "Trial count": "Number of Trials",
        "Success sum": "Success Times",
        "Success mean": "Success Rate",
        "Correct_Order mean": "Correct_Order_Rate",
        "Final_Tool_Without_Prior mean": "Shortcut_Rate"
    })

    if "Exec_Path" in df.columns:
        path_counts = df.groupby(["Model", "Quantization", "Params (B)"])["Exec_Path"].value_counts().unstack(fill_value=0)
        path_counts.columns = [f"Path_{col}_Count" for col in path_counts.columns]

        for expected in ["Path_FAILED_Count", "Path_PERFECT_Count", "Path_WANDERING_Count", "Path_SHORTCUT_Count"]:
            if expected not in path_counts.columns:
                path_counts[expected] = 0

        path_counts = path_counts.reset_index()
        summary_df = summary_df.merge(path_counts, on=["Model", "Quantization", "Params (B)"], how="left")

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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists("config.ini"):
        print("Error: config.ini not found.")
        return

    config = parse_config("config.ini")

    # ── Benchmark settings ────────────────────────────────────────────────────
    agents_enabled = [key.upper() for key, val in config["AGENTS"].items() if val.lower() == "true"]

    qwen_models  = _load_models_from_section(config, "QWEN_MODELS",  "qwen_models_dir")
    llama_models = _load_models_from_section(config, "LLAMA_MODELS", "llama_models_dir")
    all_models   = qwen_models + llama_models

    use_vllm        = config["BENCHMARK"].getboolean("use_vllm", fallback=True)
    repeated_trials = config["BENCHMARK"].getboolean("repeated_trials", fallback=True)
    num_trials      = config["BENCHMARK"].getint("number_of_trials", fallback=10) if repeated_trials else 1
    save_traces     = config["BENCHMARK"].getboolean("save_traces", fallback=True)
    output_dir_base = config["BENCHMARK"].get("output_dir", fallback="./results")
    output_prefix   = config["BENCHMARK"].get("output_prefix", fallback="benchmark")

    # ── Inference settings ────────────────────────────────────────────────────
    inf_section    = config["INFERENCE"] if "INFERENCE" in config else {}
    temperature    = float(inf_section.get("temperature",    "0.2"))
    top_p          = float(inf_section.get("top_p",          "0.95"))
    max_new_tokens = int(inf_section.get("max_new_tokens",  "512"))
    do_sample      = inf_section.get("do_sample", "True").strip().lower() == "true"
    max_steps      = int(inf_section.get("max_steps",      "15"))

    # ── Create timestamped run directory and start automated Tee logging ──────
    dataset_path = "dataset.json"
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir_base, f"{output_prefix}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    log_path = os.path.join(run_output_dir, "output.log")
    tee = OSTee(log_path)

    try:
        # ── Suite Header (printed to terminal and output.log simultaneously) ──
        print("=======================================================")
        print("       QUANTIZATION AGENT BENCHMARK SUITE")
        print("=======================================================")
        print(f"Timestamp:        {timestamp}")
        print(f"Output Directory: {run_output_dir}")
        print(f"Log File:         {log_path}")
        print(f"SET TEMPERATURE:  {temperature}")
        print("-------------------------------------------------------")
        print("Inference Hyperparameters:")
        print(f"  • Temperature:    {temperature}")
        print(f"  • Top-p:          {top_p}")
        print(f"  • Max New Tokens: {max_new_tokens}")
        print(f"  • Sampling Mode:  {'Sampling (do_sample=True)' if do_sample else 'Greedy (do_sample=False)'}")
        print(f"  • Max Steps:      {max_steps}")
        print("-------------------------------------------------------")
        print("Benchmark Configuration:")
        print(f"  • Active Agents:  {', '.join(agents_enabled) if agents_enabled else 'None'}")
        print(f"  • Qwen Models:    {len(qwen_models)} enabled")
        print(f"  • Llama Models:   {len(llama_models)} enabled")
        print(f"  • Total Models:   {len(all_models)}")
        print(f"  • Trials / Model: {num_trials}")
        print(f"  • Engine:         {'vLLM' if use_vllm else 'HuggingFace'}")
        print(f"  • Save Traces:    {save_traces}")
        print("=======================================================\n")

        # ── Run per-agent benchmarks ──────────────────────────────────────────
        for agent_id in agents_enabled:
            print(f"\n=======================================================")
            print(f"=== TESTING AGENT: {agent_id} ===")
            print(f"=======================================================\n")

            agent_results = []
            agent_output_dir = os.path.join(run_output_dir, agent_id)
            os.makedirs(agent_output_dir, exist_ok=True)

            for model_name, model_path in all_models:
                print(f"--- Benchmarking Model: {model_name} on {agent_id} ---")
                safe_model_name = model_name.replace("/", "__")
                output_json = f"temp_result_{safe_model_name}_{agent_id}.json"

                cmd = [
                    sys.executable, "-u", "run_single.py",
                    "--model_path",     model_path,
                    "--model_name",     model_name,
                    "--dataset",        dataset_path,
                    "--agent_id",       agent_id,
                    "--num_trials",     str(num_trials),
                    "--output_json",    output_json,
                    "--output_dir",     agent_output_dir,
                    "--temperature",    str(temperature),
                    "--top_p",          str(top_p),
                    "--max_new_tokens", str(max_new_tokens),
                    "--max_steps",      str(max_steps),
                ]

                if use_vllm:
                    cmd.append("--use_vllm")

                if save_traces:
                    cmd.append("--save_traces")

                if not do_sample:
                    cmd.append("--no_sample")

                try:
                    subprocess.run(cmd, check=True)

                    if os.path.exists(output_json):
                        with open(output_json, "r") as f:
                            trials_data = json.load(f)
                            agent_results.extend(trials_data)
                        os.remove(output_json)
                    else:
                        agent_results.append({"Model": model_name, "Success": False, "Error": "Subprocess failed to write output."})

                except subprocess.CalledProcessError as e:
                    print(f"Error running model {model_name}. Exit code: {e.returncode}")
                    agent_results.append({"Model": model_name, "Success": False, "Error": f"Subprocess crashed with code {e.returncode}"})

            # Aggregate and save agent results
            if agent_results:
                df = pd.DataFrame(agent_results)
                aggregate_results(df, agent_id, agent_output_dir, output_prefix)
                print(f"\n--- {agent_id} Benchmark Complete! Results saved to {agent_output_dir} ---")

        print(f"\nBenchmark run finished! All output saved to {run_output_dir}")

    finally:
        tee.close()


if __name__ == "__main__":
    main()
