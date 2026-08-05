import os
import sys
import json
import subprocess
import configparser
import pandas as pd

def parse_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity for model names
    config.read(config_file)
    return config

def aggregate_results(df, agent_id, output_dir, prefix):
    # Base aggregation on the 'Model' and 'Quantization' columns
    # We want to aggregate the raw trials into averages and calculate success rates
    if df.empty:
        return
        
    grouped = df.groupby(["Model", "Quantization", "Params (B)"])
    
    agg_funcs = {
        "Success": [("sum", "sum"), ("mean", "mean")],  # sum = success_times, mean = success_rate
    }
    if "Trial" in df.columns:
        agg_funcs["Trial"] = "count"

    
    # Only average these if they exist (numeric columns)
    numeric_cols = ["Steps", "Total Tokens", "Inference Time (s)", "Tokens / Sec", 
                    "Achieved TFLOPS", "Weights Size (GB)", "Avg Power (W)", "Peak Power (W)", 
                    "Total Energy (J)", "Avg VRAM (GB)", "Peak VRAM (GB)", "Avg GPU Util (%)",
                    # Tool-call quality metrics
                    "Expected_Steps", "Actual_Tool_Calls", "Redundant_Tool_Calls",
                    "Parse_Errors", "Tool_Call_Efficiency", "Tokens_per_Useful_Step",
                    "Tool_Repeat_Count", "Unique_Tools_Called"]
                    
    for col in numeric_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"

    # Boolean columns to average (gives a rate)
    bool_cols = ["Correct_Order", "Final_Tool_Without_Prior"]
    for col in bool_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"
            
    summary_df = grouped.agg(agg_funcs).reset_index()
    
    # Flatten multi-level columns
    summary_df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col
                          for col in summary_df.columns.values]
    
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        "Trial count": "Number of Trials",
        "Success sum": "Success Times",
        "Success mean": "Success Rate",
        "Correct_Order mean": "Correct_Order_Rate",
        "Final_Tool_Without_Prior mean": "Shortcut_Rate"
    })
    
    # Add Exec_Path distribution counts per model/quant group
    if "Exec_Path" in df.columns:
        path_counts = df.groupby(["Model", "Quantization", "Params (B)"])["Exec_Path"].value_counts().unstack(fill_value=0)
        path_counts.columns = [f"Path_{col}_Count" for col in path_counts.columns]
        
        # Ensure standard path columns always exist
        for expected in ["Path_FAILED_Count", "Path_PERFECT_Count", "Path_WANDERING_Count", "Path_SHORTCUT_Count"]:
            if expected not in path_counts.columns:
                path_counts[expected] = 0
                
        path_counts = path_counts.reset_index()
        summary_df = summary_df.merge(path_counts, on=["Model", "Quantization", "Params (B)"], how="left")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all trials raw
    df.to_csv(os.path.join(output_dir, f"{prefix}_{agent_id}_raw_trials.csv"), index=False)
    
    # Save aggregated "all" summary
    summary_df.to_csv(os.path.join(output_dir, f"{prefix}_{agent_id}_all_summary.csv"), index=False)
    
    # Save successes vs failures summaries (filter raw, then aggregate? Or just filter raw)
    # The user requested "considering only successes, considering only failures"
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

def _load_models_from_section(config, section_name: str, dir_key: str) -> list:
    """Read a model section and return a list of (model_name, model_path) tuples
    for every entry whose value is 'true' (case-insensitive).

    Args:
        config:       ConfigParser instance
        section_name: e.g. 'QWEN_MODELS' or 'LLAMA_MODELS'
        dir_key:      the key inside the section that holds the base directory,
                      e.g. 'qwen_models_dir' or 'llama_models_dir'
    """
    if section_name not in config:
        return []

    section = config[section_name]
    base_dir = section.get(dir_key, "").strip()

    models = []
    for key, val in section.items():
        if key == dir_key:
            continue                          # skip the directory key itself
        if val.strip().lower() == "true":
            full_path = os.path.join(base_dir, key) if base_dir else key
            models.append((key, full_path))

    return models

def main():
    if not os.path.exists("config.ini"):
        print("Error: config.ini not found.")
        return
        
    config = parse_config("config.ini")
    
    # Extract config
    agents_enabled = [key.upper() for key, val in config["AGENTS"].items() if val.lower() == "true"]

    # Build unified model list from both sections, preserving Qwen-then-Llama order
    qwen_models  = _load_models_from_section(config, "QWEN_MODELS",  "qwen_models_dir")
    llama_models = _load_models_from_section(config, "LLAMA_MODELS", "llama_models_dir")
    all_models   = qwen_models + llama_models
    
    use_vllm = config["BENCHMARK"].getboolean("use_vllm", fallback=True)
    repeated_trials = config["BENCHMARK"].getboolean("repeated_trials", fallback=True)
    num_trials = config["BENCHMARK"].getint("number_of_trials", fallback=10) if repeated_trials else 1
    output_dir_base = config["BENCHMARK"].get("output_dir", fallback="./results")
    output_prefix = config["BENCHMARK"].get("output_prefix", fallback="benchmark")
    dataset_path = "dataset.json"
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir_base, f"{output_prefix}_{timestamp}")
    
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Starting Benchmark Suite V2")
    print(f"Agents: {agents_enabled}")
    print(f"Qwen models enabled:  {len(qwen_models)}")
    print(f"Llama models enabled: {len(llama_models)}")
    print(f"Total models: {len(all_models)}")
    print(f"Trials per model: {num_trials}")
    print(f"Output Directory: {run_output_dir}\n")
    
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
                sys.executable, "run_single.py",
                "--model_path", model_path,
                "--model_name", model_name,
                "--dataset", dataset_path,
                "--agent_id", agent_id,
                "--num_trials", str(num_trials),
                "--output_json", output_json,
                "--output_dir", agent_output_dir
            ]
            
            if use_vllm:
                cmd.append("--use_vllm")
                
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

if __name__ == "__main__":
    main()
