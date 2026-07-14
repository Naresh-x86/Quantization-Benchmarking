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
        "Trial": "count",  # Number of trials
        "Success": ["sum", "mean"],  # sum = success_times, mean = success_rate
    }
    
    # Only average these if they exist (numeric columns)
    numeric_cols = ["Steps", "Total Tokens", "Inference Time (s)", "Tokens / Sec", 
                    "Achieved TFLOPS", "Weights Size (GB)", "Avg Power (W)", "Peak Power (W)", 
                    "Total Energy (J)", "Avg VRAM (GB)", "Peak VRAM (GB)", "Avg GPU Util (%)"]
                    
    for col in numeric_cols:
        if col in df.columns:
            agg_funcs[col] = "mean"
            
    summary_df = grouped.agg(agg_funcs).reset_index()
    
    # Flatten multi-level columns
    summary_df.columns = [' '.join(col).strip() for col in summary_df.columns.values]
    
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        "Trial count": "Number of Trials",
        "Success sum": "Success Times",
        "Success mean": "Success Rate"
    })
    
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

def main():
    if not os.path.exists("config.ini"):
        print("Error: config.ini not found.")
        return
        
    config = parse_config("config.ini")
    
    # Extract config
    agents_enabled = [key.upper() for key, val in config["AGENTS"].items() if val.lower() == "true"]
    models_dir = config["MODELS"].get("models_dir", "./models")
    
    models_enabled = [
        key for key, val in config["MODELS"].items() 
        if key != "models_dir" and val.lower() == "true"
    ]
    
    use_vllm = config["BENCHMARK"].getboolean("use_vllm", fallback=True)
    repeated_trials = config["BENCHMARK"].getboolean("repeated_trials", fallback=True)
    num_trials = config["BENCHMARK"].getint("number_of_trials", fallback=10) if repeated_trials else 1
    output_dir = config["BENCHMARK"].get("output_dir", fallback="./results")
    output_prefix = config["BENCHMARK"].get("output_prefix", fallback="benchmark")
    dataset_path = "dataset.json"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting Benchmark Suite V2")
    print(f"Agents: {agents_enabled}")
    print(f"Models: {models_enabled}")
    print(f"Trials per model: {num_trials}\n")
    
    for agent_id in agents_enabled:
        print(f"\n=======================================================")
        print(f"=== TESTING AGENT: {agent_id} ===")
        print(f"=======================================================\n")
        
        agent_results = []
        
        for model_name in models_enabled:
            model_path = os.path.join(models_dir, model_name)
            
            # Note: The model_path might be relative to config.ini's models_dir, we don't strictly check exist here
            # as it will be checked by the subprocess, but good to warn.
            
            print(f"--- Benchmarking Model: {model_name} on {agent_id} ---")
            output_json = f"temp_result_{model_name}_{agent_id}.json"
            
            cmd = [
                sys.executable, "run_single.py",
                "--model_path", model_path,
                "--model_name", model_name,
                "--dataset", dataset_path,
                "--agent_id", agent_id,
                "--num_trials", str(num_trials),
                "--output_json", output_json,
                "--output_dir", output_dir
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
            aggregate_results(df, agent_id, output_dir, output_prefix)
            print(f"\n--- {agent_id} Benchmark Complete! Results saved to {output_dir} ---")

if __name__ == "__main__":
    main()
