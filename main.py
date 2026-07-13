import os
import sys
import json
import argparse
import subprocess
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Quantization Agentic Benchmark")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory containing the models to benchmark")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to the dataset JSON")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM instead of Transformers")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory {args.models_dir} not found.")
        return
        
    models_to_test = [d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]
    
    results = []
    
    for model_name in models_to_test:
        print(f"\n=======================================================")
        print(f"--- Benchmarking Model: {model_name} ---")
        print(f"=======================================================\n")
        
        model_path = os.path.join(args.models_dir, model_name)
        output_json = f"temp_result_{model_name}.json"
        
        cmd = [
            sys.executable, "run_single.py",
            "--model_path", model_path,
            "--model_name", model_name,
            "--dataset", args.dataset,
            "--output_json", output_json
        ]
        
        if args.use_vllm:
            cmd.append("--use_vllm")
            
        try:
            # We run in a subprocess so that when the process exits, ALL VRAM is natively freed by the OS.
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_json):
                with open(output_json, "r") as f:
                    results.append(json.load(f))
                os.remove(output_json)
            else:
                results.append({"Model": model_name, "Success": False, "Error": "Subprocess failed to write output."})
                
        except subprocess.CalledProcessError as e:
            print(f"Error running model {model_name}. Exit code: {e.returncode}")
            results.append({"Model": model_name, "Success": False, "Error": f"Subprocess crashed with code {e.returncode}"})

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n--- Benchmark Complete! Results saved to {args.output} ---")
    print(df.to_string())

if __name__ == "__main__":
    main()
