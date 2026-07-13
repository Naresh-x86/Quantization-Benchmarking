import os
import json
import argparse
import pandas as pd
from inference import LLMEngine
from agent import ReActAgent
from metrics import GPUTracker

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Quantization Agentic Benchmark")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory containing the models to benchmark")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to the dataset JSON")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM instead of Transformers")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV file")
    return parser.parse_args()

def get_params_billion_from_name(model_name: str) -> float:
    # Extremely basic heuristic: extract X from 'X.XB' or 'XB' in the model name
    import re
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    if match:
        return float(match.group(1))
    return 7.0 # Default fallback if unparseable

def main():
    args = parse_args()
    
    with open(args.dataset, "r") as f:
        tasks = json.load(f)
        
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory {args.models_dir} not found.")
        return
        
    models_to_test = [d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]
    
    results = []
    
    for model_name in models_to_test:
        print(f"--- Benchmarking Model: {model_name} ---")
        model_path = os.path.join(args.models_dir, model_name)
        params_billion = get_params_billion_from_name(model_name)
        
        # We try to infer the quantization type from the folder name
        quant_type = "baseline"
        if "INT8" in model_name: quant_type = "INT8"
        elif "INT4" in model_name: quant_type = "INT4"
        elif "FP8" in model_name: quant_type = "FP8"
        elif "FP16" in model_name: quant_type = "FP16"
        
        try:
            # Initialize engine
            engine = LLMEngine(model_path, use_vllm=args.use_vllm, quant_type=quant_type)
            agent = ReActAgent(engine)
            
            # Start GPU Tracking
            tracker = GPUTracker(poll_interval=0.05)
            tracker.start()
            
            # We only run the first task for this benchmark to get a clean run
            task = tasks[0]
            print(f"Running task: {task['task_id']}")
            
            agent_result = agent.run(task['description'])
            
            # Stop tracking
            gpu_metrics = tracker.stop()
            
            # Compute FLOPS
            tflops = engine.calculate_tflops(
                params_billion, 
                agent_result["total_generated_tokens"], 
                agent_result["total_inference_time_sec"]
            )
            
            # Aggregate Results
            result_row = {
                "Model": model_name,
                "Quantization": quant_type,
                "Params (B)": params_billion,
                "Success": agent_result["success"],
                "Steps": agent_result["steps"],
                "Total Tokens": agent_result["total_generated_tokens"],
                "Inference Time (s)": agent_result["total_inference_time_sec"],
                "Tokens / Sec": agent_result["total_generated_tokens"] / agent_result["total_inference_time_sec"] if agent_result["total_inference_time_sec"] > 0 else 0,
                "Achieved TFLOPS": tflops
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
            
            results.append(result_row)
            
            # Explicit cleanup to free VRAM for the next model
            del agent
            del engine
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            results.append({
                "Model": model_name,
                "Quantization": quant_type,
                "Success": False,
                "Error": str(e)
            })

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n--- Benchmark Complete! Results saved to {args.output} ---")
    print(df.to_string())

if __name__ == "__main__":
    main()
