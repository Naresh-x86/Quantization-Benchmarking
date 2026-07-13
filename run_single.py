import os
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
        
    model_name = args.model_name
    model_path = args.model_path
    params_billion = get_params_billion_from_name(model_name)
    
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
        
        task = tasks[0]
        
        agent_result = agent.run(task['description'])
        
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
            
    except Exception as e:
        result_row = {
            "Model": model_name,
            "Quantization": quant_type,
            "Success": False,
            "Error": str(e)
        }
        
    with open(args.output_json, "w") as f:
        json.dump(result_row, f)

if __name__ == "__main__":
    main()
