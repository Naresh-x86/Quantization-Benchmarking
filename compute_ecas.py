import pandas as pd
import os
import json

RESULTS_DIR = "results/benchmark_20260810_171506"
AGENTS = ["AGENT_1_SUPPORT", "AGENT_2_IT_HELPDESK", "AGENT_3_FINANCE"]

# FP16 baseline VRAM lookup (Peak VRAM GB) per param size
# We'll compute these dynamically from the data

all_agent_data = {}
for agent in AGENTS:
    csv_path = os.path.join(RESULTS_DIR, agent, f"benchmark_{agent}_all_summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        all_agent_data[agent] = df

# Build FP16 baseline VRAM reference per (model_family, param_size)
def get_model_family(model_name):
    if "Qwen" in model_name:
        return "Qwen"
    elif "Llama" in model_name or "llama" in model_name:
        return "Llama"
    return "Unknown"

def get_fp16_vram(agent_df, model_name, params_b):
    family = get_model_family(model_name)
    # Find the FP16 baseline with same param count and family
    for _, row in agent_df.iterrows():
        row_model = str(row.get("Model", "")).strip()
        row_quant = str(row.get("Quantization", "")).strip()
        row_params = float(row.get("Params (B)", 0))
        row_family = get_model_family(row_model)
        if row_family == family and abs(row_params - params_b) < 0.5 and row_quant == "FP16":
            vram = row.get("Peak VRAM (GB) mean", 0)
            if pd.notna(vram) and float(vram) > 0:
                return float(vram)
    return None

def compute_ecas_for_row(row, agent_df):
    model = str(row.get("Model", "")).strip()
    params_b = float(row.get("Params (B)", 0))
    
    # Get number of trials
    num_trials_col = row.get("Number of Trials", None)
    if pd.isna(num_trials_col) or num_trials_col == 0:
        return None  # No data
    num_trials = float(num_trials_col)
    
    # S_accuracy
    perfect_count = float(row.get("Path_PERFECT_Count", 0)) if pd.notna(row.get("Path_PERFECT_Count", 0)) else 0
    p_perfect = perfect_count / num_trials if num_trials > 0 else 0
    s_rate = float(row.get("Success Rate", 0)) if pd.notna(row.get("Success Rate", 0)) else 0
    s_accuracy = (0.70 * p_perfect + 0.30 * s_rate) * 100
    
    # S_tool_quality
    e_tool_raw = float(row.get("Tool_Call_Efficiency mean", 0)) if pd.notna(row.get("Tool_Call_Efficiency mean", 0)) else 0
    e_tool = min(1.0, e_tool_raw)
    
    parse_errors = float(row.get("Parse_Errors mean", 0)) if pd.notna(row.get("Parse_Errors mean", 0)) else 0
    steps = float(row.get("Steps mean", 1)) if pd.notna(row.get("Steps mean", 1)) else 1
    parse_error_rate = parse_errors / steps if steps > 0 else 0
    parse_accuracy = 1 - parse_error_rate
    
    c_order = float(row.get("Correct_Order_Rate", 0)) if pd.notna(row.get("Correct_Order_Rate", 0)) else 0
    
    s_tool_quality = (0.40 * e_tool + 0.30 * parse_accuracy + 0.30 * c_order) * 100
    
    # S_efficiency
    peak_vram = float(row.get("Peak VRAM (GB) mean", 0)) if pd.notna(row.get("Peak VRAM (GB) mean", 0)) else 0
    fp16_vram = get_fp16_vram(agent_df, model, params_b)
    
    if fp16_vram and peak_vram > 0:
        r_vram = fp16_vram / peak_vram
    else:
        r_vram = 1.0
    
    tok_per_sec = float(row.get("Tokens / Sec mean", 0)) if pd.notna(row.get("Tokens / Sec mean", 0)) else 0
    throughput_norm = tok_per_sec / 100.0
    
    s_efficiency = min(100.0, 50 * r_vram + 50 * throughput_norm)
    
    ecas = (0.45 * s_accuracy) + (0.35 * s_tool_quality) + (0.20 * s_efficiency)
    
    return {
        "S_accuracy": round(s_accuracy, 2),
        "S_tool_quality": round(s_tool_quality, 2),
        "S_efficiency": round(s_efficiency, 2),
        "ECAS": round(ecas, 2),
        "P_perfect": round(p_perfect, 4),
        "S_rate": round(s_rate, 4),
        "E_tool": round(e_tool, 4),
        "ParseErrorRate": round(parse_error_rate, 4),
        "C_order": round(c_order, 4),
        "R_vram": round(r_vram, 4),
        "TokPerSec": round(tok_per_sec, 2),
        "PeakVRAM": round(peak_vram, 2),
    }

# Compute ECAS for every model in every agent
results = {}  # model -> {agent -> ecas_dict}
for agent, df in all_agent_data.items():
    for _, row in df.iterrows():
        model = str(row.get("Model", "")).strip()
        quant = str(row.get("Quantization", "")).strip()
        params = float(row.get("Params (B)", 0))
        
        ecas_result = compute_ecas_for_row(row, df)
        if ecas_result is None:
            continue
        
        key = model
        if key not in results:
            results[key] = {"quant": quant, "params": params, "agents": {}}
        results[key]["agents"][agent] = ecas_result

# Print results as structured output
print("=" * 120)
print(f"{'Model':<52} {'Quant':<10} {'Params':>6} {'Agent1':>8} {'Agent2':>8} {'Agent3':>8} {'Global':>8}")
print("=" * 120)

global_scores = []
for model, data in sorted(results.items(), key=lambda x: (-x[1]["params"], x[0])):
    agent_ecas = []
    a1 = data["agents"].get("AGENT_1_SUPPORT", {}).get("ECAS", None)
    a2 = data["agents"].get("AGENT_2_IT_HELPDESK", {}).get("ECAS", None)
    a3 = data["agents"].get("AGENT_3_FINANCE", {}).get("ECAS", None)
    
    scores = [s for s in [a1, a2, a3] if s is not None]
    global_ecas = round(sum(scores) / len(scores), 2) if scores else 0
    
    a1_str = f"{a1:.1f}" if a1 is not None else "N/A"
    a2_str = f"{a2:.1f}" if a2 is not None else "N/A"
    a3_str = f"{a3:.1f}" if a3 is not None else "N/A"
    
    print(f"{model:<52} {data['quant']:<10} {data['params']:>5.0f}B {a1_str:>8} {a2_str:>8} {a3_str:>8} {global_ecas:>8.1f}")
    
    global_scores.append({
        "model": model,
        "quant": data["quant"],
        "params": data["params"],
        "agent1": a1,
        "agent2": a2,
        "agent3": a3,
        "global": global_ecas,
        "details": data["agents"]
    })

print("=" * 120)

# Also dump JSON for consumption
with open("ecas_scores.json", "w") as f:
    json.dump(global_scores, f, indent=2)

print("\nSaved ecas_scores.json")
