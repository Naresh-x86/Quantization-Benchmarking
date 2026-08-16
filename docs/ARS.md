# Agent Readiness Score (ARS)

## Motivation & Purpose

Model selection platforms such as Claude, Jules, and OpenClaw Hermes present users with a curated list of models alongside qualitative fitness labels (e.g. *"fit for everyday tasks"*). These labels are hand-authored by the provider for fundamentally different model architectures serving general-purpose workloads.

The **Agent Readiness Score (ARS)** adapts this concept for an industrial, quantization-aware setting. Rather than comparing architecturally distinct models, the ARS evaluates the **same base model across different quantization formats** (FP16, FP8, INT4 AWQ, INT8 GPTQ, BNB-4bit, etc.) on **domain-specific agentic tasks**. It answers a single question:

> *"How reliably can this quantized model variant complete agentic tool-calling tasks on a given hardware target, and at what operational cost?"*

The score is **empirically computed** from benchmark telemetry — not hand-assigned — making it reproducible and hardware-aware.

---

## Score Definition

The ARS is a single scalar value on a **0–100 scale**, computed as a weighted combination of five normalised sub-scores (each on a 0–1 scale):

```
ARS = 100 × (w_R·R  +  w_P·P  +  w_E·E  +  w_D·D  +  w_C·C)
```

| Symbol | Pillar | Default Weight | What It Measures |
| :---: | :--- | :---: | :--- |
| **R** | Task Reliability | 35% | Does the agent complete tasks correctly without shortcuts? |
| **P** | Tool Precision | 25% | Does it call the right tools, in the right order, with valid JSON? |
| **E** | Efficiency | 20% | How resource-efficient is each useful action (tokens, energy, latency)? |
| **D** | Deployability | 15% | Can the model physically run on the target hardware (VRAM budget)? |
| **C** | Operational Cost | 5% | What is the sustained power draw and GPU utilisation? |

---

## Sub-Score Formulations

### Pillar 1 — Task Reliability (R)

```
R = Success_Rate × (1 − Shortcut_Rate)
```

* **Success_Rate**: Fraction of trials where the agent satisfied all `must_call` constraints and avoided all `must_not_call` tools.
* **Shortcut_Rate**: Fraction of trials where the agent called the final required tool without completing prerequisite steps (a "lucky guess" penalty).

The shortcut penalty prevents inflated scores from models that skip reasoning steps but happen to produce a correct final action.

---

### Pillar 2 — Tool Precision (P)

```
P = 0.30 × Correct_Order_Rate
  + 0.25 × Tool_Call_Efficiency
  + 0.20 × (1 − Parse_Error_Rate)
  + 0.25 × Perfect_Path_Rate
```

| Component | Definition |
| :--- | :--- |
| **Correct_Order_Rate** | Fraction of trials where `must_call` tools appeared in the specified sequence. |
| **Tool_Call_Efficiency** | `expected_steps / actual_tool_calls`. A value of 1.0 means no redundant calls. |
| **Parse_Error_Rate** | `parse_errors / total_steps`. Measures the model's ability to emit syntactically valid `Action:` / `Action Input:` blocks. |
| **Perfect_Path_Rate** | Fraction of trials classified as `PERFECT` — all required tools called in correct order with zero extras, zero repeats, and zero parse errors. |

This pillar captures reasoning trace quality. In industrial settings, a "wandering" success (correct outcome but 5 redundant API calls) incurs real latency and cost, so it must be penalised relative to a clean execution.

---

### Pillar 3 — Efficiency (E)

```
E = 0.40 × norm⁻¹(Tokens_per_Useful_Step)
  + 0.30 × norm⁻¹(Total_Energy_J)
  + 0.30 × norm⁻¹(Inference_Time_s)
```

All three components are **min-max normalised** across the set of model variants under evaluation, with inversion (`norm⁻¹`) so that lower raw values yield higher scores:

```
norm⁻¹(x) = (x_max − x) / (x_max − x_min)
```

| Component | Source Metric | Why It Matters |
| :--- | :--- | :--- |
| **Tokens_per_Useful_Step** | `Tokens_per_Useful_Step mean` | Token verbosity per required tool call — lower = more concise reasoning. |
| **Total_Energy_J** | `Total Energy (J) mean` | Joules consumed per task — directly maps to electricity cost ($/kWh). |
| **Inference_Time_s** | `Inference Time (s) mean` | Wall-clock latency — critical for real-time agentic loops. |

---

### Pillar 4 — Deployability (D)

```
D = 0.50 × norm⁻¹(Peak_VRAM_GB)
  + 0.30 × norm⁻¹(Weights_Size_GB)
  + 0.20 × VRAM_Headroom_Bonus
```

When a **target device VRAM** is specified (e.g. 8 GB, 16 GB, 24 GB):

```
VRAM_Headroom_Bonus = max(0, (Target_VRAM − Peak_VRAM) / Target_VRAM)
```

Models whose `Peak_VRAM_GB` exceeds the target receive a headroom bonus of **0**, effectively gating them as hardware-incompatible. When no target is specified, `VRAM_Headroom_Bonus` defaults to `norm⁻¹(Peak_VRAM_GB)`.

---

### Pillar 5 — Operational Cost (C)

```
C = 0.60 × norm⁻¹(Avg_Power_W)
  + 0.40 × norm(Avg_GPU_Util)
```

| Component | Direction | Rationale |
| :--- | :--- | :--- |
| **Avg_Power_W** | Lower is better | Sustained power draw maps to operational $/kWh costs. |
| **Avg_GPU_Util** | Higher is better | Higher utilisation means the model saturates available compute — better hardware efficiency. |

---

## Data Source Mapping

Every input to the ARS formula is already captured by the benchmark pipeline. No additional data collection is required.

| ARS Component | Source Column in `*_all_summary.csv` |
| :--- | :--- |
| Success_Rate | `Success Rate` |
| Shortcut_Rate | `Shortcut_Rate` |
| Correct_Order_Rate | `Correct_Order_Rate` |
| Tool_Call_Efficiency | `Tool_Call_Efficiency mean` |
| Parse_Error_Rate | `Parse_Errors mean` / `Steps mean` |
| Perfect_Path_Rate | `Path_PERFECT_Count` / `Number of Trials` |
| Tokens_per_Useful_Step | `Tokens_per_Useful_Step mean` |
| Total_Energy_J | `Total Energy (J) mean` |
| Inference_Time_s | `Inference Time (s) mean` |
| Peak_VRAM_GB | `Peak VRAM (GB) mean` |
| Weights_Size_GB | `Weights Size (GB) mean` |
| Avg_Power_W | `Avg Power (W) mean` |
| Avg_GPU_Util | `Avg GPU Util (%) mean` |

---

## Fitness Label Classification

The ARS maps to human-readable fitness labels for end-user presentation:

| ARS Range | Label | Interpretation |
| :---: | :--- | :--- |
| 85–100 | 🟢 **Production Ready** | Reliable for complex multi-step agentic workflows. Minimal supervision needed. |
| 70–84 | 🟡 **Capable** | Handles standard tool-calling tasks well. May need fallback logic for edge cases. |
| 50–69 | 🟠 **Limited** | Suitable for simple, single-tool lookups. Not recommended for multi-step chains. |
| 30–49 | 🔴 **Experimental** | High failure rate or excessive wandering. Use only for testing or low-stakes tasks. |
| 0–29 | ⛔ **Not Recommended** | Fundamental agentic capability is broken at this quantization level. |

Additionally, a **"Best For"** tag is auto-generated from the dominant sub-score profile:

| Condition | Tag |
| :--- | :--- |
| R > 0.9 and P > 0.8 | Complex multi-step reasoning tasks |
| R > 0.7 and E > 0.8 | High-throughput batch processing |
| D > 0.9 and R > 0.5 | Edge deployment on consumer GPUs |
| E > 0.9 | Low-latency single-step lookups |

---

## Configurable Weight Presets

The default weights prioritise task reliability. Alternative presets shift the balance for different deployment scenarios:

| Preset | R | P | E | D | C | Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **default** | 35% | 25% | 20% | 15% | 5% | Safety-critical industrial agents |
| **edge** | 25% | 15% | 15% | 40% | 5% | Edge / consumer GPU deployment |
| **throughput** | 20% | 15% | 35% | 15% | 15% | High-volume batch processing |
| **cost** | 25% | 15% | 20% | 10% | 30% | Cost-optimised datacenter operations |

---

## Empirical Results

All experiments were conducted on an NVIDIA RTX 5090 (32 GB VRAM) across 100 trials per model per agent across 3 agent archetypes (Support, IT Helpdesk, Finance) for a total of 8,100 task executions. The ARS was computed using the **edge** preset with a **16 GB target VRAM** constraint.

### Overall Ranking (Cross-Agent Average)

| Rank | Model | Quantization | Params | ARS | Label | Best For |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | Qwen2.5-3B-Instruct_INT4_AWQ | AWQ | 3B | **90.8** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 2 | Llama-3.2-3B-Instruct-GPTQ-4bit | GPTQ | 3B | **90.3** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 3 | Llama-3.2-3B-Instruct-BNB-4bit | BNB-4BIT | 3B | **89.4** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 4 | Llama-3.2-3B-Instruct-AWQ | AWQ | 3B | **88.3** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 5 | Llama-3.2-3B-Instruct-GPTQ-8bit | GPTQ-8BIT | 3B | **86.5** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 6 | Qwen2.5-7B-Instruct_INT4_AWQ | AWQ | 7B | **86.4** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 7 | Meta-Llama-3-8B-Instruct-AWQ | AWQ | 8B | **85.8** | 🟢 Production Ready | Complex multi-step reasoning tasks |
| 8 | Meta-Llama-3-8B-Instruct-GPTQ-4bit | GPTQ | 8B | **84.1** | 🟡 Capable | Complex multi-step reasoning tasks |
| 9 | Qwen2.5-3B-Instruct_FP16_Baseline | FP16 | 3B | **84.1** | 🟡 Capable | Complex multi-step reasoning tasks |
| 10 | Llama-3.2-3B-Instruct-BNB-8bit | BNB-8BIT | 3B | **82.5** | 🟡 Capable | Complex multi-step reasoning tasks |
| 11 | Qwen2.5-3B-Instruct_FP8_Static | FP8 | 3B | **79.8** | 🟡 Capable | Standard tool-calling workflows |
| 12 | Qwen2.5-14B-Instruct_INT4_AWQ | AWQ | 14B | **78.5** | 🟡 Capable | Complex multi-step reasoning tasks |
| 13 | Meta-Llama-3-8B-Instruct-GPTQ-8bit | GPTQ-8BIT | 8B | **77.4** | 🟡 Capable | High-throughput batch processing |
| 14 | Meta-Llama-3-8B-Instruct-BNB-4bit | BNB-4BIT | 8B | **76.9** | 🟡 Capable | High-throughput batch processing |
| 15 | Meta-Llama-3-8B-Instruct-BNB-8bit | BNB-8BIT | 8B | **75.8** | 🟡 Capable | Complex multi-step reasoning tasks |
| 16 | Meta-Llama-3-8B-Instruct | FP16 | 8B | **74.3** | 🟡 Capable | Complex multi-step reasoning tasks |
| 17 | Qwen2.5-7B-Instruct_FP8_Static | FP8 | 7B | **71.0** | 🟡 Capable | Complex multi-step reasoning tasks |
| 18 | Qwen2.5-7B-Instruct_FP16_Baseline | FP16 | 7B | **70.6** | 🟡 Capable | Complex multi-step reasoning tasks |
| 19 | Llama-3.2-1B-Instruct | FP16 | 1B | **65.3** | 🟠 Limited | Lightweight experimentation |
| 20 | Llama-3.2-1B-Instruct-GPTQ-8bit | GPTQ-8BIT | 1B | **65.3** | 🟠 Limited | Lightweight experimentation |
| 21 | Llama-3.2-1B-Instruct-BNB-4bit | BNB-4BIT | 1B | **63.7** | 🟠 Limited | Lightweight experimentation |
| 22 | Llama-3.2-1B-Instruct-AWQ | AWQ | 1B | **62.8** | 🟠 Limited | Lightweight experimentation |
| 23 | Llama-3.2-3B-Instruct | FP16 | 3B | **60.7** | 🟠 Limited | Standard tool-calling workflows |
| 24 | Llama-3.2-1B-Instruct-BNB-8bit | BNB-8BIT | 1B | **60.7** | 🟠 Limited | Lightweight experimentation |
| 25 | Llama-3.2-1B-Instruct-GPTQ-4bit | GPTQ | 1B | **57.6** | 🟠 Limited | Lightweight experimentation |
| 26 | Qwen2.5-14B-Instruct_FP8_Static | FP8 | 14B | **54.0** | 🟠 Limited | Complex multi-step reasoning tasks |
| 27 | Qwen2.5-14B-Instruct_FP16_Baseline | FP16 | 14B | **52.1** | 🟠 Limited | Complex multi-step reasoning tasks |

### Per-Agent Breakdown

| Model | Quant | Params | Support (Agent 1) | IT Helpdesk (Agent 2) | Finance (Agent 3) | Overall ARS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-3B-Instruct_INT4_AWQ | AWQ | 3B | 94.0 | 88.8 | 89.5 | **90.8** |
| Llama-3.2-3B-Instruct-GPTQ-4bit | GPTQ | 3B | 93.2 | 92.4 | 85.4 | **90.3** |
| Llama-3.2-3B-Instruct-BNB-4bit | BNB-4BIT | 3B | 90.6 | 90.7 | 86.8 | **89.4** |
| Llama-3.2-3B-Instruct-AWQ | AWQ | 3B | 92.2 | 89.3 | 83.5 | **88.3** |
| Llama-3.2-3B-Instruct-GPTQ-8bit | GPTQ-8BIT | 3B | 88.9 | 88.5 | 82.0 | **86.5** |
| Qwen2.5-7B-Instruct_INT4_AWQ | AWQ | 7B | 88.4 | 86.9 | 83.9 | **86.4** |
| Meta-Llama-3-8B-Instruct-AWQ | AWQ | 8B | 88.0 | 87.9 | 81.4 | **85.8** |
| Meta-Llama-3-8B-Instruct-GPTQ-4bit | GPTQ | 8B | 88.5 | 88.6 | 75.3 | **84.1** |
| Qwen2.5-3B-Instruct_FP16_Baseline | FP16 | 3B | 87.4 | 82.3 | 82.6 | **84.1** |
| Llama-3.2-3B-Instruct-BNB-8bit | BNB-8BIT | 3B | 86.1 | 81.4 | 79.9 | **82.5** |
| Qwen2.5-3B-Instruct_FP8_Static | FP8 | 3B | 83.9 | 76.2 | 79.4 | **79.8** |
| Qwen2.5-14B-Instruct_INT4_AWQ | AWQ | 14B | 80.2 | 80.3 | 75.0 | **78.5** |
| Meta-Llama-3-8B-Instruct-GPTQ-8bit | GPTQ-8BIT | 8B | 82.7 | 82.3 | 67.1 | **77.4** |
| Meta-Llama-3-8B-Instruct-BNB-4bit | BNB-4BIT | 8B | 87.3 | 87.5 | 55.9 | **76.9** |
| Meta-Llama-3-8B-Instruct-BNB-8bit | BNB-8BIT | 8B | 80.6 | 79.1 | 67.6 | **75.8** |
| Meta-Llama-3-8B-Instruct | FP16 | 8B | 77.8 | 77.6 | 67.4 | **74.3** |
| Qwen2.5-7B-Instruct_FP8_Static | FP8 | 7B | 72.6 | 72.0 | 68.3 | **71.0** |
| Qwen2.5-7B-Instruct_FP16_Baseline | FP16 | 7B | 71.8 | 72.6 | 67.5 | **70.6** |
| Llama-3.2-1B-Instruct | FP16 | 1B | 74.5 | 58.7 | 62.8 | **65.3** |
| Llama-3.2-1B-Instruct-GPTQ-8bit | GPTQ-8BIT | 1B | 71.9 | 59.6 | 64.3 | **65.3** |
| Llama-3.2-1B-Instruct-BNB-4bit | BNB-4BIT | 1B | 71.9 | 59.1 | 60.2 | **63.7** |
| Llama-3.2-1B-Instruct-AWQ | AWQ | 1B | 70.1 | 63.4 | 54.8 | **62.8** |
| Llama-3.2-3B-Instruct | FP16 | 3B | 88.1 | 12.0 | 81.9 | **60.7** |
| Llama-3.2-1B-Instruct-BNB-8bit | BNB-8BIT | 1B | 65.5 | 54.8 | 61.9 | **60.7** |
| Llama-3.2-1B-Instruct-GPTQ-4bit | GPTQ | 1B | 57.1 | 57.5 | 58.1 | **57.6** |
| Qwen2.5-14B-Instruct_FP8_Static | FP8 | 14B | 56.3 | 55.7 | 50.1 | **54.0** |
| Qwen2.5-14B-Instruct_FP16_Baseline | FP16 | 14B | 53.5 | 54.0 | 48.9 | **52.1** |

### Sub-Score Profile (Overall)

| Model | Quant | Params | R (Reliability) | P (Precision) | E (Efficiency) | D (Deployability) | C (Cost) | ARS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-3B-Instruct_INT4_AWQ | AWQ | 3B | 1.00 | 0.80 | 0.94 | 0.92 | 0.63 | **90.8** |
| Llama-3.2-3B-Instruct-GPTQ-4bit | GPTQ | 3B | 0.96 | 0.87 | 0.92 | 0.90 | 0.64 | **90.3** |
| Llama-3.2-3B-Instruct-BNB-4bit | BNB-4BIT | 3B | 0.98 | 0.87 | 0.86 | 0.90 | 0.58 | **89.4** |
| Llama-3.2-3B-Instruct-AWQ | AWQ | 3B | 0.95 | 0.86 | 0.83 | 0.90 | 0.62 | **88.3** |
| Llama-3.2-3B-Instruct-GPTQ-8bit | GPTQ-8BIT | 3B | 0.95 | 0.86 | 0.85 | 0.85 | 0.63 | **86.5** |
| Qwen2.5-7B-Instruct_INT4_AWQ | AWQ | 7B | 1.00 | 0.87 | 0.95 | 0.78 | 0.58 | **86.4** |
| Meta-Llama-3-8B-Instruct-AWQ | AWQ | 8B | 0.99 | 0.90 | 0.95 | 0.76 | 0.58 | **85.8** |
| Meta-Llama-3-8B-Instruct-GPTQ-4bit | GPTQ | 8B | 0.90 | 0.88 | 0.96 | 0.77 | 0.62 | **84.1** |
| Qwen2.5-3B-Instruct_FP16_Baseline | FP16 | 3B | 1.00 | 0.82 | 0.92 | 0.76 | 0.57 | **84.1** |
| Llama-3.2-3B-Instruct-BNB-8bit | BNB-8BIT | 3B | 0.96 | 0.85 | 0.58 | 0.85 | 0.60 | **82.5** |
| Qwen2.5-3B-Instruct_FP8_Static | FP8 | 3B | 0.98 | 0.80 | 0.64 | 0.77 | 0.60 | **79.8** |
| Qwen2.5-14B-Instruct_INT4_AWQ | AWQ | 14B | 1.00 | 0.90 | 0.87 | 0.61 | 0.57 | **78.5** |
| Meta-Llama-3-8B-Instruct-GPTQ-8bit | GPTQ-8BIT | 8B | 0.88 | 0.87 | 0.92 | 0.64 | 0.59 | **77.4** |
| Meta-Llama-3-8B-Instruct-BNB-4bit | BNB-4BIT | 8B | 0.70 | 0.83 | 0.89 | 0.77 | 0.55 | **76.9** |
| Meta-Llama-3-8B-Instruct-BNB-8bit | BNB-8BIT | 8B | 0.95 | 0.88 | 0.69 | 0.64 | 0.59 | **75.8** |
| Meta-Llama-3-8B-Instruct | FP16 | 8B | 0.94 | 0.88 | 0.89 | 0.55 | 0.43 | **74.3** |
| Qwen2.5-7B-Instruct_FP8_Static | FP8 | 7B | 1.00 | 0.90 | 0.76 | 0.46 | 0.54 | **71.0** |
| Qwen2.5-7B-Instruct_FP16_Baseline | FP16 | 7B | 1.00 | 0.89 | 0.92 | 0.41 | 0.44 | **70.6** |
| Llama-3.2-1B-Instruct | FP16 | 1B | 0.35 | 0.50 | 0.61 | 0.92 | 0.60 | **65.3** |
| Llama-3.2-1B-Instruct-GPTQ-8bit | GPTQ-8BIT | 1B | 0.31 | 0.50 | 0.61 | 0.94 | 0.65 | **65.3** |
| Llama-3.2-1B-Instruct-BNB-4bit | BNB-4BIT | 1B | 0.33 | 0.49 | 0.47 | 0.95 | 0.60 | **63.7** |
| Llama-3.2-1B-Instruct-AWQ | AWQ | 1B | 0.23 | 0.46 | 0.60 | 0.95 | 0.63 | **62.8** |
| Llama-3.2-3B-Instruct | FP16 | 3B | 0.64 | 0.55 | 0.60 | 0.64 | 0.37 | **60.7** |
| Llama-3.2-1B-Instruct-BNB-8bit | BNB-8BIT | 1B | 0.37 | 0.51 | 0.24 | 0.93 | 0.60 | **60.7** |
| Llama-3.2-1B-Instruct-GPTQ-4bit | GPTQ | 1B | 0.08 | 0.42 | 0.52 | 0.95 | 0.67 | **57.6** |
| Qwen2.5-14B-Instruct_FP8_Static | FP8 | 14B | 1.00 | 0.90 | 0.51 | 0.13 | 0.53 | **54.0** |
| Qwen2.5-14B-Instruct_FP16_Baseline | FP16 | 14B | 1.00 | 0.90 | 0.71 | 0.02 | 0.40 | **52.1** |

### Key Observations

1. **INT4 Quantized 3B Models Dominate the Production Ready Tier**:
   - The top 5 models are all 3B quantized variants (`Qwen2.5-3B AWQ` at 90.8, `Llama-3.2-3B GPTQ-4bit` at 90.3, `Llama-3.2-3B BNB-4bit` at 89.4, and `Llama-3.2-3B AWQ` at 88.3).
   - They achieve an optimal sweet spot: near-perfect Reliability ($R \ge 0.95$), high Tool Precision ($P \ge 0.80$), top-tier Efficiency ($E \ge 0.83$), and excellent Deployability ($D \ge 0.90$) with peak VRAM under 4.2 GB.

2. **Quantization Unlocks Hardware Deployability for Mid-to-Large Models**:
   - `Qwen2.5-7B-Instruct_INT4_AWQ` (ARS: 86.4, 7.2 GB VRAM) and `Meta-Llama-3-8B-Instruct-AWQ` (ARS: 85.8, 7.5 GB VRAM) achieve 🟢 **Production Ready** status, whereas their FP16 baselines (16.2 GB and 16.9 GB VRAM) exceed the 16 GB constraint, receiving heavy Deployability penalties (scoring 70.6 and 74.3).
   - `Qwen2.5-14B-Instruct_INT4_AWQ` achieves an ARS of 78.5 (🟡 Capable) with 11.3 GB VRAM and perfect Reliability ($R=1.00$), compared to the FP16 14B baseline (29.6 GB VRAM) which scores only 52.1.

3. **Sub-3B Models Suffer Fundamental Agentic Breakdowns**:
   - All 1B models (Llama-3.2-1B across FP16, AWQ, BNB-4/8bit, GPTQ-4/8bit) score between 57.6 and 65.3 (🟠 Limited). Despite high Deployability ($D > 0.92$), their Task Reliability collapses ($R = 0.08 - 0.37$) and Tool Precision is degraded ($P \approx 0.42 - 0.51$) due to persistent hallucinated tool calls, parse errors, and missing prerequisite steps.

4. **Quantization Method Dynamics**:
   - **AWQ and GPTQ-4bit** consistently outperform baseline and naive formats across all parameter classes, maintaining reasoning fidelity with minimal quantization degradation.
   - **BNB-8bit** introduces noticeable execution latency and energy overheads (e.g. Efficiency drops to $E=0.24$ on 1B and $E=0.58$ on 3B).

5. **Cross-Domain Agent Resilience**:
   - High-capacity quantized models (`Qwen2.5-7B/14B AWQ`, `Meta-Llama-3-8B AWQ`) maintain consistent $\ge 80$ ARS across all three agent domains (Support, IT Helpdesk, and Finance).
   - Unquantized `Llama-3.2-3B FP16` suffers catastrophic task failures on IT Helpdesk (ARS 12.0), while its quantized counterparts (`AWQ`, `GPTQ`, `BNB`) exhibit cross-domain stability ($>83$ ARS across all tasks).

---

## Comparison with Existing Agent Benchmarks

The ARS is evaluated against three widely cited benchmarks for LLM agent and tool-calling assessment.

### Benchmark Overview

| Benchmark | Institution | What It Measures | Primary Metric |
| :--- | :--- | :--- | :--- |
| **BFCL** (Berkeley Function-Calling Leaderboard) | UC Berkeley (Gorilla LLM) | Single-turn function-calling accuracy — correct tool selection, argument generation, and relevance detection (knowing when NOT to call a tool). | Overall Accuracy (%) |
| **TAU-Bench** (Tool-Agent-User Benchmark) | Sierra AI (2024) | End-to-end multi-turn agent task completion across realistic domains (retail, airline). | Pass Rate (%) |
| **AgentBench** | Tsinghua University | Multi-environment agentic evaluation across 8 distinct scenarios (OS, database, web, knowledge graph, etc.). | Weighted Overall Score |

### Capability Coverage Matrix

| Capability | BFCL | TAU-Bench | AgentBench | **This Framework + ARS** |
| :--- | :---: | :---: | :---: | :---: |
| Correct tool / function selection | ✅ | ✅ | ✅ | ✅ `must_call` validation |
| Correct argument generation (JSON) | ✅ | ✅ | ✅ | ✅ Regex parser + JSON validation |
| Relevance detection (when NOT to call) | ✅ | ❌ | ❌ | ✅ `must_not_call` enforcement |
| Multi-step tool chaining | ❌ | ✅ | ✅ | ✅ `Correct_Order`, `Exec_Path` |
| Multi-turn reasoning (ReAct loop) | ❌ | ✅ | ✅ | ✅ Full ReAct loop with step-level traces |
| End-to-end task pass / fail | ❌ | ✅ | ✅ | ✅ `Success_Rate` |
| Multi-domain evaluation | ❌ | ✅ (2) | ✅ (8) | ✅ (3) Support, IT Helpdesk, Finance |
| Parallel function calls | ✅ | ❌ | ❌ | ❌ (by design — 1 tool per step) |
| Redundancy / efficiency detection | ❌ | ❌ | ❌ | ✅ `Tool_Call_Efficiency`, `Redundant_Tool_Calls` |
| Execution path classification | ❌ | ❌ | ❌ | ✅ PERFECT / WANDERING / SHORTCUT / FAILED |
| Hardware telemetry (VRAM, power, utilisation) | ❌ | ❌ | ❌ | ✅ `pynvml` GPU polling at 50 ms |
| Quantization impact measurement | ❌ | ❌ | ❌ | ✅ Core evaluation axis |
| Energy cost per task | ❌ | ❌ | ❌ | ✅ Total Energy (J), Avg Power (W) |
| Single composite score | ❌ | ❌ | ❌ | ✅ ARS (0–100) |

### Key Differentiators

1. **Superset of core capabilities.** The evaluation framework captures the same fundamental capabilities measured by BFCL (function-calling accuracy, relevance detection via `must_not_call`), TAU-Bench (end-to-end multi-turn agent task completion), and AgentBench (multi-domain agent evaluation).

2. **Extends the assessment envelope.** Beyond what existing benchmarks measure, this framework adds tool-call trace quality analysis (execution path classification, redundancy detection, shortcut detection), hardware telemetry (VRAM, power, GPU utilisation), and quantization degradation measurement — dimensions not addressed by any current benchmark.

3. **Composite scoring.** BFCL, TAU-Bench, and AgentBench each produce accuracy-only metrics. The ARS compresses task reliability, tool precision, computational efficiency, hardware deployability, and operational cost into a single actionable score with configurable weights.

4. **Design trade-off.** The only BFCL capability not captured is parallel function calling (invoking multiple tools in a single generation step). This is an intentional constraint of the ReAct agent design, which enforces one tool call per response to prevent multi-call hallucinations — a deliberate safety measure for industrial deployment.

---

## Usage

The ARS is computed by `compute_ars.py` which reads the `*_all_summary.csv` files produced by the benchmark pipeline:

```bash
# Default preset (safety-critical)
python compute_ars.py --results_dir ./results/benchmark_20260810_171506

# Edge deployment with 16 GB VRAM target
python compute_ars.py --results_dir ./results/benchmark_20260810_171506 --preset edge --target_vram 16

# High-throughput batch preset
python compute_ars.py --results_dir ./results/benchmark_20260810_171506 --preset throughput
```

### Output Files

| File | Description |
| :--- | :--- |
| `ars_<AGENT_ID>.csv` | Per-model ARS and sub-scores for a single agent |
| `ars_<AGENT_ID>_cards.txt` | Human-readable ranked model cards for a single agent |
| `ars_overall.csv` | Cross-agent averaged ARS with per-agent breakdown columns |
| `ars_overall_cards.txt` | Human-readable overall model card ranking |

---

## Platform Integration

The ARS is designed to power a model selection interface for industrial agentic platforms. For each quantized model variant, the platform displays:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Qwen2.5-3B-Instruct_INT4_AWQ    ARS: 91/100  █████████             │
│  🟢 Production Ready                                                │
│  ✅ Best for: Complex multi-step reasoning tasks                    │
│  ⚡ VRAM: 3.7 GB  |  Speed: 62.2 tok/s  |  Energy: 526 J/task        │
├─────────────────────────────────────────────────────────────────────┤
│  Qwen2.5-14B-Instruct_FP16       ARS: 52/100  █████                 │
│  🟠 Limited                                                         │
│  ⚠️ Exceeds 16 GB VRAM budget (requires 29.6 GB)                    │
│  ⚡ VRAM: 29.6 GB  |  Speed: 42.7 tok/s  |  Energy: 1,700 J/task     │
└─────────────────────────────────────────────────────────────────────┘
```

Unlike provider-assigned labels (e.g. Claude's *"fit for everyday tasks"*), the ARS is:
* **Empirically computed** from the user's own benchmark data across 8,100 evaluated agent executions.
* **Hardware-aware** — models that exceed the deployment VRAM budget are automatically downranked.
* **Task-aware** — scores vary per agent archetype, enabling domain-specific recommendations.
* **Quantization-aware** — directly compares FP16 baselines against INT4, FP8, GPTQ, BNB, and other compressed variants.
