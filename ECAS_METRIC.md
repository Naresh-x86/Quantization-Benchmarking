# EKOI Composite Agent Score (ECAS)

## 1. Purpose

The **EKOI Composite Agent Score (ECAS)** is a single normalized metric on a **0–100 scale** designed to evaluate how well a quantized (or baseline) LLM variant performs within multi-step agentic workflows on the EKOI platform.

It answers one question for the end-user:

> *"Before I deploy this model locally on my device, how well will it actually work for my workflow — considering accuracy, tool-calling reliability, and hardware cost?"*

ECAS condenses the raw benchmark telemetry (success rates, path classifications, tool call sequences, VRAM usage, throughput) into a single actionable number, comparable across model families, parameter counts, and quantization formats.

---

## 2. Formula

```
ECAS = (0.45 × S_accuracy) + (0.35 × S_tool_quality) + (0.20 × S_efficiency)
```

Each sub-score is internally normalized to a **0–100** range before weighting.

---

## 3. Sub-Score Definitions

### 3.1 Task Accuracy Score — `S_accuracy` (Weight: 45%)

Measures whether the model completes the agentic task correctly and via the optimal execution path.

```
S_accuracy = (0.70 × P_perfect + 0.30 × S_rate) × 100
```

| Component | Symbol | Source Column | Description |
|:---|:---|:---|:---|
| **Path Perfection Rate** | `P_perfect` | `Path_PERFECT_Count / Number of Trials` | Fraction of trials where the model called all required tools, in the correct order, with zero redundant calls and zero parse errors. This is the strictest measure of task execution quality. |
| **Overall Success Rate** | `S_rate` | `Success Rate` | Fraction of trials where the task was completed successfully (regardless of path optimality — includes WANDERING and SHORTCUT paths). |

**Why 70/30 split?**
Path perfection is weighted more heavily because a model that succeeds via wandering (extra tool calls, retries) is still burning tokens and latency — critical concerns for production agentic deployments. However, raw success rate still matters because a wandering success is strictly better than a failure.

---

### 3.2 Tool Execution Quality Score — `S_tool_quality` (Weight: 35%)

Measures the precision and reliability of the model's tool-calling behavior.

```
S_tool_quality = (0.40 × E_tool + 0.30 × (1 - ParseErrorRate) + 0.30 × C_order) × 100
```

| Component | Symbol | Source Column | Description |
|:---|:---|:---|:---|
| **Tool Call Efficiency** | `E_tool` | `Tool_Call_Efficiency mean` | Ratio of expected tool calls to actual tool calls: `Expected_Steps / Actual_Tool_Calls`. A perfect score of 1.0 means the model made exactly the right number of calls. Values > 1.0 indicate the model skipped steps (likely a failure); values < 1.0 indicate redundant calls. Clamped to [0.0, 1.0]. |
| **Parse Accuracy** | `1 - ParseErrorRate` | `Parse_Errors mean / Steps mean` | Fraction of agent steps that did NOT result in a parse error (malformed Action/Action Input). A parse error means the model generated text the framework couldn't interpret as a valid tool call. |
| **Tool Call Order Integrity** | `C_order` | `Correct_Order_Rate` | Fraction of trials where the required tools were called in the exact expected sequence. Even if the model calls all the right tools, calling them out of order (e.g., issuing a refund before checking inventory) is a logical error. |

---

### 3.3 Hardware Efficiency Score — `S_efficiency` (Weight: 20%)

Measures the deployment cost-effectiveness of using this quantized variant vs. the FP16 baseline.

```
S_efficiency = min(100, 50 × R_vram + 50 × (Tokens_per_Sec / 100))
```

| Component | Symbol | Source Column | Description |
|:---|:---|:---|:---|
| **VRAM Compression Ratio** | `R_vram` | `FP16_Peak_VRAM / Quantized_Peak_VRAM` | How much VRAM the quantized model saves relative to its FP16 baseline of the same parameter count. A ratio of 2.0 means the quantized model uses half the VRAM. For FP16 baselines themselves, `R_vram = 1.0`. |
| **Token Throughput** | `Tokens_per_Sec` | `Tokens / Sec mean` | Raw generation speed. Divided by 100 as a normalizer (100 tok/s is treated as the "excellent" reference point for agentic workloads where latency matters). |

**Why only 20% weight?**
In an agentic model repository, the primary concern is *"does this model actually work for my task?"* — not *"is it fast?"*. A model that is blazing fast but fails 40% of tasks is useless. However, efficiency still matters because EKOI targets on-premise/edge deployments where VRAM is constrained.

---

## 4. Execution Path Classification Reference

The ECAS formula depends on the **Execution Path** taxonomy used in the benchmark pipeline. Each trial is classified into exactly one path:

| Path | Criteria | Impact on ECAS |
|:---|:---|:---|
| **PERFECT** | Task succeeded + all `must_call` tools invoked + correct order + zero redundant calls + zero parse errors | Maximizes `S_accuracy` and `S_tool_quality` |
| **WANDERING** | Task succeeded, but with redundant tool calls, repeated tools, parse errors, or out-of-order calls | Contributes to `S_rate` but penalizes `P_perfect`, `E_tool`, and `C_order` |
| **SHORTCUT** | Task succeeded, but the model called the final tool without completing prerequisite steps (lucky guess) | Contributes to `S_rate` but heavily penalizes `P_perfect` and `C_order` |
| **FAILED** | Task did not succeed | Zero contribution to both `S_accuracy` components |

---

## 5. Worked Example

Using real benchmark data from Agent 1 (Support) with 100 trials:

### Model: `Qwen2.5-14B-Instruct_INT4_AWQ`

| Metric | Value |
|:---|:---|
| Path_PERFECT_Count | 100 |
| Number of Trials | 100 |
| Success Rate | 1.0 |
| Tool_Call_Efficiency mean | 1.0 |
| Parse_Errors mean | 0.0 |
| Steps mean | 3.0 |
| Correct_Order_Rate | 1.0 |
| Peak VRAM (GB) | 11.29 |
| FP16 Baseline Peak VRAM (GB) | 29.57 |
| Tokens / Sec mean | 42.86 |

**Step 1: S_accuracy**
```
P_perfect = 100 / 100 = 1.0
S_rate    = 1.0
S_accuracy = (0.70 × 1.0 + 0.30 × 1.0) × 100 = 100.0
```

**Step 2: S_tool_quality**
```
E_tool          = min(1.0, 1.0) = 1.0
ParseErrorRate  = 0.0 / 3.0    = 0.0
Parse Accuracy  = 1 - 0.0      = 1.0
C_order         = 1.0

S_tool_quality = (0.40 × 1.0 + 0.30 × 1.0 + 0.30 × 1.0) × 100 = 100.0
```

**Step 3: S_efficiency**
```
R_vram         = 29.57 / 11.29 = 2.619
Throughput_norm = 42.86 / 100  = 0.4286

S_efficiency = min(100, 50 × 2.619 + 50 × 0.4286) = min(100, 130.95 + 21.43) = 100.0
```

**Final ECAS:**
```
ECAS = (0.45 × 100.0) + (0.35 × 100.0) + (0.20 × 100.0) = 100.0
```

---

### Model: `meta-llama/Llama-3.2-1B-Instruct` (FP16 Baseline)

| Metric | Value |
|:---|:---|
| Path_PERFECT_Count | 4 |
| Number of Trials | 100 |
| Success Rate | 0.69 |
| Tool_Call_Efficiency mean | 0.589 |
| Parse_Errors mean | 1.17 |
| Steps mean | 7.41 |
| Correct_Order_Rate | 0.74 |
| Peak VRAM (GB) | 4.15 |
| Tokens / Sec mean | 170.63 |

**Step 1: S_accuracy**
```
P_perfect = 4 / 100 = 0.04
S_rate    = 0.69
S_accuracy = (0.70 × 0.04 + 0.30 × 0.69) × 100 = (0.028 + 0.207) × 100 = 23.5
```

**Step 2: S_tool_quality**
```
E_tool          = min(1.0, 0.589) = 0.589
ParseErrorRate  = 1.17 / 7.41     = 0.158
Parse Accuracy  = 1 - 0.158       = 0.842
C_order         = 0.74

S_tool_quality = (0.40 × 0.589 + 0.30 × 0.842 + 0.30 × 0.74) × 100
              = (0.2356 + 0.2526 + 0.222) × 100 = 71.0
```

**Step 3: S_efficiency**
```
R_vram         = 1.0 (FP16 baseline, no compression)
Throughput_norm = 170.63 / 100 = 1.7063

S_efficiency = min(100, 50 × 1.0 + 50 × 1.7063) = min(100, 50 + 85.32) = 100.0
```

**Final ECAS:**
```
ECAS = (0.45 × 23.5) + (0.35 × 71.0) + (0.20 × 100.0)
     = 10.575 + 24.85 + 20.0
     = 55.4
```

---

## 6. Interpretation Scale

| ECAS Range | Label | Deployment Recommendation |
|:---:|:---|:---|
| **90 – 100** | **Excellent** | Production-ready. Deploy with confidence for the target workflow. |
| **80 – 89** | **Good** | Suitable for production with monitoring. Minor tool-call inefficiencies may occur. |
| **70 – 79** | **Acceptable** | Usable for non-critical workflows. Expect occasional wandering paths. |
| **50 – 69** | **Marginal** | Use only for simple, single-step tasks. Multi-step agentic workflows will be unreliable. |
| **< 50** | **Not Recommended** | Do not deploy for agentic workflows. High failure rate and/or severe tool-calling degradation. |

---

## 7. Edge Cases & Clamping Rules

| Scenario | Handling |
|:---|:---|
| `Actual_Tool_Calls = 0` | `E_tool = 0` (model never called any tools — complete failure). |
| `E_tool > 1.0` (fewer calls than expected) | Clamped to `1.0`. This happens when a model skips tools but still marks success (a SHORTCUT path). The shortcut penalty is already captured by `P_perfect` and `C_order`. |
| `S_efficiency > 100` | Clamped to `100`. A highly compressed, fast model shouldn't inflate the score beyond the maximum. |
| `FP16 baseline unavailable for R_vram` | Set `R_vram = 1.0` (no compression benefit assumed). |
| Model failed to load (0 trials completed) | `ECAS = 0`. No data to score. |

---

## 8. Multi-Agent Aggregation

When a model is evaluated across multiple agents (e.g., AGENT_1_SUPPORT, AGENT_2_IT_HELPDESK, AGENT_3_FINANCE), the **per-agent ECAS scores are averaged equally** to produce a global ECAS:

```
ECAS_global = (ECAS_agent1 + ECAS_agent2 + ECAS_agent3) / N_agents
```

This ensures a model that excels on one task but fails on another is not misleadingly rated.

---

## 9. Source Columns Quick Reference

All values used in ECAS computation are drawn from the `*_all_summary.csv` files generated by the benchmark pipeline:

| ECAS Variable | CSV Column(s) |
|:---|:---|
| `P_perfect` | `Path_PERFECT_Count` ÷ `Number of Trials` |
| `S_rate` | `Success Rate` |
| `E_tool` | `Tool_Call_Efficiency mean` |
| `ParseErrorRate` | `Parse_Errors mean` ÷ `Steps mean` |
| `C_order` | `Correct_Order_Rate` |
| `R_vram` | `Peak VRAM (GB) mean` (FP16 baseline) ÷ `Peak VRAM (GB) mean` (this model) |
| `Tokens_per_Sec` | `Tokens / Sec mean` |

---

## 10. ECAS Scoreboard — All Quantized Models

Computed from benchmark run `benchmark_20260810_171506` (100 trials per model per agent).
Global ECAS is the average of per-agent ECAS scores across AGENT_1_SUPPORT (A1), AGENT_2_IT_HELPDESK (A2), and AGENT_3_FINANCE (A3).

### 10.1 Qwen Model Family

| Rank | Model | Quant | Params | ECAS (A1) | ECAS (A2) | ECAS (A3) | **Global ECAS** | Rating |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Qwen2.5-14B-Instruct_INT4_AWQ | AWQ | 14B | 100.0 | 100.0 | 65.0 | **88.3** | Good |
| 2 | Qwen2.5-7B-Instruct_FP16_Baseline | FP16 | 7B | 100.0 | 97.5 | 60.7 | **86.1** | Good |
| 3 | Qwen2.5-7B-Instruct_INT4_AWQ | AWQ | 7B | 100.0 | 92.9 | 65.0 | **86.0** | Good |
| 4 | Qwen2.5-14B-Instruct_FP16_Baseline | FP16 | 14B | 94.2 | 94.3 | 59.3 | **82.6** | Good |
| 5 | Qwen2.5-7B-Instruct_FP8_Static | FP8 | 7B | 93.3 | 93.3 | 58.3 | **81.6** | Good |
| 6 | Qwen2.5-14B-Instruct_FP8_Static | FP8 | 14B | 91.9 | 91.9 | 56.9 | **80.2** | Good |
| 7 | Qwen2.5-3B-Instruct_INT4_AWQ | AWQ | 3B | 100.0 | 69.4 | 64.8 | **78.1** | Acceptable |
| 8 | Qwen2.5-3B-Instruct_FP16_Baseline | FP16 | 3B | 97.9 | 70.8 | 62.8 | **77.2** | Acceptable |
| 9 | Qwen2.5-3B-Instruct_FP8_Static | FP8 | 3B | 91.4 | 61.8 | 58.1 | **70.4** | Acceptable |

### 10.2 Llama Model Family

| Rank | Model | Quant | Params | ECAS (A1) | ECAS (A2) | ECAS (A3) | **Global ECAS** | Rating |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Meta-Llama-3-8B-Instruct-AWQ | AWQ | 8B | 100.0 | 100.0 | 64.9 | **88.3** | Good |
| 2 | Meta-Llama-3-8B-Instruct-GPTQ-4bit | GPTQ | 8B | 100.0 | 100.0 | 61.7 | **87.2** | Good |
| 3 | Meta-Llama-3-8B-Instruct-GPTQ-8bit | GPTQ-8BIT | 8B | 100.0 | 98.2 | 60.7 | **86.3** | Good |
| 4 | Llama-3.2-3B-Instruct-BNB-4bit | BNB-4BIT | 3B | 96.3 | 94.2 | 66.2 | **85.6** | Good |
| 5 | Llama-3.2-3B-Instruct-GPTQ-4bit | GPTQ | 3B | 97.5 | 96.5 | 62.8 | **85.6** | Good |
| 6 | Meta-Llama-3-8B-Instruct-BNB-8bit | BNB-8BIT | 8B | 98.4 | 95.9 | 61.3 | **85.2** | Good |
| 7 | Meta-Llama-3-8B-Instruct (FP16) | FP16 | 8B | 97.4 | 96.2 | 61.1 | **84.9** | Good |
| 8 | Llama-3.2-3B-Instruct-GPTQ-8bit | GPTQ-8BIT | 3B | 96.5 | 95.4 | 62.4 | **84.8** | Good |
| 9 | Llama-3.2-3B-Instruct-AWQ | AWQ | 3B | 99.4 | 92.1 | 62.6 | **84.7** | Good |
| 10 | Meta-Llama-3-8B-Instruct-BNB-4bit | BNB-4BIT | 8B | 100.0 | 100.0 | 49.9 | **83.3** | Good |
| 11 | Llama-3.2-3B-Instruct-BNB-8bit | BNB-8BIT | 3B | 93.2 | 89.7 | 61.2 | **81.4** | Good |
| 12 | Llama-3.2-3B-Instruct (FP16) | FP16 | 3B | 98.0 | — ¹ | 63.6 | **80.8** | Good |
| 13 | Llama-3.2-1B-Instruct-BNB-4bit | BNB-4BIT | 1B | 55.1 | 49.1 | 48.9 | **51.0** | Marginal |
| 14 | Llama-3.2-1B-Instruct-GPTQ-8bit | GPTQ-8BIT | 1B | 54.3 | 46.6 | 50.2 | **50.4** | Marginal |
| 15 | Llama-3.2-1B-Instruct (FP16) | FP16 | 1B | 55.4 | 46.1 | 49.0 | **50.2** | Marginal |
| 16 | Llama-3.2-1B-Instruct-BNB-8bit | BNB-8BIT | 1B | 49.7 | 46.8 | 48.7 | **48.4** | Not Recommended |
| 17 | Llama-3.2-1B-Instruct-AWQ | AWQ | 1B | 50.6 | 49.9 | 43.7 | **48.1** | Not Recommended |
| 18 | Llama-3.2-1B-Instruct-GPTQ-4bit | GPTQ | 1B | 45.0 | 45.7 | 39.4 | **43.4** | Not Recommended |

> ¹ Llama-3.2-3B FP16 failed to initialize on Agent 2, so its global ECAS is the average of A1 and A3 only.

### 10.3 Combined Global Ranking (Top 10)

| Rank | Model | Quant | Params | **Global ECAS** | Rating |
|:---:|:---|:---:|:---:|:---:|:---|
| 🥇 | Qwen2.5-14B-Instruct_INT4_AWQ | AWQ | 14B | **88.3** | Good |
| 🥇 | Meta-Llama-3-8B-Instruct-AWQ | AWQ | 8B | **88.3** | Good |
| 🥉 | Meta-Llama-3-8B-Instruct-GPTQ-4bit | GPTQ | 8B | **87.2** | Good |
| 4 | Meta-Llama-3-8B-Instruct-GPTQ-8bit | GPTQ-8BIT | 8B | **86.3** | Good |
| 5 | Qwen2.5-7B-Instruct_FP16_Baseline | FP16 | 7B | **86.1** | Good |
| 6 | Qwen2.5-7B-Instruct_INT4_AWQ | AWQ | 7B | **86.0** | Good |
| 7 | Llama-3.2-3B-Instruct-BNB-4bit | BNB-4BIT | 3B | **85.6** | Good |
| 7 | Llama-3.2-3B-Instruct-GPTQ-4bit | GPTQ | 3B | **85.6** | Good |
| 9 | Meta-Llama-3-8B-Instruct-BNB-8bit | BNB-8BIT | 8B | **85.2** | Good |
| 10 | Meta-Llama-3-8B-Instruct (FP16) | FP16 | 8B | **84.9** | Good |

### 10.4 Key Observations

1. **AWQ quantization consistently produces the highest ECAS** — Qwen 14B AWQ and Llama 8B AWQ both tie at 88.3, outperforming even their own FP16 baselines on global ECAS because AWQ preserves tool-calling accuracy while dramatically improving hardware efficiency.

2. **Agent 3 (Finance) is the hardest task** — Every model scores significantly lower on A3 compared to A1/A2. This 4-step workflow (price lookup → average → math comparison → conditional trade) exposes tool-ordering weaknesses that simpler tasks mask.

3. **The 1B parameter cliff is real** — All 1B Llama variants score below 52, regardless of quantization format. The model lacks sufficient capacity for reliable multi-step tool reasoning. GPTQ-4bit 1B is the worst at 43.4 (Not Recommended).

4. **3B models are the sweet spot for edge deployment** — Llama-3.2-3B variants score 81–86 with only 4 GB VRAM, making them the best option for resource-constrained on-premise EKOI deployments.

5. **FP16 baselines are NOT always the best** — Qwen 14B FP16 (82.6) scores lower than Qwen 14B AWQ (88.3) because the FP16 model's massive VRAM footprint (29.6 GB) penalizes its efficiency score without any accuracy advantage.
