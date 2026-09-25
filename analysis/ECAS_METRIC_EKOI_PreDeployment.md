# EKOI Composite Agent Score (ECAS) — Pre-Deployment Edition

## Companion document to ARS_Presentation (research) and the original ECAS_METRIC.md

## 1. Purpose

The original ECAS formula scored a model by **replaying it against three
benchmark agents** (Support, IT Helpdesk, Finance) and reading off empirical
success rates, tool-call precision, and path classifications. That is
exactly what **ARS (Agent Readiness Score)** already does, rigorously, for
research purposes — proxy agents are ARS's job, and ARS does it well.

The EKOI **Model Repository** has a different problem. A user uploads a
model — or browses one someone else uploaded — and wants to know whether it
is worth investing GPU budget and integration time into, **for a workflow
that has never been benchmarked**: a new customer's onboarding-compliance
checker, a policy-RAG assistant for a domain we've never tested, a nudge
engine no research agent resembles. There is no proxy task to run yet. The
question is not *"how did this model perform?"* — it's:

> *"Before we spend engineering time wiring this model into a workflow and
> running a real pilot, what's our best estimate of whether it's worth the
> investment?"*

This is the same logic as underwriting a loan before there's a repayment
history, or scoring an ROI projection before a project ships — the estimate
uses **whatever is knowable in advance**, is explicit about its uncertainty,
and gets replaced by real numbers (via ARS-style testing, once a proxy or
pilot exists) as soon as they're available. This document defines that
estimate.

**ECAS (Pre-Deployment Edition) never reads a proxy-agent CSV.** Every input
is either (a) publicly published about the base model, (b) established in
the quantization literature about the format, or (c) measured directly from
hardware telemetry in a way that does not depend on which task was running.

---

## 2. Formula

```
ECAS_pre = CDF × DCF × (0.45 × S_capability + 0.35 × S_format_reliability + 0.20 × S_efficiency)
```

Each sub-score is normalized to **0–100** before weighting, so the weighted
sum itself already lands on a 0–100 scale — no extra ×100 needed. `CDF` and
`DCF` are independent risk-discount multipliers, each in **[0.0, 1.0]** —
see Section 4.

The weights (45/35/20) are carried over unchanged from the original ECAS,
so a "Pre-Deployment" score and a later "Validated" score (Section 8) stay
comparable in shape even though their inputs are entirely different.

---

## 3. Sub-Score Definitions

### 3.1 General Capability Score — `S_capability` (Weight: 45%)

Estimates raw model quality **independent of any EKOI-specific task**,
using the base (unquantized) model's own published benchmark results.

```
S_capability = mean(MMLU, GSM8K)
```

| Component | Source | Description |
|:---|:---|:---|
| **MMLU** | Model card / technical report | General knowledge & reasoning across 57 subjects. 0–100 scale. |
| **GSM8K** | Model card / technical report | Multi-step arithmetic word-problem reasoning. A reasonable stand-in for the "can it follow a multi-step procedure" skill that agentic workflows lean on, without needing a live agent test. 0–100 scale. |

Use **IFEval** (instruction-following) as a third component averaged in
when published, since it correlates well with an agent's ability to respect
output-format constraints (a major real failure mode in agentic
tool-calling). Default to the two-metric average when IFEval isn't
published for a given checkpoint.

**Why not more benchmarks?** MMLU and GSM8K are near-universally published
for any serious instruction-tuned release, which keeps this score
computable for every model that reaches the Model Repository on day one —
unlike task-specific proxy testing, which requires the model to already be
running on EKOI infrastructure.

**A note on MMLU variants:** Qwen officially reports **MMLU-redux** rather
than plain MMLU for the 2.5 series; Meta reports plain **MMLU** for Llama.
The two are close enough in scale and intent (both 0–100 general-knowledge
multiple-choice accuracy) to average alongside GSM8K without renormalizing,
but they are not identically defined — treat cross-family `S_capability`
comparisons as directionally reliable, not decimal-precise.

---

### 3.2 Quantization Format Reliability Score — `S_format_reliability` (Weight: 35%)

Estimates how much of the base model's capability survives a given
quantization format, using **documented accuracy-retention figures**
rather than a measured task outcome.

```
S_format_reliability = 100 × D_retention
```

| Component | Symbol | Source | Description |
|:---|:---|:---|:---|
| **Documented Retention Fraction** | `D_retention` | Quantization method literature / vendor docs | Typical fraction of FP16 benchmark performance preserved by this format at this bit-width, independent of task. |

**Quantization Format Reference Table** (recalibrate periodically as new
data accumulates — see Section 9):

| Format | Typical `D_retention` | Notes |
|:---|:---:|:---|
| FP16 (baseline) | 1.00 | No compression, reference point |
| INT4 — AWQ | 0.98 | Activation-aware calibration preserves instruction-following well |
| INT8 — GPTQ | 0.99 | Minimal loss at 8-bit |
| INT4 — GPTQ | 0.97 | Calibration-dependent; more sensitive on small models |
| INT8 — bitsandbytes | 0.97 | |
| INT4 — bitsandbytes (NF4) | 0.95 | Least sophisticated calibration of the four INT4 options |
| FP8 — static per-tensor | 0.99 | Numerically close to FP16 — **but see 3.3**, its risk shows up as a speed/energy penalty, not an accuracy one |

This table is deliberately simple and format-level, not model-level — it
answers *"what does this quantization technique typically cost,"* not
*"what did it cost on Task X."* That's what keeps it usable for a model
nobody has tested yet.

---

### 3.3 Hardware Efficiency Score — `S_efficiency` (Weight: 20%)

Unchanged in spirit from the original ECAS — this was already
workflow-agnostic. VRAM footprint and raw token throughput are properties
of the model + quantization format + serving stack, and don't meaningfully
shift based on which downstream task is calling the model.

```
S_efficiency = min(100, 50 × R_vram + 50 × (Tokens_per_Sec / 100))
```

| Component | Symbol | Source | Description |
|:---|:---|:---|:---|
| **VRAM Compression Ratio** | `R_vram` | Measured hardware telemetry | `FP16_Peak_VRAM ÷ Quantized_Peak_VRAM`. `1.0` for FP16 baselines. |
| **Token Throughput** | `Tokens_per_Sec` | Measured hardware telemetry | Raw generation speed on EKOI's own serving stack. |

**This is where backend risk still shows up, without needing a special
factor for it.** Our own quantization study measured FP8 static per-tensor
running 55–61% slower and using up to 82% more energy than FP16 on EKOI's
stack — not because FP8 loses accuracy (`D_retention` = 0.99), but because
this stack runs FP8 through a Hugging Face Transformers fallback instead of
vLLM's optimized kernels. That penalty is *already* captured here, directly
from measured `Tokens_per_Sec`, with no separate "backend risk" term
needed. Same story for bitsandbytes-8bit, whose dequantization overhead
made it the single most energy-expensive format in the entire study —
worse even than uncompressed FP16.

---

## 4. Risk-Discount Multipliers

Two independent multipliers apply to the final score, each representing a
distinct kind of uncertainty a pre-deployment estimate has to own.

### 4.1 Capacity Discount Factor — `CDF`

Our own ARS research established that agentic tool-calling reliability
**does not degrade smoothly with parameter count** — it falls off a cliff
below roughly 3B parameters, regardless of quantization format (Llama-3.2-1B
posted a 0.13 Reliability score in ARS testing across every format tried).
`S_capability` alone won't catch this, because MMLU/GSM8K degrade far more
gently than agentic reliability does at small scale. `CDF` encodes that
known cliff as a general, cross-task finding — not a specific proxy-task
result — so it stays legitimate to use even for an untested workflow:

| Parameter Count | `CDF` | Rationale |
|:---|:---:|:---|
| ≥ 7B | 1.00 | No discount |
| 3B – 6.9B | 0.95 | Small headroom loss, still reliable in ARS testing |
| 1B – 2.9B | 0.65 | Empirically established agentic capability cliff |
| < 1B | 0.45 | Extrapolated below tested range — treat as speculative |

### 4.2 Data Confidence Factor — `DCF`

Reflects how much verified data actually backs the two estimated
sub-scores (`S_capability`, `S_format_reliability`) for this **exact**
model + quantization combination.

| Situation | `DCF` |
|:---|:---:|
| Base model has official published MMLU/GSM8K, and the quant format is one of the six in Section 3.2's reference table | 1.00 |
| Base model benchmarks are third-party-reproduced rather than vendor-published, OR the quant format is a variant not yet in the reference table (needs an analogous estimate) | 0.85 |
| Base model is a community fine-tune with no independently verified benchmarks, OR the quantization method is unfamiliar / unvalidated on EKOI's stack | 0.70 |

`DCF` exists so an obscure model doesn't get to look exactly as
trustworthy as a well-documented one just because its estimated inputs
happened to compute a similar raw score — the *uncertainty itself* is part
of the investment decision.

---

## 5. Worked Examples

Using the same two models as the original ECAS_METRIC.md worked examples,
for direct comparison.

### Model: `Qwen2.5-14B-Instruct` — INT4 (AWQ)

| Metric | Value | Source |
|:---|:---|:---|
| MMLU | 80.0 | Published (Qwen2.5 technical report) |
| GSM8K | 94.8 | Published |
| `D_retention` (AWQ INT4) | 0.98 | Reference table |
| Peak VRAM (FP16) | 29.59 GB | Measured |
| Peak VRAM (this variant) | 11.35 GB | Measured |
| Tokens / Sec | 42.9 | Measured |
| Parameters | 14B | → `CDF` = 1.00 |
| Data availability | Vendor-published + standard format | → `DCF` = 1.00 |

```
S_capability        = mean(80.0, 94.8) = 87.4
S_format_reliability = 100 × 0.98 = 98.0
R_vram               = 29.59 / 11.35 = 2.607
S_efficiency         = min(100, 50×2.607 + 50×0.429) = min(100, 151.8) = 100.0

Raw weighted = 0.45×87.4 + 0.35×98.0 + 0.20×100.0
             = 39.33 + 34.30 + 20.00 = 93.63

ECAS_pre = 1.00 × 1.00 × 93.63 = 93.6  →  High-Confidence Investment
```

### Model: `meta-llama/Llama-3.2-1B-Instruct` (FP16 Baseline)

| Metric | Value | Source |
|:---|:---|:---|
| MMLU | 49.3 | Published (Meta Llama 3.2 model card) |
| GSM8K | 44.4 | Published |
| `D_retention` (FP16) | 1.00 | Reference table |
| Peak VRAM | 4.15 GB | Measured |
| Tokens / Sec | 170.63 | Measured |
| Parameters | 1B | → `CDF` = 0.65 |
| Data availability | Vendor-published, standard format | → `DCF` = 1.00 |

```
S_capability        = mean(49.3, 44.4) = 46.85
S_format_reliability = 100 × 1.00 = 100.0
R_vram               = 1.0 (baseline)
S_efficiency         = min(100, 50×1.0 + 50×1.7063) = min(100, 135.3) = 100.0

Raw weighted = 0.45×46.85 + 0.35×100.0 + 0.20×100.0
             = 21.08 + 35.00 + 20.00 = 76.08

ECAS_pre = 0.65 × 1.00 × 76.08 = 49.5  →  High Risk
```

**Sanity check against the empirical record:** the original proxy-based
ECAS scored this exact model at **50.2** ("Marginal") after actually
running it against three benchmark agents. The pre-deployment estimate,
using zero proxy-agent data, lands at **49.5** — within a point, entirely
because `CDF` correctly encoded the known small-model agentic cliff. That
convergence is not guaranteed for every model; it's a spot-check that the
discount factors are calibrated sensibly, not a proof they always will be.

---

## 6. Full Scoreboard — Current Model Repository Candidates

All 27 model × quantization combinations from the ARS benchmark corpus,
scored with **zero proxy-agent data** — `S_capability` from each base
model's official published MMLU/GSM8K, `S_format_reliability` from the
Section 3.2 reference table, and `S_efficiency` from measured VRAM/token
throughput (workflow-agnostic hardware telemetry, sourced from the same
underlying benchmark runs but not from their task outcomes).

| Rank | Model | Format | S_cap | S_fmt | S_eff | CDF | ECAS_pre | Tier |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Qwen2.5-14B-Instruct | AWQ (INT4) | 87.4 | 98.0 | 100.0 | 1.00 | **93.6** | 🟢 High-Confidence |
| 2 | Qwen2.5-7B-Instruct | AWQ (INT4) | 83.5 | 98.0 | 100.0 | 1.00 | **91.9** | 🟢 High-Confidence |
| 3 | Qwen2.5-7B-Instruct | FP16 | 83.5 | 100.0 | 89.5 | 1.00 | **90.5** | 🟢 High-Confidence |
| 4 | Llama-3-8B-Instruct | GPTQ (INT8) | 77.0 | 99.0 | 100.0 | 1.00 | **89.3** | 🟢 High-Confidence |
| 5 | Llama-3-8B-Instruct | AWQ (INT4) | 77.0 | 98.0 | 100.0 | 1.00 | **88.9** | 🟢 High-Confidence |
| 6 | Qwen2.5-14B-Instruct | FP16 | 87.4 | 100.0 | 71.2 | 1.00 | **88.6** | 🟢 High-Confidence |
| 6 | Llama-3-8B-Instruct | GPTQ (INT4) | 77.0 | 97.0 | 100.0 | 1.00 | **88.6** | 🟢 High-Confidence |
| 8 | Llama-3-8B-Instruct | bitsandbytes (INT4) | 77.0 | 95.0 | 100.0 | 1.00 | **87.9** | 🟢 High-Confidence |
| 9 | Llama-3-8B-Instruct | FP16 | 77.0 | 100.0 | 87.7 | 1.00 | **87.2** | 🟢 High-Confidence |
| 10 | Llama-3-8B-Instruct | bitsandbytes (INT8) | 77.0 | 97.0 | 91.8 | 1.00 | **86.9** | 🟢 High-Confidence |
| 11 | Qwen2.5-14B-Instruct | FP8 static | 87.4 | 99.0 | 57.2 | 1.00 | **85.4** | 🟡 Promising |
| 12 | Qwen2.5-7B-Instruct | FP8 static | 83.5 | 99.0 | 64.2 | 1.00 | **85.1** | 🟡 Promising |
| 13 | Qwen2.5-3B-Instruct | AWQ (INT4) | 75.6 | 98.0 | 100.0 | 0.95 | **83.9** | 🟡 Promising |
| 14 | Qwen2.5-3B-Instruct | FP16 | 75.6 | 100.0 | 89.2 | 0.95 | **82.5** | 🟡 Promising |
| 15 | Llama-3.2-3B-Instruct | FP16 | 70.5 | 100.0 | 100.0 | 0.95 | **82.4** | 🟡 Promising |
| 16 | Llama-3.2-3B-Instruct | GPTQ (INT8) | 70.5 | 99.0 | 100.0 | 0.95 | **82.1** | 🟡 Promising |
| 17 | Llama-3.2-3B-Instruct | AWQ (INT4) | 70.5 | 98.0 | 100.0 | 0.95 | **81.7** | 🟡 Promising |
| 18 | Llama-3.2-3B-Instruct | GPTQ (INT4) | 70.5 | 97.0 | 100.0 | 0.95 | **81.4** | 🟡 Promising |
| 19 | Llama-3.2-3B-Instruct | bitsandbytes (INT4) | 70.5 | 95.0 | 100.0 | 0.95 | **80.7** | 🟡 Promising |
| 20 | Llama-3.2-3B-Instruct | bitsandbytes (INT8) | 70.5 | 97.0 | 88.6 | 0.95 | **79.2** | 🟡 Promising |
| 21 | Qwen2.5-3B-Instruct | FP8 static | 75.6 | 99.0 | 62.8 | 0.95 | **77.2** | 🟡 Promising |
| 22 | Llama-3.2-1B-Instruct | FP16 | 46.8 | 100.0 | 100.0 | 0.65 | **49.5** | 🔴 High Risk |
| 23 | Llama-3.2-1B-Instruct | GPTQ (INT8) | 46.8 | 99.0 | 100.0 | 0.65 | **49.2** | 🔴 High Risk |
| 24 | Llama-3.2-1B-Instruct | AWQ (INT4) | 46.8 | 98.0 | 100.0 | 0.65 | **49.0** | 🔴 High Risk |
| 25 | Llama-3.2-1B-Instruct | GPTQ (INT4) | 46.8 | 97.0 | 100.0 | 0.65 | **48.8** | 🔴 High Risk |
| 26 | Llama-3.2-1B-Instruct | bitsandbytes (INT4) | 46.8 | 95.0 | 100.0 | 0.65 | **48.3** | 🔴 High Risk |
| 27 | Llama-3.2-1B-Instruct | bitsandbytes (INT8) | 46.8 | 97.0 | 87.6 | 0.65 | **47.2** | 🔴 High Risk |

*(`DCF` = 1.00 throughout — every model here is a vendor-published release on
a quantization format already in the Section 3.2 reference table, so none
of them get the uncertainty discount.)*

**Read-outs from the full table:**

- **No model here scores "Speculative" or "Not Recommended."** Every
  candidate is either genuinely solid (≥75) or clearly risky (≤50) — there's
  no messy middle, which is itself useful signal: the 1B tier's `CDF`
  discount does most of the separating work, more than format or even raw
  capability does.
- **FP8 static is the only format that visibly drags a strong model down a
  full tier** — both Qwen 14B and 7B FP8 variants slide from
  High-Confidence to Promising purely on `S_efficiency`, despite
  `S_format_reliability` being nearly perfect (0.99). Same story as the
  original quantization study: the accuracy is fine, the backend isn't.
- **Llama-3-8B is the strongest floor in the table** — all six of its
  format variants land in High-Confidence (86.9–89.3), the tightest spread
  of any family/size. If the Model Repository wants one "safe default"
  recommendation independent of which format ships, this is it.
- **Every Llama-3.2-1B variant clusters within 2.3 points of each other**
  (47.2–49.5) — once `CDF` = 0.65 applies, the choice of quantization
  format barely matters anymore. That's the discount doing its job: at 1B,
  the format isn't the risk, the parameter count is.

---

## 7. Interpretation Scale

| ECAS_pre Range | Label | Investment Guidance |
|:---:|:---|:---|
| **90 – 100** | **High-Confidence Investment** | Safe to shortlist and pilot against any reasonably-matched workflow. |
| **75 – 89** | **Promising Candidate** | Worth piloting; monitor early results closely before wider rollout. |
| **60 – 74** | **Speculative** | Small-scale trial only — budget for the possibility it underperforms once tested. |
| **40 – 59** | **High Risk** | Avoid unless a specific cost or licensing constraint strongly justifies it. |
| **< 40** | **Not Recommended** | Do not shortlist without new evidence (updated benchmarks, a different quant format, etc.). |

---

## 8. Relationship to ARS — Estimate vs. Evidence

ECAS_pre and ARS are two stages of the same decision, not competing scores:

| | **ECAS (Pre-Deployment)** | **ARS (Agent Readiness Score)** |
|:---|:---|:---|
| Question answered | "Should we invest in testing this model?" | "How did this model actually perform?" |
| Inputs | Published benchmarks, quantization literature, hardware telemetry | Empirical proxy-agent success rate, tool precision, path classification |
| Needs a matching workflow benchmark? | **No** | Yes |
| When to use | Model Repository browsing, initial shortlisting, procurement | After a pilot or proxy-agent run exists |
| Uncertainty handling | Explicit discount multipliers (`CDF`, `DCF`) | None needed — it's measured, not estimated |

**The intended flow:** a model enters the Model Repository → ECAS_pre gives
an initial investment signal with no testing required → if it clears a
reasonable bar (typically ≥ 60), it gets piloted on a real or proxy
workflow → ARS (or an equivalent workflow-specific eval) replaces the
estimate with a measured score. ECAS_pre is a **prior**; ARS-style testing
is the **posterior**. Never treat an ECAS_pre score as a substitute for
ARS once real testing is possible — it exists specifically for the window
before that's true.

---

## 9. Recalibration

The two components most likely to drift out of date:

- **Section 3.2's `D_retention` table** — as EKOI accumulates its own ARS
  results across more models and formats, replace the literature-sourced
  defaults with EKOI's own observed accuracy-retention deltas per format,
  the same way the original ECAS was recalibrated for hardware-specific
  quirks (FP8 backend behavior, etc.).
- **Section 4.1's `CDF` table** — the 3B/1B cliff thresholds are drawn from
  the specific model families tested in ARS so far (Qwen 2.5, Llama 3/3.2).
  Revisit if a new model family shows a materially different scaling
  pattern.

Treat this document as a living spec, not a one-time formula — every ARS
run is, implicitly, a chance to check whether `CDF` and the `D_retention`
table still predict what actually happens.

---

## 10. Edge Cases & Clamping Rules

| Scenario | Handling |
|:---|:---|
| No published MMLU/GSM8K for the base model | Use the nearest same-family, same-size checkpoint's published scores and force `DCF ≤ 0.70`. If no reasonable analog exists, do not compute `ECAS_pre` — route to a manual review queue instead. |
| Quantization format not in Section 3.2's table | Estimate `D_retention` from the closest bit-width/method analog (e.g., an unlisted 4-bit method defaults to the INT4-GPTQ row) and force `DCF ≤ 0.85`. |
| `S_efficiency > 100` | Clamped to `100`, same as original ECAS. |
| Parameter count unknown or unverifiable | `CDF = 0.70` regardless of stated size — cannot verify the capacity claim. |
| Model has no FP16 baseline on record for `R_vram` | Set `R_vram = 1.0` (no compression benefit assumed), matching original ECAS behavior. |

---

## 11. Source Reference

| ECAS_pre Variable | Source |
|:---|:---|
| `MMLU`, `GSM8K` (and `IFEval` if available) | Model's official technical report / model card |
| `D_retention` | Section 3.2 reference table (or EKOI's own recalibrated version, per Section 9) |
| `R_vram`, `Tokens_per_Sec` | EKOI hardware telemetry, measured once per model + quant variant (not per task) |
| `CDF` | Section 4.1 table, keyed on parameter count |
| `DCF` | Section 4.2 table, keyed on data provenance |

No column in this document is read from `benchmark_AGENT_*_all_summary.csv`.
Those files, and the proxy-agent methodology behind them, remain ARS's
domain.

---

## 12. EKOI Workflow × Model Recommendation Matrix

The ECAS scoreboard (Section 6) answers *"which models are worth investing
in?"* — this section answers the follow-up: *"which model should we assign
to which workflow?"*

### 12.1 Workflow Demand Classification

Each of the 7 EKOI workflows is classified by three factors that determine
how sensitive it is to model quality:

- **LLM-Critical Artifacts** — how many artifacts in the chain rely on the
  LLM's language understanding (as opposed to classical ML like anomaly
  detection or forecasting, which is model-agnostic).
- **Reasoning Depth** — total artifact steps the LLM must orchestrate
  sequentially. More steps = higher compounding error risk from
  quantization degradation.
- **Output Sensitivity** — consequence of a wrong output (compliance/legal
  risk vs. internal analytics).

| # | Workflow | Steps | LLM-Critical | Output Sensitivity | Demand |
|:---:|:---|:---:|:---:|:---|:---:|
| W1 | Talent Sourcing & Candidate Matching | 4 | 2 | Medium — recruiter reviews shortlist | **MEDIUM** |
| W2 | Workforce Demand & Resource Allocation | 5 | 2 | Medium — delivery manager reviews | **MEDIUM** |
| W3 | Onboarding & Compliance Verification | 4 | 4 | **High** — compliance / legal risk | **HIGH** |
| W4 | Payroll Anomaly Audit | 2 | 1 | **High** — financial accuracy | **MEDIUM** |
| W5 | Skill Gap & Succession Planning | 4 | 3 | Medium — HR reviews recommendations | **HIGH** |
| W6 | HR Knowledge Assistant (RAG) | 3 | 3 | **High** — policy accuracy with citations | **HIGH** |
| W7 | Offboarding Analytics & Exit Intelligence | 3 | 3 | Low–Medium — analytics / clearance | **MEDIUM** |

**W4's MEDIUM classification despite high output sensitivity:** the anomaly
detection itself is classical ML (Isolation Forests / Z-Score). The LLM
only generates the NudgeEngine alert context — a narrow, well-bounded task.

### 12.2 ECAS Thresholds by Demand Class

| Demand | Min ECAS_pre | Eligible Variants (of 27) | Rationale |
|:---:|:---:|:---:|:---|
| **HIGH** | ≥ 85 | 10 | Multiple LLM-critical artifacts; compliance- or policy-sensitive outputs; quantization degradation compounds across steps. |
| **MEDIUM** | ≥ 75 | 21 | Fewer LLM-critical steps or a human-in-the-loop safeguard absorbs errors. |

### 12.3 Per-Workflow Recommendations

**W1 — Talent Sourcing (MEDIUM, ≥ 75):** 21 variants qualify. Top pick:
Qwen2.5-7B AWQ (91.9). 3B models acceptable — DocumentParser and
EntityExtractor are the only LLM-critical steps, and the recruiter reviews
the shortlist.

**W2 — Resource Allocation (MEDIUM, ≥ 75):** 21 variants qualify. Top pick:
Qwen2.5-7B AWQ (91.9). Core forecasting and anomaly detection are classical
ML; the LLM handles SemanticMatching and NudgeEngine recommendations.

**W3 — Onboarding Compliance (HIGH, ≥ 85):** 10 variants qualify. Top pick:
**Qwen2.5-14B AWQ (93.6)**. All 4 artifacts are LLM-critical. PolicyRAG
cross-references documents against state compliance policies — incorrect
answers carry legal risk. All 3B models excluded (max 83.9).

**W4 — Payroll Audit (MEDIUM, ≥ 75):** 21 variants qualify. Top pick:
Qwen2.5-7B AWQ (91.9). Simplest workflow (2 steps). Anomaly detection is
classical ML; the LLM only triages severity and generates alert context.

**W5 — Succession Planning (HIGH, ≥ 85):** 10 variants qualify. Top pick:
**Qwen2.5-14B AWQ (93.6)**. Sentiment analysis of performance reviews
requires nuanced language understanding that degrades below 7B. Succession
rankings feed C-level decisions.

**W6 — HR Knowledge RAG (HIGH, ≥ 85):** 10 variants qualify. Top pick:
**Qwen2.5-14B AWQ (93.6)**. The purest LLM-dependent workflow — answer
quality is entirely a function of instruction-following and grounded
generation capability. RAG hallucination is the primary risk. No 3B model
has `S_capability` above 75.6.

**W7 — Offboarding Analytics (MEDIUM, ≥ 75):** 21 variants qualify. Top
pick: Qwen2.5-7B AWQ (91.9). Outputs are analytics and clearance tracking,
not compliance decisions. Sentiment and thematic grouping are coarser tasks
than policy reasoning.

### 12.4 Summary — Recommended Model per Workflow

| Workflow | Demand | Min ECAS | Recommended Model | ECAS | VRAM |
|:---|:---:|:---:|:---|:---:|:---:|
| **W1** Talent Sourcing | MEDIUM | 75 | Qwen2.5-7B AWQ | 91.9 | 11.4 GB |
| **W2** Resource Allocation | MEDIUM | 75 | Qwen2.5-7B AWQ | 91.9 | 11.4 GB |
| **W3** Onboarding Compliance | HIGH | 85 | Qwen2.5-14B AWQ | 93.6 | 11.4 GB |
| **W4** Payroll Audit | MEDIUM | 75 | Qwen2.5-7B AWQ | 91.9 | 11.4 GB |
| **W5** Succession Planning | HIGH | 85 | Qwen2.5-14B AWQ | 93.6 | 11.4 GB |
| **W6** HR Knowledge RAG | HIGH | 85 | Qwen2.5-14B AWQ | 93.6 | 11.4 GB |
| **W7** Offboarding Analytics | MEDIUM | 75 | Qwen2.5-7B AWQ | 91.9 | 11.4 GB |

### 12.5 Deployment Consolidation

The 7 workflows collapse into **2 model tiers**:

| Tier | Model | Workflows | VRAM |
|:---|:---|:---|:---:|
| **Tier 1 — High** | Qwen2.5-14B-Instruct AWQ (INT4) | W3, W5, W6 | 11.4 GB |
| **Tier 2 — Medium** | Qwen2.5-7B-Instruct AWQ (INT4) | W1, W2, W4, W7 | 11.4 GB |

Both tiers use ~11.4 GB VRAM (INT4 quantization), fitting on a single 16 GB
GPU with headroom for KV cache. For operational simplicity:

> **Single-model deployment:** Qwen2.5-14B-Instruct AWQ (INT4) at ECAS 93.6
> covers all 7 workflows at 11.4 GB VRAM.

### 12.6 From Estimate to Measurement

These recommendations are ECAS estimates — a **prior**. The deployment
lifecycle (Section 8) applies:

1. Deploy the recommended model(s) to EKOI staging.
2. Run each workflow with real or representative data.
3. Measure ARS on actual task performance — the **posterior**.
4. If ARS confirms ECAS (within ~5 points), proceed to production.
5. If ARS diverges, recalibrate the demand classification and re-evaluate.

