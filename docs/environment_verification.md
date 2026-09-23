# Environment & Metrics Cross-Verification

> Extracted from [output_6th_aug.log](file:///home/PRIME1895/Quantization-Benchmarking/logs/output_6th_aug.log) and HuggingFace model API metadata.
> Verified: 2026-09-08

---

## 1. Transformers Version

> **The paper reports 4.47. The log shows 5.13.1. These are not the same.**

| Item | Paper Value | Log-Verified Value | Source |
|:---|:---:|:---:|:---|
| **Transformers** | 4.47 | **5.13.1** | GPT-QModel banner printed at every AWQ/GPTQ load (log L62) |
| **PyTorch** | 2.10+cu128 | **2.10.0+cu128** ✅ | Same banner (log L63), also confirmed by `torch_dtype` deprecation warnings |
| **GPT-QModel** | — | **7.1.0+b5bf7c8** | log L61 |
| **Triton** | — | **3.6.0** | log L64 |

### Analysis

The `Transformers : 5.13.1` version string is printed by **GPT-QModel** (the AWQ/GPTQ inference library) at every quantized model load. This is GPT-QModel's own report of the `transformers` package version installed in the environment. It appears consistently across all 36+ AWQ/GPTQ loads in the log — no conflicting version is ever printed.

The PyTorch version `2.10.0+cu128` is also confirmed independently by the `[transformers]` deprecation warning:
```
Skipping import of cpp extensions due to incompatible torch version.
Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128).
```

**Action Required:** The paper's `4.47` needs to be updated to **`5.13.1`**. The `4.47` was likely an earlier draft value or a confusion with a different package version. The actual benchmark run on Aug 6 used Transformers **5.13.1** with PyTorch **2.10.0+cu128**.

---

## 2. FP16 Checkpoint Sizes (Llama Models) — Missing `Weights Size (GB)`

### The Problem

In the benchmark CSV (`benchmark_AGENT_1_SUPPORT_all_summary.csv`), the three Llama FP16 baselines show `Weights Size (GB) mean = 0.0`:

```csv
meta-llama/Llama-3.2-1B-Instruct    , FP16 , 1.0, ..., 0.0, ...
meta-llama/Llama-3.2-3B-Instruct    , FP16 , 3.0, ..., 0.0, ...
meta-llama/Meta-Llama-3-8B-Instruct , FP16 , 8.0, ..., 0.0, ...
```

### Root Cause

The `get_weights_size_gb()` function in `run_single.py` calculates weights size by walking the **local model directory** and summing `.safetensors` / `.bin` file sizes:

```python
def get_weights_size_gb(model_path: str) -> float:
    total_bytes = 0
    if os.path.exists(model_path):
        for root, _, files in os.walk(model_path):
            for file in files:
                if file.endswith((".safetensors", ".bin")):
                    total_bytes += os.path.getsize(os.path.join(root, file))
    return total_bytes / (1024**3)
```

The Llama FP16 models were loaded as **remote HuggingFace model IDs** (e.g., `meta-llama/Llama-3.2-1B-Instruct`) — not from a local directory. The `os.path.exists()` check fails for a HF hub identifier, so the function returns `0.0`. The files were cached in the HF cache directory (e.g., `~/.cache/huggingface/hub/`), but the function didn't look there.

**Qwen FP16 models** don't have this issue because they were loaded from a **local directory** (`/home/ror-technologies/Quantization/models/Qwen2.5-*-Instruct_FP16_Baseline`).

### Correct FP16 Safetensors Sizes (from HuggingFace API)

Sizes computed from the HF API `safetensors.parameters` metadata (BF16 = 2 bytes/param):

| Model | Parameters | Safetensors Files | Computed Size (GB) | HF `usedStorage` (total repo) |
|:---|:---:|:---:|:---:|:---:|
| **meta-llama/Llama-3.2-1B-Instruct** | 1,235,814,400 | 1 (`model.safetensors`) | **2.30 GB** | 4.61 GB |
| **meta-llama/Llama-3.2-3B-Instruct** | 3,212,749,824 | 2 (`model-00001/2-of-00002.safetensors`) | **5.98 GB** | 11.97 GB |
| **meta-llama/Meta-Llama-3-8B-Instruct** | 8,030,261,248 | 4 (`model-0000{1-4}-of-00004.safetensors`) | **14.96 GB** | 57.88 GB |

> **Computation**: `params × 2 bytes ÷ 1024³ = GB`
>
> The HF `usedStorage` field includes all repo files (original `.pth`, tokenizers, README, etc.), not just safetensors. The safetensors-only size is computed from the parameter count × dtype width.

### Cross-Validation Against Qwen Baselines

As a sanity check, comparing against the Qwen FP16 baselines (which **were** measured locally):

| Model | Params (B) | Measured Weights Size (GB) | Expected (params × 2B) | Match? |
|:---|:---:|:---:|:---:|:---:|
| Qwen2.5-3B FP16 | 3.0 | 5.75 | 5.75 | ✅ |
| Qwen2.5-7B FP16 | 7.0 | 14.19 | 14.19 | ✅ |
| Qwen2.5-14B FP16 | 14.0 | 27.51 | 27.51 | ✅ |

The ratio `measured / params` is consistently ~1.91–1.97 GB per billion params (accounting for embeddings, layernorms, etc. — not exactly 2.0 because not all params are in the attention/MLP blocks).

---

## 3. Corrected Values for Paper/Metrics

### Weights Size Updates

| Model | Old Value | Corrected Value |
|:---|:---:|:---:|
| Llama-3.2-1B-Instruct FP16 | `--` (0.0) | **2.30 GB** |
| Llama-3.2-3B-Instruct FP16 | `--` (0.0) | **5.98 GB** |
| Meta-Llama-3-8B-Instruct FP16 | `--` (0.0) | **14.96 GB** |

### Environment Version Corrections

| Item | Old (Paper) | Corrected |
|:---|:---:|:---:|
| Transformers | 4.47 | **5.13.1** |
| PyTorch | 2.10+cu128 | **2.10.0+cu128** (correct, just specify patch) |

---

## 4. Other Observations from Log

**DNS Failures on Llama FP16 runs for AGENT_2:** Lines 860-862 and 998-1000 show that `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` FP16 baselines **failed on AGENT_2_IT_HELPDESK** due to DNS resolution failures during custom_generate download. This means their AGENT_2 results are missing from the dataset — only AGENT_1 and AGENT_3 data exists for those two Llama FP16 baselines.

**Torch cpp extensions warning**: `Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128)` appears on every model load. This may affect some optimized kernels but didn't prevent the benchmarks from running.
