# Enterprise Agentic Quantization Benchmarking Suite

An empirical benchmarking framework designed to evaluate the impacts of weight quantization (FP16, FP8, INT8, INT4 / AWQ, GPTQ, BNB) on the multi-step reasoning, tool-call accuracy, inference speed, energy efficiency, and VRAM utilization of enterprise ReAct AI agents.

Experiments are tuned and benchmarked on an **NVIDIA GeForce RTX 5090 (32GB VRAM)** with full containerized isolation via Docker and **vLLM**.

---

## Quickstart & Execution

All experiments are dockerized for reproducibility. The harness container does **not** auto-run tests on startup, allowing you to review your configuration and trigger benchmarks manually.

```bash
# 1. Pull the updated code
git pull

# 2. Build the benchmark container (one time, or after code changes)
docker compose build

# 3. Start the container in the background (does NOT run benchmarks yet)
docker compose up -d

# 4. Attach and trigger benchmarks manually whenever you're ready
docker exec -it quant-bench bash /app/run_benchmark.sh

# ── Optional: run in a detached tmux/screen session so SSH disconnect is safe ──
docker exec -it quant-bench bash -c "tmux new-session -d -s bench 'bash /app/run_benchmark.sh'"
docker exec -it quant-bench tmux attach -t bench

# 5. Results land on the host at ./results/benchmark_<timestamp>/
#    including output.log (all terminal output) automatically.
```

---

## Architecture & Docker Setup

The benchmarking environment uses a dual-container architecture managed through [docker-compose.yml](docker-compose.yml):

1. **Benchmark Harness Container (`quant-bench`)**:
   - Built from [Dockerfile](Dockerfile) (`python:3.11-slim`).
   - Contains the agent execution loop ([agent.py](agent.py)), tools ([tools.py](tools.py)), metrics & telemetry tracking ([metrics.py](metrics.py)), and test orchestration ([main.py](main.py)).
   - Talks to Docker via `/var/run/docker.sock` to orchestrate sibling vLLM instances per model on demand.
   - Network mode is set to `host` to communicate seamlessly with vLLM endpoints.

2. **Inference Engine (`vllm/vllm-openai`)**:
   - Model execution is powered by the official `vllm/vllm-openai` image with full NVIDIA GPU passthrough (`--gpus all`).
   - Sibling vLLM containers are spawned and torn down automatically by [main.py](main.py) per model to cleanly release GPU memory between runs.

### Volume Mounts
The stack mounts three directories from the host (configured in [docker-compose.yml](docker-compose.yml)):
- **Qwen Models**: Host directory (e.g. `/home/ror-technologies/Quantization/models`) &rarr; `/models/qwen:ro`
- **Quantized Llama Models**: Host directory (e.g. `/home/ror-technologies/IT-helpdesk-agent/quantized_models`) &rarr; `/models/llama:ro`
- **Hugging Face Cache**: Host directory (`/home/ror-technologies/.cache/huggingface`) &rarr; `/root/.cache/huggingface:ro`
  *(Unquantized baseline Llama models are loaded directly from the local HF cache without redownloading)*
- **Results Directory**: Host `./results` &rarr; Container `/results`

---

## Configuration (`config.ini`)

All benchmark parameters, agent selection, model targets, and inference settings are controlled via [config.ini](config.ini).

### 1. Agents Selection
Enable or disable target agents:
```ini
[AGENTS]
AGENT_1_SUPPORT     = False
AGENT_2_IT_HELPDESK = True
AGENT_3_FINANCE     = False
```

### 2. Model Selection
Toggle specific model checkpoints to benchmark:
```ini
[QWEN_MODELS]
qwen_models_dir = /models/qwen
Qwen2.5-3B-Instruct_FP16_Baseline        = False
Qwen2.5-3B-Instruct_INT4_AWQ             = False
Qwen2.5-7B-Instruct_FP16_Baseline        = False
...

[LLAMA_MODELS]
llama_models_dir = /models/llama
; Unquantized baseline models (resolved from mounted HF cache)
meta-llama/Llama-3.2-1B-Instruct           = False
; Quantized local models (resolved from llama_models_dir)
meta-llama__Llama-3.2-1B-Instruct-BNB-4bit = True
meta-llama__Llama-3.2-1B-Instruct-AWQ      = False
...
```

### 3. Benchmark Settings
```ini
[BENCHMARK]
use_vllm         = True
repeated_trials  = True
number_of_trials = 100
save_traces      = False
output_dir       = /results
output_prefix    = benchmark
```

### 4. Inference & VRAM Tuning
```ini
[INFERENCE]
max_new_tokens         = 512
temperature            = 0.2
do_sample              = True
gpu_memory_utilization = 0.05
max_model_len          = 4096
```

> [!IMPORTANT]
> **Accurate VRAM Telemetry**: By default, vLLM reserves 90% of available GPU memory (`--gpu-memory-utilization 0.90`) for its KV cache, causing all models to report ~28GB VRAM regardless of parameter count. By setting `gpu_memory_utilization = 0.05`, KV-cache reservation is minimized, allowing `pynvml` to measure the **true memory footprint** of each model and quantization format.

---

## Automated Logging & Results Pipeline

You no longer need to pipe output manually (e.g. `2>&1 | tee output.log`). 

When [main.py](main.py) runs:
1. It automatically tees all stdout and stderr streams directly to `output.log` inside the timestamped run folder:
   ```
   results/
   └── benchmark_YYYYMMDD_HHMMSS/
       ├── output.log                                    # Full terminal output captured
       ├── AGENT_2_IT_HELPDESK/
       │   ├── benchmark_AGENT_2_IT_HELPDESK_raw_trials.csv
       │   ├── benchmark_AGENT_2_IT_HELPDESK_all_summary.csv
       │   ├── benchmark_AGENT_2_IT_HELPDESK_successes_only.csv
       │   ├── benchmark_AGENT_2_IT_HELPDESK_failures_only.csv
       │   └── traces/                                   # If save_traces = True
       │       └── AGENT_2_IT_HELPDESK_..._Trial_1.txt
   ```
2. Metrics aggregated across trials include:
   - **Performance**: Success Rate, Shortcut Rate, Correct Order Rate, Execution Path breakdown (`PERFECT`, `WANDERING`, `SHORTCUT`, `FAILED`).
   - **Tool Quality**: Total Steps, Actual Tool Calls, Redundant Calls, Parse Errors, Tool Efficiency, Tokens per Useful Step.
   - **Compute & Speed**: Inference Time (s), Generated Tokens, Tokens/Sec, Achieved TFLOPS.
   - **Hardware Telemetry**: Average & Peak Power (W), Total Energy (Joules), Average & Peak VRAM (GB), Average GPU Utilization (%).

---

## Methodology Pipeline

### Agent Design & System Prompts
The evaluation suite implements three distinct agent archetypes across operational enterprise domains:
* **AGENT_1_SUPPORT**: Customer support agent handling orders, inventory, replacements, and refunds.
* **AGENT_2_IT_HELPDESK**: IT helpdesk engineer investigating outages, parsing service logs, and restarting systems.
* **AGENT_3_FINANCE**: Quantitative financial analyst querying prices, calculating 5-day moving averages, and executing trades.

Each agent operates within a strict **ReAct** (Reasoning and Acting) loop:
1. The agent generates thought reasoning followed by a single tool action:
   ```text
   Thought: I need to check the status of the servers.
   Action: check_server_status
   Action Input: {}
   ```
2. The environment executes the tool and injects the result:
   ```text
   Tool Result: {"payment_backend": "down", "database": "online"}
   ```
3. When the goal is met, the agent concludes:
   ```text
   Thought: I have completed the task.
   Final Answer: task is complete
   ```

### Tool Sets & Environmental State
Mutable environment state (mock databases, inventory counts, running services, and stock histories) is deep-copied and **strictly reset** before each trial to eliminate cross-trial contamination ([tools.py](tools.py)).

| Agent Archetype | Available Tools |
| :--- | :--- |
| **Customer Support** | `check_order`, `check_inventory`, `issue_replacement`, `issue_refund`, `send_email_to_customer`, `check_promotions` |
| **IT Helpdesk** | `check_server_status`, `read_service_logs`, `restart_service`, `ping_server`, `clear_browser_cache` |
| **Financial Analyst** | `get_current_price`, `get_5_day_average`, `execute_trade`, `get_company_news`, `calculate_tax`, `calculate_math` |

### Validation & Success Criteria
Runs are verified against task-specific `strict_success_criteria` defined in [dataset.json](dataset.json):
* **`must_call`**: Tools that must be executed for the task to be considered solved.
* **`must_not_call`**: Forbidden or incorrect tools that invalidate the trial if called.

### Hardware Telemetry Formulae
* **Telemetry**: Background sampling via `pynvml` at 50ms intervals captures instantaneous power (W), VRAM usage (GB), and GPU utilization (%).
* **Energy**: $\text{Total Energy (J)} = \text{Avg Power (W)} \times \text{Duration (s)}$
* **TFLOPS Proxy**:
  $$\text{TFLOPS} = \frac{2 \times N_{\text{tokens}} \times (P_{\text{params}} \times 10^9)}{10^{12} \times T_{\text{duration}}}$$
