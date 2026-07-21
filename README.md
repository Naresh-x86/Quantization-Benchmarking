# Methodology Pipeline

## Model & Quantization Framework

* **Base Architecture:** Qwen 2.5 models were selected as state-of-the-art open-weight instruction models. They were scaled across 3B, 7B, and 14B parameter sizes to observe how model scale interacts with VRAM constraints and reasoning capabilities.
* **Quantization Formats:** Benchmark evaluations include FP16 (Baseline half-precision), FP8 (8-bit floating-point, static per tensor via LLM Compressor), and INT4 (4-bit Activation-aware Weight Quantization via AutoAWQ).
* **Inference Engines:** Models were executed using vLLM and Hugging Face Transformers depending on format compatibility.

---

## Agent Design and System Prompts

The evaluation suite implements three distinct agent archetypes across different operational domains: **Support Agent** (Customer Support), **IT Helpdesk Engineer**, and **Financial Analyst**. Each agent operates within a ReAct (Reasoning and Acting) framework.

### System Directives
Each agent is initialized with a domain-specific role and explicitly instructed to resolve the user's issue by thinking step-by-step and executing available tools.

### Operational Constraints
The system prompt strictly enforces the following behavioral rules:
1. The agent must call exactly **one** tool per response to prevent multi-call hallucinations.
2. After calling a tool, the agent must immediately stop and wait for the "Tool Result" observation from the environment before continuing its reasoning.
3. The agent must not include the "Final Answer" in the same generation step as an action.

### Response Formatting
To invoke a tool, the agent must generate a precise syntactic block:

```text
Thought: I need to do [action] because [reason].
Action: [tool_name]
Action Input: {"arg1": "value1"}
```

To complete a task, the agent outputs:

```text
Thought: I have finished the task.
Final Answer: [Your final message]
```

---

## Tool Sets & Environmental State

The simulation environment provides distinct tools tailored to each agent archetype. Mutable state (such as mock databases for inventory, active services, and stock prices) is strictly reset before each trial to prevent cross-trial data contamination.

| Agent Archetype | Available Tools |
| :--- | :--- |
| **Customer Support** | `check_order`, `check_inventory`, `issue_replacement`, `issue_refund` |
| **IT Helpdesk** | `check_server_status`, `read_service_logs`, `restart_service` |
| **Financial Analyst** | `get_current_price`, `get_5_day_average`, `execute_trade` |

---

## Validation and Success Criteria

During runtime, agent outputs are evaluated dynamically:

* **Parsing Engine:** A regex parser specifically searches for the `Action:` and `Action Input:` tags. Arguments must be valid JSON matching the specified tool signature. If the JSON is invalid or the tool does not exist, the environment injects an error message into the context as the tool result.
* **Completion & Success Validation:** When the agent emits a `Final Answer`, the execution loop terminates. The run is evaluated using a `strict_success_criteria` framework containing `must_call` and `must_not_call` tool sets. A task is marked as successful if and only if:
  1. All tools in the `must_call` set were successfully executed without exceptions.
  2. Absolutely no tools in the `must_not_call` set were triggered.

---

## Hardware Telemetry & Metric Calculation

All experiments were benchmarked on an NVIDIA RTX 5090 (32GB VRAM).

### Hardware Telemetry
Power, memory, and utilization data are captured by a background thread running a custom NVIDIA Management Library (`pynvml`) polling script (`metrics.py`) every 50ms. The reported **Avg Power**, **Avg VRAM**, and **Avg GPU Util** are computed as the arithmetic mean of all readings recorded during the generation window.

### Inference Speed
Token generation speed (**Tokens / Sec**) is calculated as the total number of generated tokens divided by the total duration (in seconds) of the generation phase:

`Tokens / Sec = generated_tokens / duration_sec`

### TFLOPS Calculation
A heuristic for achieved TeraFLOPS is calculated using the following formula:

`TFLOPS = (2 * generated_tokens * (parameters_in_billions * 1e9)) / (1e12 * duration_sec)`
