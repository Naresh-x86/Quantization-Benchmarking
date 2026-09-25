"""LLM inference engine — vLLM (OpenAI-compatible) backend.

This module talks to a vLLM server running in the *same* docker-compose stack
via its OpenAI-compatible /v1/chat/completions endpoint.  All heavy model
loading is handled by the vLLM container; this process only sends HTTP
requests.

The vLLM server is launched by the benchmark runner (run_benchmark.sh) before
main.py is called, and torn down automatically after all benchmarks finish.

Key vLLM flag used for accurate VRAM measurement
─────────────────────────────────────────────────
  --gpu-memory-utilization <float>  (default 0.90)

vLLM pre-allocates 90 % of VRAM as a KV-cache reserve, so nvidia-smi always
shows ~28 GB regardless of model size.  We override this to a very small value
(e.g. 0.05) so that only the model weights occupy VRAM, giving an accurate
per-model VRAM reading from pynvml.

This value is read from config.ini → [INFERENCE] → gpu_memory_utilization and
is forwarded to the vLLM process via the VLLM_GPU_MEMORY_UTIL environment
variable set in run_benchmark.sh.
"""

import time
import openai


class LLMEngine:
    """OpenAI-compatible client that talks to a running vLLM server."""

    def __init__(
        self,
        model_path: str,
        quant_type: str = "FP16",
        use_vllm: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = True,
        vllm_base_url: str = "http://localhost:8008/v1",
        # gpu_memory_utilization is handled at server-start time (not per-request)
    ):
        self.model_path = model_path
        self.quant_type = quant_type.upper()
        self.use_vllm = use_vllm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature if do_sample else 0.0
        self.vllm_base_url = vllm_base_url

        # The model name sent to the vLLM API must match what the server loaded.
        # vLLM uses the model path as the model ID.
        self.model_id = model_path

        self.client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",  # vLLM does not require a real key
        )

        print(f"  vLLM engine ready → model: {model_path}  base_url: {vllm_base_url}")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, messages: list) -> dict:
        """Send a chat-completion request to the vLLM server.

        Args:
            messages: list of {"role": str, "content": str} dicts

        Returns:
            dict with keys:
              text, prompt_tokens, generated_tokens, duration, tokens_per_sec
        """
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        duration = time.time() - start

        choice = response.choices[0]
        text = choice.message.content or ""

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        generated_tokens = usage.completion_tokens if usage else len(text.split())
        tokens_per_sec = generated_tokens / duration if duration > 0 else 0.0

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "duration": duration,
            "tokens_per_sec": tokens_per_sec,
        }

    # ------------------------------------------------------------------
    # TFLOPS heuristic
    # ------------------------------------------------------------------

    def calculate_tflops(
        self,
        params_billion: float,
        tokens_generated: int,
        duration_sec: float,
    ) -> float:
        """Token-throughput proxy for achieved compute (TFLOPS).

        Formula:  TFLOPS = (2 × N × P) / (1e12 × T)
          N = generated tokens
          P = total parameters (params_billion × 1e9)
          T = wall-clock generation time in seconds
        """
        if duration_sec <= 0 or params_billion <= 0:
            return 0.0
        flops = 2 * tokens_generated * (params_billion * 1e9)
        return flops / 1e12 / duration_sec
