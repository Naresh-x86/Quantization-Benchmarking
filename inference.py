"""LLM inference engine for the IT-helpdesk benchmark.

Handles loading pre-quantized Llama checkpoints from the local quantized_models/
folder (AWQ 4-bit, GPTQ 4-bit/8-bit, BNB 4-bit/8-bit) AND unquantized FP16
baselines directly from HuggingFace — all via the Hugging Face Transformers
back-end.

Quantization back-end notes:
  AWQ       → autoawq reads the quantization_config from the saved config.json
  GPTQ      → auto-gptq / optimum reads the quantization_config from config.json
  BNB-*     → bitsandbytes reads the quantization_config from config.json
  FP16      → no quantization; loads in float16 directly from HF or local path

Usage:
    # Pre-quantized local checkpoint
    engine = LLMEngine("quantized_models/meta-llama__Llama-3.2-1B-Instruct-AWQ",
                       quant_type="AWQ")

    # Unquantized FP16 baseline straight from HuggingFace
    engine = LLMEngine("meta-llama/Llama-3.2-1B-Instruct", quant_type="FP16")

    output = engine.generate(messages)   # list of {"role":…, "content":…}
    tflops = engine.calculate_tflops(params_b=1.24, tokens=80, duration_sec=2.1)
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMEngine:
    """Transformers-based inference engine.

    Accepts any pre-quantized local checkpoint (AWQ / GPTQ / BNB) or an
    unquantized HuggingFace model ID for FP16 baseline comparison.
    """

    def __init__(
        self,
        model_path: str,
        quant_type: str = "AWQ",
        use_vllm: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        self.model_path    = model_path
        self.quant_type    = quant_type.upper()
        self.use_vllm      = use_vllm
        self.max_new_tokens = max_new_tokens
        self.temperature   = temperature
        self.do_sample     = do_sample

        print(f"  Loading [{self.quant_type}]  {model_path} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {
            "device_map":       "auto",
            "trust_remote_code": True,
        }

        if self.quant_type == "FP16":
            # Unquantized baseline — load in float16, no quantization_config needed.
            load_kwargs["torch_dtype"] = torch.float16

        elif self.quant_type in ("AWQ", "GPTQ", "GPTQ-4BIT", "GPTQ-8BIT"):
            # AWQ and GPTQ checkpoints embed their quantization_config in config.json.
            # AutoModelForCausalLM picks it up automatically when the right backend
            # (autoawq / auto-gptq / optimum) is installed.
            load_kwargs["torch_dtype"] = torch.float16

        elif self.quant_type in ("BNB-4BIT", "BNB-8BIT"):
            # BNB checkpoints saved by quantize_bnb.py also embed quantization_config
            # in config.json, so from_pretrained re-applies the same quantization.
            # No extra BitsAndBytesConfig needed — it's already in the saved config.
            pass  # device_map="auto" is sufficient

        else:
            # Unknown type — attempt plain float16 load and warn.
            print(f"  Warning: unknown quant_type '{self.quant_type}', loading as FP16.")
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        print(f"  ✓  Model ready.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, messages: list) -> dict:
        """Run one forward pass given a chat-template message list.

        Args:
            messages: list of {"role": str, "content": str} dicts

        Returns:
            dict with keys:
              text, prompt_tokens, generated_tokens, duration, tokens_per_sec
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=1024,
        )
        device         = next(self.model.parameters()).device
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_tokens  = int(input_ids.shape[1])

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.time() - start

        new_ids          = output_ids[0][prompt_tokens:]
        generated_tokens = int(new_ids.shape[0])
        text             = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tokens_per_sec   = generated_tokens / duration if duration > 0 else 0.0

        return {
            "text":             text,
            "prompt_tokens":    prompt_tokens,
            "generated_tokens": generated_tokens,
            "duration":         duration,
            "tokens_per_sec":   tokens_per_sec,
        }

    # ------------------------------------------------------------------
    # TFLOPS heuristic (Naresh's formula, adopted here)
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

        This is the same rough heuristic used by Naresh (and widely in ML
        benchmarking). It does NOT distinguish INT4 from INT8 or FP16
        arithmetic throughput — it's a token-throughput proxy that at least
        varies meaningfully by measured duration and actual token count,
        unlike the old static-param estimate.
        """
        if duration_sec <= 0 or params_billion <= 0:
            return 0.0
        flops = 2 * tokens_generated * (params_billion * 1e9)
        return flops / 1e12 / duration_sec
