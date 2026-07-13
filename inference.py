import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMEngine:
    def __init__(self, model_path: str, use_vllm: bool = False, quant_type: str = "baseline"):
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.quant_type = quant_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if self.use_vllm:
            from vllm import LLM, SamplingParams
            print(f"Loading {model_path} with vLLM...")
            # For AWQ or standard models, vllm generally handles it natively
            # We set gpu_memory_utilization=0.8 to avoid startup crashes if the GPU isn't 100% free
            self.model = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.8)
            self.sampling_params = SamplingParams(temperature=0.1, max_tokens=512, stop=["<|im_end|>"])
        else:
            print(f"Loading {model_path} with Transformers...")
            
            # Setup loading kwargs based on quantization
            kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }
            
            # Depending on how the models were exported, we might need specific flags.
            # Qwen models generally use safe tensors and transformers auto-detects bitsandbytes or AWQ if config.json is correct.
            # We assume config.json has the correct quantization_config.
            if "FP16" in quant_type:
                kwargs["torch_dtype"] = torch.float16
            elif "FP8" in quant_type:
                # Attempt to keep in FP8 or fallback to bfloat16
                kwargs["torch_dtype"] = "auto"
                
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            self.model.eval()

    def generate(self, prompt: str) -> dict:
        start_time = time.time()
        
        if self.use_vllm:
            from vllm import SamplingParams
            outputs = self.model.generate([prompt], self.sampling_params, use_tqdm=False)
            output_text = outputs[0].outputs[0].text
            generated_tokens = len(outputs[0].outputs[0].token_ids)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            ]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    eos_token_id=terminators,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            generated_ids = outputs[0][input_length:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_tokens = len(generated_ids)
            
        end_time = time.time()
        duration = end_time - start_time
        tokens_per_sec = generated_tokens / duration if duration > 0 else 0
        
        # Strip potential stop tokens if left over
        output_text = output_text.replace("<|im_end|>", "").strip()
        
        return {
            "text": output_text,
            "generated_tokens": generated_tokens,
            "duration": duration,
            "tokens_per_sec": tokens_per_sec
        }

    def calculate_tflops(self, params_billion: float, tokens_generated: int, duration_sec: int) -> float:
        """
        Rough heuristic for inference FLOPS: 2 * N * P / T
        N = tokens generated
        P = parameters (in billions) -> P * 1e9
        T = time in seconds
        Returns TFLOPS (TeraFLOPS)
        """
        if duration_sec <= 0:
            return 0.0
        flops = 2 * tokens_generated * (params_billion * 1e9)
        tflops = flops / 1e12 / duration_sec
        return tflops
