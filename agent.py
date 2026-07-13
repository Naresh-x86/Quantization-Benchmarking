import re
from inference import LLMEngine
from tools import AVAILABLE_TOOLS, get_tools_description

SYSTEM_PROMPT = """You are a helpful customer support agent. You must resolve the user's issue by thinking step-by-step and calling the available tools.

{tool_desc}

To use a tool, you MUST format your response exactly like this:
Thought: I need to do [action] because [reason].
Action: [tool_name]
Action Input: [arguments in JSON format, e.g., {{"order_id": "ORD-123"}}]

When you have resolved the issue or if you cannot proceed further, format your final response like this:
Thought: I have finished the task.
Final Answer: [Your final message to the user]
"""

class ReActAgent:
    def __init__(self, llm_engine: LLMEngine, max_steps: int = 10):
        self.llm = llm_engine
        self.max_steps = max_steps
        self.system_prompt = SYSTEM_PROMPT.format(tool_desc=get_tools_description())
        
    def _format_prompt(self, history: list) -> str:
        # Simple ChatML format for Qwen
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def run(self, task_description: str):
        history = [{"role": "user", "content": task_description}]
        
        total_tokens = 0
        total_time = 0.0
        steps = 0
        success = False
        
        while steps < self.max_steps:
            prompt = self._format_prompt(history)
            
            # Generate LLM response
            output = self.llm.generate(prompt)
            total_tokens += output["generated_tokens"]
            total_time += output["duration"]
            response_text = output["text"]
            
            history.append({"role": "assistant", "content": response_text})
            
            # Parse ReAct format
            if "Final Answer:" in response_text:
                success = True
                break
                
            action_match = re.search(r"Action:\s*(.+)", response_text)
            input_match = re.search(r"Action Input:\s*(.+)", response_text)
            
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input_str = input_match.group(1).strip()
                
                try:
                    import json
                    action_input = json.loads(action_input_str)
                    
                    if action in AVAILABLE_TOOLS:
                        tool_func = AVAILABLE_TOOLS[action]
                        tool_result = str(tool_func(**action_input))
                    else:
                        tool_result = f"Error: Tool '{action}' not found."
                except Exception as e:
                    tool_result = f"Error executing tool: {e}. Ensure Action Input is valid JSON."
                    
                history.append({"role": "user", "content": f"Tool Result: {tool_result}"})
            else:
                # If the model didn't follow the format properly, gently correct it
                history.append({"role": "user", "content": "Error: Could not parse Action and Action Input. Please use the exact format requested."})
                
            steps += 1
            
        return {
            "success": success,
            "steps": steps,
            "total_generated_tokens": total_tokens,
            "total_inference_time_sec": total_time,
            "final_history": history
        }
