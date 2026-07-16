import re
import json
from inference import LLMEngine
from tools import get_tools_dict, get_tools_description, reset_tool_state

def get_system_prompt(agent_id: str) -> str:
    agent_roles = {
        "AGENT_1_SUPPORT": "You are a helpful customer support agent.",
        "AGENT_2_IT_HELPDESK": "You are a highly skilled IT Helpdesk engineer.",
        "AGENT_3_FINANCE": "You are a quantitative financial analyst."
    }
    role = agent_roles.get(agent_id, "You are a helpful AI assistant.")
    
    return f"""{role} You must resolve the user's issue by thinking step-by-step and calling the available tools.

{get_tools_description(agent_id)}

IMPORTANT RULES:
1. You must call exactly ONE tool per response. Do NOT call multiple tools in the same response.
2. After calling a tool, STOP and wait for the Tool Result before proceeding.
3. Do NOT include "Final Answer:" in the same response as an "Action:". Always wait for the tool result first.

To use a tool, format your response EXACTLY like this (one tool call only):
Thought: I need to do [action] because [reason].
Action: [tool_name]
Action Input: [arguments in JSON format, e.g., {{"order_id": "ORD-123"}}]

When you have resolved the issue and all necessary actions are complete, format your final response like this (with NO Action/Action Input):
Thought: I have finished the task.
Final Answer: [Your final message to the user]
"""

class ReActAgent:
    def __init__(self, llm_engine: LLMEngine, agent_id: str, max_steps: int = 15):
        self.llm = llm_engine
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.system_prompt = get_system_prompt(agent_id)
        self.tools_dict = get_tools_dict(agent_id)
        
    def _format_prompt(self, history: list) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def run(self, task_description: str, strict_success_criteria: dict = None):
        # Reset mutable tool state before each trial to prevent cross-trial contamination
        reset_tool_state()
        
        history = [{"role": "user", "content": task_description}]
        
        total_tokens = 0
        total_time = 0.0
        steps = 0
        success = False
        tools_called_successfully = set()
        tools_attempted = set()
        
        while steps < self.max_steps:
            prompt = self._format_prompt(history)
            
            output = self.llm.generate(prompt)
            total_tokens += output["generated_tokens"]
            total_time += output["duration"]
            response_text = output["text"]
            
            history.append({"role": "assistant", "content": response_text})
            
            # Parse Action FIRST (even if Final Answer is also present)
            action_match = re.search(r"Action:\s*(.+)", response_text)
            input_match = re.search(r"Action Input:\s*(.+)", response_text)
            
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input_str = input_match.group(1).strip()
                
                tools_attempted.add(action)
                
                try:
                    action_input = json.loads(action_input_str)
                    
                    if action in self.tools_dict:
                        tool_func = self.tools_dict[action]
                        tool_result = str(tool_func(**action_input))
                        # Only count as successfully called if no exception was raised
                        tools_called_successfully.add(action)
                    else:
                        tool_result = f"Error: Tool '{action}' not found. Available tools are: {', '.join(self.tools_dict.keys())}"
                except Exception as e:
                    tool_result = f"Error executing tool: {e}. Ensure Action Input is valid JSON with the correct parameter names."
                    
                history.append({"role": "user", "content": f"Tool Result: {tool_result}"})
                
                # If Final Answer was ALSO in this response, we still executed the tool above.
                # Now check if we should terminate.
                if "Final Answer:" in response_text:
                    if strict_success_criteria:
                        must_call = set(strict_success_criteria.get("must_call", []))
                        must_not_call = set(strict_success_criteria.get("must_not_call", []))
                        if must_call.issubset(tools_called_successfully) and not must_not_call.intersection(tools_called_successfully):
                            success = True
                        else:
                            success = False
                    else:
                        success = True
                    break
                    
            elif "Final Answer:" in response_text:
                # Pure Final Answer with no Action in this response
                if strict_success_criteria:
                    must_call = set(strict_success_criteria.get("must_call", []))
                    must_not_call = set(strict_success_criteria.get("must_not_call", []))
                    if must_call.issubset(tools_called_successfully) and not must_not_call.intersection(tools_called_successfully):
                        success = True
                    else:
                        success = False
                else:
                    success = True
                break
            else:
                # Model didn't follow format at all
                history.append({"role": "user", "content": "Error: Could not parse Action and Action Input. Please use the exact format requested. Call exactly ONE tool per response."})
                
            steps += 1
            
        return {
            "success": success,
            "steps": steps,
            "total_generated_tokens": total_tokens,
            "total_inference_time_sec": total_time,
            "final_history": history,
            "tools_called": list(tools_called_successfully),
            "tools_attempted": list(tools_attempted)
        }
