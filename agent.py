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
    
    return f"""{role}
You have access to the following tools:
{get_tools_description(agent_id)}

You must solve the user's problem by calling the minimum necessary tools and no more.

Use this exact format to call a tool:
Thought: I should call the tool ...
Action: tool_name
Action Input: {{"param_name": "value"}}

CRITICAL RULES — follow these exactly:
- Call only ONE tool per response. Stop immediately after writing the Action Input and wait for the Tool Result.
- MINIMAL TOOL USE: Only call tools that are directly required by the task. Do NOT call extra tools to "notify", "confirm", "inform", or "double-check" unless the task explicitly asks for it.
- Do NOT call tools out of curiosity or as a courtesy (e.g. do not send emails or check promotions unless the task explicitly requires it).
- Do NOT repeat a tool call you have already made.
- Action Input MUST be valid JSON.
- Once the required actions are complete, go directly to Final Answer. Do not add unnecessary extra tool calls.

When the task is complete, use this format:
Thought: I have completed the task.
Final Answer: [brief summary of what was done]
"""

class ReActAgent:
    def __init__(self, llm_engine: LLMEngine, agent_id: str, max_steps: int = 15):
        self.llm = llm_engine
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.system_prompt = get_system_prompt(agent_id)
        self.tools_dict = get_tools_dict(agent_id)
        
    def _get_messages(self, history: list) -> list:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        return messages

    def run(self, task_description: str, strict_success_criteria: dict = None):
        # Reset mutable tool state before each trial to prevent cross-trial contamination
        reset_tool_state()
        
        history = [{"role": "user", "content": task_description}]
        
        total_tokens = 0
        total_time = 0.0
        steps = 0
        success = False
        tools_called_successfully = set()   # unique names (kept for backward compat)
        tools_attempted = set()             # unique names (kept for backward compat)
        tool_call_sequence = []             # ordered list with repeats
        tool_attempt_sequence = []          # ordered list with repeats
        parse_error_count = 0               # format / parse failures
        
        while steps < self.max_steps:
            messages = self._get_messages(history)
            
            output = self.llm.generate(messages)
            total_tokens += output["generated_tokens"]
            total_time += output["duration"]
            response_text = output["text"]
            
            # Truncate hallucinated results from smaller models
            for stop_word in ["\nResult:", "\nResponse:", "\nTool Result:", "\nObservation:", "\nExpected next call:"]:
                idx = response_text.find(stop_word)
                if idx != -1:
                    response_text = response_text[:idx]
            
            history.append({"role": "assistant", "content": response_text.strip()})
            
            # Parse Action FIRST (even if Final Answer is also present)
            # Enforce that action is just the function name (no brackets/params attached)
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", response_text)
            input_match_start = response_text.find("Action Input:")
            
            if action_match and input_match_start != -1:
                action = action_match.group(1).strip()
                
                # Robust JSON extraction
                json_start = response_text.find("{", input_match_start)
                if json_start != -1:
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(response_text[json_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = json_start + i
                                break
                    if json_end != -1:
                        action_input_str = response_text[json_start:json_end+1]
                    else:
                        action_input_str = response_text[input_match_start+13:].strip().split('\n')[0]
                else:
                    # fallback if no braces
                    action_input_str = response_text[input_match_start+13:].strip().split('\n')[0]
                
                tools_attempted.add(action)
                tool_attempt_sequence.append(action)
                
                try:
                    action_input = json.loads(action_input_str)
                    
                    if action in self.tools_dict:
                        tool_func = self.tools_dict[action]
                        tool_result = str(tool_func(**action_input))
                        # Only count as successfully called if no exception was raised
                        tools_called_successfully.add(action)
                        tool_call_sequence.append(action)
                    else:
                        tool_result = f"Error: Tool '{action}' not found. Available tools are: {', '.join(self.tools_dict.keys())}"
                except Exception as e:
                    tool_result = f"Error executing tool: {e}. Ensure Action Input is valid JSON with the correct parameter names."
                    
                # We want to record this execution for the human-readable trace.
                # history will store an extended object temporarily, or we just append string format.
                # To keep agent.py logic the same, we'll append the user tool result, 
                # but also add a secret field for run_single.py to format later.
                history.append({"role": "user", "content": f"Tool Result: {tool_result}", "action": action, "action_input": action_input_str})
                
                # If Final Answer was ALSO in this response, we still executed the tool above.
                # Now check if we should terminate.
                if "Final Answer:" in response_text:
                    if strict_success_criteria:
                        must_call_dicts = strict_success_criteria.get("must_call", [])
                        must_call_names = [mc["tool"] if isinstance(mc, dict) else mc for mc in must_call_dicts]
                        must_call_set = set(must_call_names)
                        must_not_call = set(strict_success_criteria.get("must_not_call", []))
                        if must_call_set.issubset(tools_called_successfully) and not must_not_call.intersection(tools_called_successfully):
                            success = True
                        else:
                            success = False
                    else:
                        success = True
                    break
                    
            elif "Final Answer:" in response_text:
                # Pure Final Answer with no Action in this response
                if strict_success_criteria:
                    must_call_dicts = strict_success_criteria.get("must_call", [])
                    must_call_names = [mc["tool"] if isinstance(mc, dict) else mc for mc in must_call_dicts]
                    must_call_set = set(must_call_names)
                    must_not_call = set(strict_success_criteria.get("must_not_call", []))
                    if must_call_set.issubset(tools_called_successfully) and not must_not_call.intersection(tools_called_successfully):
                        success = True
                    else:
                        success = False
                else:
                    success = True
                break
            else:
                # Model didn't follow format at all
                parse_error_count += 1
                history.append({"role": "user", "content": "Error: Could not parse Action and Action Input. Please use the exact format requested. Call exactly ONE tool per response."})
                
            steps += 1
            
        return {
            "success": success,
            "steps": steps,
            "total_generated_tokens": total_tokens,
            "total_inference_time_sec": total_time,
            "final_history": history,
            "tools_called": list(tools_called_successfully),
            "tools_attempted": list(tools_attempted),
            "tool_call_sequence": tool_call_sequence,
            "tool_attempt_sequence": tool_attempt_sequence,
            "parse_error_count": parse_error_count
        }
