import re
import json
import inspect
from inference import LLMEngine
from tools import get_tools_dict, get_tools_description, reset_tool_state, get_tool_signature

def _get_example_for_agent(agent_id: str) -> str:
    """Return a concrete worked example showing the one-tool-at-a-time pattern."""
    if agent_id == "AGENT_1_SUPPORT":
        return """Example (DO NOT copy — use the real values from the user's request):
User: Check order ORD-100.
Thought: I need to look up the order.
Action: check_order
Action Input: {"order_id": "ORD-100"}
[STOP and wait for Tool Result before continuing]"""
    elif agent_id == "AGENT_2_IT_HELPDESK":
        return """Example (DO NOT copy — use the real values from the user's request):
User: A service is down.
Thought: I need to check which services are down.
Action: check_server_status
Action Input: {}
[STOP and wait for Tool Result before continuing]"""
    elif agent_id == "AGENT_3_FINANCE":
        return """Example (DO NOT copy — use the real values from the user's request):
User: Is 50.00 less than 100.00?
Thought: I need to use the calculate_math tool to evaluate the expression.
Action: calculate_math
Action Input: {"expression": "50.00 < 100.00"}
[STOP and wait for Tool Result before continuing]"""
    return ""

def get_system_prompt(agent_id: str) -> str:
    agent_roles = {
        "AGENT_1_SUPPORT": "You are a helpful customer support agent.",
        "AGENT_2_IT_HELPDESK": "You are a highly skilled IT Helpdesk engineer.",
        "AGENT_3_FINANCE": "You are a quantitative financial analyst."
    }
    role = agent_roles.get(agent_id, "You are a helpful AI assistant.")
    example = _get_example_for_agent(agent_id)
    
    return f"""{role}
You have access to the following tools:
{get_tools_description(agent_id)}

You must solve the user's problem by calling the appropriate tools one by one.
Do not guess or make up information. Always use the tools.

IMPORTANT FORMAT — follow this EXACTLY:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object with the correct parameter names>

{example}

RULES (you MUST follow every rule):
1. Call exactly ONE tool per response. Do NOT call multiple tools at once.
2. Action Input MUST be valid JSON using the parameter names shown in the tool list above.
3. After writing Action Input, STOP IMMEDIATELY. Do NOT write anything else.
   Wait for the Tool Result before deciding your next step.
4. Do NOT invent or guess Tool Results. Only use results that are given to you.

When the task is fully complete, respond with:
Thought: I have completed the task.
Final Answer: task is complete
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
        consecutive_parse_errors = 0        # consecutive format failures (for early exit)
        
        while steps < self.max_steps:
            messages = self._get_messages(history)
            
            output = self.llm.generate(messages)
            total_tokens += output["generated_tokens"]
            total_time += output["duration"]
            response_text = output["text"]
            
            # Truncate hallucinated results from smaller models
            for stop_word in ["\nResult:", "\nResponse:", "\nTool Result:", "\nObservation:",
                              "\nExpected next call:", "\n TOOL RESULT:", "\n RULES:",
                              "\n[STOP"]:
                idx = response_text.find(stop_word)
                if idx != -1:
                    response_text = response_text[:idx]
            
            # Truncate multi-action dumps: if the model emitted more than one
            # "Action:" block, keep only the first one (up to its Action Input JSON).
            first_action_pos = response_text.find("Action:")
            if first_action_pos != -1:
                second_action_pos = response_text.find("\nAction:", first_action_pos + 7)
                # Only truncate if the second Action: comes AFTER an Action Input:
                first_input_pos = response_text.find("Action Input:", first_action_pos)
                if second_action_pos != -1 and first_input_pos != -1 and second_action_pos > first_input_pos:
                    # Find the closing brace of the first Action Input JSON
                    brace_start = response_text.find("{", first_input_pos)
                    if brace_start != -1 and brace_start < second_action_pos:
                        depth = 0
                        cut_pos = second_action_pos  # fallback
                        for i, ch in enumerate(response_text[brace_start:]):
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    cut_pos = brace_start + i + 1
                                    break
                        response_text = response_text[:cut_pos]
            
            history.append({"role": "assistant", "content": response_text.strip()})
            
            # Parse Action FIRST (even if Final Answer is also present)
            # Enforce that action is just the function name (no brackets/params attached)
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", response_text)
            input_match_start = response_text.find("Action Input:")
            
            # Fuzzy recovery: if no Action: match but we have Action Input:,
            # try to find a partial tool name in the response (fixes Qwen's '_refund' loop).
            # Priority: exact full name match > exact suffix match > substring match.
            if not action_match and input_match_start != -1:
                # Extract the text before Action Input for matching
                pre_input_text = response_text[:input_match_start]
                best_match = None
                best_priority = 99  # lower is better
                for tool_name in self.tools_dict:
                    if tool_name in pre_input_text:
                        # Full name found in text before Action Input (highest priority)
                        if best_priority > 0:
                            best_match = tool_name
                            best_priority = 0
                    elif tool_name.split('_', 1)[-1] in pre_input_text:
                        # Suffix match like '_refund' → 'issue_refund' (second priority)
                        # Prefer longer suffix matches
                        suffix = tool_name.split('_', 1)[-1]
                        priority = 1 if len(suffix) > 3 else 2
                        if priority < best_priority:
                            best_match = tool_name
                            best_priority = priority
                if best_match:
                    matched_name = best_match
                    class FuzzyMatch:
                        def group(self, n):
                            return matched_name
                    action_match = FuzzyMatch()
            
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
                consecutive_parse_errors = 0  # reset on successful parse
                
                try:
                    # Sanitize Python booleans/None to valid JSON before parsing
                    sanitized = action_input_str
                    sanitized = re.sub(r'\bTrue\b', 'true', sanitized)
                    sanitized = re.sub(r'\bFalse\b', 'false', sanitized)
                    sanitized = re.sub(r'\bNone\b', 'null', sanitized)
                    action_input = json.loads(sanitized)
                    
                    if action in self.tools_dict:
                        tool_func = self.tools_dict[action]
                        # Strip unknown kwargs: only pass params the tool actually accepts
                        sig = inspect.signature(tool_func)
                        valid_params = set(sig.parameters.keys())
                        filtered_input = {k: v for k, v in action_input.items() if k in valid_params}
                        tool_result = str(tool_func(**filtered_input))
                        # Only count as successfully called if no exception was raised
                        tools_called_successfully.add(action)
                        tool_call_sequence.append(action)
                    else:
                        tool_result = f"Error: Tool '{action}' not found. Available tools are: {', '.join(self.tools_dict.keys())}"
                except json.JSONDecodeError as e:
                    sig_hint = ""
                    if action in self.tools_dict:
                        sig_hint = f" Correct usage: {get_tool_signature(self.tools_dict[action])}"
                    tool_result = f"Error: Invalid JSON in Action Input. {e}.{sig_hint}"
                except Exception as e:
                    # Include the tool's actual signature so the model learns the correct params
                    sig_hint = ""
                    if action in self.tools_dict:
                        sig_hint = f" Correct usage: {get_tool_signature(self.tools_dict[action])}"
                    tool_result = f"Error: {e}.{sig_hint}"
                    
                # We want to record this execution for the human-readable trace.
                # history will store an extended object temporarily, or we just append string format.
                # To keep agent.py logic the same, we'll append the user tool result, 
                # but also add a secret field for run_single.py to format later.
                history.append({"role": "user", "content": f"Tool Result: {tool_result}", "action": action, "action_input": action_input_str})
                
                # If Final Answer was ALSO in this response, we still executed the tool above.
                # Only terminate if all required tools have been called; otherwise the
                # model dumped everything in one go (common with 1B) and we should
                # let the loop continue so subsequent tools can be called.
                if "Final Answer:" in response_text:
                    should_terminate = True
                    if strict_success_criteria:
                        must_call_dicts = strict_success_criteria.get("must_call", [])
                        must_call_names = [mc["tool"] if isinstance(mc, dict) else mc for mc in must_call_dicts]
                        must_call_set = set(must_call_names)
                        must_not_call = set(strict_success_criteria.get("must_not_call", []))
                        if must_call_set.issubset(tools_called_successfully) and not must_not_call.intersection(tools_called_successfully):
                            success = True
                        else:
                            # Not all tools called yet — don't terminate, let the loop continue
                            should_terminate = False
                            success = False
                    else:
                        success = True
                    if should_terminate:
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
                consecutive_parse_errors += 1
                
                # If the model is stuck in a loop (3+ consecutive parse errors), break out
                if consecutive_parse_errors >= 3:
                    # Check if we already completed all required tools
                    if strict_success_criteria:
                        must_call_dicts = strict_success_criteria.get("must_call", [])
                        must_call_names = [mc["tool"] if isinstance(mc, dict) else mc for mc in must_call_dicts]
                        must_call_set = set(must_call_names)
                        must_not_call = set(strict_success_criteria.get("must_not_call", []))
                        if must_call_set.issubset(tools_called_successfully) and not must_not_call.intersection(tools_called_successfully):
                            success = True
                    break
                
                history.append({"role": "user", "content": "Error: Could not parse your response. You must use this exact format:\nThought: <reasoning>\nAction: <tool_name>\nAction Input: <JSON>\n\nOr if the task is done:\nThought: I have completed the task.\nFinal Answer: task is complete"})
                
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
