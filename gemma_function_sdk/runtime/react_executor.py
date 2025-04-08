"""
ReAct (Reasoning and Acting) execution framework for Gemma SDK.

This module provides a ReAct execution framework for multi-turn, multi-API function calls.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import torch

from ..sdk_definition import FunctionCallResult, FunctionRegistry
from .gemma_runtime import GemmaRuntime


class ReActExecutor:
    """
    ReAct (Reasoning and Acting) execution framework for Gemma SDK.
    
    This class implements the ReAct pattern for multi-turn, multi-API function calls.
    """
    
    def __init__(self, runtime: Optional[GemmaRuntime] = None, model_name: str = "google/gemma-7b", 
                 hf_token: Optional[str] = None, functions: Optional[Union[List[Dict[str, Any]], FunctionRegistry]] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ReActExecutor.
        
        Args:
            runtime: Optional existing GemmaRuntime instance
            model_name: Name of the Gemma model to use if runtime is not provided
            hf_token: Hugging Face token for accessing the model if runtime is not provided
            functions: List of function definitions or a FunctionRegistry instance
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.runtime = runtime or GemmaRuntime(model_name, hf_token, functions, device)
        
        # Store conversation history
        self.conversation_history = []
    
    def create_react_prompt(self, user_query: str) -> str:
        """
        Create a ReAct prompt for Gemma.
        
        Args:
            user_query: User query to include in the prompt
            
        Returns:
            ReAct prompt for Gemma
        """
        # Get function descriptions
        function_defs = self.runtime.sdk.get_function_definitions()
        tools_description = ""
        
        for func in function_defs:
            name = func["name"]
            desc = func["description"]
            params = []
            
            for param_name, param_info in func["parameters"]["properties"].items():
                param_type = param_info.get("type", "string")
                params.append(f"{param_name}: {param_type}")
            
            param_str = ", ".join(params)
            tools_description += f"* `{name}({param_str})`: {desc}\n"
        
        # Create the ReAct prompt
        system_prompt = f"""You are an AI assistant that helps users by using tools.
You have access to the following tools:

{tools_description}

Use the following format:

Thought: I need to think about what to do
Action: I should use the tool `tool_name` with input `{{"param1": "value1", "param2": "value2"}}`
Observation: The result of the tool
... (repeat Thought/Action/Observation as needed)
Final Answer: The final response to the user's query

Begin!
"""
        
        # Add conversation history
        conversation = system_prompt
        for entry in self.conversation_history:
            conversation += f"\n{entry}"
        
        # Add the current user query
        conversation += f"\nUser query: {user_query}\nThought:"
        
        return conversation
    
    def parse_react_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a ReAct response from the model.
        
        Args:
            response: Model response to parse
            
        Returns:
            Parsed ReAct response with thought, action, and final answer
        """
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        # Extract thought
        thought_match = re.search(r"Thought:(.*?)(?:Action:|Final Answer:|$)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action: I should use the tool `(.*?)` with input `(.*?)`", response)
        if action_match:
            result["action"] = action_match.group(1).strip()
            action_input_str = action_match.group(2).strip()
            
            # Try to parse action input as JSON
            try:
                result["action_input"] = json.loads(action_input_str)
            except json.JSONDecodeError:
                # If not valid JSON, try to parse key-value pairs
                action_input = {}
                pairs = action_input_str.split(",")
                for pair in pairs:
                    if ":" in pair:
                        key, value = pair.split(":", 1)
                        action_input[key.strip()] = value.strip()
                result["action_input"] = action_input
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:(.*?)$", response, re.DOTALL)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
        
        return result
    
    def execute_action(self, action: str, action_input: Dict[str, Any]) -> str:
        """
        Execute a tool action.
        
        Args:
            action: Name of the tool to execute
            action_input: Input parameters for the tool
            
        Returns:
            Result of the tool execution as a string
        """
        function_call = {
            "name": action,
            "parameters": action_input
        }
        
        result = self.runtime.sdk.execute_function(function_call)
        
        if result.error:
            return f"Error: {result.error}"
        else:
            # Convert result to string if it's not already
            if isinstance(result.result, str):
                return result.result
            elif isinstance(result.result, (dict, list)):
                return json.dumps(result.result, indent=2)
            else:
                return str(result.result)
    
    def execute(self, user_query: str, max_turns: int = 5) -> str:
        """
        Execute a user query using the ReAct framework.
        
        Args:
            user_query: User query to execute
            max_turns: Maximum number of turns to take
            
        Returns:
            Final answer or the last response if no final answer is reached
        """
        # Create the initial prompt
        conversation = self.create_react_prompt(user_query)
        
        for turn in range(max_turns):
            # Generate model response
            inputs = self.runtime.tokenizer(conversation, return_tensors="pt")
            outputs = self.runtime.model.generate(**inputs, max_length=1024)
            response = self.runtime.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the new content only
            new_content = response[len(conversation):]
            
            # Update conversation
            conversation = response
            
            # Parse the response
            parsed = self.parse_react_response(new_content)
            
            # Record in conversation history
            self.conversation_history.append(f"Thought: {parsed['thought']}")
            
            # Check if we have a final answer
            if parsed["final_answer"]:
                self.conversation_history.append(f"Final Answer: {parsed['final_answer']}")
                return parsed["final_answer"]
            
            # Execute action if present
            if parsed["action"] and parsed["action_input"]:
                self.conversation_history.append(f"Action: I should use the tool `{parsed['action']}` with input `{json.dumps(parsed['action_input'])}`")
                
                # Execute the action
                result = self.execute_action(parsed["action"], parsed["action_input"])
                
                # Add observation to conversation
                self.conversation_history.append(f"Observation: {result}")
                conversation += f"\nObservation: {result}\nThought:"
            else:
                # No action or final answer, break the loop
                break
        
        # If we reach here, we didn't get a final answer
        return "No final answer was reached after the maximum number of turns."
    
    def clear_conversation(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history.clear()
    
    def get_conversation_history(self) -> List[str]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history
