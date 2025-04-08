"""
Gemma SDK Runtime module for executing function calls.

This module provides the runtime environment for executing function calls with Gemma models.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..sdk_definition import GemmaSDK, FunctionCallResult, FunctionRegistry


class GemmaRuntime:
    """
    Runtime environment for executing function calls with Gemma models.
    
    This class extends the basic GemmaSDK with additional runtime capabilities.
    """
    
    def __init__(self, model_name: str = "google/gemma-7b", hf_token: Optional[str] = None, 
                 functions: Optional[Union[List[Dict[str, Any]], FunctionRegistry]] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the GemmaRuntime.
        
        Args:
            model_name: Name of the Gemma model to use
            hf_token: Hugging Face token for accessing the model
            functions: List of function definitions or a FunctionRegistry instance
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=hf_token, 
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Initialize the SDK
        self.sdk = GemmaSDK(self.model, self.tokenizer, functions)
        
        # Store execution history
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_function(self, function_def: Union[Dict[str, Any], Any], implementation: Optional[Callable] = None) -> None:
        """
        Register a function definition.
        
        Args:
            function_def: Function definition to register
            implementation: Optional implementation of the function
        """
        self.sdk.register_function(function_def, implementation)
    
    def register_functions(self, function_defs: List[Dict[str, Any]], implementations: Optional[Dict[str, Callable]] = None) -> None:
        """
        Register multiple function definitions.
        
        Args:
            function_defs: List of function definition dictionaries
            implementations: Optional dictionary mapping function names to implementations
        """
        self.sdk.register_functions(function_defs, implementations)
    
    def execute(self, user_query: str, output_format: str = "json") -> Tuple[Optional[FunctionCallResult], str]:
        """
        Execute a user query with function calling.
        
        Args:
            user_query: User query to execute
            output_format: Output format for function calls ("json" or "python")
            
        Returns:
            Tuple of (function call result, model response)
        """
        result, response = self.sdk.execute(user_query, output_format)
        
        # Record execution in history
        execution_record = {
            "query": user_query,
            "response": response,
            "function_call": result.dict() if result else None,
            "timestamp": import_time().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.execution_history.append(execution_record)
        
        return result, response
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history.
        
        Returns:
            List of execution records
        """
        return self.execution_history
    
    def clear_execution_history(self) -> None:
        """
        Clear the execution history.
        """
        self.execution_history.clear()
    
    def save_execution_history(self, file_path: str) -> None:
        """
        Save the execution history to a file.
        
        Args:
            file_path: Path to save the execution history
        """
        with open(file_path, 'w') as f:
            json.dump(self.execution_history, f, indent=2)
    
    def load_execution_history(self, file_path: str) -> None:
        """
        Load execution history from a file.
        
        Args:
            file_path: Path to load the execution history from
        """
        with open(file_path, 'r') as f:
            self.execution_history = json.load(f)


# Helper function to import time module
def import_time():
    """Import time module to avoid circular imports."""
    import datetime
    return datetime.datetime.now()
