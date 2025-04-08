"""
Main SDK class for Gemma function calling.

This module provides the main GemmaSDK class that integrates all components for function calling.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json

from .function_definition import FunctionDefinition, FunctionRegistry, FunctionCallResult


class GemmaSDK:
    """
    Main SDK class for Gemma function calling.
    
    This class integrates all components for function calling with Gemma models.
    """
    
    def __init__(self, model=None, tokenizer=None, functions=None):
        """
        Initialize the GemmaSDK.
        
        Args:
            model: Gemma model instance
            tokenizer: Gemma tokenizer instance
            functions: List of function definitions or a FunctionRegistry instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.registry = FunctionRegistry()
        
        if functions:
            if isinstance(functions, FunctionRegistry):
                self.registry = functions
            elif isinstance(functions, list):
                self.registry.register_multiple(functions)
    
    def register_function(self, function_def: Union[Dict[str, Any], FunctionDefinition], implementation: Optional[Callable] = None) -> None:
        """
        Register a function definition.
        
        Args:
            function_def: Function definition to register
            implementation: Optional implementation of the function
        """
        if isinstance(function_def, dict):
            self.registry.register_from_dict(function_def, implementation)
        else:
            self.registry.register(function_def)
    
    def register_functions(self, function_defs: List[Dict[str, Any]], implementations: Optional[Dict[str, Callable]] = None) -> None:
        """
        Register multiple function definitions.
        
        Args:
            function_defs: List of function definition dictionaries
            implementations: Optional dictionary mapping function names to implementations
        """
        self.registry.register_multiple(function_defs, implementations)
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all registered function definitions as dictionaries.
        
        Returns:
            List of function definition dictionaries
        """
        return self.registry.get_all_dicts()
    
    def create_function_calling_prompt(self, user_query: str, output_format: str = "json") -> str:
        """
        Create a function calling prompt for Gemma.
        
        Args:
            user_query: User query to include in the prompt
            output_format: Output format for function calls ("json" or "python")
            
        Returns:
            Function calling prompt for Gemma
        """
        function_defs = json.dumps(self.get_function_definitions(), indent=2)
        
        if output_format == "json":
            setup = """You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of
{"name": function name, "parameters": dictionary of argument name and its value}

You SHOULD NOT include any other text in the response if you call a function
"""
        else:  # Python format
            setup = """You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

You SHOULD NOT include any other text in the response if you call a function
"""
        
        return f"{setup}\n{function_defs}\n{user_query}"
    
    def parse_function_call(self, response: str, output_format: str = "json") -> Optional[Dict[str, Any]]:
        """
        Parse a function call from a model response.
        
        Args:
            response: Model response to parse
            output_format: Output format of the function call ("json" or "python")
            
        Returns:
            Parsed function call or None if no function call was found
        """
        try:
            if output_format == "json":
                function_call = json.loads(response)
                return function_call
            else:  # Python format
                # Simple regex-based parsing for Python format
                import re
                match = re.search(r'\[(.*?)\]', response)
                if match:
                    function_text = match.group(1)
                    function_name = function_text.split('(')[0].strip()
                    params_text = function_text.split('(')[1].split(')')[0]
                    
                    # Parse parameters
                    params = {}
                    param_pairs = params_text.split(',')
                    for pair in param_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            # Remove quotes if present
                            value = value.strip()
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            params[key.strip()] = value
                    
                    return {"name": function_name, "parameters": params}
        except Exception as e:
            print(f"Error parsing function call: {e}")
            return None
        
        return None
    
    def execute_function(self, function_call: Dict[str, Any]) -> FunctionCallResult:
        """
        Execute a function call.
        
        Args:
            function_call: Function call to execute
            
        Returns:
            Result of the function call
            
        Raises:
            ValueError: If the function is not found
        """
        function_name = function_call["name"]
        parameters = function_call["parameters"]
        
        function_def = self.registry.get(function_name)
        if not function_def:
            return FunctionCallResult(
                function_name=function_name,
                parameters=parameters,
                result=None,
                error=f"Function '{function_name}' not found"
            )
        
        try:
            result = function_def(**parameters)
            return FunctionCallResult(
                function_name=function_name,
                parameters=parameters,
                result=result
            )
        except Exception as e:
            return FunctionCallResult(
                function_name=function_name,
                parameters=parameters,
                result=None,
                error=str(e)
            )
    
    def execute(self, user_query: str, output_format: str = "json") -> Tuple[Optional[FunctionCallResult], str]:
        """
        Execute a user query with function calling.
        
        Args:
            user_query: User query to execute
            output_format: Output format for function calls ("json" or "python")
            
        Returns:
            Tuple of (function call result, model response)
            
        Raises:
            ValueError: If model or tokenizer is not set
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be set before executing queries")
        
        prompt = self.create_function_calling_prompt(user_query, output_format)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        function_call = self.parse_function_call(response, output_format)
        if function_call:
            result = self.execute_function(function_call)
            return result, response
        
        return None, response
