"""
SDK Definition module for defining Gemma function calling interfaces.

This module provides the core classes and interfaces for defining functions that can be called by Gemma models.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
import json


class Parameter(BaseModel):
    """
    Definition of a parameter for a Gemma function.
    """
    type: str = Field(..., description="The type of the parameter (string, number, integer, boolean, array, object)")
    description: Optional[str] = Field(None, description="Description of the parameter")
    enum: Optional[List[Any]] = Field(None, description="List of allowed values for the parameter")
    format: Optional[str] = Field(None, description="Format of the parameter (e.g., 'binary' for images)")
    items: Optional[Dict[str, Any]] = Field(None, description="Schema for array items if type is 'array'")
    properties: Optional[Dict[str, Any]] = Field(None, description="Properties if type is 'object'")
    required: Optional[List[str]] = Field(None, description="List of required properties if type is 'object'")


class FunctionDefinition(BaseModel):
    """
    Definition of a function that can be called by Gemma models.
    """
    name: str = Field(..., description="Name of the function")
    description: str = Field(..., description="Description of what the function does")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the function")
    supports_images: bool = Field(False, description="Whether the function supports image inputs")
    implementation: Optional[Callable] = Field(None, description="Actual implementation of the function")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the function definition to a dictionary format suitable for Gemma.
        
        Returns:
            Dictionary representation of the function definition
        """
        result = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
        
        if self.supports_images:
            result["supports_images"] = True
            
        return result
    
    def to_json(self) -> str:
        """
        Convert the function definition to a JSON string.
        
        Returns:
            JSON string representation of the function definition
        """
        # Create a copy without the implementation which is not JSON serializable
        data = self.to_dict()
        return json.dumps(data, indent=2)
    
    def __call__(self, *args, **kwargs):
        """
        Call the function implementation with the provided arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the function call
            
        Raises:
            ValueError: If no implementation is provided
        """
        if self.implementation is None:
            raise ValueError(f"No implementation provided for function '{self.name}'")
        
        return self.implementation(*args, **kwargs)


class FunctionRegistry:
    """
    Registry for managing function definitions.
    """
    
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
    
    def register(self, function_def: FunctionDefinition) -> None:
        """
        Register a function definition.
        
        Args:
            function_def: Function definition to register
        """
        self.functions[function_def.name] = function_def
    
    def register_from_dict(self, function_dict: Dict[str, Any], implementation: Optional[Callable] = None) -> None:
        """
        Register a function definition from a dictionary.
        
        Args:
            function_dict: Dictionary representation of the function definition
            implementation: Optional implementation of the function
        """
        function_def = FunctionDefinition(
            name=function_dict["name"],
            description=function_dict["description"],
            parameters=function_dict["parameters"],
            supports_images=function_dict.get("supports_images", False),
            implementation=implementation
        )
        self.register(function_def)
    
    def register_from_json(self, json_str: str, implementation: Optional[Callable] = None) -> None:
        """
        Register a function definition from a JSON string.
        
        Args:
            json_str: JSON string representation of the function definition
            implementation: Optional implementation of the function
        """
        function_dict = json.loads(json_str)
        self.register_from_dict(function_dict, implementation)
    
    def register_from_file(self, file_path: str, implementation: Optional[Callable] = None) -> None:
        """
        Register a function definition from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing the function definition
            implementation: Optional implementation of the function
        """
        with open(file_path, 'r') as f:
            function_dict = json.load(f)
        
        self.register_from_dict(function_dict, implementation)
    
    def register_multiple(self, function_defs: List[Dict[str, Any]], implementations: Optional[Dict[str, Callable]] = None) -> None:
        """
        Register multiple function definitions.
        
        Args:
            function_defs: List of function definition dictionaries
            implementations: Optional dictionary mapping function names to implementations
        """
        for function_dict in function_defs:
            implementation = None
            if implementations and function_dict["name"] in implementations:
                implementation = implementations[function_dict["name"]]
            
            self.register_from_dict(function_dict, implementation)
    
    def get(self, name: str) -> Optional[FunctionDefinition]:
        """
        Get a function definition by name.
        
        Args:
            name: Name of the function
            
        Returns:
            Function definition or None if not found
        """
        return self.functions.get(name)
    
    def get_all(self) -> List[FunctionDefinition]:
        """
        Get all registered function definitions.
        
        Returns:
            List of all function definitions
        """
        return list(self.functions.values())
    
    def get_all_dicts(self) -> List[Dict[str, Any]]:
        """
        Get all registered function definitions as dictionaries.
        
        Returns:
            List of function definition dictionaries
        """
        return [func.to_dict() for func in self.functions.values()]
    
    def remove(self, name: str) -> None:
        """
        Remove a function definition by name.
        
        Args:
            name: Name of the function to remove
        """
        if name in self.functions:
            del self.functions[name]
    
    def clear(self) -> None:
        """
        Clear all registered function definitions.
        """
        self.functions.clear()


class FunctionCallResult(BaseModel):
    """
    Result of a function call.
    """
    function_name: str = Field(..., description="Name of the called function")
    parameters: Dict[str, Any] = Field(..., description="Parameters passed to the function")
    result: Any = Field(..., description="Result of the function call")
    error: Optional[str] = Field(None, description="Error message if the function call failed")
    
    def is_success(self) -> bool:
        """
        Check if the function call was successful.
        
        Returns:
            True if the function call was successful, False otherwise
        """
        return self.error is None
