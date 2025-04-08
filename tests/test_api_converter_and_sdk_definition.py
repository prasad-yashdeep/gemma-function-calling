import unittest
from unittest.mock import MagicMock, patch
import json
import os
from pathlib import Path

from gemma_sdk.api_converter import APIConverter
from gemma_sdk.sdk_definition import FunctionDefinition, FunctionRegistry, FunctionCallResult


class TestAPIConverter(unittest.TestCase):
    """Test cases for the API Converter module."""
    
    def setUp(self):
        self.converter = APIConverter()
        self.test_dir = Path("test_files")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a sample OpenAPI spec
        self.openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "paths": {
                "/weather": {
                    "get": {
                        "operationId": "getWeather",
                        "summary": "Get weather information",
                        "parameters": [
                            {
                                "name": "location",
                                "in": "query",
                                "required": True,
                                "schema": {
                                    "type": "string"
                                },
                                "description": "City name or location"
                            }
                        ]
                    }
                }
            }
        }
        
        self.openapi_path = self.test_dir / "openapi.json"
        with open(self.openapi_path, "w") as f:
            json.dump(self.openapi_spec, f)
        
        # Create a sample REST API spec
        self.rest_spec = {
            "endpoints": [
                {
                    "name": "getWeather",
                    "description": "Get weather information",
                    "parameters": [
                        {
                            "name": "location",
                            "type": "string",
                            "description": "City name or location",
                            "required": True
                        }
                    ]
                }
            ]
        }
        
        self.rest_path = self.test_dir / "rest.json"
        with open(self.rest_path, "w") as f:
            json.dump(self.rest_spec, f)
        
        # Create a sample image API spec
        self.image_spec = {
            "endpoints": [
                {
                    "name": "analyzeImage",
                    "description": "Analyze image content",
                    "image_processing": True,
                    "parameters": [
                        {
                            "name": "image",
                            "type": "image",
                            "description": "Image to analyze",
                            "required": True
                        }
                    ]
                }
            ]
        }
        
        self.image_path = self.test_dir / "image_api.json"
        with open(self.image_path, "w") as f:
            json.dump(self.image_spec, f)
    
    def tearDown(self):
        # Clean up test files
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()
    
    def test_convert_from_openapi(self):
        """Test converting from OpenAPI specification."""
        functions = self.converter.convert_from_openapi(self.openapi_path)
        
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["name"], "getWeather")
        self.assertEqual(functions[0]["description"], "Get weather information")
        self.assertIn("location", functions[0]["parameters"]["properties"])
        self.assertIn("location", functions[0]["parameters"]["required"])
    
    def test_convert_from_rest(self):
        """Test converting from REST API documentation."""
        functions = self.converter.convert_from_rest(self.rest_path)
        
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["name"], "getWeather")
        self.assertEqual(functions[0]["description"], "Get weather information")
        self.assertIn("location", functions[0]["parameters"]["properties"])
        self.assertIn("location", functions[0]["parameters"]["required"])
    
    def test_convert_from_image_api(self):
        """Test converting from image API specification."""
        functions = self.converter.convert_from_image_api(self.image_path)
        
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["name"], "analyzeImage")
        self.assertEqual(functions[0]["description"], "Analyze image content")
        self.assertIn("image", functions[0]["parameters"]["properties"])
        self.assertIn("image", functions[0]["parameters"]["required"])
        self.assertTrue(functions[0]["supports_images"])
    
    def test_save_and_load_functions(self):
        """Test saving and loading function definitions."""
        functions = [
            {
                "name": "testFunction",
                "description": "Test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string"
                        }
                    },
                    "required": ["param1"]
                }
            }
        ]
        
        output_path = self.test_dir / "functions.json"
        self.converter.save_functions(functions, output_path)
        
        loaded_functions = self.converter.load_functions(output_path)
        
        self.assertEqual(len(loaded_functions), 1)
        self.assertEqual(loaded_functions[0]["name"], "testFunction")
        self.assertEqual(loaded_functions[0]["description"], "Test function")


class TestFunctionDefinition(unittest.TestCase):
    """Test cases for the Function Definition module."""
    
    def test_function_definition(self):
        """Test creating and using a function definition."""
        # Create a function definition
        function_def = FunctionDefinition(
            name="testFunction",
            description="Test function",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string"
                    }
                },
                "required": ["param1"]
            }
        )
        
        # Test attributes
        self.assertEqual(function_def.name, "testFunction")
        self.assertEqual(function_def.description, "Test function")
        
        # Test to_dict method
        function_dict = function_def.to_dict()
        self.assertEqual(function_dict["name"], "testFunction")
        self.assertEqual(function_dict["description"], "Test function")
        
        # Test to_json method
        function_json = function_def.to_json()
        function_from_json = json.loads(function_json)
        self.assertEqual(function_from_json["name"], "testFunction")
    
    def test_function_call(self):
        """Test calling a function through its definition."""
        # Create a mock implementation
        mock_impl = MagicMock(return_value="test result")
        
        # Create a function definition with the mock implementation
        function_def = FunctionDefinition(
            name="testFunction",
            description="Test function",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string"
                    }
                },
                "required": ["param1"]
            },
            implementation=mock_impl
        )
        
        # Call the function
        result = function_def(param1="test")
        
        # Verify the mock was called with the correct arguments
        mock_impl.assert_called_once_with(param1="test")
        self.assertEqual(result, "test result")


class TestFunctionRegistry(unittest.TestCase):
    """Test cases for the Function Registry."""
    
    def setUp(self):
        self.registry = FunctionRegistry()
        
        # Create a sample function definition
        self.function_def = FunctionDefinition(
            name="testFunction",
            description="Test function",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string"
                    }
                },
                "required": ["param1"]
            }
        )
    
    def test_register_and_get(self):
        """Test registering and retrieving a function definition."""
        # Register the function
        self.registry.register(self.function_def)
        
        # Get the function
        retrieved_func = self.registry.get("testFunction")
        
        # Verify it's the same function
        self.assertEqual(retrieved_func.name, "testFunction")
        self.assertEqual(retrieved_func.description, "Test function")
    
    def test_register_from_dict(self):
        """Test registering a function from a dictionary."""
        # Create a function dictionary
        function_dict = {
            "name": "dictFunction",
            "description": "Dictionary function",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string"
                    }
                },
                "required": ["param1"]
            }
        }
        
        # Register from dictionary
        self.registry.register_from_dict(function_dict)
        
        # Get the function
        retrieved_func = self.registry.get("dictFunction")
        
        # Verify it's the correct function
        self.assertEqual(retrieved_func.name, "dictFunction")
        self.assertEqual(retrieved_func.description, "Dictionary function")
    
    def test_register_multiple(self):
        """Test registering multiple functions."""
        # Create function dictionaries
        function_dicts = [
            {
                "name": "func1",
                "description": "Function 1",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "func2",
                "description": "Function 2",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        # Register multiple functions
        self.registry.register_multiple(function_dicts)
        
        # Get all functions
        all_funcs = self.registry.get_all()
        
        # Verify we have the correct number of functions
        self.assertEqual(len(all_funcs), 2)
        
        # Verify function names
        func_names = [func.name for func in all_funcs]
        self.assertIn("func1", func_names)
        self.assertIn("func2", func_names)
    
    def test_remove_and_clear(self):
        """Test removing a function and clearing the registry."""
        # Register functions
        self.registry.register(self.function_def)
        self.registry.register_from_dict({
            "name": "func2",
            "description": "Function 2",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        })
        
        # Verify we have 2 functions
        self.assertEqual(len(self.registry.get_all()), 2)
        
        # Remove one function
        self.registry.remove("testFunction")
        
        # Verify we have 1 function left
        self.assertEqual(len(self.registry.get_all()), 1)
        self.assertIsNone(self.registry.get("testFunction"))
        self.assertIsNotNone(self.registry.get("func2"))
        
        # Clear the registry
        self.registry.clear()
        
        # Verify we have 0 functions
        self.assertEqual(len(self.registry.get_all()), 0)


class TestFunctionCallResult(unittest.TestCase):
    """Test cases for the Function Call Result."""
    
    def test_success_result(self):
        """Test a successful function call result."""
        # Create a successful result
        result = FunctionCallResult(
            function_name="testFunction",
            parameters={"param1": "test"},
            result="test result"
        )
        
        # Verify it's a success
        self.assertTrue(result.is_success())
        self.assertIsNone(result.error)
        self.assertEqual(result.result, "test result")
    
    def test_error_result(self):
        """Test an error function call result."""
        # Create an error result
        result = FunctionCallResult(
            function_name="testFunction",
            parameters={"param1": "test"},
            result=None,
            error="Test error"
        )
        
        # Verify it's not a success
        self.assertFalse(result.is_success())
        self.assertEqual(result.error, "Test error")


if __name__ == "__main__":
    unittest.main()
