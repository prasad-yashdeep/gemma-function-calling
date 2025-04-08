"""
API Converter module for converting OpenAPI specifications to Gemma function definitions.
"""

import json
import yaml
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class APIConverter:
    """
    Base class for converting API schemas to Gemma function definitions.
    """
    
    def convert_from_openapi(self, openapi_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Convert an OpenAPI specification to Gemma function definitions.
        
        Args:
            openapi_path: Path to the OpenAPI specification file (JSON or YAML)
            
        Returns:
            List of Gemma function definitions
        """
        converter = OpenAPIConverter()
        return converter.convert(openapi_path)
    
    def convert_from_rest(self, rest_docs_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Convert REST API documentation to Gemma function definitions.
        
        Args:
            rest_docs_path: Path to the REST API documentation file
            
        Returns:
            List of Gemma function definitions
        """
        converter = RESTConverter()
        return converter.convert(rest_docs_path)
    
    def save_functions(self, functions: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """
        Save Gemma function definitions to a JSON file.
        
        Args:
            functions: List of Gemma function definitions
            output_path: Path to save the function definitions
        """
        with open(output_path, 'w') as f:
            json.dump(functions, f, indent=2)
    
    def load_functions(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load Gemma function definitions from a JSON file.
        
        Args:
            input_path: Path to the function definitions file
            
        Returns:
            List of Gemma function definitions
        """
        with open(input_path, 'r') as f:
            return json.load(f)


class OpenAPIConverter:
    """
    Converter for OpenAPI specifications to Gemma function definitions.
    """
    
    def __init__(self):
        self.type_mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object"
        }
    
    def convert(self, openapi_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Convert an OpenAPI specification to Gemma function definitions.
        
        Args:
            openapi_path: Path to the OpenAPI specification file (JSON or YAML)
            
        Returns:
            List of Gemma function definitions
        """
        # Load OpenAPI specification
        openapi_spec = self._load_openapi_spec(openapi_path)
        
        # Convert to Gemma function definitions
        return self._convert_openapi_to_gemma(openapi_spec)
    
    def _load_openapi_spec(self, openapi_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an OpenAPI specification from a file.
        
        Args:
            openapi_path: Path to the OpenAPI specification file (JSON or YAML)
            
        Returns:
            OpenAPI specification as a dictionary
        """
        with open(openapi_path, 'r') as f:
            content = f.read()
            
        if str(openapi_path).endswith('.json'):
            return json.loads(content)
        else:  # Assume YAML
            return yaml.safe_load(content)
    
    def _clean_description(self, description: Optional[str]) -> str:
        """
        Clean and format description text.
        
        Args:
            description: Description text to clean
            
        Returns:
            Cleaned description text
        """
        if not description:
            return ""
        
        # Remove HTML tags
        description = re.sub(r'<[^>]+>', '', description)
        
        # Remove excessive whitespace
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description
    
    def _convert_parameter(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an OpenAPI parameter to a Gemma function parameter.
        
        Args:
            param: OpenAPI parameter definition
            
        Returns:
            Gemma function parameter definition
        """
        param_type = param.get("type", "string")
        if param_type not in self.type_mapping:
            param_type = "string"  # Default to string for unknown types
        
        result = {
            "type": self.type_mapping[param_type]
        }
        
        # Add description if available
        if "description" in param:
            result["description"] = self._clean_description(param["description"])
        
        # Handle enums
        if "enum" in param:
            result["enum"] = param["enum"]
        
        # Handle array items
        if param_type == "array" and "items" in param:
            items = param["items"]
            items_type = items.get("type", "string")
            if items_type in self.type_mapping:
                result["items"] = {"type": self.type_mapping[items_type]}
        
        return result
    
    def _convert_schema_to_parameters(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an OpenAPI schema to Gemma function parameters.
        
        Args:
            schema: OpenAPI schema definition
            
        Returns:
            Gemma function parameters definition
        """
        properties = {}
        required = []
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                prop_type = prop_schema.get("type", "string")
                if prop_type not in self.type_mapping:
                    prop_type = "string"  # Default to string for unknown types
                
                prop_def = {
                    "type": self.type_mapping[prop_type]
                }
                
                # Add description if available
                if "description" in prop_schema:
                    prop_def["description"] = self._clean_description(prop_schema["description"])
                
                # Handle enums
                if "enum" in prop_schema:
                    prop_def["enum"] = prop_schema["enum"]
                
                # Handle array items
                if prop_type == "array" and "items" in prop_schema:
                    items = prop_schema["items"]
                    items_type = items.get("type", "string")
                    if items_type in self.type_mapping:
                        prop_def["items"] = {"type": self.type_mapping[items_type]}
                
                properties[prop_name] = prop_def
        
        # Get required properties
        if "required" in schema:
            required = schema["required"]
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _convert_openapi_to_gemma(self, openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert an OpenAPI specification to Gemma function definitions.
        
        Args:
            openapi_spec: OpenAPI specification as a dictionary
            
        Returns:
            List of Gemma function definitions
        """
        gemma_functions = []
        
        # Process paths
        paths = openapi_spec.get("paths", {})
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method not in ["get", "post", "put", "delete", "patch"]:
                    continue
                
                # Create function name from operationId or path
                function_name = operation.get("operationId", "")
                if not function_name:
                    # Generate name from path and method
                    path_parts = [p for p in path.split("/") if p and not p.startswith("{")]
                    function_name = f"{method}_{'_'.join(path_parts)}"
                    function_name = function_name.replace("-", "_").lower()
                
                # Create function description
                description = self._clean_description(operation.get("summary", ""))
                if not description and "description" in operation:
                    description = self._clean_description(operation["description"])
                if not description:
                    description = f"{method.upper()} {path}"
                
                # Process parameters
                parameters = {}
                required = []
                
                # Path and query parameters
                for param in operation.get("parameters", []):
                    param_name = param["name"]
                    parameters[param_name] = self._convert_parameter(param)
                    
                    if param.get("required", False):
                        required.append(param_name)
                
                # Request body
                if "requestBody" in operation:
                    content = operation["requestBody"].get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        body_params = self._convert_schema_to_parameters(schema)
                        
                        # Merge body parameters with other parameters
                        parameters.update(body_params["properties"])
                        required.extend(body_params["required"])
                
                # Create Gemma function definition
                gemma_function = {
                    "name": function_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required
                    }
                }
                
                gemma_functions.append(gemma_function)
        
        return gemma_functions


class RESTConverter:
    """
    Converter for REST API documentation to Gemma function definitions.
    """
    
    def convert(self, rest_docs_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Convert REST API documentation to Gemma function definitions.
        
        Args:
            rest_docs_path: Path to the REST API documentation file
            
        Returns:
            List of Gemma function definitions
        """
        # This is a simplified implementation
        # A real implementation would need to parse the REST API documentation format
        
        # For now, we'll assume a simple JSON format with endpoint definitions
        with open(rest_docs_path, 'r') as f:
            rest_docs = json.load(f)
        
        gemma_functions = []
        
        for endpoint in rest_docs.get("endpoints", []):
            function_name = endpoint.get("name", "")
            description = endpoint.get("description", "")
            
            parameters = {}
            required = []
            
            for param in endpoint.get("parameters", []):
                param_name = param["name"]
                param_type = param.get("type", "string")
                
                param_def = {
                    "type": param_type
                }
                
                if "description" in param:
                    param_def["description"] = param["description"]
                
                if "enum" in param:
                    param_def["enum"] = param["enum"]
                
                parameters[param_name] = param_def
                
                if param.get("required", False):
                    required.append(param_name)
            
            gemma_function = {
                "name": function_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
            
            gemma_functions.append(gemma_function)
        
        return gemma_functions


def convert_rest_endpoint_to_gemma(
    endpoint_name: str,
    endpoint_description: str,
    parameters: Dict[str, Dict[str, Any]],
    required_params: List[str] = None
) -> Dict[str, Any]:
    """
    Convert a REST API endpoint to a Gemma function definition.
    
    Args:
        endpoint_name: Name of the endpoint/function
        endpoint_description: Description of what the endpoint does
        parameters: Dictionary of parameters with their types and descriptions
        required_params: List of required parameter names
        
    Returns:
        Gemma function definition
    """
    properties = {}
    
    for param_name, param_info in parameters.items():
        param_type = param_info.get("type", "string")
        param_def = {
            "type": param_type
        }
        
        if "description" in param_info:
            param_def["description"] = param_info["description"]
        
        if "enum" in param_info:
            param_def["enum"] = param_info["enum"]
        
        properties[param_name] = param_def
    
    return {
        "name": endpoint_name,
        "description": endpoint_description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required_params or []
        }
    }
