"""
Image API support for the API Converter module.

This module extends the API Converter to support image-based APIs.
"""

import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .api_converter import APIConverter


class ImageAPIConverter:
    """
    Converter for image-based API specifications to Gemma function definitions.
    """
    
    def convert_image_api(self, api_spec_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Convert an image-based API specification to Gemma function definitions.
        
        Args:
            api_spec_path: Path to the image API specification file
            
        Returns:
            List of Gemma function definitions with image support
        """
        # Load the API specification
        with open(api_spec_path, 'r') as f:
            api_spec = json.load(f)
        
        gemma_functions = []
        
        for endpoint in api_spec.get("endpoints", []):
            function_name = endpoint.get("name", "")
            description = endpoint.get("description", "")
            
            parameters = {}
            required = []
            
            for param in endpoint.get("parameters", []):
                param_name = param["name"]
                param_type = param.get("type", "string")
                
                # Handle image parameters
                if param_type == "image":
                    param_def = {
                        "type": "string",
                        "format": "binary",
                        "description": param.get("description", "Image file or URL")
                    }
                else:
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
            
            # Add image processing metadata if needed
            if endpoint.get("image_processing", False):
                parameters["image_processing_options"] = {
                    "type": "object",
                    "description": "Options for image processing",
                    "properties": {
                        "resize": {
                            "type": "boolean",
                            "description": "Whether to resize the image"
                        },
                        "max_size": {
                            "type": "integer",
                            "description": "Maximum size for the image (pixels)"
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format for the image",
                            "enum": ["jpeg", "png", "webp"]
                        }
                    }
                }
            
            gemma_function = {
                "name": function_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                },
                "supports_images": True
            }
            
            gemma_functions.append(gemma_function)
        
        return gemma_functions


# Extend the main APIConverter class with image API support
def extend_api_converter():
    """
    Extend the APIConverter class with image API support.
    """
    original_init = APIConverter.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.image_converter = ImageAPIConverter()
    
    def convert_from_image_api(self, api_spec_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Convert an image-based API specification to Gemma function definitions.
        
        Args:
            api_spec_path: Path to the image API specification file
            
        Returns:
            List of Gemma function definitions with image support
        """
        return self.image_converter.convert_image_api(api_spec_path)
    
    # Add the new method to the APIConverter class
    APIConverter.__init__ = new_init
    APIConverter.convert_from_image_api = convert_from_image_api


# Extend the APIConverter class when this module is imported
extend_api_converter()
