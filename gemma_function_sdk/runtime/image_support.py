"""
Image API support for the Gemma SDK runtime.

This module provides support for image-based APIs in the Gemma SDK runtime.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import base64
import io
import json
import requests
from pathlib import Path
from PIL import Image
import numpy as np

from ..sdk_definition import FunctionCallResult


class ImageProcessor:
    """
    Processor for handling images in function calls.
    
    This class provides utilities for processing images in function calls.
    """
    
    @staticmethod
    def load_image(image_path_or_url: str) -> Image.Image:
        """
        Load an image from a file path or URL.
        
        Args:
            image_path_or_url: Path to an image file or URL
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If the image cannot be loaded
        """
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_path_or_url, stream=True)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            else:
                # Load from file
                return Image.open(image_path_or_url)
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path_or_url}: {str(e)}")
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            max_size: Maximum size (width or height) for the resized image
            
        Returns:
            Resized PIL Image object
        """
        width, height = image.size
        
        if width <= max_size and height <= max_size:
            return image
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    @staticmethod
    def convert_image_format(image: Image.Image, format: str = "JPEG") -> Image.Image:
        """
        Convert an image to a specific format.
        
        Args:
            image: PIL Image object
            format: Target format (JPEG, PNG, WEBP)
            
        Returns:
            Converted PIL Image object
        """
        if image.format == format:
            return image
        
        # Convert to RGB if needed (for JPEG)
        if format == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
        
        # Create a new image in the target format
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        return Image.open(buffer)
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert an image to a base64-encoded string.
        
        Args:
            image: PIL Image object
            format: Image format (JPEG, PNG, WEBP)
            
        Returns:
            Base64-encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Image.Image:
        """
        Convert a base64-encoded string to an image.
        
        Args:
            base64_str: Base64-encoded string
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If the string cannot be decoded
        """
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 string: {str(e)}")
    
    @staticmethod
    def process_image_parameter(param_value: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Process an image parameter for a function call.
        
        Args:
            param_value: Image path, URL, or base64-encoded string
            options: Optional processing options
                - resize: Whether to resize the image (default: False)
                - max_size: Maximum size for the image (default: 1024)
                - format: Output format for the image (default: "JPEG")
                
        Returns:
            Base64-encoded string of the processed image
        """
        options = options or {}
        resize = options.get("resize", False)
        max_size = options.get("max_size", 1024)
        format = options.get("format", "JPEG")
        
        # Check if the parameter is already a base64-encoded string
        if len(param_value) > 100 and ',' in param_value and ';base64,' in param_value:
            # Extract the base64 part
            base64_str = param_value.split(',', 1)[1]
            image = ImageProcessor.base64_to_image(base64_str)
        else:
            # Load from path or URL
            image = ImageProcessor.load_image(param_value)
        
        # Process the image
        if resize:
            image = ImageProcessor.resize_image(image, max_size)
        
        image = ImageProcessor.convert_image_format(image, format)
        
        # Convert to base64
        return ImageProcessor.image_to_base64(image, format)


class ImageFunctionHandler:
    """
    Handler for image-based function calls.
    
    This class provides utilities for handling image-based function calls.
    """
    
    def __init__(self):
        """
        Initialize an image function handler.
        """
        self.image_processor = ImageProcessor()
    
    def preprocess_function_call(self, function_call: Dict[str, Any], function_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a function call with image parameters.
        
        Args:
            function_call: Function call to preprocess
            function_def: Function definition
            
        Returns:
            Preprocessed function call
        """
        if not function_def.get("supports_images", False):
            return function_call
        
        parameters = function_call.get("parameters", {})
        processed_parameters = {}
        
        for param_name, param_value in parameters.items():
            param_def = function_def.get("parameters", {}).get("properties", {}).get(param_name, {})
            
            # Check if this is an image parameter
            if param_def.get("format") == "binary" or param_def.get("type") == "image":
                # Process the image
                options = parameters.get("image_processing_options", {})
                processed_parameters[param_name] = self.image_processor.process_image_parameter(param_value, options)
            else:
                # Pass through non-image parameters
                processed_parameters[param_name] = param_value
        
        # Create a new function call with processed parameters
        return {
            "name": function_call["name"],
            "parameters": processed_parameters
        }
    
    def postprocess_function_result(self, result: FunctionCallResult, function_def: Dict[str, Any]) -> FunctionCallResult:
        """
        Postprocess a function result with image outputs.
        
        Args:
            result: Function call result to postprocess
            function_def: Function definition
            
        Returns:
            Postprocessed function call result
        """
        # Check if the function has image outputs
        if not function_def.get("has_image_output", False):
            return result
        
        # Process image outputs if needed
        # This is a placeholder for custom image output processing
        
        return result
