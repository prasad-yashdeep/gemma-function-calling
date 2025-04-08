"""
Example implementation of a weather API with the Gemma SDK.

This example demonstrates how to use the Gemma SDK to create a weather information system.
"""

import os
import json
from pathlib import Path
import requests
from typing import Dict, Any, Optional

from gemma_sdk import GemmaSDK, APIConverter
from gemma_sdk.models import load_gemma_model
from gemma_sdk.runtime import ReActExecutor


# Define the weather API functions
WEATHER_API_SPEC = {
    "endpoints": [
        {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
            "parameters": [
                {
                    "name": "location",
                    "type": "string",
                    "description": "City name or location",
                    "required": True
                },
                {
                    "name": "units",
                    "type": "string",
                    "description": "Temperature units (celsius or fahrenheit)",
                    "enum": ["celsius", "fahrenheit"],
                    "required": False
                }
            ]
        },
        {
            "name": "get_weather_forecast",
            "description": "Get the weather forecast for a location",
            "parameters": [
                {
                    "name": "location",
                    "type": "string",
                    "description": "City name or location",
                    "required": True
                },
                {
                    "name": "days",
                    "type": "integer",
                    "description": "Number of days for the forecast (1-7)",
                    "required": False
                },
                {
                    "name": "units",
                    "type": "string",
                    "description": "Temperature units (celsius or fahrenheit)",
                    "enum": ["celsius", "fahrenheit"],
                    "required": False
                }
            ]
        }
    ]
}


# Function implementations
def get_current_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather for a location.
    
    Args:
        location: City name or location
        units: Temperature units (celsius or fahrenheit)
        
    Returns:
        Current weather information
    """
    # In a real implementation, this would call a weather API
    # For this example, we'll return mock data
    
    if units == "celsius":
        temp = 22
        temp_unit = "°C"
    else:
        temp = 72
        temp_unit = "°F"
    
    return {
        "location": location,
        "temperature": f"{temp}{temp_unit}",
        "condition": "Sunny",
        "humidity": "45%",
        "wind": "10 km/h"
    }


def get_weather_forecast(location: str, days: int = 3, units: str = "celsius") -> Dict[str, Any]:
    """
    Get the weather forecast for a location.
    
    Args:
        location: City name or location
        days: Number of days for the forecast (1-7)
        units: Temperature units (celsius or fahrenheit)
        
    Returns:
        Weather forecast information
    """
    # In a real implementation, this would call a weather API
    # For this example, we'll return mock data
    
    days = min(max(1, days), 7)  # Ensure days is between 1 and 7
    
    if units == "celsius":
        temps = [22, 20, 18, 19, 21, 23, 22]
        temp_unit = "°C"
    else:
        temps = [72, 68, 64, 66, 70, 74, 72]
        temp_unit = "°F"
    
    conditions = ["Sunny", "Partly Cloudy", "Rain", "Cloudy", "Sunny", "Sunny", "Partly Cloudy"]
    
    forecast = []
    for i in range(days):
        forecast.append({
            "day": f"Day {i+1}",
            "temperature": f"{temps[i]}{temp_unit}",
            "condition": conditions[i]
        })
    
    return {
        "location": location,
        "forecast": forecast
    }


def main():
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Please set the HF_TOKEN environment variable")
        return
    
    # Save the weather API spec to a file
    api_spec_path = Path("weather_api_spec.json")
    with open(api_spec_path, "w") as f:
        json.dump(WEATHER_API_SPEC, f, indent=2)
    
    # Convert the API spec to Gemma function definitions
    converter = APIConverter()
    functions = converter.convert_from_rest(api_spec_path)
    
    # Create function implementations
    implementations = {
        "get_current_weather": get_current_weather,
        "get_weather_forecast": get_weather_forecast
    }
    
    # Create a ReAct executor
    executor = ReActExecutor(
        model_name="google/gemma-7b",
        hf_token=hf_token,
        functions=functions
    )
    
    # Register function implementations
    for func_def in functions:
        func_name = func_def["name"]
        if func_name in implementations:
            executor.runtime.register_function(func_def, implementations[func_name])
    
    # Execute a query
    print("Executing query: What's the weather like in New York?")
    result = executor.execute("What's the weather like in New York?")
    print("\nResult:")
    print(result)
    
    print("\nExecuting query: What will the weather be like in London for the next 5 days?")
    result = executor.execute("What will the weather be like in London for the next 5 days?")
    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()
