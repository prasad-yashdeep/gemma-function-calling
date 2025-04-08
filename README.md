# Gemma SDK Documentation

## Overview

Gemma SDK is a Python SDK that allows Gemma LLM models (4B/7B/12B) to seamlessly perform multi-turn, multi-API function calls, including support for image-based APIs. This SDK makes onboarding new APIs intuitive and fast by directly converting existing API definitions into Gemma function-callable formats.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [API Converter Module](#api-converter-module)
5. [SDK Definition Module](#sdk-definition-module)
6. [Runtime Module](#runtime-module)
7. [Image API Support](#image-api-support)
8. [Examples](#examples)
9. [Advanced Usage](#advanced-usage)
10. [API Reference](#api-reference)



## Quick Start

```python
import os
from gemma_sdk import GemmaSDK, APIConverter
from gemma_sdk.runtime import ReActExecutor

# Get Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Convert an API specification to Gemma function definitions
converter = APIConverter()
functions = converter.convert_from_openapi("path/to/openapi.json")

# Create a ReAct executor with the Gemma model
executor = ReActExecutor(
    model_name="google/gemma-7b",
    hf_token=hf_token,
    functions=functions
)

# Register function implementations
def get_weather(location, units="celsius"):
    # Implementation
    return {"temperature": "22°C", "condition": "Sunny"}

executor.runtime.register_function(functions[0], get_weather)

# Execute a query
result = executor.execute("What's the weather like in New York?")
print(result)
```

## Architecture

The Gemma SDK follows a modular architecture with three main components:

1. **API Converter**: Converts existing API schemas (OpenAPI, REST) to Gemma function definitions
2. **SDK Definition**: Core SDK function definitions and interfaces
3. **Runtime**: Gemma SDK runtime with ReAct execution framework

The workflow is as follows:

```
Existing API Schema (OpenAPI, REST Docs)
        │
        ▼
api_converter.py (Gemma JSON definition generator)
        │
        ▼
SDK Function Definition (Python module)
        │
        ▼
Gemma SDK runtime (multi-turn, ReAct execution)
```

## API Converter Module

The API Converter module provides tools for converting existing API schemas to Gemma function definitions.

### Converting OpenAPI Specifications

```python
from gemma_sdk import APIConverter

converter = APIConverter()
functions = converter.convert_from_openapi("path/to/openapi.json")
```

### Converting REST API Documentation

```python
from gemma_sdk import APIConverter

converter = APIConverter()
functions = converter.convert_from_rest("path/to/rest_docs.json")
```

### Converting Image-based APIs

```python
from gemma_sdk import APIConverter

converter = APIConverter()
functions = converter.convert_from_image_api("path/to/image_api_spec.json")
```

### Manual Function Definition

```python
from gemma_sdk import APIConverter

# Manually define a function
function_def = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location"
            },
            "units": {
                "type": "string",
                "description": "Temperature units (celsius or fahrenheit)",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
}

# Save to a file
converter = APIConverter()
converter.save_functions([function_def], "weather_functions.json")
```

## SDK Definition Module

The SDK Definition module provides the core classes and interfaces for defining functions that can be called by Gemma models.

### Function Definition

```python
from gemma_sdk.sdk_definition import FunctionDefinition

# Create a function definition
function_def = FunctionDefinition(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location"
            },
            "units": {
                "type": "string",
                "description": "Temperature units (celsius or fahrenheit)",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
)
```

### Function Registry

```python
from gemma_sdk.sdk_definition import FunctionRegistry

# Create a function registry
registry = FunctionRegistry()

# Register a function
registry.register(function_def)

# Register a function from a dictionary
registry.register_from_dict({
    "name": "get_forecast",
    "description": "Get the weather forecast for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location"
            },
            "days": {
                "type": "integer",
                "description": "Number of days for the forecast"
            }
        },
        "required": ["location"]
    }
})

# Get all function definitions
functions = registry.get_all_dicts()
```

### Main SDK Class

```python
from gemma_sdk import GemmaSDK
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a Gemma model
model_name = "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# Create the SDK
sdk = GemmaSDK(model, tokenizer, functions)

# Execute a query
result, response = sdk.execute("What's the weather like in New York?")
```

## Runtime Module

The Runtime module provides the execution environment for function calls with Gemma models.

### Gemma Runtime

```python
from gemma_sdk.runtime import GemmaRuntime

# Create a runtime
runtime = GemmaRuntime(
    model_name="google/gemma-7b",
    hf_token=hf_token,
    functions=functions
)

# Register function implementations
def get_weather(location, units="celsius"):
    # Implementation
    return {"temperature": "22°C", "condition": "Sunny"}

runtime.register_function(functions[0], get_weather)

# Execute a query
result, response = runtime.execute("What's the weather like in New York?")
```

### ReAct Executor

```python
from gemma_sdk.runtime import ReActExecutor

# Create a ReAct executor
executor = ReActExecutor(
    model_name="google/gemma-7b",
    hf_token=hf_token,
    functions=functions
)

# Register function implementations
executor.runtime.register_function(functions[0], get_weather)

# Execute a query with ReAct framework
result = executor.execute("What's the weather like in New York?")
```

### Conversation Support

```python
from gemma_sdk.runtime import Conversation, ConversationManager

# Create a conversation
conversation = Conversation(system_prompt="You are a helpful assistant.")

# Add messages
conversation.add_user_message("What's the weather like in New York?")
conversation.add_assistant_message("I'll check the weather for you.")

# Get formatted messages for model input
messages = conversation.get_formatted_messages()

# Create a conversation manager for multiple conversations
manager = ConversationManager()
manager.create_conversation("user1", "You are a helpful assistant.")
manager.get_conversation("user1").add_user_message("Hello!")
```

## Image API Support

The SDK provides support for image-based APIs through the ImageProcessor and ImageFunctionHandler classes.

### Image Processing

```python
from gemma_sdk.runtime import ImageProcessor

# Create an image processor
processor = ImageProcessor()

# Load an image
image = processor.load_image("path/to/image.jpg")

# Resize an image
resized_image = processor.resize_image(image, max_size=512)

# Convert to base64
base64_str = processor.image_to_base64(resized_image)

# Process an image parameter
processed_image = processor.process_image_parameter(
    "path/to/image.jpg",
    options={"resize": True, "max_size": 512, "format": "JPEG"}
)
```

### Image Function Handler

```python
from gemma_sdk.runtime import ImageFunctionHandler

# Create an image function handler
handler = ImageFunctionHandler()

# Preprocess a function call with image parameters
function_call = {
    "name": "analyze_image",
    "parameters": {
        "image": "path/to/image.jpg",
        "analysis_type": "objects"
    }
}

function_def = {
    "name": "analyze_image",
    "description": "Analyze an image",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "format": "binary",
                "description": "Image file or URL"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis"
            }
        },
        "required": ["image"]
    },
    "supports_images": True
}

processed_call = handler.preprocess_function_call(function_call, function_def)
```

## Examples

The SDK includes several example implementations to demonstrate its capabilities.

### Weather API Example

```python
# See examples/weather_api_example.py
```

### E-commerce API Example

```python
# See examples/ecommerce_api_example.py
```

### Image API Example

```python
# See examples/image_api_example.py
```

## Advanced Usage

### Custom Model Configuration

```python
from gemma_sdk.runtime import GemmaRuntime
import torch

# Create a runtime with custom model configuration
runtime = GemmaRuntime(
    model_name="google/gemma-7b",
    hf_token=hf_token,
    device="cuda:0",
    functions=functions
)

# Use half-precision for better performance on GPU
runtime.model = runtime.model.half()
```

### Custom Function Parsing

```python
from gemma_sdk import GemmaSDK

# Create the SDK
sdk = GemmaSDK(model, tokenizer, functions)

# Custom function call parsing
response = "I'll use the get_weather function with location=New York"
function_call = sdk.parse_function_call(response, output_format="python")
```

### Error Handling

```python
from gemma_sdk.sdk_definition import FunctionCallResult

# Execute a function
function_call = {
    "name": "get_weather",
    "parameters": {
        "location": "New York"
    }
}

result = sdk.execute_function(function_call)

# Check for errors
if result.error:
    print(f"Error: {result.error}")
else:
    print(f"Result: {result.result}")
```

## API Reference

For detailed API reference, please see the [API Reference]() document.
