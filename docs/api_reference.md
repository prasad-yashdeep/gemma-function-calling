# API Reference

This document provides a detailed API reference for the Gemma SDK.

## Table of Contents

1. [API Converter Module](#api-converter-module)
2. [SDK Definition Module](#sdk-definition-module)
3. [Runtime Module](#runtime-module)
4. [Models Module](#models-module)
5. [Utils Module](#utils-module)

## API Converter Module

### APIConverter

The main class for converting API schemas to Gemma function definitions.

#### Methods

- `convert_from_openapi(openapi_path)`: Convert an OpenAPI specification to Gemma function definitions
- `convert_from_rest(rest_docs_path)`: Convert REST API documentation to Gemma function definitions
- `convert_from_image_api(api_spec_path)`: Convert an image-based API specification to Gemma function definitions
- `save_functions(functions, output_path)`: Save Gemma function definitions to a JSON file
- `load_functions(input_path)`: Load Gemma function definitions from a JSON file

### OpenAPIConverter

Specialized converter for OpenAPI specifications.

#### Methods

- `convert(openapi_path)`: Convert an OpenAPI specification to Gemma function definitions
- `_load_openapi_spec(openapi_path)`: Load an OpenAPI specification from a file
- `_clean_description(description)`: Clean and format description text
- `_convert_parameter(param)`: Convert an OpenAPI parameter to a Gemma function parameter
- `_convert_schema_to_parameters(schema)`: Convert an OpenAPI schema to Gemma function parameters
- `_convert_openapi_to_gemma(openapi_spec)`: Convert an OpenAPI specification to Gemma function definitions

### RESTConverter

Specialized converter for REST API documentation.

#### Methods

- `convert(rest_docs_path)`: Convert REST API documentation to Gemma function definitions

### ImageAPIConverter

Specialized converter for image-based API specifications.

#### Methods

- `convert_image_api(api_spec_path)`: Convert an image-based API specification to Gemma function definitions

### Utility Functions

- `convert_rest_endpoint_to_gemma(endpoint_name, endpoint_description, parameters, required_params)`: Convert a REST API endpoint to a Gemma function definition

## SDK Definition Module

### Parameter

Definition of a parameter for a Gemma function.

#### Attributes

- `type`: The type of the parameter (string, number, integer, boolean, array, object)
- `description`: Description of the parameter
- `enum`: List of allowed values for the parameter
- `format`: Format of the parameter (e.g., 'binary' for images)
- `items`: Schema for array items if type is 'array'
- `properties`: Properties if type is 'object'
- `required`: List of required properties if type is 'object'

### FunctionDefinition

Definition of a function that can be called by Gemma models.

#### Attributes

- `name`: Name of the function
- `description`: Description of what the function does
- `parameters`: Parameters for the function
- `supports_images`: Whether the function supports image inputs
- `implementation`: Actual implementation of the function

#### Methods

- `to_dict()`: Convert the function definition to a dictionary format suitable for Gemma
- `to_json()`: Convert the function definition to a JSON string
- `__call__(*args, **kwargs)`: Call the function implementation with the provided arguments

### FunctionRegistry

Registry for managing function definitions.

#### Methods

- `register(function_def)`: Register a function definition
- `register_from_dict(function_dict, implementation)`: Register a function definition from a dictionary
- `register_from_json(json_str, implementation)`: Register a function definition from a JSON string
- `register_from_file(file_path, implementation)`: Register a function definition from a JSON file
- `register_multiple(function_defs, implementations)`: Register multiple function definitions
- `get(name)`: Get a function definition by name
- `get_all()`: Get all registered function definitions
- `get_all_dicts()`: Get all registered function definitions as dictionaries
- `remove(name)`: Remove a function definition by name
- `clear()`: Clear all registered function definitions

### FunctionCallResult

Result of a function call.

#### Attributes

- `function_name`: Name of the called function
- `parameters`: Parameters passed to the function
- `result`: Result of the function call
- `error`: Error message if the function call failed

#### Methods

- `is_success()`: Check if the function call was successful

### GemmaSDK

Main SDK class for Gemma function calling.

#### Methods

- `register_function(function_def, implementation)`: Register a function definition
- `register_functions(function_defs, implementations)`: Register multiple function definitions
- `get_function_definitions()`: Get all registered function definitions as dictionaries
- `create_function_calling_prompt(user_query, output_format)`: Create a function calling prompt for Gemma
- `parse_function_call(response, output_format)`: Parse a function call from a model response
- `execute_function(function_call)`: Execute a function call
- `execute(user_query, output_format)`: Execute a user query with function calling

## Runtime Module

### GemmaRuntime

Runtime environment for executing function calls with Gemma models.

#### Methods

- `register_function(function_def, implementation)`: Register a function definition
- `register_functions(function_defs, implementations)`: Register multiple function definitions
- `execute(user_query, output_format)`: Execute a user query with function calling
- `get_execution_history()`: Get the execution history
- `clear_execution_history()`: Clear the execution history
- `save_execution_history(file_path)`: Save the execution history to a file
- `load_execution_history(file_path)`: Load execution history from a file

### ReActExecutor

ReAct (Reasoning and Acting) execution framework for Gemma SDK.

#### Methods

- `create_react_prompt(user_query)`: Create a ReAct prompt for Gemma
- `parse_react_response(response)`: Parse a ReAct response from the model
- `execute_action(action, action_input)`: Execute a tool action
- `execute(user_query, max_turns)`: Execute a user query using the ReAct framework
- `clear_conversation()`: Clear the conversation history
- `get_conversation_history()`: Get the conversation history

### Message

A message in a conversation.

#### Attributes

- `role`: Role of the message sender ("user", "assistant", or "function")
- `content`: Content of the message
- `function_call`: Optional function call made by the assistant
- `function_result`: Optional result of a function call
- `timestamp`: Timestamp of the message

### Conversation

A conversation with a Gemma model.

#### Methods

- `add_user_message(content)`: Add a user message to the conversation
- `add_assistant_message(content, function_call)`: Add an assistant message to the conversation
- `add_function_result(function_result)`: Add a function result to the conversation
- `get_messages()`: Get all messages in the conversation
- `get_formatted_messages()`: Get messages formatted for model input
- `clear()`: Clear all messages in the conversation
- `to_dict()`: Convert the conversation to a dictionary
- `to_json()`: Convert the conversation to a JSON string
- `from_dict(data)`: Create a conversation from a dictionary
- `from_json(json_str)`: Create a conversation from a JSON string

### ConversationManager

Manager for multiple conversations with Gemma models.

#### Methods

- `create_conversation(conversation_id, system_prompt)`: Create a new conversation
- `get_conversation(conversation_id)`: Get a conversation by ID
- `delete_conversation(conversation_id)`: Delete a conversation
- `get_all_conversation_ids()`: Get all conversation IDs
- `save_conversations(file_path)`: Save all conversations to a file
- `load_conversations(file_path)`: Load conversations from a file

### ImageProcessor

Processor for handling images in function calls.

#### Methods

- `load_image(image_path_or_url)`: Load an image from a file path or URL
- `resize_image(image, max_size)`: Resize an image while maintaining aspect ratio
- `convert_image_format(image, format)`: Convert an image to a specific format
- `image_to_base64(image, format)`: Convert an image to a base64-encoded string
- `base64_to_image(base64_str)`: Convert a base64-encoded string to an image
- `process_image_parameter(param_value, options)`: Process an image parameter for a function call

### ImageFunctionHandler

Handler for image-based function calls.

#### Methods

- `preprocess_function_call(function_call, function_def)`: Preprocess a function call with image parameters
- `postprocess_function_result(result, function_def)`: Postprocess a function result with image outputs

## Models Module

### Functions

- `load_gemma_model(model_name, hf_token)`: Load a Gemma model and tokenizer

## Utils Module

### FunctionCallParser

Utility for parsing function calls from model responses.

#### Methods

- `parse_json_function_call(response)`: Parse a JSON function call from a model response
- `parse_python_function_call(response)`: Parse a Python-style function call from a model response
- `parse_function_call(response, output_format)`: Parse a function call from a model response using the specified format
