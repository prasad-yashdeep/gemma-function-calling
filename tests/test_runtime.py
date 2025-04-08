import unittest
from unittest.mock import MagicMock, patch
import json
import os
from pathlib import Path

from gemma_sdk.runtime import GemmaRuntime, ReActExecutor, Conversation, ConversationManager, ImageProcessor


class TestGemmaRuntime(unittest.TestCase):
    """Test cases for the Gemma Runtime module."""
    
    @patch('gemma_sdk.runtime.gemma_runtime.AutoTokenizer')
    @patch('gemma_sdk.runtime.gemma_runtime.AutoModelForCausalLM')
    def test_runtime_initialization(self, mock_model_class, mock_tokenizer_class):
        """Test initializing the Gemma Runtime."""
        # Create mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Initialize runtime
        runtime = GemmaRuntime(
            model_name="google/gemma-7b",
            hf_token="test_token",
            functions=[]
        )
        
        # Verify tokenizer and model were loaded
        mock_tokenizer_class.from_pretrained.assert_called_once_with("google/gemma-7b", token="test_token")
        mock_model_class.from_pretrained.assert_called_once()
        
        # Verify SDK was initialized
        self.assertIsNotNone(runtime.sdk)
        self.assertEqual(runtime.model, mock_model)
        self.assertEqual(runtime.tokenizer, mock_tokenizer)
    
    @patch('gemma_sdk.runtime.gemma_runtime.AutoTokenizer')
    @patch('gemma_sdk.runtime.gemma_runtime.AutoModelForCausalLM')
    def test_register_function(self, mock_model_class, mock_tokenizer_class):
        """Test registering a function with the runtime."""
        # Create mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Initialize runtime
        runtime = GemmaRuntime(
            model_name="google/gemma-7b",
            hf_token="test_token"
        )
        
        # Create a mock for the SDK's register_function method
        runtime.sdk.register_function = MagicMock()
        
        # Register a function
        function_def = {
            "name": "testFunction",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        implementation = lambda: "test result"
        
        runtime.register_function(function_def, implementation)
        
        # Verify SDK's register_function was called
        runtime.sdk.register_function.assert_called_once_with(function_def, implementation)
    
    @patch('gemma_sdk.runtime.gemma_runtime.AutoTokenizer')
    @patch('gemma_sdk.runtime.gemma_runtime.AutoModelForCausalLM')
    def test_execution_history(self, mock_model_class, mock_tokenizer_class):
        """Test execution history functionality."""
        # Create mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Initialize runtime
        runtime = GemmaRuntime(
            model_name="google/gemma-7b",
            hf_token="test_token"
        )
        
        # Mock the SDK's execute method
        mock_result = MagicMock()
        mock_result.dict.return_value = {"function_name": "testFunction", "result": "test result"}
        runtime.sdk.execute = MagicMock(return_value=(mock_result, "test response"))
        
        # Execute a query
        runtime.execute("test query")
        
        # Verify execution was recorded in history
        self.assertEqual(len(runtime.execution_history), 1)
        self.assertEqual(runtime.execution_history[0]["query"], "test query")
        self.assertEqual(runtime.execution_history[0]["response"], "test response")
        self.assertEqual(runtime.execution_history[0]["function_call"]["function_name"], "testFunction")
        
        # Clear history
        runtime.clear_execution_history()
        
        # Verify history was cleared
        self.assertEqual(len(runtime.execution_history), 0)


class TestReActExecutor(unittest.TestCase):
    """Test cases for the ReAct Executor."""
    
    @patch('gemma_sdk.runtime.react_executor.GemmaRuntime')
    def test_react_executor_initialization(self, mock_runtime_class):
        """Test initializing the ReAct Executor."""
        # Create mock runtime
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime
        
        # Initialize executor
        executor = ReActExecutor(
            model_name="google/gemma-7b",
            hf_token="test_token",
            functions=[]
        )
        
        # Verify runtime was created
        mock_runtime_class.assert_called_once_with("google/gemma-7b", "test_token", [], "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(executor.runtime, mock_runtime)
    
    @patch('gemma_sdk.runtime.react_executor.GemmaRuntime')
    def test_create_react_prompt(self, mock_runtime_class):
        """Test creating a ReAct prompt."""
        # Create mock runtime and SDK
        mock_runtime = MagicMock()
        mock_sdk = MagicMock()
        mock_runtime.sdk = mock_sdk
        mock_runtime_class.return_value = mock_runtime
        
        # Mock function definitions
        mock_sdk.get_function_definitions.return_value = [
            {
                "name": "testFunction",
                "description": "Test function",
                "parameters": {
                    "properties": {
                        "param1": {"type": "string"}
                    }
                }
            }
        ]
        
        # Initialize executor
        executor = ReActExecutor(runtime=mock_runtime)
        
        # Create a prompt
        prompt = executor.create_react_prompt("test query")
        
        # Verify prompt contains expected elements
        self.assertIn("You are an AI assistant", prompt)
        self.assertIn("testFunction", prompt)
        self.assertIn("Test function", prompt)
        self.assertIn("param1: string", prompt)
        self.assertIn("test query", prompt)
    
    @patch('gemma_sdk.runtime.react_executor.GemmaRuntime')
    def test_parse_react_response(self, mock_runtime_class):
        """Test parsing a ReAct response."""
        # Create mock runtime
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime
        
        # Initialize executor
        executor = ReActExecutor(runtime=mock_runtime)
        
        # Test response with thought and action
        response = """
        Thought: I need to get the weather for New York.
        Action: I should use the tool `get_weather` with input `{"location": "New York"}`
        """
        
        parsed = executor.parse_react_response(response)
        
        self.assertEqual(parsed["thought"], "I need to get the weather for New York.")
        self.assertEqual(parsed["action"], "get_weather")
        self.assertEqual(parsed["action_input"], {"location": "New York"})
        self.assertIsNone(parsed["final_answer"])
        
        # Test response with final answer
        response = """
        Thought: I have the weather information.
        Final Answer: The weather in New York is sunny with a temperature of 72°F.
        """
        
        parsed = executor.parse_react_response(response)
        
        self.assertEqual(parsed["thought"], "I have the weather information.")
        self.assertIsNone(parsed["action"])
        self.assertIsNone(parsed["action_input"])
        self.assertEqual(parsed["final_answer"], "The weather in New York is sunny with a temperature of 72°F.")
    
    @patch('gemma_sdk.runtime.react_executor.GemmaRuntime')
    def test_execute_action(self, mock_runtime_class):
        """Test executing an action."""
        # Create mock runtime and SDK
        mock_runtime = MagicMock()
        mock_sdk = MagicMock()
        mock_runtime.sdk = mock_sdk
        mock_runtime_class.return_value = mock_runtime
        
        # Mock execute_function
        mock_result = MagicMock()
        mock_result.error = None
        mock_result.result = {"temperature": "72°F", "condition": "sunny"}
        mock_sdk.execute_function.return_value = mock_result
        
        # Initialize executor
        executor = ReActExecutor(runtime=mock_runtime)
        
        # Execute an action
        result = executor.execute_action("get_weather", {"location": "New York"})
        
        # Verify SDK's execute_function was called
        mock_sdk.execute_function.assert_called_once_with({
            "name": "get_weather",
            "parameters": {"location": "New York"}
        })
        
        # Verify result
        self.assertIn("temperature", result)
        self.assertIn("72°F", result)


class TestConversation(unittest.TestCase):
    """Test cases for the Conversation module."""
    
    def test_conversation_messages(self):
        """Test adding and retrieving messages in a conversation."""
        # Create a conversation
        conversation = Conversation(system_prompt="You are a helpful assistant.")
        
        # Add messages
        conversation.add_user_message("Hello!")
        conversation.add_assistant_message("Hi there! How can I help you?")
        
        # Get messages
        messages = conversation.get_messages()
        
        # Verify messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "Hello!")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].content, "Hi there! How can I help you?")
    
    def test_formatted_messages(self):
        """Test getting formatted messages for model input."""
        # Create a conversation
        conversation = Conversation(system_prompt="You are a helpful assistant.")
        
        # Add messages
        conversation.add_user_message("What's the weather like?")
        conversation.add_assistant_message("I'll check the weather for you.")
        
        # Get formatted messages
        formatted = conversation.get_formatted_messages()
        
        # Verify formatted messages
        self.assertEqual(len(formatted), 3)  # System prompt + 2 messages
        self.assertEqual(formatted[0]["role"], "system")
        self.assertEqual(formatted[0]["content"], "You are a helpful assistant.")
        self.assertEqual(formatted[1]["role"], "user")
        self.assertEqual(formatted[1]["content"], "What's the weather like?")
        self.assertEqual(formatted[2]["role"], "assistant")
        self.assertEqual(formatted[2]["content"], "I'll check the weather for you.")
    
    def test_function_results(self):
        """Test adding function results to a conversation."""
        # Create a conversation
        conversation = Conversation()
        
        # Create a function result
        function_result = {
            "function_name": "get_weather",
            "parameters": {"location": "New York"},
            "result": {"temperature": "72°F", "condition": "sunny"}
        }
        
        # Add function result
        conversation.add_function_result(function_result)
        
        # Get messages
        messages = conversation.get_messages()
        
        # Verify function result message
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "function")
        self.assertIn("get_weather", messages[0].content)
        self.assertEqual(messages[0].function_result, function_result)
    
    def test_serialization(self):
        """Test serializing and deserializing a conversation."""
        # Create a conversation
        conversation = Conversation(system_prompt="You are a helpful assistant.")
        conversation.add_user_message("Hello!")
        conversation.add_assistant_message("Hi there!")
        
        # Convert to dictionary
        conv_dict = conversation.to_dict()
        
        # Create a new conversation from the dictionary
        new_conversation = Conversation.from_dict(conv_dict)
        
        # Verify the new conversation
        self.assertEqual(new_conversation.system_prompt, "You are a helpful assistant.")
        self.assertEqual(len(new_conversation.messages), 2)
        self.assertEqual(new_conversation.messages[0].role, "user")
        self.assertEqual(new_conversation.messages[0].content, "Hello!")
        
        # Test JSON serialization
        json_str = conversation.to_json()
        json_conversation = Conversation.from_json(json_str)
        
        self.assertEqual(json_conversation.system_prompt, "You are a helpful assistant.")
        self.assertEqual(len(json_conversation.messages), 2)


class TestConversationManager(unittest.TestCase):
    """Test cases for the Conversation Manager."""
    
    def test_conversation_management(self):
        """Test creating, retrieving, and deleting conversations."""
        # Create a manager
        manager = ConversationManager()
        
        # Create conversations
        conv1 = manager.create_conversation("user1", "You are a helpful assistant.")
        conv2 = manager.create_conversation("user2", "You are a weather assistant.")
        
        # Add messages
        conv1.add_user_message("Hello!")
        conv2.add_user_message("What's the weather?")
        
        # Get conversations
        retrieved_conv1 = manager.get_conversation("user1")
        retrieved_conv2 = manager.get_conversation("user2")
        
        # Verify conversations
        self.assertEqual(retrieved_conv1.system_prompt, "You are a helpful assistant.")
        self.assertEqual(retrieved_conv1.messages[0].content, "Hello!")
        self.assertEqual(retrieved_conv2.system_prompt, "You are a weather assistant.")
        self.assertEqual(retrieved_conv2.messages[0].content, "What's the weather?")
        
        # Get all conversation IDs
        ids = manager.get_all_conversation_ids()
        self.assertEqual(len(ids), 2)
        self.assertIn("user1", ids)
        self.assertIn("user2", ids)
        
        # Delete a conversation
        manager.delete_conversation("user1")
        
        # Verify deletion
        self.assertIsNone(manager.get_conversation("user1"))
        self.assertIsNotNone(manager.get_conversation("user2"))
        self.assertEqual(len(manager.get_all_conversation_ids()), 1)


class TestImageProcessor(unittest.TestCase):
    """Test cases for the Image Processor."""
    
    def setUp(self):
        self.processor = ImageProcessor()
        self.test_dir = Path("test_images")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a test image
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_path = self.test_dir / "test_image.jpg"
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        # Clean up test files
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()
    
    @patch('gemma_sdk.runtime.image_support.requests.get')
    def test_load_image(self, mock_get):
        """Test loading an image from a file path or URL."""
        # Test loading from file path
        image = self.processor.load_image(str(self.test_image_path))
        self.assertEqual(image.size, (100, 100))
        
        # Mock response for URL
        mock_response = MagicMock()
        mock_response.content = open(self.test_image_path, 'rb').read()
        mock_get.return_value = mock_response
        
        # Test loading from URL
        image = self.processor.load_image("https://example.com/image.jpg")
        mock_get.assert_called_once_with("https://example.com/image.jpg", stream=True)
    
    def test_resize_image(self):
        """Test resizing an image."""
        from PIL import Image
        # Create a large test image
        large_image = Image.new('RGB', (1000, 500), color='blue')
        
        # Resize the image
        resized = self.processor.resize_image(large_image, max_size=200)
        
        # Verify the image was resized correctly
        self.assertEqual(resized.size, (200, 100))  # Maintains aspect ratio
    
    def test_image_to_base64(self):
        """Test converting an image to base64."""
        from PIL import Image
        # Create a test image
        test_image = Image.new('RGB', (10, 10), color='green')
        
        # Convert to base64
        base64_str = self.processor.image_to_base64(test_image)
        
        # Verify it's a valid base64 string
        self.assertTrue(len(base64_str) > 0)
        
        # Verify we can decode it back to an image
        decoded_image = self.processor.base64_to_image(base64_str)
        self.assertEqual(decoded_image.size, (10, 10))
    
    def test_process_image_parameter(self):
        """Test processing an image parameter."""
        # Process an image path
        processed = self.processor.process_image_parameter(
            str(self.test_image_path),
            options={"resize": True, "max_size": 50}
        )
        
        # Verify it's a valid base64 string
        self.assertTrue(len(processed) > 0)
        
        # Decode and verify the image was resized
        decoded = self.processor.base64_to_image(processed)
        self.assertEqual(decoded.size, (50, 50))


if __name__ == "__main__":
    unittest.main()
