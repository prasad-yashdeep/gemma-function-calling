"""
Conversation module for multi-turn interactions with Gemma models.

This module provides support for multi-turn conversations with Gemma models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
from dataclasses import dataclass, field
from datetime import datetime

from ..sdk_definition import FunctionCallResult


@dataclass
class Message:
    """
    A message in a conversation.
    """
    role: str  # "user", "assistant", or "function"
    content: str
    function_call: Optional[Dict[str, Any]] = None
    function_result: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Conversation:
    """
    A conversation with a Gemma model.
    
    This class manages multi-turn conversations with Gemma models.
    """
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize a conversation.
        
        Args:
            system_prompt: Optional system prompt to set the behavior of the model
        """
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.messages: List[Message] = []
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            content: Content of the user message
        """
        self.messages.append(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str, function_call: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: Content of the assistant message
            function_call: Optional function call made by the assistant
        """
        self.messages.append(Message(role="assistant", content=content, function_call=function_call))
    
    def add_function_result(self, function_result: FunctionCallResult) -> None:
        """
        Add a function result to the conversation.
        
        Args:
            function_result: Result of a function call
        """
        # Convert function result to a dictionary
        result_dict = function_result.dict() if hasattr(function_result, "dict") else function_result
        
        # Create a message with the function result
        content = f"Function {result_dict['function_name']} returned: {result_dict['result']}"
        if result_dict.get("error"):
            content = f"Function {result_dict['function_name']} error: {result_dict['error']}"
        
        self.messages.append(Message(
            role="function",
            content=content,
            function_result=result_dict
        ))
    
    def get_messages(self) -> List[Message]:
        """
        Get all messages in the conversation.
        
        Returns:
            List of messages
        """
        return self.messages
    
    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """
        Get messages formatted for model input.
        
        Returns:
            List of formatted messages
        """
        formatted_messages = [{"role": "system", "content": self.system_prompt}]
        
        for message in self.messages:
            formatted_message = {"role": message.role, "content": message.content}
            
            if message.function_call:
                formatted_message["function_call"] = message.function_call
            
            if message.function_result:
                formatted_message["function_result"] = message.function_result
            
            formatted_messages.append(formatted_message)
        
        return formatted_messages
    
    def clear(self) -> None:
        """
        Clear all messages in the conversation.
        """
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            "system_prompt": self.system_prompt,
            "messages": [vars(message) for message in self.messages]
        }
    
    def to_json(self) -> str:
        """
        Convert the conversation to a JSON string.
        
        Returns:
            JSON string representation of the conversation
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Create a conversation from a dictionary.
        
        Args:
            data: Dictionary representation of a conversation
            
        Returns:
            Conversation instance
        """
        conversation = cls(system_prompt=data.get("system_prompt"))
        
        for message_data in data.get("messages", []):
            message = Message(
                role=message_data["role"],
                content=message_data["content"],
                function_call=message_data.get("function_call"),
                function_result=message_data.get("function_result"),
                timestamp=message_data.get("timestamp", datetime.now().isoformat())
            )
            conversation.messages.append(message)
        
        return conversation
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Conversation':
        """
        Create a conversation from a JSON string.
        
        Args:
            json_str: JSON string representation of a conversation
            
        Returns:
            Conversation instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class ConversationManager:
    """
    Manager for multiple conversations with Gemma models.
    
    This class manages multiple conversations with Gemma models.
    """
    
    def __init__(self):
        """
        Initialize a conversation manager.
        """
        self.conversations: Dict[str, Conversation] = {}
    
    def create_conversation(self, conversation_id: str, system_prompt: Optional[str] = None) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            system_prompt: Optional system prompt to set the behavior of the model
            
        Returns:
            New conversation instance
        """
        conversation = Conversation(system_prompt)
        self.conversations[conversation_id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Conversation instance or None if not found
        """
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_all_conversation_ids(self) -> List[str]:
        """
        Get all conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        return list(self.conversations.keys())
    
    def save_conversations(self, file_path: str) -> None:
        """
        Save all conversations to a file.
        
        Args:
            file_path: Path to save the conversations
        """
        data = {
            conversation_id: conversation.to_dict()
            for conversation_id, conversation in self.conversations.items()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_conversations(self, file_path: str) -> None:
        """
        Load conversations from a file.
        
        Args:
            file_path: Path to load the conversations from
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.conversations = {
            conversation_id: Conversation.from_dict(conversation_data)
            for conversation_id, conversation_data in data.items()
        }
