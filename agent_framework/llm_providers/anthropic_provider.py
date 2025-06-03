# agent_framework/llm_providers/anthropic_provider.py
import os
from typing import List, Dict, Optional
from .base_provider import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, thinking: Optional[bool] = False):
        """
        Initialize the Anthropic provider.
        
        Args:
            model: The model identifier to use
            api_key: Anthropic API key (if None, will look for ANTHROPIC_API_KEY environment variable)
        """
        self.model = model
        self.thinking = thinking
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        # Import here to avoid dependency if not using Anthropic
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install it with 'pip install anthropic'")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using Anthropic models.
        
        Args:
            messages: List of message dictionaries
            kwargs: Additional arguments to pass to the Anthropic API
            
        Returns:
            Generated text response
        """
        # Convert messages format to Anthropic's format
        system_message = None
        anthropic_messages = []
        
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            elif message["role"] == "user":
                anthropic_messages.append({"role": "user", "content": message["content"]})
            elif message["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": message["content"]})
        
        response = self.client.messages.create(
            model=self.model,
            system=system_message,
            messages=anthropic_messages,
            thinking={
                "type": "enabled",
                "budget_tokens": 16000
            } if self.thinking else {"type": "disabled"},
            **kwargs
        )
        return response.content[1].text if self.thinking else response.content[0].text