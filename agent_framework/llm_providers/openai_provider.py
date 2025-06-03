# agent_framework/llm_providers/openai_provider.py
import os
from typing import List, Dict, Any, Optional
import json
from .base_provider import BaseLLMProvider

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model: The model identifier to use
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Import here to avoid dependency if not using OpenAI
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install it with 'pip install openai'")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using OpenAI models.
        
        Args:
            messages: List of message dictionaries
            kwargs: Additional arguments to pass to the OpenAI API
            
        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
