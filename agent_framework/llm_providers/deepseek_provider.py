# agent_framework/llm_providers/deepseek_provider.py
import os
from typing import List, Dict, Any, Optional
from .base_provider import BaseLLMProvider

class DeepseekProvider(BaseLLMProvider):
    """Provider for Deepseek models."""
    
    def __init__(self, model: str = "deepseek-coder-r1", api_key: Optional[str] = None):
        """
        Initialize the Deepseek provider.
        
        Args:
            model: The model identifier to use
            api_key: Deepseek API key (if None, will look for DEEPSEEK_API_KEY environment variable)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Deepseek API key must be provided or set as DEEPSEEK_API_KEY environment variable")
        
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
            self.api_url = "https://api.deepseek.com/v1/chat/completions"
        except ImportError:
            raise ImportError("Requests package not installed. Install it with 'pip install requests'")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using Deepseek models.
        
        Args:
            messages: List of message dictionaries
            kwargs: Additional arguments to pass to the Deepseek API
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        response = self.session.post(self.api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"]