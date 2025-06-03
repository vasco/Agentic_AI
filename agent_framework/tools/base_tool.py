# agent_framework/tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable

class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str = None, description: str = None):
        """
        Initialize a tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
        """
        self.name = name or self.__class__.__name__
        self.description = description or "No description provided"
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            kwargs: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for this tool's parameters.
        
        Returns:
            Parameter schema as a dictionary
        """
        import inspect
        sig = inspect.signature(self.execute)
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {}
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'kwargs':
                continue
                
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list or param.annotation is List:
                    param_type = "array"
                elif param.annotation is dict or param.annotation is Dict:
                    param_type = "object"
            
            required = param.default == inspect.Parameter.empty
            
            schema["parameters"][param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}",
                "required": required
            }
            
            if not required and param.default is not None:
                schema["parameters"][param_name]["default"] = param.default
        
        return schema
