# agent_framework/tools/tool_registry.py
from typing import Dict, Type, Any, Optional
from .base_tool import BaseTool

class ToolRegistry:
    """Registry for storing and retrieving tool instances."""
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        """
        Register a tool in the registry.
        
        Args:
            tool: The tool instance to register
        """
        cls._tools[tool.name] = tool
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[BaseTool]:
        """
        Get a tool from the registry by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            The tool instance or None if not found
        """
        return cls._tools.get(name)
    
    @classmethod
    def list_tools(cls) -> Dict[str, BaseTool]:
        """
        List all registered tools.
        
        Returns:
            Dictionary of all registered tools
        """
        return cls._tools.copy()
