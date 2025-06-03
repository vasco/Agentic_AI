# agent_framework/tools/__init__.py
from .base_tool import BaseTool
from .tool_registry import ToolRegistry
from .web_tools import WebSearch, WebScraper
from .markdown_to_pdf import MarkdownToPDF
from .gmail_sender import GmailSender

# Pre-register common tools
ToolRegistry.register(WebSearch())
ToolRegistry.register(WebScraper())
ToolRegistry.register(MarkdownToPDF())
ToolRegistry.register(GmailSender())

__all__ = [
    'BaseTool', 
    'ToolRegistry',
    'WebSearch',
    'WebScraper',
    'MarkdownToPDF',
    'GmailSender',
]
