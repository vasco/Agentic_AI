import yaml
import importlib
import logging
from typing import Dict, Optional, Union
import json
import inspect
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    """Base Agent class that can be instantiated from YAML config files."""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize an agent from either a YAML file path or a config dictionary.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary containing agent configuration
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        self.name = self.config.get('name', 'Unnamed Agent')
        self.description = self.config.get('description', '')
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize tools
        self.tools = []
        self._initialize_tools()
        
        # System prompt
        self.system_prompt = self.config.get('system_prompt', '')
        
        # Output format
        self.output_format = self.config.get('output_format', None)
        
        # Max tool iteration
        self.max_tool_iterations = self.config.get('max_tool_iterations', 10)
        
    def _initialize_llm(self):
        """Initialize the LLM based on the configuration."""
        llm_config = self.config.get('llm', {})
        llm_type = llm_config.get('type', 'openai')
        
        if llm_type == 'openai':
            from .llm_providers.openai_provider import OpenAIProvider
            model = llm_config.get('model', 'gpt-4o')
            api_key = llm_config.get('api_key', None)
            self.llm = OpenAIProvider(model=model, api_key=api_key)
        
        elif llm_type == 'deepseek':
            from .llm_providers.deepseek_provider import DeepseekProvider
            model = llm_config.get('model', 'deepseek-coder-r1')
            api_key = llm_config.get('api_key', None)
            self.llm = DeepseekProvider(model=model, api_key=api_key)
            
        elif llm_type == 'anthropic':
            from .llm_providers.anthropic_provider import AnthropicProvider
            model = llm_config.get('model', 'claude-3-sonnet-20240229')
            api_key = llm_config.get('api_key', None)
            self.llm = AnthropicProvider(model=model, api_key=api_key)
        
        else:
            # Load custom LLM provider
            try:
                module_path, class_name = llm_type.rsplit('.', 1)
                module = importlib.import_module(module_path)
                LLMClass = getattr(module, class_name)
                self.llm = LLMClass(**llm_config.get('params', {}))
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load LLM provider: {llm_type}. Error: {e}")
    
    def _initialize_tools(self):
        """Initialize tools based on the configuration."""
        from .tools import ToolRegistry
        
        tools_config = self.config.get('tools', [])
        logger.info(f"Initializing {len(tools_config)} tools for agent '{self.name}'")
        
        for tool_config in tools_config:
            if isinstance(tool_config, str):
                # Reference to a predefined tool
                try:
                    tool = ToolRegistry.get_tool(tool_config)
                    if tool:
                        logger.info(f"Successfully loaded tool '{tool_config}' for agent '{self.name}'")
                        self.tools.append(tool)
                    else:
                        logger.warning(f"Tool '{tool_config}' not found in registry for agent '{self.name}'")
                except Exception as e:
                    logger.error(f"Error loading tool '{tool_config}' for agent '{self.name}': {str(e)}")
            elif isinstance(tool_config, dict):
                # Custom tool definition
                tool_type = tool_config.get('type')
                if not tool_type:
                    logger.warning(f"Tool configuration missing 'type' field for agent '{self.name}'")
                    continue
                
                try:
                    if '.' in tool_type:
                        # Import custom tool class
                        module_path, class_name = tool_type.rsplit('.', 1)
                        logger.info(f"Attempting to import custom tool class {class_name} from {module_path}")
                        module = importlib.import_module(module_path)
                        ToolClass = getattr(module, class_name)
                    else:
                        # Use builtin tool
                        logger.info(f"Attempting to load built-in tool {tool_type}")
                        from . import tools
                        ToolClass = getattr(tools, tool_type)
                    
                    tool_params = tool_config.get('params', {})
                    tool_instance = ToolClass(**tool_params)
                    self.tools.append(tool_instance)
                    logger.info(f"Successfully initialized custom tool {tool_type} for agent '{self.name}'")
                except ImportError as e:
                    # More specific handling for import errors
                    logger.error(f"Missing dependency for tool '{tool_type}': {str(e)}")
                    if "serpapi" in str(e):
                        logger.error("SerpAPI module missing. Install with: pip install google-search-results")
                    elif "beautifulsoup4" in str(e) or "bs4" in str(e):
                        logger.error("BeautifulSoup missing. Install with: pip install beautifulsoup4")
                    elif "requests" in str(e):
                        logger.error("Requests module missing. Install with: pip install requests")
                except AttributeError as e:
                    logger.error(f"Could not find tool class: {tool_type}. Error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error initializing tool '{tool_type}': {str(e)}", exc_info=True)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run the agent with the given input.
        
        Args:
            input_text: The input text/query to the agent
            
        Returns:
            The agent's response
        """
        # Prepare the prompt with system message and tools
        messages = []
        
        # Add system message
        if self.system_prompt:
            messages.append({"role": "system", "content": self._build_system_prompt()})
        
        # Add user message
        messages.append({"role": "user", "content": input_text})
        
        # Iterative tool calling
        iteration = 0
        final_answer = None
        
        while iteration < self.max_tool_iterations:
            iteration += 1
            logger.info(f"Tool iteration {iteration}/{self.max_tool_iterations}")
            
            # Call LLM
            response = self.llm.generate(messages, **kwargs)
            # logger.info(f"LLM response: {response}")

            # Check if response contains a tool call or final answer
            tool_call = self._extract_tool_call(response)
            
            if tool_call:
                logger.info(f"Tool call detected: {tool_call.get('tool')}")
                tool_name = tool_call.get('tool')
                tool_params = tool_call.get('params', {})
                
                try:
                    # Execute the tool
                    tool_result = self.call_tool(tool_name, **tool_params)
                    # logger.info(f"Tool '{tool_name}' result: {tool_result}")

                    # Add the tool call and result to the conversation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "system", 
                        "content": f"Tool '{tool_name}' was called with params: {json.dumps(tool_params, indent=2)}.\nResult: {json.dumps(tool_result, indent=2)}"
                    })
                except Exception as e:
                    error_message = f"Error executing tool '{tool_name}': {str(e)}"
                    logger.error(error_message)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "system", 
                        "content": error_message
                    })
            else:
                # Check if this is a final answer (no tool call detected)
                final_answer = response
                logger.info("Final answer detected (no tool call)")
                break
        
        # If we reached max iterations without a final answer, use the last response
        if final_answer is None:
            logger.warning(f"Reached maximum tool iterations ({self.max_tool_iterations}) without final answer")
            final_answer = str(messages)
        
        # Format output if needed
        if self.output_format and 'format' in self.output_format:
            try:
                final_answer = self._format_output(final_answer)
            except Exception as e:
                logger.error(f"Failed to format output: {e}")
        
        return final_answer
    
    def _extract_tool_call(self, response: str) -> Optional[Dict]:
        """
        Extract tool call information from the response if present.
        
        Args:
            response: The response from the LLM
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call is found
        """
        # Try to find JSON in triple backticks
        json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and 'tool' in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        # Try to parse the entire response as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict) and 'tool' in data:
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to find a JSON-like structure without backticks
        # This handles cases like: '{\n "tool": "web_search",\n "params": {...}\n}'
        try:
            # Look for patterns that start with { and end with }
            json_pattern = r'(\{[\s\S]*\})'
            potential_json = re.search(json_pattern, response)
            if potential_json:
                json_str = potential_json.group(1)
                data = json.loads(json_str)
                if isinstance(data, dict) and 'tool' in data:
                    return data
        except (json.JSONDecodeError, AttributeError):
            pass
            
        # Try to find any JSON object containing a "tool" field
        try:
            # Look for JSON-like patterns: {..."tool": "tool_name"...}
            json_pattern = r'\{[^}]*"tool"\s*:\s*"[^"]*"[^}]*\}'
            potential_json = re.search(json_pattern, response)
            if potential_json:
                data = json.loads(potential_json.group(0))
                if 'tool' in data:
                    return data
        except (json.JSONDecodeError, AttributeError):
            pass
            
        return None
    
    def _build_system_prompt(self) -> str:
        """Build the complete system prompt including tools information."""
        prompt = self.system_prompt
        
        if self.tools:
            tools_desc = "\n\nYou have access to the following tools:\n"
            for i, tool in enumerate(self.tools, 1):
                # Get tool signature
                tool_signature = self._get_tool_signature(tool)
                tools_desc += f"{i}. {tool.name}: {tool.description}\n"
                tools_desc += f"   Signature: {tool_signature}\n\n"
            
            tools_desc += "To use a tool, respond with ONLY the following JSON format:\n"
            tools_desc += "```json\n{\n  \"tool\": \"tool_name\",\n  \"params\": {\"param1\": \"value1\", ...}\n}\n```\n"
            tools_desc += "Make sure to use the correct parameter names and types for each tool as specified in its signature.\n\n"
            tools_desc += "IMPORTANT: You can make multiple tool calls in sequence. For each tool call:\n"
            tools_desc += "1. Respond with only the tool call JSON format\n"
            tools_desc += "2. Receive the tool result\n"
            tools_desc += "3. Decide whether to call another tool or provide a final answer\n\n"
            tools_desc += "When you have all the information needed to provide a complete answer, respond with your final answer WITHOUT any tool call JSON. The absence of a tool call indicates that your response is the final answer.\n"
            
            prompt += tools_desc
        
        if self.output_format:
            format_desc = f"\n\nYour final answer should conform to the following format: {self.output_format.get('description', '')}"
            if 'example' in self.output_format:
                format_desc += f"\n\nExample:\n{self.output_format['example']}"
            prompt += format_desc
        
        return prompt
    
    def _get_tool_signature(self, tool) -> str:
        """
        Get a formatted signature for a tool's execute method.
        
        Args:
            tool: The tool instance
            
        Returns:
            Formatted string representation of the tool's execute method signature
        """
        try:
            # Get the signature of the execute method
            sig = inspect.signature(tool.execute)
            
            # Format the parameters
            params = []
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                
                param_str = name
                if param.annotation != inspect.Parameter.empty:
                    # Convert annotation to string representation
                    if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is Union:
                        # Handle Union types (e.g., Optional)
                        types = [t.__name__ if hasattr(t, '__name__') else str(t).replace('typing.', '') 
                                for t in param.annotation.__args__]
                        type_str = " | ".join(types)
                    else:
                        type_str = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation).replace('typing.', '')
                    
                    param_str += f": {type_str}"
                
                if param.default != inspect.Parameter.empty:
                    default_val = param.default
                    if isinstance(default_val, str):
                        default_val = f'"{default_val}"'
                    param_str += f" = {default_val}"
                
                params.append(param_str)
            
            # Get return annotation
            return_anno = ""
            if sig.return_annotation != inspect.Signature.empty:
                if hasattr(sig.return_annotation, '__name__'):
                    return_anno = f" -> {sig.return_annotation.__name__}"
                else:
                    return_anno = f" -> {str(sig.return_annotation).replace('typing.', '')}"
            
            return f"{tool.name}.execute({', '.join(params)}){return_anno}"
        
        except (AttributeError, TypeError):
            # Fallback if we can't get the signature
            return f"{tool.name}.execute(...)"
    
    def _format_output(self, response: str) -> str:
        """Format the output according to the specified format."""
        format_type = self.output_format.get('format')
        
        if format_type == 'json':
            # Try to extract JSON from the response
            try:
                # Look for JSON between triple backticks
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                
                # Try to parse the entire response as JSON
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Could not parse response as JSON")
                return response
            
        elif format_type == 'yaml':
            try:
                # Look for YAML between triple backticks
                import re
                yaml_match = re.search(r'```(?:yaml)?\s*([\s\S]*?)\s*```', response)
                if yaml_match:
                    yaml_str = yaml_match.group(1)
                    return yaml.safe_load(yaml_str)
                
                # Try to parse the entire response as YAML
                return yaml.safe_load(response)
            except yaml.YAMLError:
                logger.warning("Could not parse response as YAML")
                return response
            
        return response
    
    def call_tool(self, tool_name: str, **params):
        """
        Call a tool by name with the given parameters.
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        for tool in self.tools:
            if tool.name == tool_name:
                logger.info(f"Executing tool: {tool_name} with params: {params}")
                result = tool.execute(**params)
                logger.info(f"Tool execution completed: {tool_name}")
                return result
        
        raise ValueError(f"Tool '{tool_name}' not found")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Agent':
        """
        Create an Agent instance from a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            An Agent instance
        """
        return cls(config_path=yaml_path)
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'Agent':
        """
        Create an Agent instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing agent configuration
            
        Returns:
            An Agent instance
        """
        return cls(config_dict=config)