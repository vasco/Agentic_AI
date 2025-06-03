import os
import argparse
import yaml
import json
from agent_framework import Agent, MasterAgent
import openai
import shutil
from dotenv import load_dotenv
from agent_framework.llm_providers.anthropic_provider import AnthropicProvider

# Load environment variables from .env file
load_dotenv()

def create_sample_agents():
    """Create sample agent configuration files in the 'agents' directory."""
    
    # Create agents directory if it doesn't exist
    os.makedirs("agents", exist_ok=True)
    
    # Researcher agent config
    researcher_config = {
        "name": "Researcher",
        "description": "Specializes in information gathering and research",
        "llm": {
            "type": "openai",
            "model": "gpt-4o"
        },
        "tools": [
            "web_search",
            "web_scraper"
        ],
        "system_prompt": """You are a research specialist agent. Your role is to gather information, analyze data, 
and provide well-researched answers. Always cite your sources and prioritize reliable information.
When using tools, be specific in your queries and extract the most relevant information.""",
        "output_format": {
            "format": "json",
            "description": "Include the research question, findings, and sources",
            "example": """
{
  "question": "What are the health benefits of turmeric?",
  "findings": [
    "Contains curcumin, a compound with anti-inflammatory properties",
    "May help reduce risk of heart disease",
    "Has antioxidant effects"
  ],
  "sources": [
    {"title": "Turmeric and Its Major Compound Curcumin", "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664031/"}
  ]
}"""
        }
    }
    
    # Coder agent config
    coder_config = {
        "name": "Coder",
        "description": "Specializes in writing, analyzing, and debugging code",
        "llm": {
            "type": "deepseek",
            "model": "deepseek-coder-r1"
        },
        "tools": [
            "code_executor",
            "github_tool"
        ],
        "system_prompt": """You are a coding specialist agent. Your role is to write clean, efficient code,
debug issues, and explain technical concepts clearly. Always include comments in your code
and ensure your solutions are well-tested and secure.""",
        "output_format": {
            "format": "markdown",
            "description": "Provide code solutions with explanations and testing notes"
        }
    }
    
    # Writer agent config
    writer_config = {
        "name": "Writer",
        "description": "Specializes in creating and editing written content",
        "llm": {
            "type": "anthropic",
            "model": "claude-3-sonnet-20240229"
        },
        "system_prompt": """You are a writing specialist agent. Your role is to craft compelling, clear, and
engaging content. You can write in various styles and formats, from formal reports
to creative stories. Focus on proper structure, grammar, and tailoring content to the
intended audience.""",
        "output_format": {
            "format": "text",
            "description": "Provide well-formatted text content with clear structure"
        }
    }
    
    # Save configs to YAML files
    with open("agents/researcher.yaml", "w") as f:
        yaml.dump(researcher_config, f, default_flow_style=False)
    
    with open("agents/coder.yaml", "w") as f:
        yaml.dump(coder_config, f, default_flow_style=False)
    
    with open("agents/writer.yaml", "w") as f:
        yaml.dump(writer_config, f, default_flow_style=False)
    
    print("Sample agent configurations created in 'agents' directory")

def create_master_config():
    """Create a sample master agent configuration file."""
    
    master_config = {
        "name": "MasterCoordinator",
        "description": "Coordinates multiple specialized agents to solve complex tasks",
        "llm": {
            "type": "openai",
            "model": "gpt-4o"
        },
        "system_prompt": """You are a master coordinator agent that orchestrates multiple specialized agents to solve complex tasks.
Your job is to:
1. Break down the main problem into smaller, manageable tasks
2. Select the most appropriate agent for each task from the available agents
3. Execute tasks in an optimal sequence
4. Integrate results from different agents
5. Plan next steps based on the current state of the solution
6. Provide a coherent final answer

Always think step-by-step and delegate tasks efficiently to the most suitable agent.
"""
    }
    
    # Save config to YAML file
    with open("master_config.yaml", "w") as f:
        yaml.dump(master_config, f, default_flow_style=False)
    
    print("Master agent configuration created as 'master_config.yaml'")

def create_agent_from_description(name, description, output_path=None):
    """
    Create a new agent configuration file based on a description using OpenAI.
    
    Args:
        name: Name of the agent
        description: Detailed description of what the agent should do
        output_path: Path to save the agent config (default: agents/{name_lowercase}.yaml)
    """
    try:
        # Ensure the OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return
            
        client = openai.OpenAI(api_key=api_key)
        
        # Define the prompt for OpenAI
        prompt = f"""
        Create a detailed agent configuration YAML for an AI agent with the following details:
        - Name: {name}
        - Purpose/Description: {description}
        
        The configuration should include:
        1. A suitable LLM provider and model
        2. A detailed system prompt that guides the agent's behavior
        3. Appropriate tools the agent might need (from: web_search, web_scraper, code_executor, github_tool, etc.)
        4. A suitable output format (json, markdown, or text)
        
        Format the response as a valid YAML configuration similar to this example:
        ```yaml
        name: "Agent Name"
        description: "Detailed description"
        llm:
          type: "openai"
          model: "gpt-4o"
        tools:
          - "tool1"
          - "tool2"
        system_prompt: "Detailed instructions for the agent..."
        output_format:
          format: "json/markdown/text"
          description: "Description of expected output format"
          example: "Optional example of the output format"
        ```
        
        Provide only the YAML content without any other explanations.
        """
        
        # Call OpenAI API to generate the agent configuration
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates detailed and appropriate agent configurations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract YAML content
        yaml_content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if yaml_content.startswith("```yaml"):
            yaml_content = yaml_content.replace("```yaml", "", 1)
        if yaml_content.endswith("```"):
            yaml_content = yaml_content[:-3]
        
        yaml_content = yaml_content.strip()
        
        # Parse to ensure valid YAML
        config_dict = yaml.safe_load(yaml_content)
        
        # Ensure the directory exists
        os.makedirs("agents", exist_ok=True)
        
        # Determine output path
        if not output_path:
            output_path = f"agents/{name.lower().replace(' ', '_')}.yaml"
        
        # Write to file
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Agent configuration created and saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error creating agent from description: {str(e)}")
        return None

def create_tool_from_description(name, description):
    """
    Create a new tool Python file based on a description using Anthropic Claude.
    
    Args:
        name: Name of the tool (will be used as class name)
        description: Detailed description of what the tool should do
    """
    try:
        # Ensure the Anthropic API key is set
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            return
            
        llm = AnthropicProvider(model="claude-3-7-sonnet-20250219", api_key=api_key, thinking=False)
        
        # Clean the tool name
        tool_class_name = ''.join(word.capitalize() for word in name.split('_'))
        tool_file_name = name.lower().replace(' ', '_')
        
        # Define the prompt for OpenAI
        prompt = f'''
        Create a Python tool class for an AI agent framework with the following details:
        - Tool name: {name}
        - Tool purpose: {description}
        
        The tool should:
        1. Inherit from the BaseTool class
        2. Have a proper __init__ method with appropriate parameters
        3. Implement an execute method that performs the tool's function
        4. Include proper error handling and documentation
        5. Follow the pattern shown in the example below
        6. Try to avoid using complex to install libraries (e.g. weasyprint)
        7. Use dotevn wheneever it is necessary to load environment variables:
        
        ```python
        from typing import Dict, List, Any, Optional
        from .base_tool import BaseTool
        import os
        
        class WebSearch(BaseTool):
            """Tool for performing web searches using Serper API."""
            
            def __init__(self, api_key: Optional[str] = None):
                """
                Initialize the web search tool with Serper API.
                
                Args:
                    api_key: API key for the Serper API service (required)
                """
                super().__init__(
                    name="web_search",
                    description="Search the web for information on a given query using Serper API"
                )
                self.api_key = api_key or os.environ.get("SERPER_API_KEY")
            
            def execute(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
                """
                Execute a web search using Serper API.
                
                Args:
                    query: The search query
                    num_results: Number of results to return
                    
                Returns:
                    List of search results, each containing title, snippet, and URL
                """
                try:            
                    if not self.api_key:
                        return {{"success": False, "error": "Serper API key is required. Please provide an API key."}}
        
                    params = {{
                        "api_key": self.api_key,
                        "engine": "duckduckgo",
                        "q": query,
                        "kl": "us-en",
                        "num": num_results
                    }}
        
                    search = GoogleSearch(params)
                    data = search.get_dict()
                    
                    content = []
                    if 'organic_results' in data:
                        for item in data['organic_results'][:num_results]:
                            content.append({{
                                'title': item.get('title', ''),
                                'snippet': item.get('snippet', ''),
                                'url': item.get('link', '')
                            }})
                        return {{"success": True, "content": content}}
                    else:
                        return {{"success": False, "error": "No search results found."}}
                except ImportError:
                    return {{"success": False, "error": "Requests package not installed. Install with 'pip install requests'"}}
                except Exception as e:
                    return {{"success": False, "error": f"Search failed: {{str(e)}}"}}
        ```
        
        Provide only the Python code without any other explanations.
        '''
        
        # Prepare the prompt with system message and tools
        messages = []
        messages.append({"role": "system", "content": "You are an AI assistant that creates Python tool classes for an agent framework."})
        messages.append({"role": "user", "content": prompt})

        # Call Anthropic API to generate the tool class
        response = llm.generate(messages, max_tokens=20000)
        
        # Extract code content
        python_content = response.replace("```python", "").replace("```", "").strip()
        
        # Ensure the tools directory exists
        os.makedirs("agent_framework/tools", exist_ok=True)
        
        # Write to file
        output_path = f"agent_framework/tools/{tool_file_name}.py"
        with open(output_path, "w") as f:
            f.write(python_content)
        
        # Update the __init__.py file properly
        init_path = "agent_framework/tools/__init__.py"
        
        if os.path.exists(init_path):
            with open(init_path, "r") as f:
                init_content = f.read()
            
            # Parse the current init file
            # Add the import statement after the last import
            import_statement = f"from .{tool_file_name} import {tool_class_name}\n"
            
            # Find the spot to insert the import
            import_section_end = init_content.find("\n# Pre-register")
            if import_section_end == -1:
                # If the structure is different, add to the top
                new_init_content = import_statement + init_content
            else:
                # Insert after the last import
                new_init_content = init_content[:import_section_end] + import_statement + init_content[import_section_end:]
            
            # Add to the tool registry
            registry_line = f"ToolRegistry.register({tool_class_name}())\n"
            
            # Find the last registry line
            registry_section_end = new_init_content.find("\n__all__")
            if registry_section_end == -1:
                # If __all__ is not found, add at the end
                new_init_content += f"\n{registry_line}"
            else:
                # Insert before __all__
                new_init_content = new_init_content[:registry_section_end] + registry_line + new_init_content[registry_section_end:]
            
            # Update __all__ list
            all_section = new_init_content.find("__all__ = [")
            if all_section != -1:
                # Find the closing bracket
                all_end = new_init_content.find("]", all_section)
                if all_end != -1:
                    # Insert the new tool class name
                    new_all_list = new_init_content[:all_end] + f"    '{tool_class_name}',\n" + new_init_content[all_end:]
                    new_init_content = new_all_list
            
            # Write the updated content
            with open(init_path, "w") as f:
                f.write(new_init_content)
        else:
            # Create a new __init__.py file if it doesn't exist
            basic_init = f"""from .base_tool import BaseTool
            from .tool_registry import ToolRegistry
            from .{tool_file_name} import {tool_class_name}

            # Pre-register common tools
            ToolRegistry.register({tool_class_name}())

            __all__ = [
                'BaseTool',
                'ToolRegistry',
                '{tool_class_name}',
            ]
            """
            with open(init_path, "w") as f:
                f.write(basic_init)
        
        print(f"Tool created and saved to {output_path}")
        print(f"Updated {init_path} to import and register the new tool")
        return output_path
        
    except Exception as e:
        print(f"Error creating tool from description: {str(e)}")
        return None

def start_web_app():
    """Start the Flask web application."""
    from app import app
    app.run(debug=True)

def main():
    parser = argparse.ArgumentParser(description="Agent Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Initialize sample agent configurations")
    
    # Create agent command
    create_agent_parser = subparsers.add_parser("create-agent", help="Create a new agent from description")
    create_agent_parser.add_argument("name", help="Name of the agent")
    create_agent_parser.add_argument("description", help="Detailed description of what the agent should do")
    create_agent_parser.add_argument("--output", help="Output path for the agent config file", default=None)
    
    # Create tool command
    create_tool_parser = subparsers.add_parser("create-tool", help="Create a new tool from description")
    create_tool_parser.add_argument("name", help="Name of the tool (snake_case)")
    create_tool_parser.add_argument("description", help="Detailed description of what the tool should do")
    
    # Run single agent command
    agent_parser = subparsers.add_parser("run-agent", help="Run a single agent")
    agent_parser.add_argument("config", help="Path to agent configuration YAML file")
    agent_parser.add_argument("input", help="Input text/query for the agent")
    
    # Run master agent command
    master_parser = subparsers.add_parser("run-master", help="Run master agent with multiple agents")
    master_parser.add_argument("--config", help="Path to master agent configuration YAML file", default="master_config.yaml")
    master_parser.add_argument("--agents-dir", help="Directory containing agent YAML files", default="agents")
    master_parser.add_argument("input", help="Input problem/query for the master agent")
    master_parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of steps for the master agent")
    
    # Web app command
    web_parser = subparsers.add_parser("web", help="Start the web application")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        create_sample_agents()
        create_master_config()
    
    elif args.command == "create-agent":
        create_agent_from_description(args.name, args.description, args.output)
    
    elif args.command == "create-tool":
        create_tool_from_description(args.name, args.description)
    
    elif args.command == "run-agent":
        agent = Agent.from_yaml(args.config)
        result = agent.run(args.input)
        print("\nAgent Result:")
        print(result)
    
    elif args.command == "run-master":
        master = MasterAgent(config_path=args.config, agents_dir=args.agents_dir)
        result = master.run(args.input, max_steps=args.max_steps)
        
        print("\nMaster Agent Result:")
        print(f"Answer: {result['answer']}")
        
        print("\nExecution Trace:")
        for step in result["execution_trace"]:
            print(f"Step {step['step']}: Task '{step['task']}' executed by '{step['agent']}'")
    
    elif args.command == "web":
        start_web_app()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()