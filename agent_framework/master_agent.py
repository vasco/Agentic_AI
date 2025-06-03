# agent_framework/master_agent.py
import logging
import yaml
import os
import re
import json
import queue
from typing import Dict, List, Any, Optional, Union
from .agent import Agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Queue for signaling task completion
execution_updates = queue.Queue()

class MasterAgent:
    """
    Master agent that coordinates multiple specialized agents.
    It breaks down problems, assigns tasks to agents, and integrates results.
    """
    
    def __init__(self, config_path: Optional[str] = None, agents_dir: Optional[str] = None):
        """
        Initialize the master agent.
        
        Args:
            config_path: Path to master agent configuration
            agents_dir: Directory containing agent YAML configurations
        """
        # Load master configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "name": "MasterAgent",
                "description": "Coordinates multiple specialized agents to solve complex tasks",
                "llm": {
                    "type": "openai",
                    "model": "gpt-4o"
                },
                "system_prompt": """You are a master agent that coordinates multiple specialized agents to solve complex tasks.
Your job is to:
1. Break down the main problem into smaller, manageable tasks
2. Select the most appropriate agent for each task
3. Execute tasks in an optimal sequence
4. Integrate results from different agents
5. Plan next steps based on the current state of the solution
6. Provide a coherent final answer

Available agents and their capabilities are provided in the context.
"""
            }
        
        # Initialize master LLM
        self._initialize_master_llm()
        
        # Load agents
        self.agents = {}
        if agents_dir:
            self._load_agents_from_directory(agents_dir)
    
    def _initialize_master_llm(self):
        """Initialize the LLM for the master agent."""
        llm_config = self.config.get('llm', {})
        llm_type = llm_config.get('type', 'openai')
        
        if llm_type == 'openai':
            from .llm_providers.openai_provider import OpenAIProvider
            model = llm_config.get('model', 'gpt-4o')
            api_key = llm_config.get('api_key', None)
            self.llm = OpenAIProvider(model=model, api_key=api_key)
        
        elif llm_type == 'anthropic':
            from .llm_providers.anthropic_provider import AnthropicProvider
            model = llm_config.get('model', 'claude-3-7-sonnet-20250219')
            api_key = llm_config.get('api_key', None)
            self.llm = AnthropicProvider(model=model, api_key=api_key)
        
        else:
            # Load custom LLM provider
            try:
                import importlib
                module_path, class_name = llm_type.rsplit('.', 1)
                module = importlib.import_module(module_path)
                LLMClass = getattr(module, class_name)
                self.llm = LLMClass(**llm_config.get('params', {}))
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load LLM provider: {llm_type}. Error: {e}")
    
    def _load_agents_from_directory(self, agents_dir: str):
        """
        Load all agent configurations from a directory.
        
        Args:
            agents_dir: Directory containing agent YAML files
        """
        if not os.path.exists(agents_dir):
            logger.warning(f"Agents directory not found: {agents_dir}")
            return
        
        for filename in os.listdir(agents_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                try:
                    agent_path = os.path.join(agents_dir, filename)
                    agent = Agent.from_yaml(agent_path)
                    self.agents[agent.name] = agent
                    logger.info(f"Loaded agent: {agent.name}")
                except Exception as e:
                    logger.error(f"Failed to load agent from {filename}: {e}")
    
    def add_agent(self, agent: Union[Agent, str, Dict]):
        """
        Add an agent to the master agent.
        
        Args:
            agent: Agent instance, path to agent YAML, or agent config dictionary
        """
        if isinstance(agent, Agent):
            self.agents[agent.name] = agent
        elif isinstance(agent, str) and (agent.endswith('.yaml') or agent.endswith('.yml')):
            agent_instance = Agent.from_yaml(agent)
            self.agents[agent_instance.name] = agent_instance
        elif isinstance(agent, dict):
            agent_instance = Agent.from_dict(agent)
            self.agents[agent_instance.name] = agent_instance
        else:
            raise ValueError("Agent must be an Agent instance, a path to a YAML file, or a configuration dictionary")
    
    def list_agents(self) -> List[str]:
        """
        List all available agents.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def _build_agents_info(self) -> str:
        """
        Build a string describing all available agents.
        
        Returns:
            String with information about all agents
        """
        agents_info = "Available Agents:\n\n"
        
        for name, agent in self.agents.items():
            agents_info += f"- Agent: {name}\n"
            agents_info += f"  Description: {agent.description}\n"
            
            if agent.tools:
                agents_info += "  Tools:\n"
                for tool in agent.tools:
                    agents_info += f"    - {tool.name}: {tool.description}\n"
            
            agents_info += "\n"
        
        return agents_info
    
    def run(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Run the master agent to solve a complex problem.
        
        Args:
            query: The problem or query to solve
            max_steps: Maximum number of steps to take
            
        Returns:
            Dictionary containing the solution and execution trace
        """
        try:
            logger.info(f"Starting Master Agent run with query: '{query[:100]}...' (truncated)")
            logger.info(f"Available agents: {list(self.agents.keys())}")
            
            # Debug: Print the full query for better understanding
            logger.info(f"FULL QUERY: {query}")
            
            if not self.agents:
                logger.error("No agents available to run the task!")
                return {"error": "No agents available. Add agents before running the master agent."}
            
            # History of steps taken
            execution_trace = []
            
            # Context to track state
            context = {
                "original_query": query,
                "current_step": 0,
                "max_steps": max_steps,
                "completed_tasks": [],
                "results": {}
            }
            
            # Signal that processing has started
            execution_updates.put({
                "type": "started",
                "message": "Processing task"
            })
            
            # Main execution loop
            while context["current_step"] < context["max_steps"]:
                # Increment step counter
                context["current_step"] += 1
                
                # Plan next action
                logger.info(f"Master Agent: Step {context['current_step']}: Planning next action")
                try:
                    action_plan = self._plan_action(query, context)
                    logger.info(f"Action plan received: {json.dumps(action_plan)}")
                except Exception as e:
                    logger.error(f"Error in planning action: {str(e)}", exc_info=True)
                    return {"error": f"Error planning action: {str(e)}"}
                
                # Extract action details
                task = action_plan.get("task", "")
                agent_name = action_plan.get("agent", "")
                agent_input = action_plan.get("input", "")
                
                if not task or not agent_name or agent_name not in self.agents:
                    error_message = f"Invalid action plan: Missing task, agent, or agent '{agent_name}' not found"
                    logger.error(error_message)
                    execution_trace.append({
                        "step": context["current_step"],
                        "action": "error",
                        "details": error_message
                    })
                    break
                
                # Execute task with the selected agent
                logger.info(f"Executing task '{task}' with agent '{agent_name}'")
                
                try:
                    agent = self.agents[agent_name]
                    logger.info(f"Agent tools: {[tool.name for tool in agent.tools]}")
                    result = agent.run(agent_input)
                    logger.info(f"Agent execution completed, result length: {len(result)}")
                except Exception as e:
                    logger.error(f"Error executing agent '{agent_name}': {str(e)}", exc_info=True)
                    result = f"Error executing agent task: {str(e)}"
                
                # Update context
                task_id = f"task_{context['current_step']}"
                context["completed_tasks"].append({
                    "id": task_id,
                    "task": task,
                    "agent": agent_name,
                    "input": agent_input,
                    "result": result
                })
                
                context["results"][task_id] = result
                
                # Create trace entry
                trace_entry = {
                    "step": context["current_step"],
                    "task": task,
                    "agent": agent_name,
                    "input": agent_input,
                    "result": result
                }
                
                # Add to execution trace
                execution_trace.append(trace_entry)
                
                # Log step completion
                logger.info(f"Completed step {context['current_step']}")
                
                # Check for completion
                is_complete = self._check_completion(context)
                if is_complete:
                    break
            
            # Generate final answer
            logger.info("Generating final answer...")
            final_answer = self._generate_final_answer(context)
            
            # Signal completion in the execution updates queue
            try:
                execution_updates.put({
                    "type": "complete",
                    "message": "Task processing complete"
                })
                logger.info("Published completion message to queue")
            except Exception as e:
                logger.error(f"Failed to publish completion message: {e}")
            
            return {
                "answer": final_answer,
                "execution_trace": execution_trace,
                "context": context
            }
        except Exception as e:
            logger.error(f"Critical error during master agent execution: {str(e)}", exc_info=True)
            execution_updates.put({
                "type": "error",
                "message": f"Critical error during processing: {str(e)}",
                "details": "Check logs for more information"
            })
            return {"error": f"Critical error during execution: {str(e)}"}
    
    def _plan_action(self, query: str, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Plan the next action based on the current context.
        
        Args:
            query: Original query
            context: Current execution context
            
        Returns:
            Dictionary with the next action details
        """
        try:
            # Log planning step
            logger.info(f"Planning step {context['current_step']}")
                
            agents_info = self._build_agents_info()
            
            # Prepare prompt for the master LLM
            system_prompt = f"{self.config.get('system_prompt', '')}\n\n{agents_info}"
            
            completed_tasks = ""
            for task in context.get("completed_tasks", []):
                completed_tasks += f"Task: {task['task']}\n"
                completed_tasks += f"Agent: {task['agent']}\n"
                completed_tasks += f"Input: {task['input']}\n"
                completed_tasks += f"Result: {task['result']}\n\n"
            
            user_prompt = f"""
Original Problem: {query}

Current Step: {context['current_step']} / {context['max_steps']}

Completed Tasks:
{completed_tasks if completed_tasks else "None yet."}

Based on the above, plan the next step:
1. What task needs to be done next?
2. Which agent is best suited for this task?
3. What specific input should be given to the agent?

Return your plan in JSON format:
```json
{{
  "task": "description of the task",
  "agent": "name of the selected agent",
  "input": "input for the agent in text format without code wrappers or JSON objects"
}}
```
"""
            
            # Call the master LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            logger.info(f"Sending planning request to LLM for step {context['current_step']}")
            
            try:
                response = self.llm.generate(messages)
                logger.info("LLM planning response received successfully")
            except Exception as e:
                logger.error(f"Error calling LLM API: {str(e)}")
                # Return a default action plan using the first available agent
                return {
                    "task": "API_ERROR_RECOVERY",
                    "agent": list(self.agents.keys())[0] if self.agents else "Researcher",
                    "input": f"There was an error connecting to the LLM API. Please check your API keys and connection. Original query: {query}"
                }
            
            # Extract JSON plan from the response
            action_plan = {}
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                if json_match:
                    action_plan = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire response as JSON
                    action_plan = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse action plan as JSON: {response}")
                action_plan = {
                    "task": "error_recovery",
                    "agent": list(self.agents.keys())[0] if self.agents else "",
                    "input": f"There was an error parsing the previous step. Original query: {query}"
                }
            
            logger.info(f"Action Plan:\n{action_plan}")
            
            return action_plan
            
        except Exception as e:
            logger.error(f"Unexpected error in _plan_action: {str(e)}", exc_info=True)
            # Return a fallback plan
            default_agent = list(self.agents.keys())[0] if self.agents else "Researcher"
            return {
                "task": "SYSTEM_ERROR_RECOVERY",
                "agent": default_agent,
                "input": f"There was a system error in processing. Original query: {query}"
            }
    
    def _check_completion(self, context: Dict[str, Any]) -> bool:
        """
        Check if the problem has been solved.
        
        Args:
            context: Current execution context
            
        Returns:
            True if the problem is solved, False otherwise
        """
        try:
            # If no tasks have been completed, we can't be done
            if not context.get("completed_tasks", []):
                logger.info("No tasks completed yet, returning not complete")
                return False
            
            # Log completion check
            logger.info(f"Checking if task is complete after step {context['current_step']}")
            
            # Prepare prompt for the master LLM
            system_prompt = self.config.get('system_prompt', '')
            
            completed_tasks = ""
            for task in context.get("completed_tasks", []):
                completed_tasks += f"Task: {task['task']}\n"
                completed_tasks += f"Agent: {task['agent']}\n"
                completed_tasks += f"Input: {task['input']}\n"
                completed_tasks += f"Result: {task['result']}\n\n"
            
            user_prompt = f"""
Original Problem: {context['original_query']}

Completed Tasks:
{completed_tasks}

Based on the above, is the problem solved completely?
Return ONLY "yes" if the problem is solved, or "no" if more steps are needed.
"""
            
            # Call the master LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            logger.info("Sending completion check request to LLM")
            
            try:
                response = self.llm.generate(messages)
                logger.info(f"LLM completion check response: {response}")
            except Exception as e:
                logger.error(f"Error calling LLM API during completion check: {str(e)}")
                # Default to continuing with more steps
                return False
            
            # Check if the response indicates completion
            is_complete = response.strip().lower() == "yes"
            logger.info(f"Completion status: {is_complete}")
            return is_complete
            
        except Exception as e:
            logger.error(f"Unexpected error in _check_completion: {str(e)}", exc_info=True)
            # Default to continuing with more steps
            return False
    
    def _generate_final_answer(self, context: Dict[str, Any]) -> str:
        """
        Generate the final answer based on all completed tasks.
        
        Args:
            context: Current execution context
            
        Returns:
            Final answer to the original query
        """
        try:
            # Log final answer generation
            logger.info("Generating final answer from completed tasks")
            
            # Prepare prompt for the master LLM
            system_prompt = self.config.get('system_prompt', '')
            
            completed_tasks = ""
            for task in context.get("completed_tasks", []):
                completed_tasks += f"Task: {task['task']}\n"
                completed_tasks += f"Agent: {task['agent']}\n"
                completed_tasks += f"Input: {task['input']}\n"
                completed_tasks += f"Result: {task['result']}\n\n"
            
            user_prompt = f"""
Original Problem: {context['original_query']}

Completed Tasks:
{completed_tasks}

Based on all the steps taken, provide a comprehensive final answer to the original problem.
Make sure to integrate all relevant results from the individual agents.
"""
            
            # Call the master LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            logger.info("Sending final answer request to LLM")
            
            try:
                response = self.llm.generate(messages)
                logger.info("LLM final answer response received successfully")
                return response
            except Exception as e:
                logger.error(f"Error calling LLM API during final answer generation: {str(e)}")
                
                # Fallback: Generate a simple summary from the completed tasks
                fallback_answer = f"Error generating comprehensive answer due to API connectivity issues. Here's a summary of completed tasks:\n\n"
                
                for task in context.get("completed_tasks", []):
                    fallback_answer += f"- {task['task']} (by {task['agent']}): "
                    # Get a short result summary (first 100 chars)
                    result_summary = task['result'][:100] + "..." if len(task['result']) > 100 else task['result']
                    fallback_answer += f"{result_summary}\n"
                
                fallback_answer += f"\nPlease check your API keys in the .env file and try again."
                return fallback_answer
                
        except Exception as e:
            logger.error(f"Unexpected error in _generate_final_answer: {str(e)}", exc_info=True)
            return f"An unexpected error occurred while generating the final answer: {str(e)}. Please check the logs for more information."