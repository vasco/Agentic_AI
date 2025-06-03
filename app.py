import os
import yaml
import json
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import time
import json
import threading
import queue
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired
from agent_framework.master_agent import MasterAgent, execution_updates
from agent_framework.agent import Agent
from agent_framework.tools import ToolRegistry
from agent_framework.tools.base_tool import BaseTool

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['AGENTS_DIR'] = os.path.join(os.path.dirname(__file__), 'agents')
app.config['MASTER_CONFIG'] = os.path.join(os.path.dirname(__file__), 'master_config.yaml')

# Initialize the master agent
master_agent = MasterAgent(
    config_path=app.config['MASTER_CONFIG'], 
    agents_dir=app.config['AGENTS_DIR']
)

# Form for submitting tasks to the master agent
class TaskForm(FlaskForm):
    task = TextAreaField('Task', validators=[DataRequired()])
    submit = SubmitField('Submit Task')

# Form for editing master agent configuration
class MasterConfigForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    description = TextAreaField('Description', validators=[DataRequired()])
    llm_model = StringField('LLM Model', validators=[DataRequired()])
    llm_type = SelectField('LLM Type', choices=[
        ('openai', 'OpenAI'),
        ('anthropic', 'Anthropic'),
        ('deepseek', 'DeepSeek')
    ])
    system_prompt = TextAreaField('System Prompt', validators=[DataRequired()])
    submit = SubmitField('Save Changes')

# Form for creating/editing agents
class AgentForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    description = TextAreaField('Description', validators=[DataRequired()])
    system_prompt = TextAreaField('System Prompt', validators=[DataRequired()])
    llm_type = SelectField('LLM Type', choices=[
        ('openai', 'OpenAI'), 
        ('anthropic', 'Anthropic'),
        ('deepseek', 'DeepSeek')
    ])
    llm_model = StringField('LLM Model')
    tools = SelectField('Tools', choices=[], render_kw={"multiple": True})
    submit = SubmitField('Save Agent')

# Form for creating/editing tools
class ToolForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    description = TextAreaField('Description', validators=[DataRequired()])
    tool_type = StringField('Tool Type', validators=[DataRequired()])
    params = TextAreaField('Parameters (JSON)')
    submit = SubmitField('Save Tool')

# Route removed to fix duplicate endpoint error

# Global variable to store task results
task_result = None

def run_master_agent(query):
    """Run the master agent in a background thread"""
    global task_result
    try:
        # Clear any previous updates
        logger.info(f"Starting master agent run for query: {query[:50]}...")
        try:
            while not execution_updates.empty():
                execution_updates.get_nowait()
            logger.info("Cleared existing queue items")
        except Exception as e:
            logger.error(f"Error clearing queue before run: {e}")
            
        # Debug environment variables (remove sensitive info in production)
        env_vars = {k: v for k, v in os.environ.items() if 'key' in k.lower() or 'api' in k.lower()}
        masked_vars = {k: v[:5] + '*****' if len(v) > 10 else '****' for k, v in env_vars.items()}
        logger.info(f"Environment variables related to APIs: {masked_vars}")
        
        # Debug available tools
        from agent_framework.tools import ToolRegistry
        available_tools = ToolRegistry.list_tools()
        logger.info(f"Registered tools: {list(available_tools.keys())}")
        
        # Add a debug notification to the queue
        execution_updates.put({
            'type': 'progress',
            'step': 0,
            'title': 'Debug Information',
            'message': f'Available agents: {list(master_agent.agents.keys())}',
            'details': f'Available tools: {list(available_tools.keys())}'
        })
            
        # Run the master agent
        logger.info("Executing master agent run")
        logger.info("Calling master_agent.run() method...")
        task_result = master_agent.run(query)
        logger.info("Master agent run completed successfully")
    except Exception as e:
        logger.error(f"Error in master agent run: {e}", exc_info=True)
        # Add detailed error to the queue
        execution_updates.put({
            'type': 'error',
            'title': 'Master Agent Error',
            'message': f'Error executing master agent: {str(e)}',
            'details': 'Check server logs for detailed stack trace'
        })
        task_result = {"error": str(e)}

@app.route('/run_task', methods=['POST'])
def run_task():
    global task_result
    task_form = TaskForm()
    
    try:
        logger.info("Task submission received")
        if task_form.validate_on_submit():
            query = task_form.task.data
            logger.info(f"Task validated, query: {query[:50]}...")
            
            # Run the master agent in a background thread
            thread = threading.Thread(target=run_master_agent, args=(query,))
            thread.daemon = True
            thread.start()
            logger.info("Started master agent thread")
            
            # Redirect to the index page - results will show when processing is complete
            return redirect(url_for('index'))
        else:
            logger.error(f"Form validation failed: {task_form.errors}")
            flash("Invalid form submission", "error")
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Critical error in run_task: {str(e)}", exc_info=True)
        flash(f"Critical error processing task: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/')
def index():
    task_form = TaskForm()
    config_form = MasterConfigForm()
    
    # Load master agent configuration for the sidebar
    with open(app.config['MASTER_CONFIG'], 'r') as f:
        config = yaml.safe_load(f)
    
    # Populate the config form
    config_form.name.data = config.get('name', '')
    config_form.description.data = config.get('description', '')
    config_form.llm_model.data = config.get('llm', {}).get('model', '')
    config_form.llm_type.data = config.get('llm', {}).get('type', 'openai')
    config_form.system_prompt.data = config.get('system_prompt', '')
    
    return render_template('index.html', form=task_form, config_form=config_form,
                          result=task_result, master_agent=master_agent, config=config)

@app.route('/update_master_config', methods=['POST'])
def update_master_config():
    try:
        # Get current config
        with open(app.config['MASTER_CONFIG'], 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config with form data
        config['name'] = request.form.get('name')
        config['description'] = request.form.get('description')
        config['llm']['model'] = request.form.get('llm_model')
        config['llm']['type'] = request.form.get('llm_type')
        config['system_prompt'] = request.form.get('system_prompt')
        
        # Save updated config
        with open(app.config['MASTER_CONFIG'], 'w') as f:
            yaml.dump(config, f)
        
        # Reinitialize the master agent with the new config
        global master_agent
        master_agent = MasterAgent(
            config_path=app.config['MASTER_CONFIG'],
            agents_dir=app.config['AGENTS_DIR']
        )
        
        flash("Master agent configuration updated successfully", "success")
    except Exception as e:
        flash(f"Error updating master agent configuration: {str(e)}", "error")
    
    return redirect(url_for('index'))
@app.route('/agents')
def list_agents():
    agents = []
    for agent_name in master_agent.list_agents():
        agent = master_agent.agents[agent_name]
        agents.append({
            'name': agent.name,
            'description': agent.description
        })
    
    return render_template('agents.html', agents=agents, master_agent=master_agent)

@app.route('/agent/<name>')
def view_agent(name):
    if name in master_agent.agents:
        agent = master_agent.agents[name]
        return render_template('agent_detail.html', agent=agent, master_agent=master_agent, ToolRegistry=ToolRegistry)
    else:
        flash(f"Agent '{name}' not found", "error")
        return redirect(url_for('list_agents'))

@app.route('/agent/new', methods=['GET', 'POST'])
def new_agent():
    form = AgentForm()
    form.tools.choices = [(tool.name, tool.name) for tool in ToolRegistry.list_tools().values()]
    
    if form.validate_on_submit():
        # Create agent config dictionary
        agent_config = {
            'name': form.name.data,
            'description': form.description.data,
            'system_prompt': form.system_prompt.data,
            'llm': {
                'type': form.llm_type.data,
                'model': form.llm_model.data
            },
            'tools': form.tools.data
        }
        
        # Save agent to YAML file
        agent_path = os.path.join(app.config['AGENTS_DIR'], f"{form.name.data.lower()}.yaml")
        with open(agent_path, 'w') as f:
            yaml.dump(agent_config, f)
        
        # Add agent to master agent
        try:
            master_agent.add_agent(agent_path)
            flash(f"Agent '{form.name.data}' created successfully", "success")
            return redirect(url_for('list_agents'))
        except Exception as e:
            flash(f"Error creating agent: {str(e)}", "error")
    
    return render_template('agent_form.html', form=form, is_new=True, master_agent=master_agent)

@app.route('/agent/edit/<name>', methods=['GET', 'POST'])
def edit_agent(name):
    if name not in master_agent.agents:
        flash(f"Agent '{name}' not found", "error")
        return redirect(url_for('list_agents'))
    
    agent = master_agent.agents[name]
    form = AgentForm()
    form.tools.choices = [(tool.name, tool.name) for tool in ToolRegistry.list_tools().values()]
    
    if request.method == 'GET':
        # Pre-populate form with agent data
        form.name.data = agent.name
        form.description.data = agent.description
        form.system_prompt.data = agent.system_prompt
        form.llm_type.data = agent.config.get('llm', {}).get('type', 'openai')
        form.llm_model.data = agent.config.get('llm', {}).get('model', '')
        form.tools.data = [tool.name for tool in agent.tools]
    
    if form.validate_on_submit():
        # Update agent config
        agent_config = {
            'name': form.name.data,
            'description': form.description.data,
            'system_prompt': form.system_prompt.data,
            'llm': {
                'type': form.llm_type.data,
                'model': form.llm_model.data
            },
            'tools': form.tools.data
        }
        
        # Save updated agent to YAML file
        agent_path = os.path.join(app.config['AGENTS_DIR'], f"{name.lower()}.yaml")
        with open(agent_path, 'w') as f:
            yaml.dump(agent_config, f)
        
        # Reload the agent in the master agent
        try:
            master_agent.add_agent(agent_path)
            flash(f"Agent '{form.name.data}' updated successfully", "success")
            return redirect(url_for('list_agents'))
        except Exception as e:
            flash(f"Error updating agent: {str(e)}", "error")
    
    return render_template('agent_form.html', form=form, is_new=False, agent=agent, master_agent=master_agent)

@app.route('/tools')
def list_tools():
    tools = []
    for tool_name, tool in ToolRegistry.list_tools().items():
        tools.append({
            'name': tool.name,
            'description': tool.description
        })
    
    return render_template('tools.html', tools=tools, master_agent=master_agent)

@app.route('/tool/<name>')
def view_tool(name):
    tool = ToolRegistry.get_tool(name)
    if tool:
        schema = tool.get_schema()
        return render_template('tool_detail.html', tool=tool, schema=schema, master_agent=master_agent)
    else:
        flash(f"Tool '{name}' not found", "error")
        return redirect(url_for('list_tools'))

@app.route('/tool/new', methods=['GET', 'POST'])
def new_tool():
    form = ToolForm()
    
    if form.validate_on_submit():
        try:
            # Parse params
            params = {}
            if form.params.data:
                params = json.loads(form.params.data)
            
            # Import the tool class
            module_path, class_name = form.tool_type.data.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            ToolClass = getattr(module, class_name)
            
            # Create tool instance
            tool = ToolClass(name=form.name.data, description=form.description.data, **params)
            
            # Register tool
            ToolRegistry.register(tool)
            flash(f"Tool '{form.name.data}' created successfully", "success")
            return redirect(url_for('list_tools'))
        except Exception as e:
            flash(f"Error creating tool: {str(e)}", "error")
    
    return render_template('tool_form.html', form=form, is_new=True, master_agent=master_agent)

@app.route('/tool/edit/<name>', methods=['GET', 'POST'])
def edit_tool(name):
    tool = ToolRegistry.get_tool(name)
    if not tool:
        flash(f"Tool '{name}' not found", "error")
        return redirect(url_for('list_tools'))
    
    form = ToolForm()
    
    if request.method == 'GET':
        # Pre-populate form with tool data
        form.name.data = tool.name
        form.description.data = tool.description
        form.tool_type.data = tool.__class__.__module__ + '.' + tool.__class__.__name__
        # Try to extract params if available
        if hasattr(tool, 'params'):
            form.params.data = json.dumps(tool.params, indent=2)
    
    if form.validate_on_submit():
        try:
            # Parse params
            params = {}
            if form.params.data:
                params = json.loads(form.params.data)
            
            # Import the tool class
            module_path, class_name = form.tool_type.data.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            ToolClass = getattr(module, class_name)
            
            # Create updated tool instance
            updated_tool = ToolClass(name=form.name.data, description=form.description.data, **params)
            
            # Re-register tool
            ToolRegistry.register(updated_tool)
            flash(f"Tool '{form.name.data}' updated successfully", "success")
            return redirect(url_for('list_tools'))
        except Exception as e:
            flash(f"Error updating tool: {str(e)}", "error")
    
    return render_template('tool_form.html', form=form, is_new=False, tool=tool, master_agent=master_agent)

@app.route('/task_progress')
def task_progress():
    """Server-sent events endpoint for task completion notification."""
    
    def generate_sse():
        logger.info("Starting SSE connection for task progress")
        
        # Send initial connection message
        yield "retry: 5000\n"
        yield "event: message\n"
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to server'})}\n\n"
        
        # Listen for started and complete messages only
        while True:
            try:
                # Try to get an update from the queue with a 1-second timeout
                update = execution_updates.get(timeout=1)
                logger.info(f"Received update from queue: {update.get('type', 'unknown')}")
                
                # Only process started and complete messages
                if update.get('type') in ['started', 'complete', 'error']:
                    yield f"event: message\n"
                    yield f"data: {json.dumps(update)}\n\n"
                
                # If this is a completion message, end the stream
                if update.get('type') == 'complete':
                    logger.info("Received completion message, ending SSE stream")
                    yield f"event: message\n"
                    yield f"data: {json.dumps({'type': 'refresh', 'message': 'Task complete, refreshing page...'})}\n\n"
                    break
                    
                # If this is an error message, also end the stream
                if update.get('type') == 'error':
                    logger.info("Received error message, ending SSE stream")
                    break
                    
            except queue.Empty:
                # Send a ping every 2 seconds to keep the connection alive
                yield f"event: message\n"
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    
    # Create response with proper headers for SSE
    response = Response(generate_sse(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering for Nginx proxies
    response.headers['Connection'] = 'keep-alive'
    return response

if __name__ == '__main__':
    app.run(debug=True)