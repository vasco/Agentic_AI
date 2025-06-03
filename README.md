# Agentic AI Framework

This project is an AI agent framework that allows coordinating multiple specialized agents to solve complex tasks.

## Credits and License

This code and license belong to **João Melo de Jesus**.

## How to Run

### Web Interface
```bash
python app.py
```

### Terminal
```bash
python main.py 'run-master' 'What is Agentic AI, and What Are Its Types? How Does It Benefit Financial Services? Write a report about it, save it in pdf format and send it to joao.melo.jesus@gmail.com'
```

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Agentic_AI
```

### 2. Install dependencies
You will need to install several packages:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root and add your API keys:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# DeepSeek API Configuration (if using DeepSeek models)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Web Search API Configuration
SERPER_API_KEY=your_serper_api_key_here

# Gmail Configuration (for email sending)
GMAIL_EMAIL=your_gmail_address@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password_here

# Flask Configuration
SECRET_KEY=your_secret_key_for_flask
```

**Important:** You need to set up the API keys in the .env file as shown above.

### 4. Additional system dependencies

For PDF conversion, you may need to install `wkhtmltopdf`:

#### Ubuntu/Debian:
```bash
sudo apt-get install wkhtmltopdf
```

#### macOS:
```bash
brew install wkhtmltopdf
```

#### Windows:
Download and install from: https://wkhtmltopdf.org/downloads.html

## Project Structure

- `app.py` - Flask web interface
- `main.py` - Command line interface
- `agent_framework/` - Main agent framework
  - `master_agent.py` - Main coordinator agent
  - `agent.py` - Base class for individual agents
  - `tools/` - Available tools for agents
  - `llm_providers/` - Language model providers
- `agents/` - Specialized agent configurations
- `templates/` - HTML templates for web interface
- `static/` - Static files (CSS, JS)
- `output/` - Directory for output files (PDFs, etc.)

## Available Commands

### Initial setup
```bash
python main.py setup
```

### Create new agent
```bash
python main.py create-agent "Agent Name" "Detailed description of what the agent should do"
```

### Create new tool
```bash
python main.py create-tool "tool_name" "Tool description"
```

### Run single agent
```bash
python main.py run-agent config/agent.yaml "Your query here"
```

### Run master agent
```bash
python main.py run-master "Your complex query here"
```

### Start web interface
```bash
python main.py web
```

## Features

- **Specialized Agents**: Researcher, Coder, Writer
- **Intelligent Coordination**: Master agent that coordinates multiple agents
- **Integrated Tools**: 
  - Web search
  - Website scraping
  - Email sending
  - Markdown to PDF conversion
- **Web Interface**: User-friendly interface to manage agents and tasks
- **Secure API Keys**: Configuration via environment variables
- **Multiple LLM Support**: OpenAI, Anthropic, DeepSeek

## Usage Example

The framework can be used for complex tasks such as:
- Research information on a topic
- Write reports based on research
- Convert reports to PDF
- Send results via email

The entire process is automated through intelligent coordination of specialized agents.

## Contact

For questions or support, contact João Melo de Jesus: joao.melo.jesus@gmail.com 