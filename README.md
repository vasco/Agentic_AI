# Agentic AI Framework

Este projeto é um framework de agentes de IA que permite coordenar múltiplos agentes especializados para resolver tarefas complexas.

## Créditos e Licença

Este código e licença pertencem a **João Melo de Jesus**.

## Como executar

### Web Interface
```bash
python app.py
```

### Terminal
```bash
python main.py 'run-master' 'What is Agentic AI, and What Are Its Types? How Does It Benefit Financial Services? Write a report about it, save it in pdf format and send it to joao.melo.jesus@gmail.com'
```

## Instalação

### 1. Clone o repositório
```bash
git clone <repository-url>
cd Agentic_AI
```

### 2. Instale as dependências
Vais precisar de uns quantos pip install:
```bash
pip install -r requirements.txt
```

### 3. Configure as API Keys
Crie um arquivo `.env` na raiz do projeto e adicione suas chaves de API:

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

**Importante:** Você precisará colocar as api keys no .env conforme mostrado acima.

### 4. Dependências adicionais do sistema

Para a conversão de PDF, você pode precisar instalar o `wkhtmltopdf`:

#### Ubuntu/Debian:
```bash
sudo apt-get install wkhtmltopdf
```

#### macOS:
```bash
brew install wkhtmltopdf
```

#### Windows:
Baixe e instale de: https://wkhtmltopdf.org/downloads.html

## Estrutura do Projeto

- `app.py` - Interface web Flask
- `main.py` - Interface de linha de comando
- `agent_framework/` - Framework principal dos agentes
  - `master_agent.py` - Agente coordenador principal
  - `agent.py` - Classe base para agentes individuais
  - `tools/` - Ferramentas disponíveis para os agentes
  - `llm_providers/` - Provedores de modelos de linguagem
- `agents/` - Configurações dos agentes especializados
- `templates/` - Templates HTML para a interface web
- `static/` - Arquivos estáticos (CSS, JS)
- `output/` - Diretório para arquivos de saída (PDFs, etc.)

## Comandos Disponíveis

### Setup inicial
```bash
python main.py setup
```

### Criar novo agente
```bash
python main.py create-agent "Nome do Agente" "Descrição detalhada do que o agente deve fazer"
```

### Criar nova ferramenta
```bash
python main.py create-tool "nome_da_ferramenta" "Descrição da ferramenta"
```

### Executar agente único
```bash
python main.py run-agent config/agente.yaml "Sua consulta aqui"
```

### Executar master agent
```bash
python main.py run-master "Sua consulta complexa aqui"
```

### Iniciar interface web
```bash
python main.py web
```

## Funcionalidades

- **Agentes Especializados**: Pesquisador, Programador, Escritor
- **Coordenação Inteligente**: Master agent que coordena múltiplos agentes
- **Ferramentas Integradas**: 
  - Busca na web
  - Raspagem de websites
  - Envio de emails
  - Conversão Markdown para PDF
- **Interface Web**: Interface amigável para gerenciar agentes e tarefas
- **API Keys Seguras**: Configuração via variáveis de ambiente
- **Suporte a Múltiplos LLMs**: OpenAI, Anthropic, DeepSeek

## Exemplo de Uso

O framework pode ser usado para tarefas complexas como:
- Pesquisar informações sobre um tópico
- Escrever relatórios baseados na pesquisa
- Converter o relatório para PDF
- Enviar o resultado por email

Todo o processo é automatizado através da coordenação inteligente dos agentes especializados.

## Contato

Para dúvidas ou suporte, entre em contato com João Melo de Jesus: joao.melo.jesus@gmail.com 