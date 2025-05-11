# OpenManus Enhanced

This is an enhanced version of the OpenManus project with improved support for multiple LLM providers, including:

- OpenAI (GPT models)
- Azure OpenAI
- Anthropic (Claude models)
- Google (Gemini models)
- Mistral AI
- Ollama (local models)

## Key Enhancements

1. **Multi-Provider Support**: Seamlessly use any major LLM provider with a unified interface.
2. **Simplified Configuration**: Easy provider switching through the config file.
3. **Provider-Specific Adapters**: Properly formats requests and responses for each provider.
4. **Streaming Support**: Streaming responses for all supported providers.
5. **Tool/Function Calling**: Advanced capabilities with providers that support it.

## Installation

Follow the same installation process as the original OpenManus:

### Method 1: Using conda

```bash
conda create -n open_manus python=3.12
conda activate open_manus
git clone https://github.com/your-username/OpenManus-Enhanced.git
cd OpenManus-Enhanced
pip install -r requirements.txt
```

### Method 2: Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/your-username/OpenManus-Enhanced.git
cd OpenManus-Enhanced
uv venv
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Configuration

Configuration has been enhanced to support multiple LLM providers more easily:

1. Copy the example configuration:

```bash
cp config/config.example.toml config/config.toml
```

2. Edit your configuration file with your preferred provider and API keys:

```toml
# Global configuration
[llm]
provider = "openai"  # Set your default provider here
model = "gpt-4o"
api_key = "sk-..."

# Provider-specific configurations
[openai]
model = "gpt-4o"
api_key = "sk-..."

[anthropic]
model = "claude-3-5-sonnet"
api_key = "sk-ant-..."

[google]
model = "gemini-pro"
api_key = "YOUR_API_KEY"

[mistral]
model = "mistral-large-latest"
api_key = "YOUR_API_KEY"

[ollama]
model = "llama3:latest"
base_url = "http://localhost:11434"
```

## Usage

Usage remains the same as the original OpenManus:

```bash
python main.py
```

Or for the unstable version:

```bash
python run_flow.py
```

## Provider-Specific Notes

### OpenAI
- Full support for all features including tools/function calling
- Streaming support

### Anthropic (Claude)
- High-quality completions
- Support for streaming
- Limited tool/function calling

### Google (Gemini)
- Advanced multimodal capabilities
- Note that system messages are passed as user messages with `[system]` prefix

### Mistral AI
- High-performance models with low latency
- Full support for streaming

### Ollama
- Local model support for privacy and offline use
- Reduced functionality compared to cloud providers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Same as the original OpenManus project - MIT License.
