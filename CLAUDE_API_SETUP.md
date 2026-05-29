# PinPoint Claude API Setup

PinPoint now supports both **Ollama** (local LLM) and **Claude API** for inference.

## Quick Start

### Option 1: Use Claude API (Recommended for School Laptop)

```bash
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=sk-ant-...your-api-key...
python main.py
```

### Option 2: Use Ollama (Default)

```bash
export LLM_PROVIDER=ollama
# Ollama will auto-start on localhost:11434
python main.py
```

## Environment Variables

### Required for Claude API

- **`LLM_PROVIDER`**: Set to `claude` to use Claude API. Default is `ollama`.
- **`ANTHROPIC_API_KEY`**: Your Anthropic API key (starts with `sk-ant-`)

### Optional

- **`ANTHROPIC_MODEL`**: Claude model to use (default: `claude-3-5-sonnet-20241022`)
  - Other options: `claude-3-5-haiku-20241022`, `claude-3-opus-20250219`, etc.

### Ollama-specific

- **`OLLAMA_BASE_URL`**: Ollama API endpoint (default: `http://localhost:11434/v1`)
- **`OLLAMA_MODEL`**: Model to use (default: `qwen2.5-coder:14b`)

## Getting Your API Key

1. Go to https://console.anthropic.com/
2. Sign in or create an account
3. Create a new API key in the dashboard
4. Copy the key (looks like `sk-ant-...`)

## Windows Setup Example

Create a `.env` file in the Pinpoint directory:

```
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

Then in PowerShell:

```powershell
# Load environment from .env file
Get-Content .env | ForEach-Object {
    if ($_ -and -not $_.StartsWith('#')) {
        $key, $value = $_ -split '=', 2
        [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim())
    }
}
python main.py
```

Or use a batch file:

```batch
set LLM_PROVIDER=claude
set ANTHROPIC_API_KEY=sk-ant-...your-key...
python main.py
```

## Linux/Mac Setup

Add to your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=sk-ant-...your-api-key...
```

Then reload:

```bash
source ~/.bashrc
python main.py
```

## Performance Notes

- **Claude API**: Instant responses (cloud-based), costs per-API-call
- **Ollama (qwen2.5-coder:14b)**: Slower but free (runs locally)
- **Claude Haiku** (`claude-3-5-haiku`): Fastest Claude model, lower cost
- **Claude Sonnet** (default): Good balance of speed and intelligence

## Switching Between Providers

You can switch between Ollama and Claude API by simply changing the `LLM_PROVIDER` environment variable:

```bash
# Use Claude
export LLM_PROVIDER=claude
python main.py

# Switch back to Ollama
export LLM_PROVIDER=ollama
python main.py
```

## Troubleshooting

**"ANTHROPIC_API_KEY environment variable not set"**
- Make sure you've exported the `ANTHROPIC_API_KEY` with your actual API key
- Check that `LLM_PROVIDER=claude` is set

**"Invalid API key"**
- Verify your API key starts with `sk-ant-`
- Check if your key has expired or been revoked in the Anthropic dashboard

**Claude API is slow**
- Try switching to `claude-3-5-haiku` (faster, cheaper)
- Network latency may cause delays — this is expected with cloud APIs

**Ollama won't start**
- Make sure Ollama is installed: https://ollama.ai
- Run manually: `ollama serve`
- Check if port 11434 is available

## How It Works

PinPoint internally detects which LLM provider to use and adapts:
- **Ollama requests** use OpenAI-compatible API format
- **Claude API requests** use Anthropic SDK with automatic format conversion
- **Streaming responses** work seamlessly with both providers
