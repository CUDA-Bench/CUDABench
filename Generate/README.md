# Generation

A parallelized framework for evaluating LLMs on CUDA programming tasks. This tool supports multiple API providers (DeepSeek, OpenAI, Google, Anthropic) and uses multiprocessing to accelerate the generation process.

## 1. Directory Structure

```text
.
├── main.py             # Entry point: handles arguments and process orchestration
├── config.py           # Configuration management (Dataclass & settings)
├── llm_api.py          # API client wrappers and call logic
├── prompt_builder.py   # Logic for constructing CUDA prompts
├── utils.py            # Helper functions (JSON IO, code extraction)
├── prompt.py           # Raw prompt templates
└── results/            # Output directory (auto-generated)

```

## 2. API Keys

Set your API keys as environment variables before running:

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
# export GOOGLE_API_KEY="..."
# export MINIMAX_API_KEY="..."

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your_key_here"

```

## 3. Basic Usage

You can run the script using the default configuration defined in `config.py`, or override settings via command-line arguments.

### Run with Defaults

```bash
python main.py

```

### Run with Custom Settings

**1. Using DeepSeek Reasoner (R1):**

```bash
python main.py \
  --api_option deepseek \
  --model_name deepseek-reasoner \
  --level 3 \
  --samples 3

```

**2. Using OpenAI GPT-4o:**

```bash
python main.py \
  --api_option openai \
  --model_name gpt-4o \
  --samples 3

```

### Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--api_option` | The API provider (`deepseek`, `openai`, `google`, `anthropic`). | `deepseek` |
| `--model_name` | The specific model ID (e.g., `gpt-4o`, `claude-3-5-sonnet`). | `deepseek-reasoner` |
| `--level` | Prompt detail level (`level1_prompt` to `level3_prompt`). | `level3_prompt` |
| `--samples` | Number of samples to generate per task (pass@k). | `3` |

## Output

Results are automatically saved to the `results/` directory with the following structure:
`Results/{api_option}/{model_name}_{level}_pass{samples}.jsonl`

The script automatically handles **resuming**: if you interrupt the process, restart it with the same arguments, and it will skip already completed tasks.