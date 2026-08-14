# HiddenBench

A benchmark for evaluating collective reasoning in multi-agent LLM systems, based on the Hidden Profile paradigm from social psychology.

## Overview

HiddenBench assesses whether groups of language models can successfully integrate distributed information when each agent holds asymmetric knowledge pieces. The benchmark is based on the research paper ["HiddenBench: Evaluating Collective Reasoning in Multi-Agent LLM Systems"](https://arxiv.org/html/2505.11556v2).

### Key Concepts

- **Hidden Profile**: A decision-making scenario where information is distributed such that no single agent can determine the correct answer alone
- **Shared Information**: Facts available to all agents
- **Unshared Information**: Unique facts given only to specific agents
- **Collective Reasoning**: The ability of agents to pool their knowledge through discussion

### Evaluation Protocol

1. **Pre-discussion Phase**: Each agent makes a decision based only on their individual information
2. **Discussion Phase**: Agents exchange messages over multiple rounds (stops early when consensus is reached)
3. **Post-discussion Phase**: Each agent makes a final decision after the discussion
4. **Full Profile Baseline**: Each agent receives ALL information to establish an upper bound

### Consensus Detection

During discussion, HiddenBench automatically detects when all agents have reached unanimous agreement on an answer. By default, the discussion stops early once consensus is reached, saving tokens and time. This behavior can be disabled with `--no-early-stop` to match the original paper's methodology of always running all 15 rounds.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hiddenbench.git
cd hiddenbench

# Install the package
pip install -e .

# For local Llama support (optional)
pip install -e ".[local]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### 1. Download Official HiddenBench Data

The official benchmark contains 65 tasks from the HuggingFace dataset:

```bash
# Create data directory and download
mkdir -p data/hiddenbench_official
curl -L -o data/hiddenbench_official/benchmark.json \
  https://huggingface.co/datasets/YuxuanLi1225/HiddenBench/resolve/main/benchmark.json
```

### 2. Configure API Keys

```bash
# Copy the example configuration
cp config.example.yaml config.yaml

# Edit config.yaml and add your API keys
```

### 3. Verify Tasks

```bash
# Check that tasks are loaded correctly
hiddenbench tasks

# Show detailed task list
hiddenbench tasks --verbose
```

### 4. Run the Benchmark

```bash
# Run with default settings (all 65 official tasks)
hiddenbench run

# Run with specific provider
hiddenbench run --provider openai --model gpt-4o

# Run only official tasks
hiddenbench run --official-only

# Run only your custom tasks
hiddenbench run --custom-only

# Run with custom options
hiddenbench run --agents 3 --rounds 15 --verbose

# Run all rounds without early stopping (like the original paper)
hiddenbench run --no-early-stop
```

### 5. View Results

Reports are saved to the `reports/` directory in both JSON and Markdown formats.

## Data Sources

HiddenBench supports two data sources:

### Official HiddenBench Data (65 tasks)

Downloaded from HuggingFace: [YuxuanLi1225/HiddenBench](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench)

- Stored in `data/hiddenbench_official/benchmark.json`
- Includes tasks from psychology studies and auto-generated scenarios
- Uses the official format with `shared_information` and `hidden_information` fields

### Custom Tasks

Create your own tasks in the `tasks/` directory using either:
- **Official format**: Compatible with HuggingFace data
- **Custom format**: Pre-divided information per agent

## Configuration

Edit `config.yaml` to customize the benchmark. See `config.example.yaml` for detailed documentation.

### Provider Options

| Provider | Description | API Key Required |
|----------|-------------|------------------|
| `anthropic` | Claude models (Claude 3 Opus, Sonnet, Haiku) | Yes |
| `openai` | GPT models (GPT-4o, GPT-4, GPT-3.5) | Yes |
| `grok` | xAI Grok models | Yes |
| `local` | Local Llama models via llama-cpp-python | No (model file required) |

### Benchmark Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `num_agents` | 4 | Number of agents in each scenario (matches original paper) |
| `num_rounds` | 15 | Number of discussion rounds |
| `temperature` | 0.7 | LLM sampling temperature |
| `run_full_profile` | true | Run Full Profile baseline |
| `data_dir` | `./data/hiddenbench_official` | Path to official data |
| `tasks_dir` | `./tasks` | Path to custom tasks |
| `use_official_data` | true | Include official 65 tasks |
| `use_custom_tasks` | true | Include custom tasks |

## Creating Custom Tasks

You can create your own Hidden Profile tasks to evaluate LLMs on domain-specific scenarios, test particular reasoning patterns, or extend the benchmark with additional challenges. Custom tasks are placed in the `tasks/` directory and can be used:

- **Alongside official tasks**: Set `use_official_data: true` and `use_custom_tasks: true` in config.yaml
- **Instead of official tasks**: Use `hiddenbench run --custom-only` or set `use_official_data: false`
- **Mixed runs**: Use CLI flags like `--official-only` or `--custom-only` to control which tasks run

### Designing Effective Hidden Profile Tasks

A good Hidden Profile task should:
1. Have a **clear correct answer** that requires integrating information from multiple agents
2. Distribute **hidden information** such that no single agent can determine the correct answer alone
3. Include some **shared information** that all agents know (to establish common ground)
4. Have **plausible distractor options** that might seem correct with incomplete information

### Official Format (recommended)

Compatible with HuggingFace data format. Hidden information is automatically distributed across agents:

```json
{
  "id": 1,
  "name": "my_scenario",
  "description": "Scenario description for agents...",
  "shared_information": [
    "Fact known by all agents"
  ],
  "hidden_information": [
    "Hidden fact 1",
    "Hidden fact 2",
    "Hidden fact 3"
  ],
  "possible_answers": ["Option A", "Option B", "Option C"],
  "correct_answer": "Option B"
}
```

### Custom Format

Pre-divided information per agent:

```json
{
  "id": "example_001",
  "name": "Example Decision Task",
  "description": "Scenario description for agents...",
  "options": ["Option A", "Option B", "Option C"],
  "correct_answer": "Option B",
  "shared_info": [
    {"content": "Fact known by all agents", "is_shared": true}
  ],
  "unshared_info": [
    [{"content": "Fact only Agent 1 knows", "is_shared": false}],
    [{"content": "Fact only Agent 2 knows", "is_shared": false}],
    [{"content": "Fact only Agent 3 knows", "is_shared": false}]
  ]
}
```

### Creating Tasks

**Interactive CLI** (guided creation):
```bash
hiddenbench create-task --output tasks/my_task.json
```

**Manual creation**: Create a JSON file in the `tasks/` directory following either format above.

**Validation tips**:
- Run `hiddenbench tasks --verbose` to verify your task loads correctly
- Test with `hiddenbench run --custom-only` to run only your custom tasks
- Check that agents with partial information tend to choose wrong answers (the Hidden Profile effect)

## CLI Commands

```bash
# Show configured tasks and data sources
hiddenbench tasks
hiddenbench tasks --verbose
hiddenbench tasks --official-only

# Run the benchmark
hiddenbench run [OPTIONS]
hiddenbench run --official-only
hiddenbench run --custom-only
hiddenbench run --no-early-stop  # Run all rounds (like original paper)

# List available providers
hiddenbench list-providers

# Create a new task interactively
hiddenbench create-task --output path/to/task.json

# Initialize a new project
hiddenbench init
```

### Run Command Options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to configuration file |
| `--provider`, `-p` | LLM provider (anthropic, openai, grok, local) |
| `--model`, `-m` | Model to use |
| `--agents`, `-a` | Number of agents (default: 4) |
| `--rounds`, `-r` | Number of discussion rounds (default: 15) |
| `--num-tasks`, `-n` | Number of tasks to run (default: 1) |
| `--all` | Run all available tasks |
| `--seed` | Random seed for task selection |
| `--no-early-stop` | Don't stop when consensus is reached |
| `--no-full-profile` | Skip Full Profile baseline |
| `--official-only` | Only run official HiddenBench tasks |
| `--custom-only` | Only run custom tasks |
| `--verbose`, `-v` | Show detailed output |

## Output Reports

Reports are automatically saved to the `reports/` directory after each benchmark run. Both JSON and Markdown formats are generated with matching filenames.

### Example Report

See [`reports/results-opus-4.5.md`](reports/results-opus-4.5.md) for a complete example report from running Claude Opus 4.5 on all 63 official tasks. Key results from this run:

| Metric | Value |
|--------|-------|
| Pre-Discussion Accuracy | 13.1% |
| Post-Discussion Accuracy | 89.3% |
| Full Profile Accuracy | 95.6% |
| Information Integration Gain | +76.2% |
| Tasks with Consensus | 56/63 |
| Avg Consensus Round | 3.7 |

### JSON Report Format

The JSON report (`results-*.json`) contains structured data for programmatic analysis:

```json
{
  "metadata": {
    "timestamp": "2026-02-01T14:54:27.133367",
    "version": "0.1.0",
    "status": "complete",
    "completed_at": "2026-02-01T17:22:25.623706"
  },
  "config": {
    "provider": "anthropic",
    "model": "claude-opus-4-5-20251101",
    "num_agents": 4,
    "num_rounds": 15,
    "temperature": 0.7,
    "run_full_profile": true,
    "token_usage": {
      "input": 8996525,
      "output": 250424,
      "total": 9246949
    }
  },
  "summary": {
    "num_tasks": 63,
    "average_pre_accuracy": 0.131,
    "average_post_accuracy": 0.893,
    "average_full_profile_accuracy": 0.956,
    "average_information_gain": 0.762
  },
  "results": [
    {
      "task": { "id": "1", "name": "task_name", ... },
      "pre_discussion_decisions": [...],
      "post_discussion_decisions": [...],
      "discussion_history": [...],
      "full_profile_decisions": [...],
      "consensus_round": 3,
      "stopped_early": true,
      "metrics": { ... }
    }
  ]
}
```

**Key sections:**
- `metadata`: Timestamp, version, and completion status
- `config`: Full benchmark configuration including token usage
- `summary`: Aggregate metrics across all tasks
- `results`: Per-task results including:
  - Task definition (scenario, options, correct answer)
  - Pre-discussion decisions with rationales
  - Full discussion transcript (all rounds, all agents)
  - Post-discussion decisions with rationales
  - Full Profile baseline decisions (if enabled)
  - Consensus detection results
  - Initial prompts shown to each agent

### Markdown Report Format

The Markdown report (`results-*.md`) is human-readable and includes:

- **Summary table**: Aggregate metrics at a glance
- **Per-task sections** with:
  - Scenario description
  - Correct answer
  - Metrics table (pre/post accuracy, consensus round, etc.)
  - Initial prompts (showing what each agent saw)
  - Pre-discussion decisions (each agent's initial vote and rationale)
  - Discussion transcript (organized by round, showing all agent messages)
  - Post-discussion decisions (final votes after discussion)
  - Full Profile results (baseline with complete information)

The Markdown format is ideal for reviewing individual task results, understanding agent reasoning, and identifying patterns in successful vs. failed information integration.

## Metrics

| Metric | Description |
|--------|-------------|
| Pre-Discussion Accuracy | Proportion correct before discussion (Yᵖʳᵉ) |
| Post-Discussion Accuracy | Proportion correct after discussion (Yᵖᵒˢᵗ) |
| Full Profile Accuracy | Accuracy with complete information (Yᶠᵘˡˡ) |
| Information Integration Gain | Post - Pre accuracy (improvement from discussion) |
| Collective Reasoning Gap | Full Profile - Post accuracy (room for improvement) |
| Consensus Round | Round number when all agents first agreed (if applicable) |
| Tasks with Consensus | Number of tasks where unanimous agreement was reached |

## Project Structure

```
hiddenbench/
├── src/hiddenbench/
│   ├── __init__.py
│   ├── benchmark.py      # Core evaluation logic
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration management
│   ├── prompts.py        # LLM prompt templates
│   ├── report.py         # Report generation
│   ├── task.py           # Task definitions
│   └── providers/        # LLM provider plugins
│       ├── __init__.py
│       ├── base.py       # Provider interface
│       ├── anthropic.py  # Anthropic/Claude
│       ├── openai.py     # OpenAI/GPT
│       ├── grok.py       # xAI/Grok
│       ├── local.py      # Local Llama
│       └── factory.py    # Provider factory
├── data/
│   └── hiddenbench_official/  # Official benchmark data (65 tasks)
│       ├── .gitattributes
│       ├── README.md
│       └── benchmark.json
├── tasks/                # Custom task definitions
├── reports/              # Output reports
├── config.example.yaml   # Example configuration
├── pyproject.toml        # Package configuration
└── README.md
```

## License

MIT License

## Acknowledgments

This implementation is based on the HiddenBench benchmark developed by Yuxuan Li, Aoi Naito, and Hirokazu Shirado. The official benchmark data is hosted on HuggingFace.

### Data Source

The official benchmark data (65 tasks) is sourced from:

**HuggingFace Dataset:** [YuxuanLi1225/HiddenBench](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench)

This dataset is released under the MIT License (as indicated in the dataset's metadata) and contains decision-making scenarios grounded in the Hidden Profile paradigm from social psychology research.

## Citation

If you use HiddenBench in your research, please cite the original paper:

**Paper:** Yuxuan Li, Aoi Naito, and Hirokazu Shirado. "HiddenBench: Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks." arXiv preprint arXiv:2505.11556, 2025.

```bibtex
@article{li2025hiddenbench,
  title={HiddenBench: Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks},
  author={Li, Yuxuan and Naito, Aoi and Shirado, Hirokazu},
  journal={arXiv preprint arXiv:2505.11556},
  year={2025},
  url={https://arxiv.org/abs/2505.11556},
  doi={10.48550/arXiv.2505.11556}
}
```

### HuggingFace Dataset Citation

```bibtex
@dataset{li2025hiddenbench_data,
  title={HiddenBench Dataset},
  author={Li, Yuxuan},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/YuxuanLi1225/HiddenBench}
}
```
