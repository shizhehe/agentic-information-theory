# Deep Research System

**Deep Research** is a flexible predictor-compressor architecture for AI research tasks. Mix and match cloud and local LLMs to conduct comprehensive research using premium search infrastructure (SerpAPI + Jina Reader) and evaluate with DeepResearch Bench.

## Setup

### Prerequisites

1. **Clone DeepResearch Bench evaluation framework:**
   ```bash
   git clone https://github.com/Ayanami0730/deep_research_bench.git
   ```

2. **Install dependencies:**
   ```bash
   cd res_swarm
   pip install -r requirements.txt
   ```

### API Configuration

Configure the required API keys in your environment:

```bash
# Search and content extraction (required)
export SERPAPI_KEY="your-serpapi-key"      # Get from https://serpapi.com
export JINA_API_KEY="your-jina-key"        # Get from https://jina.ai

# Model providers (at least one predictor required)
export OPENAI_API_KEY="sk-..."             # Get from https://platform.openai.com
export ANTHROPIC_API_KEY="..."             # Get from https://console.anthropic.com
export FIREWORKS_API_KEY="..."             # Get from https://fireworks.ai

# Evaluation (required)
export GEMINI_API_KEY="..."                # Get from https://makersuite.google.com
```

Apply the configuration:
```bash
source ~/.bashrc  # or restart your terminal
```

---

## Running Experiments

### Basic Experiment

Run a standard experiment with 20 tasks × 5 runs:

```bash
python run_full_experiment.py \
    --predictor gpt-4o \
    --compressor qwen2.5-7b-modal \
    --research-mode multi_query \
    --experiment-name "my_experiment" \
    --n-tasks 20 \
    --n-runs-per-task 5
```

**Available research modes:**
- `evaluation` - Fast, minimal research
- `adaptive` - Iterative, deep research (default)
- `multi_query` - 5 parallel queries, comprehensive coverage

### Evaluation Pipeline

After experiments complete, process and evaluate results:

1. **Convert to RACE format:**
   ```bash
   python convert_to_race_format_all_runs.py --experiment-name "my_experiment"
   ```
   
2. **Run evaluation:**
   ```bash
   ./run_all_race_evaluations.sh my_experiment
   ```

**Output locations:**
- Raw results: `deep_research_bench/data/test_data/raw_data/`
- Evaluation results: `deep_research_bench/results/race/my_experiment_run{1-5}/`

---