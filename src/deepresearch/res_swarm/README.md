# ResSwarm - Research Swarm Intelligence

A comprehensive supervisor-worker architecture for evaluating LLM pairs on complex research tasks using the DeepResearch Bench benchmark. Features support for **98 different model combinations** across OpenAI, Anthropic, Fireworks, and SGLang Modal endpoints.

## 🚀 Quick Start

```bash
# 1. Setup environment
cd res_swarm
pip install -r requirements.txt

# 2. Configure API keys (already in your .bashrc)
# Your API keys should be available:
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $FIREWORKS_API_KEY

# 3. Test comprehensive model support
python test_integration.py

# 4. Run experiments with any combination
python scripts/run_resswarm.py --supervisor gpt-4o-latest --worker qwen2.5-3b-modal -n 1
python scripts/run_resswarm.py --supervisor claude-3-5-sonnet-latest --worker llama3.2-3b-modal -n 1
python scripts/run_resswarm.py --supervisor llama4-maverick-fireworks --worker qwen2.5-14b-modal -n 1
```

## 🤖 Comprehensive Model Support

### 📡 **14 Supervisor Models Available**

#### **OpenAI Supervisors (3 models)**
- `gpt-4o` - Standard GPT-4o model
- `gpt-4o-latest` - **Latest GPT-4o (2024-11-20)**
- `gpt-4o-mini` - GPT-4o Mini model

#### **Anthropic Supervisors (5 models)** 
- `claude-3-opus` - Claude 3 Opus (most capable)
- `claude-3-sonnet` - Claude 3 Sonnet
- `claude-3-haiku` - Claude 3 Haiku (fastest)
- `claude-3-5-sonnet` - Claude 3.5 Sonnet (specific date)
- `claude-3-5-sonnet-latest` - **Latest Claude 3.5 Sonnet (Claude 4 equivalent)**

#### **Fireworks Supervisors (6 models)**
- `llama-v3p3-70b-fireworks` - Llama 3.3 70B (latest)
- `llama-v3p1-405b-fireworks` - Llama 3.1 405B (largest)
- `llama-v3p1-8b-fireworks` - Llama 3.1 8B
- `llama4-maverick-fireworks` - **Llama4 Maverick** (cutting-edge)
- `deepseek-v3-fireworks` - DeepSeek V3
- `qwen2p5-72b-fireworks` - Qwen2.5 72B

### 🔄 **16 Worker Models Available**

#### **SGLang Modal Workers (7 models)** - *No API keys required*
- `llama3.2-1b-modal` - Llama 3.2 1B (fastest)
- `llama3.2-3b-modal` - Llama 3.2 3B
- `llama3.1-8b-modal` - Llama 3.1 8B
- `qwen2.5-1.5b-modal` - Qwen2.5 1.5B
- `qwen2.5-3b-modal` - **Qwen2.5 3B** (recommended)
- `qwen2.5-7b-modal` - Qwen2.5 7B
- `qwen2.5-14b-modal` - Qwen2.5 14B (largest)

#### **Local Workers (9 models)** - *Local model servers*
- `llama3.2`, `llama3.2:3b`, `llama3.1`, `mixtral`, `phi3` (Ollama)
- `llama3.2-lemonade` (AMD via Lemonade)
- `llama3.2-tokasaurus` (NVIDIA via Tokasaurus)
- `gpt-3.5-turbo`, `claude-3-haiku-worker` (Cloud workers for comparison)

## 🎯 **98 Total Combinations!**
**14 supervisors × 7 SGLang Modal workers = 98 unique experiments**

## 📋 Usage Examples

### Model Discovery
```bash
# List all available models
python scripts/run_resswarm.py --list-models

# Test model configurations
python test_integration.py

# Check environment setup
python scripts/run_resswarm.py --check-env
```

### Recommended Experiments

#### **🥇 Premium Combinations** 
```bash
# Latest OpenAI + Qwen2.5 3B
python scripts/run_resswarm.py --supervisor gpt-4o-latest --worker qwen2.5-3b-modal -n 5

# Latest Claude + Llama 3.2 3B  
python scripts/run_resswarm.py --supervisor claude-3-5-sonnet-latest --worker llama3.2-3b-modal -n 5

# Llama4 Maverick + Qwen2.5 3B
python scripts/run_resswarm.py --supervisor llama4-maverick-fireworks --worker qwen2.5-3b-modal -n 5
```

#### **🚀 Large Model Combinations**
```bash
# Largest Llama + Largest Qwen
python scripts/run_resswarm.py --supervisor llama-v3p1-405b-fireworks --worker qwen2.5-14b-modal -n 3

# Claude Opus + Qwen2.5 7B
python scripts/run_resswarm.py --supervisor claude-3-opus --worker qwen2.5-7b-modal -n 3

# Qwen2.5 72B Supervisor + Llama 3.1 8B Worker
python scripts/run_resswarm.py --supervisor qwen2p5-72b-fireworks --worker llama3.1-8b-modal -n 3
```

#### **💰 Cost-Effective Combinations**
```bash
# GPT-4o Mini + Small Qwen
python scripts/run_resswarm.py --supervisor gpt-4o-mini --worker qwen2.5-1.5b-modal -n 10

# Claude Haiku + Llama 3.2 1B
python scripts/run_resswarm.py --supervisor claude-3-haiku --worker llama3.2-1b-modal -n 10
```

### Batch Experiments
```bash
# Run multiple experiments
python scripts/run_resswarm.py -es gpt-4o-latest claude-3-5-sonnet-latest llama4-maverick-fireworks -n 5

# All enabled predefined experiments
python scripts/run_resswarm.py --max-queries 20

# Specific query range
python scripts/run_resswarm.py --supervisor gpt-4o-latest --worker qwen2.5-3b-modal --start-from 10 -n 10
```

## 🔧 Configuration

### Environment Setup

Your API keys are already configured in `.bashrc`. Verify with:
```bash
echo $OPENAI_API_KEY      # For OpenAI models
echo $ANTHROPIC_API_KEY   # For Anthropic models  
echo $FIREWORKS_API_KEY   # For Fireworks models
```

### Modal Endpoints

SGLang Modal workers require you to set up your own Modal endpoints. Update the URLs in `configs/model_config.py`:
```python
# Example: Replace "YOUR_MODAL_ENDPOINT_HERE" with your actual Modal endpoint
"qwen2.5-7b-modal": ModelConfig(
    provider=ModelProvider.SGLANG_MODAL,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
    temperature=0.7,
    max_tokens=2000
),
```

### Web Search Integration

ResSwarm uses **SerpAPI** for web search and **Jina Reader** for content extraction (API keys required):
- Built-in web scraping and content extraction
- No external dependencies beyond basic Python packages
- Automatically available to all worker models

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│    Supervisor       │    │       Worker        │    │   Web Search    │
│ (OpenAI/Anthropic/  │───▶│ (SGLang Modal/     │◄──▶│    Tools        │
│    Fireworks)       │    │  Local Models)     │    │                 │
│                     │    │                     │    │ • SerpAPI       │
│ • Task decomp       │    │ • Subquery exec     │    │ • Web Scraping  │
│ • Final synthesis   │    │ • Web research      │    │ • Content Parse │
└─────────────────────┘    └─────────────────────┘    └─────────────────┘
```

### Protocol Flow (max_rounds=1)

1. **Task Decomposition**: Supervisor breaks research task into focused subqueries
2. **Worker Research**: Worker processes subqueries using SerpAPI search and Jina Reader extraction
3. **Final Synthesis**: Supervisor combines worker findings into comprehensive research report

## 📊 Output & Evaluation

### Result Formats

1. **Detailed Results** (`experiments/experiment_name.jsonl`):
   ```json
   {
     "id": 1,
     "prompt": "research_query", 
     "article": "final_response_with_citations",
     "conversation_log": [...],
     "usage_stats": {...}
   }
   ```

2. **DeepResearch Bench Compatible**:
   ```bash
   # Copy results for evaluation
   python scripts/run_resswarm.py --copy-to-eval experiment_name
   
   # Run RACE/FACT evaluation
   cd ../deep_research_bench
   bash run_benchmark.sh
   ```

### Performance Recommendations

#### **Quality Hierarchy (Supervisors)**
🥇 **Tier 1**: `gpt-4o-latest`, `claude-3-5-sonnet-latest`, `llama-v3p1-405b-fireworks`
🥈 **Tier 2**: `gpt-4o`, `claude-3-opus`, `llama4-maverick-fireworks`  
🥉 **Tier 3**: `claude-3-sonnet`, `qwen2p5-72b-fireworks`, `deepseek-v3-fireworks`

#### **Speed vs Quality (Workers)**
⚡ **Fastest**: `llama3.2-1b-modal`, `qwen2.5-1.5b-modal`
⚖️ **Balanced**: `llama3.2-3b-modal`, `qwen2.5-3b-modal`
🎯 **Quality**: `llama3.1-8b-modal`, `qwen2.5-7b-modal`
🚀 **Premium**: `qwen2.5-14b-modal`

## 🛠️ Development

### Adding New Models

1. **Add to ModelProvider enum** in `configs/model_config.py`
2. **Add model configuration**:
   ```python
   "new-model": ModelConfig(
       provider=ModelProvider.FIREWORKS,
       model_name="accounts/fireworks/models/new-model",
       api_key_env="FIREWORKS_API_KEY"
   )
   ```
3. **Update client factory** if needed
4. **Test integration** with `python test_integration.py`

### Project Structure

```
res_swarm/
├── src/                          # Core implementation
│   ├── clients/                  # Custom model clients
│   │   ├── fireworks.py         # Fireworks AI client
│   │   ├── sglang_modal.py      # SGLang Modal client
│   │   └── base.py              # Base client class
│   ├── deepres_minion.py        # Supervisor-worker protocol
│   ├── deepres_runner.py        # Experiment orchestrator
│   └── deepres_tools.py         # SerpAPI search and Jina Reader tools
├── configs/                      # Configuration
│   ├── model_config.py          # All 98 model combinations
│   └── environment_template.env # Modal URLs & API keys
├── scripts/                      # CLI interfaces
├── tests/                        # Test suite
├── test_integration.py          # Comprehensive model test
├── requirements.txt             # All dependencies
└── README.md                    # This file
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing dependencies**: `pip install -r requirements.txt`
2. **API key errors**: Check your `.bashrc` exports
3. **Modal endpoint timeouts**: Modal endpoints may have cold starts
4. **Import errors**: Ensure you're in `res_swarm/` directory

### Debug Commands

```bash
# Test all model configurations
python test_integration.py

# Check specific model
python -c "from configs.model_config import get_config_manager; print(get_config_manager().get_supervisor_config('gpt-4o-latest'))"

# Validate environment
python scripts/run_resswarm.py --check-env
```

## 📈 Performance Tips

### Cost Optimization
- **SGLang Modal workers** don't charge per token (use these!)
- **Mini supervisors**: `gpt-4o-mini`, `claude-3-haiku`  
- **Small workers**: `qwen2.5-1.5b-modal`, `llama3.2-1b-modal`
- **Limit queries**: Use `--max-queries` for testing

### Speed Optimization
- **Modal workers** have faster inference than local models
- **Smaller models** generally respond faster
- **Parallel experiments**: Run multiple experiments simultaneously

### Quality Optimization
- **Premium supervisors**: `gpt-4o-latest`, `claude-3-5-sonnet-latest`, `llama-v3p1-405b-fireworks`
- **Capable workers**: `qwen2.5-7b-modal`, `llama3.1-8b-modal`
- **Web search enabled** by default for comprehensive research

## 🎉 What's New

### Latest Features (v2.0)
✅ **98 Model Combinations** - Mix any supervisor with any worker
✅ **SGLang Modal Integration** - 8 modal worker configurations (requires your own Modal endpoints)
✅ **Fireworks AI Support** - 6 cutting-edge models including Llama4 Maverick  
✅ **Latest OpenAI/Anthropic** - GPT-4o latest + Claude 3.5 Sonnet latest  
✅ **Premium Search APIs** - SerpAPI (Google) and Jina Reader (PDF support) required  
✅ **Self-Contained Clients** - Independent of minions-factory  
✅ **Comprehensive Testing** - Full integration validation  

### Model Highlights
🆕 **Llama4 Maverick** via Fireworks  
🆕 **Qwen2.5 3B** via SGLang Modal  
🆕 **GPT-4o (2024-11-20)** latest version  
🆕 **Claude 3.5 Sonnet Latest** (Claude 4 equivalent)  

## 📚 Further Reading

- **Integration Test**: Run `python test_integration.py` for comprehensive validation
- **Model Configurations**: See `configs/model_config.py` for all 98 combinations  
- **Environment Template**: See `configs/environment_template.env` for setup
- **MinionS Protocol**: Original implementation in `../minions/` directory
- **DeepResearch Bench**: Evaluation framework in `../deep_research_bench/`

## 🤝 Contributing

1. Test new models with `python test_integration.py`
2. Add model configurations in `configs/model_config.py`
3. Extend client support in `src/clients/`
4. Update documentation and tests

---

**Ready to explore 98 different AI combinations for research tasks!** 🚀

Choose your supervisor, pick your worker, and let ResSwarm conduct comprehensive research at scale.