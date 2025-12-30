"""
Model Configuration System for DeepResearch Experiments

Provides provider-agnostic configuration for predictor-compressor model pairs
and manages API keys and model parameters.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import sys

# Add minions to path for importing clients - using relative path from project root
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "minions"))

# Import will be done dynamically to avoid typing issues
# from minions.clients.base import MinionsClient


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    TOGETHER = "together"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    LEMONADE = "lemonade"
    TOKASAURUS = "tokasaurus"
    SGLANG_MODAL = "sglang_modal"
    FIREWORKS = "fireworks"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    additional_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class ExperimentConfig:
    """Configuration for a predictor-compressor experiment."""
    name: str
    predictor_config: ModelConfig
    compressor_config: ModelConfig
    description: Optional[str] = None
    enabled: bool = True


class ModelConfigManager:
    """
    Manages model configurations and creates client instances.
    """
    
    def __init__(self):
        """Initialize with predefined model configurations."""
        self.predictor_configs = self._get_predictor_configs()
        self.compressor_configs = self._get_compressor_configs()
        self.experiment_configs = self._get_experiment_configs()
    
    def _get_predictor_configs(self) -> Dict[str, ModelConfig]:
        """Define available predictor model configurations."""
        return {
            # OpenAI models
            "gpt-4o": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                temperature=0.6,
                max_tokens=16000
            ),
            "gpt-4o-mini": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                temperature=0.6,
                max_tokens=16000
            ),
            
            # Anthropic models
            "claude-3-opus": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "claude-3-sonnet": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "claude-3-haiku": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "claude-3-5-sonnet": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "claude-3-5-sonnet-latest": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-latest",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            
            # Gemini models
            "gemini-pro": ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name="gemini-pro",
                api_key_env="GEMINI_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "gemini-1.5-pro": ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name="gemini-1.5-pro",
                api_key_env="GEMINI_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            
            # Together AI models
            "llama-3-405b": ModelConfig(
                provider=ModelProvider.TOGETHER,
                model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                api_key_env="TOGETHER_API_KEY",
                temperature=0.6,
                max_tokens=16000
            ),
            "llama-3-70b": ModelConfig(
                provider=ModelProvider.TOGETHER,
                model_name="meta-llama/Llama-3-70b-chat-hf",
                api_key_env="TOGETHER_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "mixtral-8x7b": ModelConfig(
                provider=ModelProvider.TOGETHER,
                model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                api_key_env="TOGETHER_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            
            # Groq models
            "llama-3-70b-groq": ModelConfig(
                provider=ModelProvider.GROQ,
                model_name="llama3-70b-8192",
                api_key_env="GROQ_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            
            # DeepSeek models
            "deepseek-chat": ModelConfig(
                provider=ModelProvider.DEEPSEEK,
                model_name="deepseek-chat",
                api_key_env="DEEPSEEK_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            
            # Fireworks models - comprehensive list from minions-factory
            "llama-v3p3-70b-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/llama-v3p3-70b-instruct",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "llama-v3p1-70b-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/llama-v3p1-70b-instruct",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "llama-v3p1-405b-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/llama-v3p1-405b-instruct",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "llama-v3p1-8b-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/llama-v3p1-8b-instruct",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "llama4-maverick-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/llama4-maverick-instruct-basic",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "deepseek-v3-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/deepseek-v3-0324",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
            "qwen2p5-72b-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/qwen2p5-72b-instruct",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.6,
                max_tokens=4000
            ),
        }
    
    def _get_worker_configs(self) -> Dict[str, ModelConfig]:
        """Define available worker model configurations."""
        return {
            # Ollama local models
            "llama3.2": ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="llama3.2",
                temperature=0.7,
                max_tokens=2000
            ),
            "llama3.2:3b": ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="llama3.2:3b",
                temperature=0.7,
                max_tokens=2000
            ),
            "llama3.1": ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="llama3.1",
                temperature=0.7,
                max_tokens=2000
            ),
            "mixtral": ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="mixtral",
                temperature=0.7,
                max_tokens=2000
            ),
            "phi3": ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="phi3",
                temperature=0.7,
                max_tokens=2000
            ),
            
            # Tokasaurus local models (NVIDIA)
            "llama3.2-tokasaurus": ModelConfig(
                provider=ModelProvider.TOKASAURUS,
                model_name="llama3.2",
                base_url_env="TOKASAURUS_BASE_URL",
                temperature=0.7,
                max_tokens=2000
            ),
            
            # SGLang Modal models - users need to provide their own Modal endpoints
            "llama3.2-1b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="meta-llama/Llama-3.2-1B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "llama3.2-3b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="meta-llama/Llama-3.2-3B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "llama3.1-8b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "qwen2.5-0.5b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "qwen2.5-1.5b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "qwen2.5-3b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="Qwen/Qwen2.5-3B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "qwen2.5-7b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            "qwen2.5-14b-modal": ModelConfig(
                provider=ModelProvider.SGLANG_MODAL,
                model_name="Qwen/Qwen2.5-14B-Instruct",
                additional_params={"base_url": "YOUR_MODAL_ENDPOINT_HERE"},
                temperature=0.7,
                max_tokens=2000
            ),
            
            # Cloud models as workers (for comparison)
            "gpt-4o": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                temperature=0.7,
                max_tokens=2000
            ),
            "gpt-3.5-turbo": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY",
                temperature=0.7,
                max_tokens=2000
            ),
            "claude-3-haiku-worker": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.7,
                max_tokens=2000
            ),
            "llama-v3p1-405b-fireworks": ModelConfig(
                provider=ModelProvider.FIREWORKS,
                model_name="accounts/fireworks/models/llama-v3p1-405b-instruct",
                api_key_env="FIREWORKS_API_KEY",
                temperature=0.7,
                max_tokens=4000
            ),
        }
    
    def _get_experiment_configs(self) -> List[ExperimentConfig]:
        """Define experiment configurations for supervisor-worker pairs."""
        return [
            # High-end supervisor with local workers
            ExperimentConfig(
                name="gpt4o_llama32",
                supervisor_config=self.supervisor_configs["gpt-4o"],
                worker_config=self.worker_configs["llama3.2"],
                description="GPT-4o supervisor with Llama 3.2 local worker"
            ),
            ExperimentConfig(
                name="claude3opus_llama32",
                supervisor_config=self.supervisor_configs["claude-3-opus"],
                worker_config=self.worker_configs["llama3.2"],
                description="Claude 3 Opus supervisor with Llama 3.2 local worker"
            ),
            ExperimentConfig(
                name="gemini15pro_llama32",
                supervisor_config=self.supervisor_configs["gemini-1.5-pro"],
                worker_config=self.worker_configs["llama3.2"],
                description="Gemini 1.5 Pro supervisor with Llama 3.2 local worker"
            ),
            
            # Mid-tier supervisor with local workers
            ExperimentConfig(
                name="claude3sonnet_phi3",
                supervisor_config=self.supervisor_configs["claude-3-sonnet"],
                worker_config=self.worker_configs["phi3"],
                description="Claude 3 Sonnet supervisor with Phi-3 local worker"
            ),
            ExperimentConfig(
                name="gpt4omini_mixtral",
                supervisor_config=self.supervisor_configs["gpt-4o-mini"],
                worker_config=self.worker_configs["mixtral"],
                description="GPT-4o Mini supervisor with Mixtral local worker"
            ),
            
            # Cross-provider combinations
            ExperimentConfig(
                name="deepseek_llama32",
                supervisor_config=self.supervisor_configs["deepseek-chat"],
                worker_config=self.worker_configs["llama3.2"],
                description="DeepSeek supervisor with Llama 3.2 local worker"
            ),
            ExperimentConfig(
                name="llama70b_llama32",
                supervisor_config=self.supervisor_configs["llama-3-70b"],
                worker_config=self.worker_configs["llama3.2:3b"],
                description="Llama 3 70B supervisor with Llama 3.2 3B worker"
            ),
            
            # Cloud-cloud comparisons
            ExperimentConfig(
                name="gpt4o_gpt35",
                supervisor_config=self.supervisor_configs["gpt-4o"],
                worker_config=self.worker_configs["gpt-3.5-turbo"],
                description="GPT-4o supervisor with GPT-3.5-turbo worker",
                enabled=False  # Disabled by default (higher cost)
            ),
            ExperimentConfig(
                name="claude3opus_claude3haiku",
                supervisor_config=self.supervisor_configs["claude-3-opus"],
                worker_config=self.worker_configs["claude-3-haiku-worker"],
                description="Claude 3 Opus supervisor with Claude 3 Haiku worker",
                enabled=False  # Disabled by default (higher cost)
            ),
        ]
    
    def create_client(self, config: ModelConfig):
        """
        Create a client instance from a model configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Initialized client instance
            
        Raises:
            ValueError: If provider is not supported or required env vars are missing
        """
        # Get API key if required
        api_key = None
        if config.api_key_env:
            api_key = os.getenv(config.api_key_env)
            if not api_key:
                raise ValueError(f"Environment variable {config.api_key_env} not set")
        
        # Get base URL if specified
        base_url = None
        if config.base_url_env:
            base_url = os.getenv(config.base_url_env)
        
        # Create client based on provider
        if config.provider == ModelProvider.OPENAI:
            from minions.clients.openai import OpenAIClient
            return OpenAIClient(
                model_name=config.model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.ANTHROPIC:
            from minions.clients.anthropic import AnthropicClient
            return AnthropicClient(
                model_name=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.GEMINI:
            from minions.clients.gemini import GeminiClient
            return GeminiClient(
                model_name=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.TOGETHER:
            from res_swarm.src.clients.together import TogetherClient
            return TogetherClient(
                model_name=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.GROQ:
            from minions.clients.groq import GroqClient
            return GroqClient(
                model_name=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.DEEPSEEK:
            from minions.clients.deepseek import DeepSeekClient
            return DeepSeekClient(
                model_name=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.OLLAMA:
            from minions.clients.ollama import OllamaClient
            return OllamaClient(
                model_name=config.model_name,
                temperature=config.temperature,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.TOKASAURUS:
            from minions.clients.tokasaurus import TokasaurusClient
            return TokasaurusClient(
                model_name=config.model_name,
                base_url=base_url,
                temperature=config.temperature,
                **config.additional_params
            )
        
        elif config.provider == ModelProvider.SGLANG_MODAL:
            from res_swarm.src.clients.sglang_modal import SGLangModalClient
            # Use hardcoded URL from additional_params if base_url is None
            modal_url = base_url or config.additional_params.get("base_url")
            if not modal_url:
                raise ValueError(f"No base_url found for SGLang Modal model {config.model_name}")
            
            # Remove base_url from additional_params to avoid duplicate parameter
            additional_params = {k: v for k, v in config.additional_params.items() if k != "base_url"}
            
            return SGLangModalClient(
                model_name=config.model_name,
                api_base_url=modal_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **additional_params
            )
        
        elif config.provider == ModelProvider.FIREWORKS:
            from res_swarm.src.clients.fireworks import FireworksClient
            return FireworksClient(
                model_name=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.additional_params
            )
        
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def get_experiment_config(self, name: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration by name."""
        for config in self.experiment_configs:
            if config.name == name:
                return config
        return None
    
    def get_enabled_experiments(self) -> List[ExperimentConfig]:
        """Get all enabled experiment configurations."""
        return [config for config in self.experiment_configs if config.enabled]
    
    def get_predictor_config(self, name: str) -> Optional[ModelConfig]:
        """Get predictor model configuration by name."""
        return self.predictor_configs.get(name)
    
    def get_compressor_config(self, name: str) -> Optional[ModelConfig]:
        """Get compressor model configuration by name."""
        return self.compressor_configs.get(name)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """Get lists of available predictor and compressor models."""
        return {
            "predictors": list(self.predictor_configs.keys()),
            "compressors": list(self.compressor_configs.keys())
        }
    
    def create_custom_experiment(self, predictor_name: str, compressor_name: str, experiment_name: Optional[str] = None) -> Optional[ExperimentConfig]:
        """
        Create a custom experiment configuration from predictor and compressor names.
        
        Args:
            predictor_name: Name of predictor model
            compressor_name: Name of compressor model
            
        Returns:
            ExperimentConfig or None if models not found
        """
        predictor_config = self.get_predictor_config(predictor_name)
        compressor_config = self.get_compressor_config(compressor_name)
        
        if not predictor_config or not compressor_config:
            return None
        
        # Use provided experiment name or generate one
        if experiment_name is None:
            experiment_name = f"{predictor_name.replace('-', '').replace('.', '')}_" \
                             f"{compressor_name.replace('-', '').replace('.', '').replace(':', '')}_custom"
        
        description = f"Custom: {predictor_name} predictor with {compressor_name} compressor"
        
        return ExperimentConfig(
            name=experiment_name,
            predictor_config=predictor_config,
            compressor_config=compressor_config,
            description=description,
            enabled=True
        )
    
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate that required environment variables are set.
        
        Returns:
            Dictionary mapping provider to availability status
        """
        env_status = {}
        
        # Check API keys for cloud providers
        cloud_providers = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY", 
            "Gemini": "GEMINI_API_KEY",
            "Together": "TOGETHER_API_KEY",
            "Groq": "GROQ_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "SerpAPI": "SERPAPI_API_KEY",
            "Firecrawl": "FIRECRAWL_API_KEY",
        }
        
        for provider, env_var in cloud_providers.items():
            env_status[provider] = bool(os.getenv(env_var))
        
        # Check local model servers (these don't need API keys but should be running)
        local_providers = ["Ollama", "Lemonade", "Tokasaurus"]
        for provider in local_providers:
            # For now, assume they're available (would need to check if servers are running)
            env_status[provider] = True
        
        return env_status
    
    def print_environment_status(self):
        """Print the status of environment variables and model availability."""
        status = self.validate_environment()
        
        print("\n=== ENVIRONMENT STATUS ===")
        print("\nCloud Providers:")
        for provider in ["OpenAI", "Anthropic", "Gemini", "Together", "Groq", "DeepSeek"]:
            status_icon = "✅" if status.get(provider, False) else "❌"
            print(f"  {status_icon} {provider}")
        
        print("\nSearch Tools:")
        for tool in ["SerpAPI", "Firecrawl"]:
            status_icon = "✅" if status.get(tool, False) else "❌"
            print(f"  {status_icon} {tool}")
        
        print("\nLocal Model Servers:")
        for provider in ["Ollama", "Lemonade", "Tokasaurus"]:
            status_icon = "✅" if status.get(provider, False) else "❓"
            print(f"  {status_icon} {provider} (check if server is running)")
        
        print("\nEnabled Experiments:")
        enabled = self.get_enabled_experiments()
        for exp in enabled:
            print(f"  • {exp.name}: {exp.description}")
        
        print(f"\nTotal enabled experiments: {len(enabled)}")
        

# Convenience function to create a manager instance
def get_config_manager() -> ModelConfigManager:
    """Get a configured ModelConfigManager instance."""
    return ModelConfigManager()