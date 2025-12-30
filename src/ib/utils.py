from typing import Any, Dict, Optional, List
import yaml
import requests
import numpy as np
import logging

from pydantic import Field
from pydrantic import BaseConfig
import wandb
import torch


class WandBConfig(BaseConfig):
    project: str = "butternut"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None


def prepare_wandb(
    config: WandBConfig,
    config_dict: Dict[str, Any],
):
    wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.name,
        tags=config.tags,
        notes=config.notes,
        group=config.group,
        config=config_dict,
    )


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Borrowed from pytorch lightning"""
    import random
    import numpy as np
    import torch

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)


def get_logger(name: str, **kwargs) -> logging.Logger:
    import sys

    handler = logging.StreamHandler(sys.stdout)  # Send logs to stdout
    handler.setLevel(logging.INFO)  # Set the log level for this handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )  # Customize format
    handler.setFormatter(formatter)

    logger = logging.getLogger(name, **kwargs)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def load_wandb_peft_model(
    artifact_path: str,
    base_model_name: str,
    project: str = "minion-finetune",
    entity: str = "hazy-research",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load a PEFT model from a W&B artifact.

    Args:
        artifact_path: Path to the artifact in format "entity/project/artifact_name:version"
                       or just "artifact_name:version" if entity and project are provided
        base_model_name: Name of the base model to load from HuggingFace
        project: W&B project name (if not included in artifact_path)
        entity: W&B entity name (if not included in artifact_path)
        device: Device to load the model on ("cuda", "cpu", etc.)

    Returns:
        tuple: (model, tokenizer)


    ### Sample Usage
    from mfactory.utils import load_wandb_peft_model

    model, tokenizer = load_wandb_peft_model(
        artifact_path="model-lr-2e-05-3.2-minion-toy:v56",
        base_model_name="meta-llama/Llama-3.2-3B-Instruct"
    )
    """
    import wandb
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Initialize W&B run
    run = wandb.init(
        project=project,
        entity=entity,
    )

    # Download the artifact
    artifact = run.use_artifact(artifact_path, type="model")
    adapter_dir = artifact.download()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    # Load adapter config
    adapter_config = PeftConfig.from_pretrained(adapter_dir)

    # Load adapter weights
    model = PeftModel.from_pretrained(model, adapter_dir)

    model = model.merge_and_unload()

    return model, tokenizer

def compute_bpc(log_probs, context, tokenizer):
    T = len(tokenizer.encode(context, return_tensors="pt"))
    return torch.sum(-log_probs) / T


QWEN_SUMMARY_URLs = [
    "https://hazyresearch--sglang-qwen2-5-1-5b-instruct-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen2-5-3b-instruct-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen2-5-7b-instruct-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen2-5-14b-instruct-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen2-5-32b-instruct-a100-80gb-serve.modal.run",
]

QWEN3_SUMMARY_URLs = [
    "https://hazyresearch--sglang-qwen3-1-7b-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen3-4b-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen3-8b-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-qwen3-14b-a100-80gb-serve.modal.run",
]

LLAMA_SUMMARY_URLs = [
    "https://hazyresearch--sglang-llama-3-2-1b-instruct-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-llama-3-2-3b-instruct-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-llama-3-1-8b-instruct-a100-80gb-serve.modal.run",
]

LLAMA_REMOTE_SIZE_URLs = [
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
]

GEMMA_SUMMARY_URLs = [
    "https://hazyresearch--sglang-gemma-3-1b-it-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-gemma-3-4b-it-a100-80gb-serve.modal.run",
    "https://hazyresearch--sglang-gemma-3-12b-it-a100-80gb-serve.modal.run",
]


def stats_from_url(u: str) -> float:
    # return (model_size, num_layers, num_heads)
    if u == QWEN_SUMMARY_URLs[0]:
        return 1.5, 28, 12
    elif u == QWEN_SUMMARY_URLs[1]:
        return 3.0, 36, 16
    elif u == QWEN_SUMMARY_URLs[2]:
        return 7.0, 28, 28
    elif u == QWEN_SUMMARY_URLs[3]:
        return 14.0, 48, 40
    elif u == QWEN_SUMMARY_URLs[4]:
        return 32.0, 64, 40
    elif u == LLAMA_SUMMARY_URLs[0]:
        return 1.0, 16, 32
    elif u == LLAMA_SUMMARY_URLs[1]:
        return 3.0, 28, 24
    elif u == LLAMA_SUMMARY_URLs[2]:
        return 8.0, 32, 8
    elif u == QWEN3_SUMMARY_URLs[0]:
        return 1.7, 28, 16
    elif u == QWEN3_SUMMARY_URLs[1]:
        return 4.0, 36, 32
    elif u == QWEN3_SUMMARY_URLs[2]:
        return 8.0, 36, 32
    elif u == QWEN3_SUMMARY_URLs[3]:
        return 14.0, 40, 40
    elif u == LLAMA_REMOTE_SIZE_URLs[0]:
        return 405.0, 128, 126
    elif u == LLAMA_REMOTE_SIZE_URLs[1]:
        return 70.0, 64, 80
    elif u == LLAMA_REMOTE_SIZE_URLs[2]:
        return 8.0, 32, 32
    elif u == GEMMA_SUMMARY_URLs[0]:
        return 1.0, 26, 4 
    elif u == GEMMA_SUMMARY_URLs[1]:
        return 4.0, 34, 16 
    elif u == GEMMA_SUMMARY_URLs[2]:
        return 12.0, 48, 16
    else:
        return np.nan, 0, 0

def moe_stats_from_url(u: str) -> float:
    # every model here has one shared expert, which we need to account for when computing FLOPs
    # return (num_experts_activated, num_shared_experts, total_num_experts, expert_hidden_dim, total_model_size, activated_model_size, number of layers, number of heads, num_expert_layers, model_hidden_dim)
    # num_expert_layers is total_num_layers / moe_layer_freq
    # shared expert is not part of the router, so we don't need to account for it
    if u == "https://hazyresearch--sglang-qwen3-30b-a3b-instruct-2507-a100-80-d0a0fa.modal.run":
        return 8, 0, 128, 768, 30.5, 3.3, 48, 32, 48, 2048
    elif u == "accounts/fireworks/models/kimi-k2-instruct":
        return 8, 1, 384, 2048, 1000, 32, 61, 64, 60, 7168
    elif u == "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507":
        return 8, 0, 128, 1536, 235, 22, 94, 64, 94, 4096
    elif u == "accounts/fireworks/models/llama4-scout-instruct-basic":
        return 1, 1, 16, 8192, 109, 17, 48, 40, 48, 5120
    elif u == "accounts/fireworks/models/llama4-maverick-instruct-basic":
        return 1, 1, 128, 8192, 400, 17, 48, 40, 24, 5120
    else:
        return np.nan, 0, 0, 0, 0, 0, 0, 0

def flops_per_token(model_size: float, num_layers: int, num_heads: int, num_input_tokens: int) -> float:
    return (2 * model_size * 10**9) + (2 * num_layers * num_heads * num_input_tokens)

def moe_flops_per_token(activated_model_size: float, num_layers: int, num_heads: int, num_input_tokens: int, num_expert_layers: int, num_total_experts: int, model_hidden_dim: int) -> float:
    return (2 * activated_model_size * 10**9) + (2 * num_layers * num_heads * num_input_tokens) + (2 * num_expert_layers * model_hidden_dim * num_total_experts)

def moe_total_flops_url(model_url: str, num_output_tokens: int, num_input_tokens: Optional[int] = None) -> float:
    """
    Based on https://arxiv.org/pdf/2001.08361

    # FLOPS per token ~= 2 * model_size_in_B ~= 2 * model_size * 10^9
    """
    num_experts_activated, num_shared_experts, total_num_experts, expert_hidden_dim, total_model_size, activated_model_size, num_layers, num_heads, num_expert_layers, model_hidden_dim = moe_stats_from_url(model_url)
    if num_input_tokens is None:
        num_input_tokens = 0
    return moe_flops_per_token(activated_model_size, num_layers, num_heads, num_input_tokens, num_expert_layers, total_num_experts, model_hidden_dim) * num_output_tokens

def total_flops(model_size: float, num_tokens: int, num_layers: int, num_heads: int, num_input_tokens: Optional[int] = None) -> float:
    if num_input_tokens is None:
        return flops_per_token(model_size, num_layers, num_heads, 0) * num_tokens
    return flops_per_token(model_size, num_layers, num_heads, num_input_tokens) * num_tokens

def total_flops_url(model_url: str, num_tokens: int, num_input_tokens: Optional[int] = None) -> float:
    """
    Based on https://arxiv.org/pdf/2001.08361

    # FLOPS per token ~= 2 * model_size_in_B ~= 2 * model_size * 10^9
    """
    model_size, num_layers, num_heads = stats_from_url(model_url)
    return total_flops(model_size, num_tokens, num_layers, num_heads, num_input_tokens)


# Define a function to print messages to both console and log file        
def check_sglang_model(local_model, port=30000):
    try:
        response = requests.get(f"http://localhost:{port}/get_model_info")
        if response.status_code == 200:
            running_model = response.json().get("model_path")
            if running_model != local_model:
                raise RuntimeError(f"SGLang server is running wrong model: {running_model} (expected {local_model})")
            print(f"Confirmed SGLang server running correct model: {local_model}")
        else:
            raise RuntimeError("Failed to get model info from SGLang server")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"SGLang server not running on port {port}")

