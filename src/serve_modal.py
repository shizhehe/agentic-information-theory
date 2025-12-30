"""
Deploy SGLang server on Modal with support for Llama, Qwen, and Gemma models.

Features:
- One file, no external dependencies
- Explicit, commented steps (image build → volumes → web_server)  
- Standard SGLang OpenAI-style REST endpoints (`/v1/chat/completions`)
- Autoscaling with configurable GPU types and counts

USAGE
-----
# Interactive test (tears down when you quit):
modal run serve_modal.py

# Long-running autoscaling service:
modal deploy serve_modal.py

# Point any OpenAI-compatible client to the printed URL:
curl $WEB_URL/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"any","messages":[{"role":"user","content":"Hello!"}]}'
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
MODEL_REV = os.environ.get("MODEL_REV", None)  # Optional HF revision/commit ID

# Hardware configuration  
GPU_TYPE = os.environ.get("GPU_TYPE", "a100-80gb")  # "h100", "a100-80gb", "l40s"
GPU_COUNT = int(os.environ.get("GPU_COUNT", 2))  # Tensor-parallel shards

# Server configuration
PORT = 8000
MINUTES = 60  # Seconds to minutes conversion helper
SGL_VERSION = "0.4.6.post2"

# Scaling configuration
MIN_CONTAINERS = 0
MAX_CONTAINERS = 6
ALLOW_CONCURRENT_INPUTS = 256

# Memory and performance
MEM_FRACTION_STATIC = 0.7  # Memory fraction for static allocation
TOOL_CALL_PARSER = (
    "llama3" if "Llama" in MODEL_PATH 
    else "qwen25" if "Qwen" in MODEL_PATH 
    else None
)

print(f"MODEL_PATH: {MODEL_PATH}")
print(f"GPU_TYPE: {GPU_TYPE}, GPU_COUNT: {GPU_COUNT}")

# ---------------------------------------------------------------------------
# Container image build
# ---------------------------------------------------------------------------

BASE_CUDA = "12.4.0"

# Core dependencies for SGLang server
CORE_DEPENDENCIES = [
    "transformers",
    "numpy", 
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
    "torch==2.6.0",
    f"sglang=={SGL_VERSION}",
    "orjson",
    "uvicorn",
    "uvloop",
]

# Additional ML and CUDA dependencies
ML_DEPENDENCIES = [
    "zmq",
    "psutil", 
    "decord",
    "httpx",
    "scipy",
    "pillow",
    "tqdm",
    "sentencepiece",
    "accelerate",
    "safetensors",
    "xformers",
    "sgl-kernel==0.1.1",
    "compressed-tensors",
    "dill",
    "partial-json-parser",
    "einops",
    "torchao",
    "nvidia-cuda-nvrtc-cu12",
    "cuda-python",
    "xgrammar==0.1.19",
    "flashinfer-python==0.2.5",
]

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{BASE_CUDA}-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git")
    .pip_install(*(CORE_DEPENDENCIES + ML_DEPENDENCIES))
)

# Persistent storage volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal app and web server
# ---------------------------------------------------------------------------

app_name = f"sglang-{Path(MODEL_PATH).name.replace('/','-')}-{GPU_TYPE}-3"
app = modal.App(app_name)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    timeout=20 * MINUTES,
    scaledown_window=int(1 * MINUTES),
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("your-secrets")],
)
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def serve():
    """
    Launch SGLang server inside the container.
    Modal web server forwards external traffic to this port.
    """
    # System diagnostics
    subprocess.run("nvidia-smi", shell=True, check=False)
    subprocess.run("apt install libnuma1 libnuma-dev", shell=True, check=False)

    # Build SGLang command
    cmd = [
        "python", "-m", "sglang.launch_server",
        f"--model-path={MODEL_PATH}",
        f"--port={PORT}",
        f"--dp={GPU_COUNT}",
        f"--mem-fraction-static={MEM_FRACTION_STATIC}",
        "--host=0.0.0.0",
        "--attention-backend=fa3",
    ]
    
    # Model-specific options
    if MODEL_REV:
        cmd.append(f"--revision={MODEL_REV}")
    if "gemma-3" in MODEL_PATH:
        cmd.append("--attention-backend=fa3")
    if TOOL_CALL_PARSER is not None:
        cmd.append(f"--tool-call-parser={TOOL_CALL_PARSER}")

    # Launch server (non-blocking)
    print("Launching SGLang with:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)
    print("SGLang server launched successfully!")


# ---------------------------------------------------------------------------
# Testing functions
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(prompt: str = "Explain the moon landings in one sentence."):
    """
    Local test entrypoint - spins up container, tests server, then exits.
    Useful for smoke testing.
    
    Args:
        prompt: Test prompt to send to the server
    """
    import requests

    payload = {
        "model": "any",
        "messages": [{"role": "user", "content": prompt}],
    }

    t0 = time.time()
    resp = requests.post(
        serve.web_url + "/v1/chat/completions",
        json=payload,
        timeout=240
    )
    resp.raise_for_status()
    
    print(json.dumps(resp.json(), indent=2))
    print(f"\n[Served in {time.time() - t0:.1f}s]")
