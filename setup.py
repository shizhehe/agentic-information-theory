from setuptools import setup, find_packages

setup(
    name="agentic-information-bottleneck",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama",  # for local LLM
        "streamlit==1.42.2",  # for the UI
        "openai",  # for OpenAI client
        "anthropic",  # for Anthropic client
        "together",  # for Together client
        "groq",  # for Groq client
        "requests",  # for API calls
        "tiktoken",  # for token counting
        "pymupdf",  # for PDF processing
        "st-theme",
        "mcp",
        "spacy",  # for PII extraction, worked on python 3.11 and not 3.13
        "rank_bm25",  # for smart retrieval
        "PyMuPDF",  # for PDF handling
        "firecrawl-py",  # for scraping urls
        "google-genai",  # for Gemini client
        "serpapi",  # for web search
        "google_search_results",  # for web search
        "psutil",
        "flask",  # for the worker server
        "orjson",
        "twilio",
        "pyjwt",  # for JWT utilities
        "torch",
        "nv-attestation-sdk",
        "nv-local-gpu-verifier",
        "azure-security-attestation",
        "azure-identity",
        "gitingest",
        # Core dependencies for Information Bottleneck research
        "pandas",  # for data manipulation
        "numpy",  # for numerical computations
        "pydantic",  # for data validation
        "tqdm",  # for progress bars
    ],
    extras_require={
        "experiments": [
            # Additional Information Bottleneck research dependencies
            "datasets",  # for HuggingFace datasets
            "transformers",  # for tokenizers and models
            "pydrantic",  # for configuration management
            "wandb",  # for experiment tracking
            "modal",  # for cloud deployment
            "scipy",  # for scientific computing
            "pyarrow",  # for feather file format
        ],
        "mlx": ["mlx-lm"],
        "csm-mlx": ["csm-mlx @ git+https://github.com/senstella/csm-mlx.git"],
    },
    author="Shizhe, Avanika, Ishan, Scott, Chris, and Dan",
    description="A package for running agentic information bottleneck protocols with local and remote LLMs",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "agentic-information-bottleneck=agentic_information_bottleneck_cli:main",
        ],
    },
)
