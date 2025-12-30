# ResSwarm - Research Swarm Intelligence Makefile
# Provides convenient commands for common operations

.PHONY: help setup test run list check clean install

# Default target
help:
	@echo "ResSwarm - Research Swarm Intelligence"
	@echo "====================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup    - Run setup and validation"
	@echo "  make test     - Run test suite"
	@echo "  make check    - Check environment status"
	@echo "  make list     - List available experiments"
	@echo "  make run      - Run all enabled experiments (limited)"
	@echo "  make install  - Install Python dependencies"
	@echo "  make clean    - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS='-e gpt4o_llama32 -n 5'"
	@echo "  make run ARGS='--max-queries 10'"

# Setup and validation
setup:
	python scripts/setup_resswarm.py

# Run test suite
test:
	python scripts/run_tests.py

# Check environment
check:
	python scripts/run_resswarm.py --check-env

# List experiments
list:
	python scripts/run_resswarm.py --list-experiments

# Run experiments (with optional ARGS)
run:
	python scripts/run_resswarm.py $(ARGS)

# Install dependencies
install:
	pip install -e ../minions
	pip install firecrawl-py serpapi google-genai anthropic openai together groq mistralai

# Clean generated files
clean:
	rm -rf experiments/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

# Quick development cycle
dev: setup test
	@echo "✅ Development environment ready!"
	@echo "Run: make run ARGS='-e gpt4o_llama32 -n 2'"

# Production setup
prod: install setup
	@echo "✅ Production environment ready!"
	@echo "Configure .env file and run experiments"