#!/bin/bash

echo "Starting Qwen2.5 7B experiments..."
echo "Supervisor: gpt-4o"
echo "Workers: qwen2.5-7b"
echo "Research Mode: multi_query"
echo "Tasks: 20, Runs per task: 5"
echo "============================================="

# Run with qwen2.5-7b-modal
echo ""
echo "Starting experiment 1/1: qwen2.5-7b-modal"
python run_full_experiment.py \
    --supervisor gpt-4o \
    --worker qwen2.5-7b-modal \
    --research-mode multi_query \
    --experiment-name "gpt4o_qwen7b_multiquery" \
    --n-tasks 20 \
    --n-runs-per-task 5 \
    2>&1 | tee gpt4o_qwen7b_multiquery.log

echo ""
echo "All experiments completed!"
echo "Logs saved to:"
echo "  - gpt4o_qwen7b_multiquery.log"
echo ""
echo "Results will be available in:"
echo "  - outputs/experiments/gpt4o_qwen7b_multiquery/"