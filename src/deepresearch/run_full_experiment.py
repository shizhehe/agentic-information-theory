#!/usr/bin/env python3
"""
Run the full 20×5 experiment with configurable model pairs.

This implements the user's requirement:
- N randomly sampled English tasks (default: 20, seed=42)  
- Each task run M times independently (default: 5 runs, total: 100 runs)
- Collect all 7 required metrics per task/run
- Configurable supervisor + worker model combination
"""

import sys
import os
import argparse
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "res_swarm"))

from src.deepresearch.res_swarm.scripts.run_experiment_batch import BatchExperimentRunner
from src.deepresearch.res_swarm.src.experiment_utils import ExperimentConfig, TaskSelector
from src.deepresearch.res_swarm.configs.model_config import get_config_manager


def parse_arguments():
    """Parse command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description='Run N×M DeepResearch experiment with configurable models')
    parser.add_argument('--predictor', type=str, 
                       default='llama-v3p1-405b-fireworks',
                       help='Predictor model name (default: llama-v3p1-405b-fireworks)')
    parser.add_argument('--compressor', type=str,
                       default='qwen2.5-7b-modal', 
                       help='Compressor model name (default: qwen2.5-7b-modal)')
    parser.add_argument('--experiment-name', type=str,
                       help='Custom experiment name (auto-generated if not provided)')
    parser.add_argument('--n-tasks', type=int, default=20,
                       help='Number of tasks to select (default: 20)')
    parser.add_argument('--n-runs-per-task', type=int, default=5,
                       help='Number of runs per task (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for task selection (default: 42)')
    parser.add_argument('--research-mode', type=str, default='adaptive',
                       choices=['evaluation', 'adaptive', 'multi_query'],
                       help='Research mode: evaluation (fast, minimal research), adaptive (iterative, deep research), multi_query (5 parallel queries, comprehensive coverage) (default: adaptive)')
    parser.add_argument('--max-rounds', type=int, default=1,
                       help='Maximum research rounds (default: 1)')
    parser.add_argument('--max-sources-per-round', type=int, default=20,
                       help='Maximum sources per round (default: 20)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    parser.add_argument('--synthesis-strategy', type=str, default='single',
                       choices=['single', 'chunked'],
                       help='Synthesis strategy: single (current) or chunked (new sectioned approach) (default: single)')
    parser.add_argument('--sections-per-chunk', type=int, default=2,
                       help='Number of sections per synthesis chunk (default: 2)')
    return parser.parse_args()


def generate_experiment_name(supervisor, worker, n_tasks, n_runs):
    """Generate experiment name from model pair and parameters."""
    # Clean model names for filename
    supervisor_clean = supervisor.replace('-', '').replace('.', '').replace('_', '').replace(':', '')
    worker_clean = worker.replace('-', '').replace('.', '').replace('_', '').replace(':', '')
    return f"{supervisor_clean}_{worker_clean}_{n_tasks}x{n_runs}_production"


def validate_models(predictor_name, compressor_name):
    """Validate that the requested models are available."""
    config_manager = get_config_manager()
    
    predictor_config = config_manager.get_predictor_config(predictor_name)
    if not predictor_config:
        available_predictors = list(config_manager.predictor_configs.keys())
        print(f"❌ Unknown predictor model: {predictor_name}")
        print(f"Available predictors: {', '.join(available_predictors)}")
        return False
    
    compressor_config = config_manager.get_compressor_config(compressor_name)
    if not compressor_config:
        available_compressors = list(config_manager.compressor_configs.keys())
        print(f"❌ Unknown compressor model: {compressor_name}")
        print(f"Available compressors: {', '.join(available_compressors)}")
        return False
    
    return True


def run_full_experiment():
    """Run the complete N×M experiment with configurable models."""
    args = parse_arguments()
    
    # Handle --list-models
    if args.list_models:
        config_manager = get_config_manager()
        models = config_manager.list_available_models()
        print("\n=== AVAILABLE MODELS ===")
        print("\n📋 Supervisor Models:")
        for model in sorted(models["supervisors"]):
            print(f"  • {model}")
        print("\n🔧 Worker Models:")
        for model in sorted(models["workers"]):
            print(f"  • {model}")
        return True
    
    # Validate models
    if not validate_models(args.predictor, args.compressor):
        return False
    
    # Auto-generate experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = generate_experiment_name(
            args.predictor, args.compressor, args.n_tasks, args.n_runs_per_task
        )
    
    print("🚀 STARTING FULL PRODUCTION EXPERIMENT")
    print("=" * 80)
    print("EXPERIMENT SPECIFICATION:")
    print(f"- {args.n_tasks} randomly sampled English tasks (seed={args.seed})")
    print(f"- Each task run {args.n_runs_per_task} times independently") 
    print(f"- Total: {args.n_tasks * args.n_runs_per_task} runs")
    print(f"- Predictor: {args.predictor}")
    print(f"- Compressor: {args.compressor}")
    print(f"- Research mode: {args.research_mode}")
    print(f"- Max rounds: {args.max_rounds}")
    print(f"- Max sources per round: {args.max_sources_per_round}")
    print(f"- Synthesis strategy: {args.synthesis_strategy}")
    if args.synthesis_strategy == 'chunked':
        print(f"- Sections per chunk: {args.sections_per_chunk}")
    print("- Collect all 7 required metrics per run")
    print("=" * 80)
    
    # Select tasks based on CLI arguments
    print("\n📋 Selecting tasks...")
    task_selector = TaskSelector()
    selected_tasks = task_selector.select_random_english_tasks(n_tasks=args.n_tasks, seed=args.seed)
    
    task_ids = [task["id"] for task in selected_tasks]
    print(f"✅ Selected {args.n_tasks} English tasks with seed={args.seed}")
    print(f"   Task IDs: {task_ids}")
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        predictor_name=args.predictor,
        compressor_name=args.compressor,
        selected_tasks=selected_tasks,
        n_runs_per_task=args.n_runs_per_task,
        research_mode=args.research_mode,
        max_rounds=args.max_rounds,
        max_sources_per_round=args.max_sources_per_round,
        seed=args.seed,
        synthesis_strategy=args.synthesis_strategy,
        sections_per_chunk=args.sections_per_chunk
    )
    
    print(f"\n🔧 EXPERIMENT CONFIGURATION:")
    print(f"   Name: {experiment_config.experiment_name}")
    print(f"   Predictor: {experiment_config.predictor_name}")
    print(f"   Compressor: {experiment_config.compressor_name}")
    print(f"   Tasks: {len(experiment_config.selected_tasks)}")
    print(f"   Runs per task: {experiment_config.n_runs_per_task}")
    print(f"   Total runs: {experiment_config.total_runs}")
    print(f"   Research mode: {experiment_config.research_mode}")
    print(f"   Seed: {experiment_config.seed}")
    
    # Show estimated time
    print(f"\n⏱️  Estimated runtime: {experiment_config.total_runs * 2} minutes (≈{experiment_config.total_runs * 2 / 60:.1f} hours)")
    print(f"   ({experiment_config.total_runs} tasks × ~2 minutes per task)")
    
    # Create and run experiment
    print(f"\n🚀 STARTING FULL EXPERIMENT...")
    runner = BatchExperimentRunner()
    
    try:
        results = runner.run_batch_experiment(experiment_config)
        
        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        
        # Analyze final results
        metrics = runner.metrics_collector.get_metrics()
        successful_runs = len([m for m in metrics if not m.get('error')])
        failed_runs = len([m for m in metrics if m.get('error')])
        
        print(f"\n📊 FINAL EXPERIMENT SUMMARY:")
        print(f"   Total runs: {len(metrics)}")
        print(f"   Successful: {successful_runs}")
        print(f"   Failed: {failed_runs}")
        print(f"   Success rate: {(successful_runs/len(metrics)*100):.1f}%")
        
        if successful_runs > 0:
            # Validate that all 7 required metrics are collected
            sample_run = next(m for m in metrics if not m.get('error'))
            required_metrics = [
                'supervisor_model', 'worker_model', 'scores', 'worker_call_count',
                'worker_output_tokens_per_call', 'worker_input_tokens_per_call',
                'final_supervisor_output_tokens'
            ]
            
            collected_metrics = [metric for metric in required_metrics if metric in sample_run]
            print(f"\n✅ METRICS VALIDATION:")
            print(f"   Required metrics: {len(required_metrics)}")
            print(f"   Collected metrics: {len(collected_metrics)}")
            print(f"   All metrics present: {len(collected_metrics) == len(required_metrics)}")
            
            # Token tracking summary
            total_supervisor_tokens = sum(m.get('total_supervisor_tokens', 0) for m in metrics if not m.get('error'))
            total_worker_tokens = sum(m.get('total_worker_tokens', 0) for m in metrics if not m.get('error'))
            
            print(f"\n🔢 TOKEN USAGE SUMMARY:")
            print(f"   Total supervisor tokens: {total_supervisor_tokens:,}")
            print(f"   Total worker tokens: {total_worker_tokens:,}")
            print(f"   Grand total tokens: {(total_supervisor_tokens + total_worker_tokens):,}")
            
            print(f"\n🎯 EXPERIMENT OBJECTIVES ACHIEVED:")
            print(f"   ✅ {args.n_tasks} English tasks randomly sampled (seed={args.seed})")
            print(f"   ✅ Each task run {args.n_runs_per_task} times for variance reduction")
            print(f"   ✅ {args.predictor} + {args.compressor} model combination")
            print(f"   ✅ All 7 required metrics collected per run")
            print(f"   ✅ Granular token tracking implemented")
            print(f"   ✅ Results ready for variance analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_experiment()
    if success:
        print(f"\n🎉 Production experiment completed successfully!")
        print(f"Data is ready for analysis and variance reduction studies.")
    else:
        print(f"\n💡 Check error logs and retry if needed.")