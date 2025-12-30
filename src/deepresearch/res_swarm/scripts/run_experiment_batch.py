#!/usr/bin/env python3
"""
Batch Experiment Runner for ResSwarm

Runs multiple repetitions of selected tasks with detailed metrics tracking
for variance analysis and comprehensive evaluation.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "res_swarm"))
sys.path.insert(0, str(project_root / "minions"))

# Import ResSwarm components
from src.deepresearch.res_swarm.src.deepres_minion import DeepResearchMinion
from src.deepresearch.res_swarm.src.deepres_tools import WebSearchTool
from src.deepresearch.res_swarm.src.experiment_utils import (
    create_20_task_experiment_config, 
    MetricsCollector,
    VarianceAnalyzer,
    ExperimentConfig
)
from src.deepresearch.res_swarm.configs.model_config import get_config_manager


class BatchExperimentRunner:
    """
    Runner for batch experiments with detailed metrics tracking.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/experiments",
        web_search_enabled: bool = True
    ):
        """
        Initialize batch experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
            web_search_enabled: Whether to enable web search
        """
        # Resolve paths relative to project root
        self.output_dir = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.web_search_enabled = web_search_enabled
        
        # Initialize components
        self.config_manager = get_config_manager()
        if web_search_enabled:
            self.search_tool = WebSearchTool()
        else:
            self.search_tool = None
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()
        
        print(f"Batch Experiment Runner initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Web search enabled: {web_search_enabled}")
    
    def run_batch_experiment(
        self,
        experiment_config: ExperimentConfig,
        resume_from_run: Optional[int] = None,
        enable_auto_retry: bool = True,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> Dict[str, Any]:
        """
        Run a batch experiment with multiple runs per task.
        
        Args:
            experiment_config: Configuration for the experiment
            resume_from_run: Optional run number to resume from (0-indexed)
            
        Returns:
            Dictionary with experiment results and metrics
        """
        print(f"\n{'='*80}")
        print(f"STARTING BATCH EXPERIMENT: {experiment_config.experiment_name}")
        print(f"Predictor: {experiment_config.predictor_name}")
        print(f"Compressor: {experiment_config.compressor_name}")
        print(f"Tasks: {len(experiment_config.selected_tasks)}")
        print(f"Runs per task: {experiment_config.n_runs_per_task}")
        print(f"Total runs: {experiment_config.total_runs}")
        print(f"Research mode: {experiment_config.research_mode}")
        print(f"Max rounds: {experiment_config.max_rounds}")
        print(f"Seed: {experiment_config.seed}")
        print(f"{'='*80}")
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_config.experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        experiment_config.save(exp_dir)
        
        # Create clients
        try:
            predictor_config = self.config_manager.get_predictor_config(experiment_config.predictor_name)
            compressor_config = self.config_manager.get_compressor_config(experiment_config.compressor_name)
            
            if not predictor_config:
                raise ValueError(f"Unknown predictor model: {experiment_config.predictor_name}")
            if not compressor_config:
                raise ValueError(f"Unknown compressor model: {experiment_config.compressor_name}")
            
            predictor_client = self.config_manager.create_client(predictor_config)
            compressor_client = self.config_manager.create_client(compressor_config)
            
        except Exception as e:
            error_msg = f"Failed to create clients: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Initialize minion
        minion = DeepResearchMinion(
            supervisor_client=supervisor_client,
            worker_client=worker_client,
            web_search_enabled=self.web_search_enabled,
            log_dir=str(exp_dir / "logs"),
            research_mode=experiment_config.research_mode,
            max_rounds=experiment_config.max_rounds,
            max_sources_per_round=experiment_config.max_sources_per_round,
            worker_batch_size=10,
            synthesis_strategy=getattr(experiment_config, 'synthesis_strategy', 'single'),
            sections_per_chunk=getattr(experiment_config, 'sections_per_chunk', 2)
        )
        
        # Setup progress tracking
        start_time = time.time()
        total_runs = experiment_config.total_runs
        run_number = 0
        successful_runs = 0
        failed_runs = 0
        
        # Resume logic
        if resume_from_run is not None:
            run_number = resume_from_run
            print(f"📂 Resuming from run {run_number + 1}/{total_runs}")
        
        # Create raw results directory
        raw_results_dir = exp_dir / "raw_results"
        raw_results_dir.mkdir(exist_ok=True)
        
        # Run experiments
        with tqdm(total=total_runs, initial=run_number, desc="Experiment Progress", unit="run") as pbar:
            for task_idx, task in enumerate(experiment_config.selected_tasks):
                for run_idx in range(experiment_config.n_runs_per_task):
                    # Check if we should skip this run (resume logic)
                    if run_number < (resume_from_run or 0):
                        run_number += 1
                        pbar.update(1)
                        continue
                    
                    task_id = task["id"]
                    task_prompt = task["prompt"]
                    run_id = f"task_{task_id}_run_{run_idx + 1}"
                    
                    pbar.set_description(f"Task {task_id} Run {run_idx + 1}/{experiment_config.n_runs_per_task}")
                    
                    print(f"\n--- Run {run_number + 1}/{total_runs}: {run_id} ---")
                    print(f"Task: {task_prompt[:100]}...")
                    
                    try:
                        # Retry logic for transient errors
                        result = None
                        
                        for attempt in range(max_retries if enable_auto_retry else 1):
                            try:
                                # Run the experiment
                                result = minion(
                                    task=task_prompt,
                                    query_id=task_id,
                                    logging_id=run_id
                                )
                                
                                # Check if this is the specific error we want to retry
                                if (result.get("final_answer", "").startswith("Error processing task: max_workers must be greater than 0") 
                                    and attempt < max_retries - 1):
                                    print(f"⚠️ Encountered max_workers error, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(retry_delay)
                                    continue
                                
                                # Success - break out of retry loop
                                break
                                
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    print(f"⚠️ Exception occurred: {str(e)}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(retry_delay)
                                    continue
                                else:
                                    # Final attempt failed, re-raise the exception
                                    raise
                        
                        # After retry loop, save the result
                        if result:
                            # Save individual run result
                            run_result_file = raw_results_dir / f"{run_id}.json"
                            run_result = {
                                "run_id": run_id,
                                "task_id": task_id,
                                "run_number": run_number + 1,
                                "task_index": task_idx,
                                "run_index": run_idx,
                                "task_prompt": task_prompt,
                                "supervisor_model": experiment_config.supervisor_name,
                                "worker_model": experiment_config.worker_name,
                                "final_answer": result["final_answer"],
                                "conversation_log": result["conversation_log"],
                                "supervisor_calls": result.get("supervisor_calls", []),
                                "worker_calls": result.get("worker_calls", []),
                                "timing": result.get("timing", {}),
                                "success": True,
                                "error": None
                            }
                            
                            with open(run_result_file, 'w', encoding='utf-8') as f:
                                json.dump(run_result, f, indent=2, ensure_ascii=False)
                            
                            # Record metrics using new granular token tracking format
                            self.metrics_collector.record_run(
                                run_id=run_id,
                                task_id=task_id,
                                task_prompt=task_prompt,
                                result=result,
                                task_score=None  # Will be filled by evaluation later
                            )
                            
                            successful_runs += 1
                            print(f"✅ {run_id} completed successfully")
                            
                            # Print granular token tracking summary for this run
                            supervisor_calls = result.get("supervisor_calls", [])
                            worker_calls = result.get("worker_calls", [])
                            
                            supervisor_total = sum(call.get('input_tokens', 0) + call.get('output_tokens', 0) for call in supervisor_calls)
                            worker_total = sum(call.get('input_tokens', 0) + call.get('output_tokens', 0) for call in worker_calls)
                            
                            print(f"   Supervisor: {len(supervisor_calls)} calls, {supervisor_total} total tokens")
                            print(f"   Worker: {len(worker_calls)} calls, {worker_total} total tokens")
                            print(f"   Worker calls: {[call.get('output_tokens', 0) for call in worker_calls]} output tokens")
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"❌ Error in {run_id}: {error_msg}")
                        
                        # Save error result
                        run_result_file = raw_results_dir / f"{run_id}_error.json"
                        error_result = {
                            "run_id": run_id,
                            "task_id": task_id,
                            "run_number": run_number + 1,
                            "task_index": task_idx,
                            "run_index": run_idx,
                            "task_prompt": task_prompt,
                            "supervisor_model": experiment_config.supervisor_name,
                            "worker_model": experiment_config.worker_name,
                            "success": False,
                            "error": error_msg
                        }
                        
                        with open(run_result_file, 'w', encoding='utf-8') as f:
                            json.dump(error_result, f, indent=2, ensure_ascii=False)
                        
                        # Record error in metrics using new format
                        empty_result = {
                            "supervisor_model": experiment_config.supervisor_name,
                            "worker_model": experiment_config.worker_name,
                            "supervisor_calls": [],
                            "worker_calls": [],
                            "worker_call_count": 0,
                            "timing": {},
                            "final_answer": ""
                        }
                        
                        self.metrics_collector.record_run(
                            run_id=run_id,
                            task_id=task_id,
                            task_prompt=task_prompt,
                            result=empty_result,
                            task_score=None,
                            error=error_msg
                        )
                        
                        failed_runs += 1
                    
                    run_number += 1
                    pbar.update(1)
                    
                    # Save checkpoint every 10 runs
                    if run_number % 10 == 0:
                        self._save_checkpoint(exp_dir, run_number)
        
        total_time = time.time() - start_time
        
        # Save final metrics
        metrics_file = exp_dir / "detailed_metrics.json"
        self.metrics_collector.save_metrics(metrics_file)
        
        # Generate variance analysis - commented out as not needed for small runs
        # analyzer = VarianceAnalyzer(self.metrics_collector.get_metrics())
        # variance_file = exp_dir / "variance_analysis.json"
        # analyzer.save_analysis(variance_file)
        
        # Create experiment summary
        summary = {
            "experiment_config": experiment_config.to_dict(),
            "results": {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "total_time": total_time,
                "avg_time_per_run": total_time / total_runs if total_runs > 0 else 0
            },
            # "overall_statistics": analyzer.get_overall_statistics(),  # Commented out with variance analyzer
            "timestamp": experiment_config.timestamp
        }
        
        # Save summary
        summary_file = exp_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"🎉 BATCH EXPERIMENT COMPLETED: {experiment_config.experiment_name}")
        print(f"Success rate: {successful_runs}/{total_runs} ({summary['results']['success_rate']:.1%})")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"Average time per run: {summary['results']['avg_time_per_run']:.1f}s")
        print(f"Results saved to: {exp_dir}")
        print(f"Detailed metrics: {metrics_file}")
        # print(f"Variance analysis: {variance_file}")  # Commented out with variance analyzer
        print(f"{'='*80}")
        
        return summary
    
    def _save_checkpoint(self, exp_dir: Path, run_number: int):
        """Save a checkpoint with current progress."""
        checkpoint = {
            "completed_runs": run_number,
            "timestamp": time.time(),
            "metrics_count": len(self.metrics_collector.get_metrics())
        }
        
        checkpoint_file = exp_dir / f"checkpoint_run_{run_number}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"📝 Checkpoint saved: {checkpoint_file}")


def main():
    """Command-line interface for batch experiments."""
    parser = argparse.ArgumentParser(description="Run batch ResSwarm experiments with variance analysis")
    
    parser.add_argument(
        "--experiment-name", "-e",
        type=str,
        required=True,
        help="Name for the experiment"
    )
    parser.add_argument(
        "--supervisor", "-s",
        type=str,
        required=True,
        help="Supervisor model name"
    )
    parser.add_argument(
        "--worker", "-w",
        type=str,
        required=True,
        help="Worker model name"
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=20,
        help="Number of tasks to select (default: 20)"
    )
    parser.add_argument(
        "--n-runs-per-task",
        type=int,
        default=5,
        help="Number of runs per task (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task selection (default: 42)"
    )
    parser.add_argument(
        "--research-mode",
        type=str,
        choices=["evaluation", "adaptive", "multi_query"],
        default="adaptive",
        help="Research mode: evaluation (fast, minimal research), adaptive (iterative, deep research), multi_query (5 parallel queries, comprehensive coverage) (default: adaptive)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum research rounds (default: 5)"
    )
    parser.add_argument(
        "--max-sources-per-round",
        type=int,
        default=20,
        help="Maximum sources per round (default: 20)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/experiments",
        help="Output directory (default: outputs/experiments)"
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search"
    )
    parser.add_argument(
        "--resume-from-run",
        type=int,
        help="Resume from specific run number (1-indexed)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment and exit"
    )
    parser.add_argument(
        "--enable-auto-retry",
        action="store_true",
        default=True,
        help="Enable automatic retry for transient errors (default: True)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed tasks (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay in seconds between retries (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BatchExperimentRunner(
        output_dir=args.output_dir,
        web_search_enabled=not args.no_web_search
    )
    
    # Handle utility commands
    if args.check_env:
        runner.config_manager.print_environment_status()
        return
    
    if args.list_models:
        models = runner.config_manager.list_available_models()
        print("\n=== AVAILABLE MODELS ===")
        print("\n📋 Supervisor Models:")
        for model in sorted(models["supervisors"]):
            config = runner.config_manager.get_supervisor_config(model)
            provider = config.provider.value if config else "unknown"
            print(f"  • {model} ({provider})")
        
        print("\n🔧 Worker Models:")
        for model in sorted(models["workers"]):
            config = runner.config_manager.get_worker_config(model)
            provider = config.provider.value if config else "unknown"
            print(f"  • {model} ({provider})")
        return
    
    # Validate supervisor and worker models
    if not runner.config_manager.get_supervisor_config(args.supervisor):
        print(f"❌ Unknown supervisor model: {args.supervisor}")
        print("Use --list-models to see available options")
        return
    
    if not runner.config_manager.get_worker_config(args.worker):
        print(f"❌ Unknown worker model: {args.worker}")
        print("Use --list-models to see available options")
        return
    
    try:
        # Create experiment configuration
        print(f"🔄 Creating experiment configuration...")
        experiment_config = create_20_task_experiment_config(
            experiment_name=args.experiment_name,
            supervisor_name=args.supervisor,
            worker_name=args.worker,
            seed=args.seed,
            n_runs_per_task=args.n_runs_per_task,
            research_mode=args.research_mode,
            max_rounds=args.max_rounds,
            max_sources_per_round=args.max_sources_per_round
        )
        
        # Adjust for different number of tasks if specified
        if args.n_tasks != 20:
            print(f"⚠️  Adjusting to {args.n_tasks} tasks instead of 20")
            from res_swarm.src.experiment_utils import TaskSelector
            selector = TaskSelector()
            selected_tasks = selector.select_random_english_tasks(n_tasks=args.n_tasks, seed=args.seed)
            experiment_config.selected_tasks = selected_tasks
            experiment_config.task_ids = [task['id'] for task in selected_tasks]
            experiment_config.total_runs = len(selected_tasks) * args.n_runs_per_task
        
        # Convert resume_from_run to 0-indexed if provided
        resume_from = (args.resume_from_run - 1) if args.resume_from_run else None
        
        # Run the batch experiment
        summary = runner.run_batch_experiment(
            experiment_config, 
            resume_from_run=resume_from,
            enable_auto_retry=args.enable_auto_retry,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        if "error" in summary:
            print(f"❌ Batch experiment failed: {summary['error']}")
            sys.exit(1)
        else:
            print(f"🎉 Batch experiment completed successfully!")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Batch experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()