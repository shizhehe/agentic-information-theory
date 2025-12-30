"""
DeepResearch Task Runner

Main orchestrator for running supervisor-worker experiments on DeepResearch Bench tasks.
Loads queries, runs experiments, and outputs results in evaluation-compatible format.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
from pathlib import Path
from tqdm import tqdm

from src.deepresearch.res_swarm.src.deepres_minion import DeepResearchMinion
from src.deepresearch.res_swarm.configs.model_config import get_config_manager, ExperimentConfig
from src.deepresearch.res_swarm.src.deepres_tools import WebSearchTool


class DeepResearchRunner:
    """
    Main runner for DeepResearch experiments.
    
    Loads queries from DeepResearch Bench, runs supervisor-worker experiments,
    and outputs results in format compatible with evaluation pipeline.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/experiments",
        log_dir: str = "outputs/logs",
        web_search_enabled: bool = True,
        research_mode: str = "adaptive",
        max_rounds: int = 3,
        max_sources_per_round: int = 20,
        worker_batch_size: int = 10,
        content_quality_threshold: float = 0.3,
        parallel_scraping: bool = True,
        synthesis_strategy: str = "single",
        sections_per_chunk: int = 2,
    ):
        """
        Initialize the DeepResearch runner.
        
        Args:
            output_dir: Directory to save experiment results
            log_dir: Directory for detailed conversation logs
            web_search_enabled: Whether to enable web search for workers
            research_mode: Research mode ('evaluation' or 'adaptive')
            max_rounds: Maximum research rounds for adaptive mode
            max_sources_per_round: Maximum web sources per research round
            worker_batch_size: Batch size for worker processing
            content_quality_threshold: Minimum content quality threshold
            parallel_scraping: Whether to enable parallel web scraping
            synthesis_strategy: Synthesis strategy ('single' or 'chunked')
            sections_per_chunk: Number of sections per chunk for chunked synthesis
        """
        # Resolve paths relative to project root to ensure consistency
        project_root = Path(__file__).parent.parent.parent
        self.output_dir = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        self.log_dir = project_root / log_dir if not Path(log_dir).is_absolute() else Path(log_dir)
        self.web_search_enabled = web_search_enabled
        self.research_mode = research_mode
        self.max_rounds = max_rounds
        self.max_sources_per_round = max_sources_per_round
        self.worker_batch_size = worker_batch_size
        self.content_quality_threshold = content_quality_threshold
        self.parallel_scraping = parallel_scraping
        self.synthesis_strategy = synthesis_strategy
        self.sections_per_chunk = sections_per_chunk
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize model configuration manager
        self.config_manager = get_config_manager()
        
        # Initialize web search tool if enabled
        if web_search_enabled:
            self.search_tool = WebSearchTool()
        else:
            self.search_tool = None
        
        print(f"DeepResearch Runner initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Web search enabled: {web_search_enabled}")
    
    def load_queries(
        self, 
        query_file: str = "deep_research_bench/data/prompt_data/query.jsonl",
        exclude_zh: bool = True,
        include_zh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load queries from DeepResearch Bench with language filtering.
        
        Args:
            query_file: Path to query.jsonl file
            exclude_zh: Whether to exclude Chinese ('zh') queries (default: True)
            include_zh: Whether to include Chinese ('zh') queries (overrides exclude_zh)
            
        Returns:
            List of query dictionaries
        """
        query_path = Path(query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        
        queries = []
        total_loaded = 0
        zh_count = 0
        en_count = 0
        
        with open(query_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    query = json.loads(line.strip())
                    total_loaded += 1
                    
                    # Get language from query
                    language = query.get('language', 'en')  # Default to 'en' if not specified
                    
                    # Count languages
                    if language == 'zh':
                        zh_count += 1
                    elif language == 'en':
                        en_count += 1
                    
                    # Apply language filtering
                    if include_zh:
                        # If explicitly including Chinese, include all languages
                        queries.append(query)
                    elif exclude_zh and language == 'zh':
                        # Skip Chinese queries if excluding them
                        continue
                    else:
                        # Include the query
                        queries.append(query)
        
        print(f"Loaded {len(queries)} queries from {query_path}")
        print(f"Total queries in file: {total_loaded} (zh: {zh_count}, en: {en_count})")
        if exclude_zh and not include_zh:
            print(f"Excluded {zh_count} Chinese queries (exclude_zh=True)")
        elif include_zh:
            print(f"Included all languages (include_zh=True)")
            
        return queries
    
    def run_single_experiment(
        self, 
        experiment_config: ExperimentConfig,
        queries: List[Dict[str, Any]],
        max_queries: Optional[int] = None,
        start_from: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single supervisor-worker experiment on all queries.
        
        Args:
            experiment_config: Configuration for the experiment
            queries: List of DeepResearch queries
            max_queries: Maximum number of queries to process (None for all)
            start_from: Index to start processing from
            
        Returns:
            Experiment results summary
        """
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {experiment_config.name}")
        print(f"Description: {experiment_config.description}")
        print(f"{'='*60}")
        
        # Create clients
        try:
            supervisor_client = self.config_manager.create_client(experiment_config.supervisor_config)
            worker_client = self.config_manager.create_client(experiment_config.worker_config)
        except Exception as e:
            print(f"❌ Failed to create clients for {experiment_config.name}: {e}")
            return {"error": str(e), "completed": 0, "total": len(queries)}
        
        # Initialize DeepResearch Minion
        minion = DeepResearchMinion(
            supervisor_client=supervisor_client,
            worker_client=worker_client,
            web_search_enabled=self.web_search_enabled,
            log_dir=str(self.log_dir / experiment_config.name),
            research_mode=self.research_mode,
            max_rounds=self.max_rounds,
            max_sources_per_round=self.max_sources_per_round,
            worker_batch_size=self.worker_batch_size,
            synthesis_strategy=self.synthesis_strategy,
            sections_per_chunk=self.sections_per_chunk
        )
        
        # Determine query subset
        query_subset = queries[start_from:]
        if max_queries:
            query_subset = query_subset[:max_queries]
        
        print(f"Processing {len(query_subset)} queries (starting from index {start_from})")
        
        # Process queries
        results = []
        successful = 0
        total_supervisor_time = 0
        total_worker_time = 0
        total_search_time = 0
        
        start_time = time.time()
        
        for i, query in enumerate(tqdm(query_subset, desc="Processing Queries", unit="query")):
            query_id = query.get("id", start_from + i)
            task = query.get("prompt", "")
            
            print(f"\n--- Processing Query {query_id} ({i+1}/{len(query_subset)}) ---")
            print(f"Task: {task[:100]}...")
            
            try:
                # Run the minion protocol
                result = minion(
                    task=task,
                    query_id=query_id,
                    logging_id=f"{experiment_config.name}_query_{query_id}"
                )
                
                # Create result in DeepResearch Bench format
                bench_result = {
                    "id": query_id,
                    "prompt": task,
                    "article": result["final_answer"]
                }
                
                # Add metadata
                bench_result["metadata"] = {
                    "experiment": experiment_config.name,
                    "predictor": experiment_config.predictor_config.model_name,
                    "compressor": experiment_config.compressor_config.model_name,
                    "timing": result.get("timing", {}),
                    "usage": {
                        "predictor": result["predictor_usage"].to_dict(),
                        "compressor": result["compressor_usage"].to_dict()
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(bench_result)
                successful += 1
                
                # Accumulate timing stats
                timing = result.get("timing", {})
                total_supervisor_time += timing.get("supervisor_time", 0)
                total_worker_time += timing.get("worker_time", 0) 
                total_search_time += timing.get("search_time", 0)
                
                print(f"✅ Query {query_id} completed successfully")
                
            except Exception as e:
                print(f"❌ Error processing query {query_id}: {e}")
                
                # Create error result
                error_result = {
                    "id": query_id,
                    "prompt": task,
                    "article": f"Error processing query: {str(e)}",
                    "metadata": {
                        "experiment": experiment_config.name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                results.append(error_result)
        
        total_time = time.time() - start_time
        
        # Save results
        output_file = self.output_dir / f"{experiment_config.name}.jsonl"
        self._save_results(results, output_file)
        
        # Create summary
        summary = {
            "experiment": experiment_config.name,
            "description": experiment_config.description,
            "predictor": experiment_config.predictor_config.model_name,
            "compressor": experiment_config.compressor_config.model_name,
            "completed": successful,
            "total": len(query_subset),
            "success_rate": successful / len(query_subset) if query_subset else 0,
            "timing": {
                "total_time": total_time,
                "avg_time_per_query": total_time / len(query_subset) if query_subset else 0,
                "total_supervisor_time": total_supervisor_time,
                "total_worker_time": total_worker_time,
                "total_search_time": total_search_time,
            },
            "output_file": str(output_file),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n✅ EXPERIMENT COMPLETED: {experiment_config.name}")
        print(f"Success rate: {successful}/{len(query_subset)} ({summary['success_rate']:.1%})")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per query: {summary['timing']['avg_time_per_query']:.1f}s")
        print(f"Results saved to: {output_file}")
        
        return summary
    
    def run_multiple_experiments(
        self,
        experiment_names: Optional[List[str]] = None,
        max_queries: Optional[int] = None,
        start_from: int = 0,
        exclude_zh: bool = True,
        include_zh: bool = False
    ) -> Dict[str, Any]:
        """
        Run multiple experiments sequentially.
        
        Args:
            experiment_names: List of experiment names to run (None for all enabled)
            max_queries: Maximum queries per experiment
            start_from: Index to start processing from
            exclude_zh: Whether to exclude Chinese ('zh') queries (default: True)
            include_zh: Whether to include Chinese ('zh') queries (overrides exclude_zh)
            
        Returns:
            Overall results summary
        """
        # Load queries
        queries = self.load_queries(exclude_zh=exclude_zh, include_zh=include_zh)
        
        # Determine experiments to run
        if experiment_names:
            experiments = []
            for name in experiment_names:
                exp = self.config_manager.get_experiment_config(name)
                if exp:
                    experiments.append(exp)
                else:
                    print(f"⚠️  Unknown experiment: {name}")
        else:
            experiments = self.config_manager.get_enabled_experiments()
        
        if not experiments:
            print("❌ No experiments to run")
            return {"error": "No experiments configured"}
        
        print(f"\n🚀 STARTING {len(experiments)} EXPERIMENTS")
        print(f"Queries per experiment: {max_queries or len(queries)}")
        print(f"Starting from query index: {start_from}")
        
        # Run experiments
        experiment_results = {}
        overall_start = time.time()
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(experiments)}: {experiment.name}")
            print(f"{'='*80}")
            
            try:
                result = self.run_single_experiment(
                    experiment, queries, max_queries, start_from
                )
                experiment_results[experiment.name] = result
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted during experiment: {experiment.name}")
                break
            except Exception as e:
                print(f"❌ Failed experiment: {experiment.name} - {e}")
                experiment_results[experiment.name] = {"error": str(e)}
        
        overall_time = time.time() - overall_start
        
        # Create overall summary
        successful_experiments = sum(1 for r in experiment_results.values() if "error" not in r)
        total_queries_processed = sum(r.get("completed", 0) for r in experiment_results.values())
        
        overall_summary = {
            "total_experiments": len(experiments),
            "successful_experiments": successful_experiments,
            "total_queries_processed": total_queries_processed,
            "overall_time": overall_time,
            "experiment_results": experiment_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save overall summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"🎉 ALL EXPERIMENTS COMPLETED")
        print(f"Successful experiments: {successful_experiments}/{len(experiments)}")
        print(f"Total queries processed: {total_queries_processed}")
        print(f"Overall time: {overall_time:.1f}s")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*80}")
        
        return overall_summary
    
    def run_custom_experiment(
        self,
        supervisor_name: str,
        worker_name: str,
        max_queries: Optional[int] = None,
        start_from: int = 0,
        exclude_zh: bool = True,
        include_zh: bool = False,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a custom supervisor-worker experiment.
        
        Args:
            supervisor_name: Name of supervisor model
            worker_name: Name of worker model
            max_queries: Maximum number of queries to process
            start_from: Index to start processing from
            exclude_zh: Whether to exclude Chinese ('zh') queries (default: True)
            include_zh: Whether to include Chinese ('zh') queries (overrides exclude_zh)
            
        Returns:
            Experiment results dictionary
        """
        # Create custom experiment configuration
        experiment_config = self.config_manager.create_custom_experiment(supervisor_name, worker_name, experiment_name)
        
        if not experiment_config:
            return {"error": f"Invalid supervisor '{supervisor_name}' or worker '{worker_name}'"}
        
        print(f"\n{'='*80}")
        print(f"🚀 RUNNING CUSTOM EXPERIMENT: {experiment_config.name}")
        print(f"Supervisor: {supervisor_name}")
        print(f"Worker: {worker_name}")
        print(f"{'='*80}")
        
        # Load queries
        try:
            queries = self.load_queries(exclude_zh=exclude_zh, include_zh=include_zh)
        except Exception as e:
            return {"error": f"Failed to load queries: {e}"}
        
        # Run the experiment
        try:
            result = self.run_single_experiment(
                experiment_config=experiment_config,
                queries=queries,
                max_queries=max_queries,
                start_from=start_from
            )
            
            print(f"\n✅ Custom experiment '{experiment_config.name}' completed successfully!")
            print(f"Results saved to: {self.output_dir / f'{experiment_config.name}.jsonl'}")
            print(f"Use --copy-to-eval {experiment_config.name} to prepare for evaluation")
            
            return result
            
        except Exception as e:
            error_msg = f"Custom experiment failed: {e}"
            print(f"\n❌ {error_msg}")
            return {"error": error_msg}
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: Path):
        """Save results in JSONL format compatible with DeepResearch Bench."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def copy_to_eval_format(self, experiment_name: str):
        """
        Copy experiment results to DeepResearch Bench evaluation format.
        
        Args:
            experiment_name: Name of experiment to copy
        """
        source_file = self.output_dir / f"{experiment_name}.jsonl"
        if not source_file.exists():
            print(f"❌ Source file not found: {source_file}")
            return
        
        # Target directory for DeepResearch Bench evaluation
        target_dir = Path("deep_research_bench/data/test_data/raw_data")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / f"{experiment_name}.jsonl"
        
        # Read source and write target with only required fields
        with open(source_file, 'r', encoding='utf-8') as src, \
             open(target_file, 'w', encoding='utf-8') as tgt:
            
            for line in src:
                data = json.loads(line.strip())
                eval_format = {
                    "id": data["id"],
                    "prompt": data["prompt"],
                    "article": data["article"]
                }
                tgt.write(json.dumps(eval_format, ensure_ascii=False) + '\n')
        
        print(f"✅ Copied {experiment_name} results to evaluation format: {target_file}")
        print(f"   Now you can run: cd deep_research_bench && bash run_benchmark.sh")
        print(f"   (Make sure to add '{experiment_name}' to TARGET_MODELS in run_benchmark.sh)")


def main():
    """Command-line interface for running DeepResearch experiments."""
    parser = argparse.ArgumentParser(description="Run DeepResearch supervisor-worker experiments")
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Specific experiment to run (default: run all enabled)"
    )
    parser.add_argument(
        "--experiments", "-es",
        nargs="+",
        help="Multiple specific experiments to run"
    )
    parser.add_argument(
        "--max-queries", "-n",
        type=int,
        help="Maximum number of queries to process per experiment"
    )
    parser.add_argument(
        "--start-from", "-s",
        type=int,
        default=0,
        help="Index to start processing from (default: 0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/experiments",
        help="Output directory for results (default: outputs/experiments)"
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search for workers"
    )
    parser.add_argument(
        "--list-experiments", "-l",
        action="store_true",
        help="List available experiments and exit"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment variables and exit"
    )
    parser.add_argument(
        "--copy-to-eval",
        type=str,
        help="Copy experiment results to evaluation format"
    )
    parser.add_argument(
        "--supervisor",
        type=str,
        help="Supervisor model name for custom experiment (use with --worker)"
    )
    parser.add_argument(
        "--worker",
        type=str,
        help="Worker model name for custom experiment (use with --supervisor)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Custom experiment name (overrides auto-generated name)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available supervisor and worker models"
    )
    parser.add_argument(
        "--exclude-zh",
        action="store_true",
        default=True,
        help="Exclude Chinese ('zh') queries from experiments (default: True)"
    )
    parser.add_argument(
        "--include-zh",
        action="store_true",
        help="Include Chinese ('zh') queries in experiments (overrides --exclude-zh)"
    )
    parser.add_argument(
        "--research-mode",
        type=str,
        choices=["evaluation", "adaptive", "multi_query"],
        default="adaptive",
        help="Research mode: 'adaptive' (multi-round intelligent research), 'evaluation' (single-round, backward compatible), or 'multi_query' (5 queries x 4 sources) (default: adaptive)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum research rounds for adaptive mode (default: 3, ignored in evaluation mode)"
    )
    parser.add_argument(
        "--max-sources-per-round",
        type=int,
        default=20,
        help="Maximum web sources per research round (default: 20)"
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=10,
        help="Batch size for worker processing (default: 10)"
    )
    parser.add_argument(
        "--content-quality-threshold",
        type=float,
        default=0.3,
        help="Minimum content quality threshold (0.0-1.0, default: 0.3)"
    )
    parser.add_argument(
        "--parallel-scraping",
        action="store_true",
        default=True,
        help="Enable parallel web scraping (default: True)"
    )
    parser.add_argument(
        "--no-parallel-scraping",
        action="store_true",
        help="Disable parallel web scraping (use sequential)"
    )
    parser.add_argument(
        "--synthesis-strategy",
        type=str,
        choices=["single", "chunked"],
        default="single",
        help="Synthesis strategy: 'single' (default) or 'chunked' for sectioned synthesis"
    )
    parser.add_argument(
        "--sections-per-chunk",
        type=int,
        default=2,
        help="Number of sections per synthesis chunk for chunked synthesis (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Handle parallel scraping argument
    parallel_scraping = args.parallel_scraping and not args.no_parallel_scraping
    
    # Initialize runner
    runner = DeepResearchRunner(
        output_dir=args.output_dir,
        web_search_enabled=not args.no_web_search,
        research_mode=args.research_mode,
        max_rounds=args.max_rounds,
        max_sources_per_round=args.max_sources_per_round,
        worker_batch_size=args.worker_batch_size,
        content_quality_threshold=args.content_quality_threshold,
        parallel_scraping=parallel_scraping,
        synthesis_strategy=args.synthesis_strategy,
        sections_per_chunk=args.sections_per_chunk
    )
    
    # Handle special commands
    if args.check_env:
        runner.config_manager.print_environment_status()
        return
    
    if args.list_experiments:
        experiments = runner.config_manager.get_enabled_experiments()
        print("\n=== AVAILABLE EXPERIMENTS ===")
        for exp in experiments:
            print(f"• {exp.name}: {exp.description}")
        print(f"\nTotal: {len(experiments)} enabled experiments")
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
        
        print(f"\nUsage: --supervisor SUPERVISOR --worker WORKER")
        print(f"Example: --supervisor gpt-4o --worker llama3.2")
        return
    
    if args.copy_to_eval:
        runner.copy_to_eval_format(args.copy_to_eval)
        return
    
    # Check for custom supervisor-worker experiment
    if args.supervisor or args.worker:
        # Validate that both supervisor and worker are provided
        if not args.supervisor or not args.worker:
            print("❌ Error: Both --supervisor and --worker must be provided for custom experiments")
            print("Use --list-models to see available options")
            return
        
        # Validate that supervisor and worker exist
        if not runner.config_manager.get_supervisor_config(args.supervisor):
            print(f"❌ Error: Unknown supervisor model '{args.supervisor}'")
            print("Use --list-models to see available supervisors")
            return
        
        if not runner.config_manager.get_worker_config(args.worker):
            print(f"❌ Error: Unknown worker model '{args.worker}'") 
            print("Use --list-models to see available workers")
            return
        
        # Run custom experiment
        try:
            result = runner.run_custom_experiment(
                supervisor_name=args.supervisor,
                worker_name=args.worker,
                max_queries=args.max_queries,
                start_from=args.start_from,
                exclude_zh=args.exclude_zh and not args.include_zh,
                include_zh=args.include_zh,
                experiment_name=args.experiment_name
            )
            
            if "error" not in result:
                print(f"\n🎉 Custom experiment completed successfully!")
            else:
                print(f"\n❌ Custom experiment failed: {result['error']}")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Custom experiment interrupted by user")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
        
        return
    
    # Determine predefined experiments to run
    if args.experiment:
        experiment_names = [args.experiment]
    elif args.experiments:
        experiment_names = args.experiments
    else:
        experiment_names = None  # Run all enabled
    
    # Run predefined experiments
    try:
        results = runner.run_multiple_experiments(
            experiment_names=experiment_names,
            max_queries=args.max_queries,
            start_from=args.start_from,
            exclude_zh=args.exclude_zh and not args.include_zh,
            include_zh=args.include_zh
        )
        
        if "error" not in results:
            print(f"\n🎉 All experiments completed successfully!")
            print(f"Check results in: {args.output_dir}/")
        else:
            print(f"\n❌ Experiments failed: {results['error']}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")


if __name__ == "__main__":
    main()