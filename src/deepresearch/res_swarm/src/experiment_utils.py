"""
Experiment Utilities for ResSwarm

Tools for selecting tasks, managing experiments, and analyzing results.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import random


class TaskSelector:
    """
    Utility for selecting tasks from the DeepResearch Bench dataset.
    """
    
    def __init__(self, query_file: str = "deep_research_bench/data/prompt_data/query.jsonl"):
        """
        Initialize task selector.
        
        Args:
            query_file: Path to query.jsonl file
        """
        self.query_file = Path(query_file)
        self._tasks = None
        self._english_tasks = None
        
    def load_all_tasks(self) -> List[Dict[str, Any]]:
        """Load all tasks from the query file."""
        if self._tasks is None:
            self._tasks = []
            with open(self.query_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line.strip())
                        self._tasks.append(task)
        return self._tasks
    
    def get_english_tasks(self) -> List[Dict[str, Any]]:
        """Get only English tasks (language='en')."""
        if self._english_tasks is None:
            all_tasks = self.load_all_tasks()
            self._english_tasks = [task for task in all_tasks if task.get('language') == 'en']
        return self._english_tasks
    
    def select_random_english_tasks(self, n_tasks: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Randomly select n English tasks with given seed for reproducibility.
        
        Args:
            n_tasks: Number of tasks to select
            seed: Random seed for reproducibility
            
        Returns:
            List of selected task dictionaries
        """
        english_tasks = self.get_english_tasks()
        
        if n_tasks > len(english_tasks):
            raise ValueError(f"Requested {n_tasks} tasks but only {len(english_tasks)} English tasks available")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Get indices for random selection
        total_english = len(english_tasks)
        selected_indices = np.random.choice(total_english, n_tasks, replace=False)
        
        # Sort indices to maintain order
        selected_indices = sorted(selected_indices)
        
        selected_tasks = [english_tasks[i] for i in selected_indices]
        
        print(f"Selected {n_tasks} English tasks with seed={seed}")
        print(f"Selected task IDs: {[task['id'] for task in selected_tasks]}")
        
        return selected_tasks
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about available tasks."""
        all_tasks = self.load_all_tasks()
        english_tasks = self.get_english_tasks()
        
        # Count by language
        lang_counts = {}
        for task in all_tasks:
            lang = task.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Count by topic for English tasks
        topic_counts = {}
        for task in english_tasks:
            topic = task.get('topic', 'unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_tasks": len(all_tasks),
            "english_tasks": len(english_tasks),
            "language_distribution": lang_counts,
            "english_topic_distribution": topic_counts
        }


class ExperimentConfig:
    """Configuration for batch experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        predictor_name: str,
        compressor_name: str,
        selected_tasks: List[Dict[str, Any]],
        n_runs_per_task: int = 5,
        research_mode: str = "adaptive",
        max_rounds: int = 5,
        max_sources_per_round: int = 20,
        seed: int = 42,
        synthesis_strategy: str = "single",
        sections_per_chunk: int = 2
    ):
        """
        Initialize experiment configuration.
        
        Args:
            experiment_name: Name for the experiment
            predictor_name: Name of predictor model
            compressor_name: Name of compressor model  
            selected_tasks: List of selected tasks
            n_runs_per_task: Number of runs per task
            research_mode: Research mode ('adaptive' or 'evaluation')
            max_rounds: Maximum research rounds
            max_sources_per_round: Max sources per round
            seed: Random seed for reproducibility
            synthesis_strategy: Synthesis strategy ('single' or 'chunked')
            sections_per_chunk: Number of sections per chunk for chunked synthesis
        """
        self.experiment_name = experiment_name
        self.predictor_name = predictor_name
        self.compressor_name = compressor_name
        self.selected_tasks = selected_tasks
        self.n_runs_per_task = n_runs_per_task
        self.research_mode = research_mode
        self.max_rounds = max_rounds
        self.max_sources_per_round = max_sources_per_round
        self.seed = seed
        self.synthesis_strategy = synthesis_strategy
        self.sections_per_chunk = sections_per_chunk
        
        # Calculate derived properties
        self.total_runs = len(selected_tasks) * n_runs_per_task
        self.task_ids = [task['id'] for task in selected_tasks]
        
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "predictor_name": self.predictor_name,
            "compressor_name": self.compressor_name,
            "task_ids": self.task_ids,
            "n_runs_per_task": self.n_runs_per_task,
            "total_runs": self.total_runs,
            "research_mode": self.research_mode,
            "max_rounds": self.max_rounds,
            "max_sources_per_round": self.max_sources_per_round,
            "seed": self.seed,
            "synthesis_strategy": self.synthesis_strategy,
            "sections_per_chunk": self.sections_per_chunk,
            "timestamp": self.timestamp,
            "selected_tasks": self.selected_tasks
        }
    
    def save(self, output_dir: Path):
        """Save configuration to JSON file."""
        config_file = output_dir / "experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Experiment config saved to: {config_file}")


class MetricsCollector:
    """Collects and stores detailed metrics for experiment runs."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.run_metrics = []
    
    def record_run(
        self,
        run_id: str,
        task_id: int,
        task_prompt: str,
        result: Dict[str, Any],
        task_score: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Record metrics for a single run using new granular token tracking format.
        
        Args:
            run_id: Unique identifier for this run
            task_id: ID of the task
            task_prompt: The task prompt text
            result: Full result from DeepResearch minion with granular token tracking
            task_score: Optional task score from evaluation
            error: Error message if run failed
        """
        # Extract the 7 required metrics from the new result format
        predictor_calls = result.get('predictor_calls', [])
        compressor_calls = result.get('compressor_calls', [])
        
        # Calculate final predictor output tokens (last predictor call)
        final_predictor_output_tokens = 0
        if predictor_calls:
            final_predictor_output_tokens = predictor_calls[-1].get('output_tokens', 0)
        
        # Build the metrics record with all 7 required fields
        metrics = {
            "run_id": run_id,
            "task_id": task_id,
            "task_prompt": task_prompt,
            "task_score": task_score,  # Will be filled by evaluation
            "error": error,
            
            # The 7 required metrics:
            "predictor_model": result.get('predictor_model', 'unknown'),           # 1. Predictor model
            "compressor_model": result.get('compressor_model', 'unknown'),                   # 2. Compressor model  
            "scores": {"task_score": task_score} if task_score else {},              # 3. Task scores
            "compressor_call_count": result.get('compressor_call_count', 0),                 # 4. Number of compressor calls
            "compressor_output_tokens_per_call": [                                       # 5. Output tokens per compressor call
                call.get('output_tokens', 0) for call in compressor_calls
            ],
            "compressor_input_tokens_per_call": [                                        # 6. Input tokens per compressor call
                call.get('input_tokens', 0) for call in compressor_calls
            ],
            "final_predictor_output_tokens": final_predictor_output_tokens,        # 7. Final predictor output tokens
            
            # Additional useful metrics
            "predictor_calls": predictor_calls,
            "compressor_calls": compressor_calls,
            "total_predictor_tokens": sum(
                call.get('input_tokens', 0) + call.get('output_tokens', 0) 
                for call in predictor_calls
            ),
            "total_compressor_tokens": sum(
                call.get('input_tokens', 0) + call.get('output_tokens', 0) 
                for call in compressor_calls
            ),
            "timing": result.get('timing', {}),
            "final_answer": result.get('final_answer', ''),
            "timestamp": datetime.now().isoformat()
        }
        
        self.run_metrics.append(metrics)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded metrics."""
        return self.run_metrics
    
    def save_metrics(self, output_file: Path):
        """Save metrics to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.run_metrics, f, indent=2, ensure_ascii=False)
        print(f"Detailed metrics saved to: {output_file}")


class VarianceAnalyzer:
    """Analyzes variance across multiple runs of the same task."""
    
    def __init__(self, metrics: List[Dict[str, Any]]):
        """
        Initialize variance analyzer.
        
        Args:
            metrics: List of run metrics from MetricsCollector
        """
        self.metrics = metrics
    
    def analyze_by_task(self) -> Dict[str, Any]:
        """
        Analyze variance by task across multiple runs.
        
        Returns:
            Dictionary with variance analysis for each task
        """
        # Group metrics by task_id
        task_groups = {}
        for metric in self.metrics:
            task_id = metric["task_id"]
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(metric)
        
        analysis = {}
        
        for task_id, runs in task_groups.items():
            if len(runs) < 2:
                continue
                
            # Extract metrics for variance analysis
            predictor_tokens = [r.get("total_predictor_tokens", 0) for r in runs]
            compressor_tokens = [r.get("total_compressor_tokens", 0) for r in runs]
            compressor_calls = [r.get("compressor_call_count", 0) for r in runs]
            search_rounds = [1 for r in runs]  # Each run is 1 research round in current setup
            success_rates = [1 if not r.get("error") else 0 for r in runs]
            
            # Calculate statistics
            analysis[task_id] = {
                "task_prompt": runs[0]["task_prompt"][:100] + "...",
                "num_runs": len(runs),
                "success_rate": np.mean(success_rates),
                "predictor_tokens": {
                    "mean": np.mean(predictor_tokens),
                    "std": np.std(predictor_tokens),
                    "min": np.min(predictor_tokens),
                    "max": np.max(predictor_tokens),
                    "cv": np.std(predictor_tokens) / np.mean(predictor_tokens) if np.mean(predictor_tokens) > 0 else 0
                },
                "compressor_tokens": {
                    "mean": np.mean(compressor_tokens),
                    "std": np.std(compressor_tokens),
                    "min": np.min(compressor_tokens),
                    "max": np.max(compressor_tokens),
                    "cv": np.std(compressor_tokens) / np.mean(compressor_tokens) if np.mean(compressor_tokens) > 0 else 0
                },
                "compressor_calls": {
                    "mean": np.mean(compressor_calls),
                    "std": np.std(compressor_calls),
                    "min": np.min(compressor_calls),
                    "max": np.max(compressor_calls),
                    "cv": np.std(compressor_calls) / np.mean(compressor_calls) if np.mean(compressor_calls) > 0 else 0
                },
                "search_rounds": {
                    "mean": np.mean(search_rounds),
                    "std": np.std(search_rounds),
                    "min": np.min(search_rounds),
                    "max": np.max(search_rounds),
                    "cv": np.std(search_rounds) / np.mean(search_rounds) if np.mean(search_rounds) > 0 else 0
                }
            }
        
        return analysis
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall experiment statistics."""
        successful_runs = [m for m in self.metrics if not m.get("error")]
        failed_runs = [m for m in self.metrics if m.get("error")]
        
        if not successful_runs:
            return {"error": "No successful runs to analyze"}
        
        # Overall metrics
        total_predictor_tokens = sum(r.get("total_predictor_tokens", 0) for r in successful_runs)
        total_compressor_tokens = sum(r.get("total_compressor_tokens", 0) for r in successful_runs)
        total_compressor_calls = sum(r.get("compressor_call_count", 0) for r in successful_runs)
        
        return {
            "total_runs": len(self.metrics),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.metrics),
            "total_tokens": {
                "predictor": total_predictor_tokens,
                "compressor": total_compressor_tokens,
                "total": total_predictor_tokens + total_compressor_tokens
            },
            "total_compressor_calls": total_compressor_calls,
            "average_per_run": {
                "predictor_tokens": total_predictor_tokens / len(successful_runs),
                "compressor_tokens": total_compressor_tokens / len(successful_runs),
                "compressor_calls": total_compressor_calls / len(successful_runs)
            }
        }
    
    def save_analysis(self, output_file: Path):
        """Save variance analysis to JSON file."""
        analysis = {
            "task_analysis": self.analyze_by_task(),
            "overall_statistics": self.get_overall_statistics(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Variance analysis saved to: {output_file}")


def create_20_task_experiment_config(
    experiment_name: str,
    supervisor_name: str,
    worker_name: str,
    seed: int = 42,
    n_runs_per_task: int = 5,
    research_mode: str = "adaptive",
    max_rounds: int = 5,
    max_sources_per_round: int = 20,
    synthesis_strategy: str = "single",
    sections_per_chunk: int = 2
) -> ExperimentConfig:
    """
    Create a configuration for the 20-task experiment.
    
    Args:
        experiment_name: Name for the experiment
        supervisor_name: Name of supervisor model
        worker_name: Name of worker model
        seed: Random seed for task selection
        n_runs_per_task: Number of runs per task
        research_mode: Research mode
        max_rounds: Maximum research rounds
        max_sources_per_round: Max sources per round
        
    Returns:
        ExperimentConfig instance
    """
    # Select 20 random English tasks
    selector = TaskSelector()
    selected_tasks = selector.select_random_english_tasks(n_tasks=20, seed=seed)
    
    # Print task selection summary
    stats = selector.get_task_statistics()
    print(f"\nTask Selection Summary:")
    print(f"Total available tasks: {stats['total_tasks']}")
    print(f"English tasks available: {stats['english_tasks']}")
    print(f"Selected: {len(selected_tasks)} tasks")
    print(f"Topic distribution of selected tasks:")
    
    topic_counts = {}
    for task in selected_tasks:
        topic = task.get('topic', 'unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")
    
    return ExperimentConfig(
        experiment_name=experiment_name,
        supervisor_name=supervisor_name,
        worker_name=worker_name,
        selected_tasks=selected_tasks,
        n_runs_per_task=n_runs_per_task,
        research_mode=research_mode,
        max_rounds=max_rounds,
        max_sources_per_round=max_sources_per_round,
        seed=seed,
        synthesis_strategy=synthesis_strategy,
        sections_per_chunk=sections_per_chunk
    )