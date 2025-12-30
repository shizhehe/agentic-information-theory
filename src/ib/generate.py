from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
import json
import yaml

from tqdm import tqdm
from pydrantic import RunConfig, ObjectConfig

from src.ib.pipeline.base import BaseProtocol, ProtocolResponse
from src.ib.tasks import BaseTask, Problem
from src.ib.clients import ClientConfig, ClientResponse, Usage
from src.ib.utils import get_logger, seed_everything, save_yaml, load_yaml

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class GenerateConfig(RunConfig):
    """Configuration for generation tasks."""

    protocol: BaseProtocol.Config
    task: BaseTask.Config

    max_context_tokens: Optional[int] = 128_000

    seed: int = 0
    max_batch_size: int = 256
    num_samples: int = 1
    num_processes: int = 0  # Number of worker processes

    device: str = "cuda"

    output_dir: Optional[str] = None
    task_metadata: Optional[str] = None

    def run(self):
        return main(self)


@dataclass
class GenerationRequest:
    id: str
    problem: Problem


@dataclass
class Generation:
    id: str
    problem: Problem
    response: ProtocolResponse
    score: float


class Worker:
    def __init__(self, config: GenerateConfig):
        self.config = config
        self.protocol: BaseProtocol = config.protocol.instantiate()
        self.task: BaseTask = config.task.instantiate()

    def generate(
        self,
        request: GenerationRequest,
    ) -> Dict[str, Any]:
        """Process a single problem of data using the generator."""
        problem = request.problem
        response: ProtocolResponse = self.protocol(
            id=request.id,
            query=problem.query,
            context=problem.context,
            answer_choices=problem.answer_choices,
        )
        score = self.task.score(response.text, problem)

        result = Generation(
            id=request.id, problem=problem, response=response, score=score
        )

        self.save_result(request.id, result)
        return result

    def save_result(self, row_id: str, generation: Generation):
        outpath = self.config.run_dir / f"{row_id}.yaml"
        print(f"Saving problem to: {outpath}")
        save_yaml(asdict(generation), outpath)


# -----------------------------------------------------------------
# Begin multiprocessing utilities
# -----------------------------------------------------------------
_global_worker = None
# Will hold the Worker(...) instance
_global_out_dir = None  # Will hold reference to the output directory


def init_worker_fn(
    config: GenerateConfig,
):
    """
    Called once in each child process. Initializes the Worker and
    any other references needed by that process.
    """
    global _global_worker
    _global_worker = Worker(config)


def process_row_mp(row):
    """
    Called for each item in the dataset. Uses the global references
    initialized by `init_worker_fn`.
    """
    global _global_worker, _global_out_dir

    try:
        # Call the generate method instead of process_row
        result = _global_worker.generate(row)

        # Convert the Generation object to a dictionary to make it iterable
        return asdict(result)
    except Exception:
        # Capture full traceback for easier debugging in the main process
        tb = traceback.format_exc()
        return {"id": row.id if hasattr(row, "id") else "unknown", "error": tb}


# -----------------------------------------------------------------
# End multiprocessing utilities
# -----------------------------------------------------------------


def main(config: GenerateConfig):
    """Main function to run generation tasks."""
    logger = get_logger("Generate")
    seed_everything(config.seed)
    config.run_dir = Path(config.run_dir)
    logger.info(f"Saving samples to: {config.run_dir}")
    os.makedirs(config.run_dir, exist_ok=True)

    task = config.task.instantiate()
    dataset: List[Problem] = task.build_dataset()

    # Duplicate dataset for multiple samples
    dataset = [
        GenerationRequest(id=f"{item.id}_sample{idx}", problem=item)
        for idx in range(config.num_samples)
        for item in dataset
    ]

    results = []
    # Process using multiprocessing
    if config.num_processes > 0:
        from multiprocessing import Pool

        logger.info(f"Using multiprocessing with {config.num_processes} workers")

        # Define the pool, passing references to top-level functions
        with Pool(
            processes=config.num_processes,
            initializer=init_worker_fn,
            initargs=(config,),
        ) as pool:

            # Distribute dataset rows to processes
            results_iter = pool.imap(process_row_mp, dataset)

            for res in tqdm(results_iter, total=len(dataset)):
                # Check for errors
                if isinstance(res, dict) and "error" in res:
                    logger.error(f"[multiprocessing] Error processing row {res['id']}:")
                    logger.error(res["error"])  # Full traceback
                    continue

                # Convert dictionary back to Generation object if needed
                if isinstance(res, dict) and "error" not in res:
                    # Create a Generation object from the dictionary
                    res["response"]["text"] = str(res["response"]["text"])
                    generation = Generation(
                        id=res["id"],
                        problem=res["problem"],
                        response=res[
                            "response"
                        ],  # ProtocolResponse(**res["response"]),  # Convert back to ProtocolResponse
                        score=res["score"],
                    )
                    results.append(generation)
                else:
                    # Already a Generation object
                    results.append(res)

    else:
        logger.info("Processing sequentially")
        worker = Worker(config)
        for row in tqdm(dataset):
            result = worker.generate(row)
            results.append(result)

    # Save results to feather file
    df = pd.DataFrame(results)

    # Before saving to feather, ensure all columns are serializable
    for col in df.columns:
        if df[col].dtype == "object":
            # Convert any non-serializable objects to strings
            df[col] = df[col].apply(lambda x: str(x) if not _is_serializable(x) else x)

    df.to_feather(config.run_dir / "results.feather")
    logger.info(f"Saved results to: {config.run_dir / 'results.feather'}")
    if "score" in df.columns:
        logger.info(f"Avg score: {df['score'].mean()}")
    else:
        logger.info("No scores available in results")

    decompression_perplexity = []
    for i in range(len(df)):
        try:
            decompression_perplexity.extend(
                df.iloc[i]["response"]["meta"]["supervisor"]["decompression_perplexity"]
            )
        except Exception as e:
            decompression_perplexity.append(0)
            continue
    logger.info(
        f"Avg decompression perplexity: {sum(decompression_perplexity) / len(decompression_perplexity) if decompression_perplexity else 0}"
    )

    return results


def _is_serializable(obj):
    """Check if an object is serializable for PyArrow."""
    try:
        # Try to serialize to JSON as a basic test
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def reconstruct_folder(folder_path):
    """
    Reconstruct feather result file based on yaml results in folder.
    """
    results = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = f.read()

                    data = yaml.safe_load(data)

                    result = Generation(
                        id=data["id"],
                        problem=data["problem"],
                        response=data["response"],
                        score=data["score"],
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    # Sort results by the numeric part of the id (format: [number]_sample0)
    results.sort(key=lambda x: int(x.id.split("_")[0]))
    # Convert to DataFrame and save as feather

    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(folder_path, "reconstructed_results.feather")
        df.to_feather(output_path)
        print(f"Saved processed results to {output_path}")
        return df
    else:
        print(f"No YAML files found in {folder_path}")
        return None
