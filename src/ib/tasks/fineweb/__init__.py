from typing import List, Union, Optional
import random

from src.ib.tasks.base import Document, Problem, BaseTask
from .dataset import load_fineweb_dataset, load_fineweb_questions
from src.ib.utils import get_logger


class FinewebTask(BaseTask):
    name: str = "Fineweb"

    class Config(BaseTask.Config):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.qa_data_path: str = kwargs.get('qa_data_path', "data/fineweb/qa/fineweb_questions.feather")
            self.raw_data_path: Optional[str] = kwargs.get('raw_data_path', None)
            self.doc_ids: Optional[List[str]] = kwargs.get('doc_ids', None)
            self.question_types: Optional[List[str]] = kwargs.get('question_types', ["qa", "generation"])
            self.cache_dir: str = kwargs.get('cache_dir', "data/cache")

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self._dataset = None  # Cache the loaded dataset

    def load_dataset(self):
        """
        Load the FineWeb dataset with automatic data location handling.
        This follows the LongHealth pattern for dataset loading.
        """
        if self._dataset is not None:
            return self._dataset
            
        try:
            self._dataset = load_fineweb_dataset(
                doc_ids=self.config.doc_ids,
                question_types=self.config.question_types,
                qa_data_path=self.config.qa_data_path,
                raw_data_path=self.config.raw_data_path,
                cache_dir=self.config.cache_dir
            )
            self.logger.info(f"Loaded {len(self._dataset)} documents from fineweb dataset")
            return self._dataset
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Fineweb data not found at {self.config.qa_data_path}. "
                "Please run the QA generation script first: "
                "python src/ib/tasks/fineweb/questions.py"
            )
    
    def build_dataset(self) -> List[Problem]:
        """Build the dataset by loading documents and converting to Problem objects."""
        # Load the dataset first
        documents = self.load_dataset()
        
        # Convert to Problem objects
        problems = []
        for doc in documents:
            if doc.qa_pairs is None:
                self.logger.warning(f"Document {doc.doc_id} has no QA pairs, skipping")
                continue
                
            # Create a Problem for each QA pair
            for i, qa_pair in enumerate(doc.qa_pairs):
                document = Document(
                    title=f"Document {doc.doc_id}",
                    content=doc.text
                )
                
                problem = Problem(
                    id=f"{doc.doc_id}_{i}",
                    query=qa_pair.question,
                    context=[document],
                    target=qa_pair.answer,
                    metadata={
                        "doc_id": doc.doc_id,
                        "topic": qa_pair.topic,
                        "type": qa_pair.type,
                        "token_count": doc.token_count
                    }
                )
                problems.append(problem)
        
        self.logger.info(f"Built {len(problems)} problems from fineweb dataset")
        
        if self.config.shuffle:
            random.seed(42)
            random.shuffle(problems)
            
        return self._subset_dataset(problems)

    def score(self, pred: str, answer: str) -> Union[float, int]:
        """Simple string matching for correctness."""
        if not pred or not answer:
            return 0
        
        # Basic substring matching for QA tasks
        pred_clean = pred.lower().strip()
        answer_clean = answer.lower().strip()
        
        return 1 if (answer_clean in pred_clean or pred_clean in answer_clean) else 0