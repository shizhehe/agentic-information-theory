from typing import List, Union, Optional
import random

from src.ib.tasks.base import Document, Problem, BaseTask
from .dataset import load_wildchat_dataset, load_wildchat_questions
from src.ib.utils import get_logger


class WildchatTask(BaseTask):
    name: str = "Wildchat"

    class Config(BaseTask.Config):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.qa_data_path: str = kwargs.get('qa_data_path', "data/wildchat/qa/wildchat_questions.feather")
            self.raw_data_path: Optional[str] = kwargs.get('raw_data_path', None)
            self.user_ids: Optional[List[int]] = kwargs.get('user_ids', None)
            self.cache_dir: str = kwargs.get('cache_dir', "data/cache")

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self._dataset = None  # Cache the loaded dataset

    def load_dataset(self):
        """
        Load the WildChat dataset with automatic data location handling.
        This follows the LongHealth pattern for dataset loading.
        """
        if self._dataset is not None:
            return self._dataset
            
        try:
            self._dataset = load_wildchat_dataset(
                user_ids=self.config.user_ids,
                qa_data_path=self.config.qa_data_path,
                raw_data_path=self.config.raw_data_path,
                cache_dir=self.config.cache_dir
            )
            self.logger.info(f"Loaded {len(self._dataset)} users from wildchat dataset")
            return self._dataset
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Wildchat data not found at {self.config.qa_data_path}. "
                "Please run the QA generation script first: "
                "python src/ib/tasks/wildchat/questions.py"
            )
    
    def build_dataset(self) -> List[Problem]:
        """Build the dataset by loading users and converting to Problem objects."""
        # Load the dataset first
        users = self.load_dataset()
        
        # Convert to Problem objects
        problems = []
        for user in users:
            if user.question is None or user.answer is None:
                self.logger.warning(f"User {user.user_id} has no QA pair, skipping")
                continue
                
            # Create document from conversation history
            document = Document(
                title=f"User {user.user_id} Conversations",
                content=user.get_conversation_text()
            )
            
            problem = Problem(
                id=f"wildchat_user_{user.user_id}",
                query=user.question,
                context=[document],
                target=user.answer,
                metadata={
                    "user_id": user.user_id,
                    "num_conversations": user.get_total_conversations()
                }
            )
            problems.append(problem)
        
        self.logger.info(f"Built {len(problems)} problems from wildchat dataset")
        
        if self.config.shuffle:
            random.seed(42)
            random.shuffle(problems)
            
        return self._subset_dataset(problems)

    def score(self, pred: str, answer: str) -> Union[float, int]:
        """Evaluate correctness for wildchat QA pairs with flexible matching."""
        if not pred or not answer:
            return 0
        
        # For conversation-based QA, use more flexible matching
        pred_clean = pred.lower().strip()
        answer_clean = answer.lower().strip()
        
        # Check for semantic overlap (basic approach)
        pred_words = set(pred_clean.split())
        answer_words = set(answer_clean.split())
        
        # If there's significant word overlap, consider it correct
        if len(pred_words & answer_words) / max(len(answer_words), 1) > 0.3:
            return 1
        
        # Fallback to substring matching
        return 1 if (answer_clean in pred_clean or pred_clean in answer_clean) else 0