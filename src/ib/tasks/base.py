from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import os
from dataclasses import dataclass


os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass 
class Document:
    """This represents a single document"""
    title: str
    content: str

@dataclass
class Problem:
    id: str
    query: str
    context: List[Document]
    target: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    answer_choices: Optional[List[str]] = None  # for multiple choice tasks
    task: Optional[str] = None


class BaseTaskConfig:
    """Simple config class without pydrantic dependency"""
    def __init__(self, **kwargs):
        self.max_context_tokens: Optional[int] = kwargs.get('max_context_tokens', 128_000)
        self.tokenizer_model: str = kwargs.get('tokenizer_model', 'gpt2')
        self.seed: int = kwargs.get('seed', 0)
        self.shuffle: bool = kwargs.get('shuffle', False)
        self.problem_ids: Optional[List[str]] = kwargs.get('problem_ids', None)
        self.limit: Optional[int] = kwargs.get('limit', None)
        self.offset: Optional[int] = kwargs.get('offset', None)
        self.stride: Optional[int] = kwargs.get('stride', None)
        self.task_metadata: Optional[str] = kwargs.get('task_metadata', None)
        
        # Allow additional kwargs to be set as attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class BaseTask(ABC): 

    name: str = "BaseTask"
    Config = BaseTaskConfig
        
        
    def __init__(self, config: Config):
        self.config = config
    
    def _subset_dataset(self, dataset: List[Problem]) -> List[Problem]:
        if self.config.shuffle:
            import random
            random.shuffle(dataset)

        if self.config.problem_ids is not None:
            dataset = [row for row in dataset if row.id in self.config.problem_ids]

        # Apply limit, stride, and offset
        limit = self.config.limit if self.config.limit is not None else len(dataset)
        stride = self.config.stride if self.config.stride is not None else 1
        offset = self.config.offset if self.config.offset is not None else 0
        dataset = dataset[offset:limit:stride]
        return dataset

    @abstractmethod
    def score(self, pred: str, answer: str) -> Union[float, int]:
        # Can be regex, LLM as a judge, reward model, text similarity
        raise NotImplementedError()

    @abstractmethod
    def build_dataset(self) -> List[Problem]:
        """
        You should call self._subset_dataset(...) at the end of this function.
        """
        raise NotImplementedError()
    
