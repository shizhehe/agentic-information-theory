from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydrantic import ObjectConfig
from src.ib.clients.usage import Usage
from dataclasses import dataclass, asdict


@dataclass
class ClientSample:
    text: str  # Does NOT include eos_token
    tokens: Sequence[str]  # Includes eos_token
    token_ids: Sequence[int]
    log_prob: Sequence[float]
    input_log_prob: Sequence[float]
    stop_reason: Literal["max_tokens", "stop", "length", "error"]


@dataclass
class ClientResponse:
    samples: List[ClientSample]
    usages: Optional[List[Usage]] = None
    usage: Optional[Usage] = None

    timings: Optional[List[Dict[str, Any]]] = None

    def to_dict(self):
        return asdict(self)


class ClientConfig(ObjectConfig):
    _pass_as_config: bool = True

    model_name: str

    def instantiate(self, *args, **kwargs) -> "Client":
        return super().instantiate(*args, **kwargs)


class Client(ABC):
    def __init__(self, config: ClientConfig):
        self.config = config

    @abstractmethod
    def complete(
        self, prompts: List[Union[str, List[int]]], **kwargs
    ) -> ClientResponse:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
    ) -> ClientResponse:
        raise NotImplementedError
