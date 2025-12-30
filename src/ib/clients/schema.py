from dataclasses import dataclass
from typing import Literal, Type

from pydrantic import BaseConfig

@dataclass
class SampleResponse:
    tokens: list[str]
    token_ids: list[int]
    log_prob: list[float]
    stop_reason: Literal["max_tokens", "stop_string"]

    @property
    def text(self):
        return "".join(self.tokens)

@dataclass
class ProposalOutput:
    tokens: list[str]
    log_prob: float

@dataclass
class EvaluateResponse:
    tokens: list[str]
    log_prob: list[float]
    # last_token_data: SampleResponse



class ClientConfig(BaseConfig):
    model_name: str
    target: Type 