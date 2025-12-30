from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydrantic import ObjectConfig

from src.ib.clients.usage import Usage


@dataclass
class ProtocolResponse:
    text: str
    remote_usage: Usage
    edge_usage: Usage
    meta: Dict[str, Any]


class BaseProtocol(ABC):
    
    class Config(ObjectConfig):
        _pass_as_config: bool = True
    
    @abstractmethod
    def __call__(
        self,
        id: str,
        query: str,
        context: str,
        **kwargs
    ) -> ProtocolResponse:
        pass

