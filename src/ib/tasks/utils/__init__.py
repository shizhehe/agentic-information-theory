from typing import Any, Dict, Optional, List
from pydantic import Field
from pydrantic import BaseConfig
import wandb    

class WandBConfig(BaseConfig):
    project: str = "butternut"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None
    

def prepare_wandb(
    config: WandBConfig,
    config_dict: Dict[str, Any],
):
    wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.name,
        tags=config.tags,
        notes=config.notes,
        group=config.group,
        config=config_dict,
    )


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Borrowed from pytorch lightning"""
    import random
    import numpy as np
    import torch
    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.