from dataclasses import dataclass
from typing import Callable

import os

from torch import Tensor


def is_available():
    return (
        int(os.environ.get("RANK", -1)) != -1
        and int(os.environ.get("LOCAL_RANK", -1)) != -1
        and int(os.environ.get("WORLD_SIZE", -1)) != -1
        and os.environ.get("MASTER_ADDR", "-1") != "-1"
        and int(os.environ.get("MASTER_PORT", -1)) != -1
    )


@dataclass(frozen=True)
class ProcessConfiguration:
    rank: int
    local_rank: int
    world_size: int
    backend: str = "gloo"

    @staticmethod
    def from_env():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return ProcessConfiguration(
            rank=rank, local_rank=local_rank, world_size=world_size
        )


@dataclass(frozen=True)
class PipelineConfiguration:
    split_points: list[str]
    num_microbatches: int
    microbatch_sample: Tensor
    loss_fn: Callable
