from dataclasses import dataclass
from typing import Callable

import logging
import os

from torch import Tensor

logger = logging.getLogger(__name__)


def is_available():
    rank = os.environ.get("RANK", -1)
    local_rank = os.environ.get("LOCAL_RANK", -1)
    world_size = os.environ.get("WORLD_SIZE", -1)
    master_addr = os.environ.get("MASTER_ADDR", "-1")
    master_port = os.environ.get("MASTER_PORT", -1)

    logger.info(
        f"Got RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}"
    )

    return (
        int(rank) != -1
        and int(local_rank) != -1
        and int(world_size) != -1
        and master_addr != "-1"
        and int(master_port) != -1
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


@dataclass(frozen=True)
class HybridPipelineConfiguration:
    num_microbatches: int
    microbatch_sample: Tensor
    loss_fn: Callable
