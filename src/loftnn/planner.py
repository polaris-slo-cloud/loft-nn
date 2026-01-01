from typing import Protocol

import torch.nn as nn

from loftnn.types import WorkerGroup


class Planner(Protocol):
    def plan(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        num_micro_batches: int,
    ): ...
