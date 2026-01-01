from typing import Protocol

import torch.nn as nn

from loftnn.types import Device, WorkerGroup

DEFAULT_DEVICE = Device.cpu
DEFAULT_REPETITIONS = 10


class Profiler(Protocol):
    def profile(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        device: Device = DEFAULT_DEVICE,
        repetitions: int = DEFAULT_REPETITIONS,
    ): ...


class NoopProfiler(Profiler):
    def profile(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        device: Device = DEFAULT_DEVICE,
        repetitions: int = DEFAULT_REPETITIONS,
    ):
        pass
