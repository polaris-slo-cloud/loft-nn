from enum import Enum

import torch.nn as nn

from loftnn.types import Worker, WorkerGroup

ForwardExecutionTimeCache = dict[tuple[nn.Module, Worker, int], float]
BackwardExecutionTimeCache = dict[tuple[nn.Module, Worker, int], float]
TransmissionTimeCache = dict[tuple[nn.Module, int, bool, bool], float]
SampleAllocationCache = dict[tuple[WorkerGroup, int], dict[Worker, int]]


class CacheType(str, Enum):
    forward_execution_time = "forward_execution_time"
    backward_execution_time = "backward_execution_time"
    transmission_time = "transmission_time"
    sample_allocation_cache = "sample_allocation_cache"


class CacheProvider:
    forward_execution_time_cache: ForwardExecutionTimeCache
    backward_execution_time_cache: BackwardExecutionTimeCache
    transmission_time_cache: TransmissionTimeCache
    sample_allocation_cache: SampleAllocationCache

    def __init__(self):
        self.forward_execution_time_cache = {}
        self.backward_execution_time_cache = {}
        self.transmission_time_cache = {}
        self.sample_allocation_cache = {}

    def get_cache(self, cache_type: CacheType):
        if cache_type == CacheType.forward_execution_time:
            return self.forward_execution_time_cache
        elif cache_type == CacheType.backward_execution_time:
            return self.backward_execution_time_cache
        elif cache_type == CacheType.transmission_time:
            return self.transmission_time_cache
        elif cache_type == CacheType.sample_allocation_cache:
            return self.sample_allocation_cache
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
