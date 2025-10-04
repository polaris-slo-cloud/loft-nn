import copy

from dataclasses import dataclass
from enum import Enum


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


@dataclass(frozen=True)
class Worker:
    rank: int
    compute_capacity: int
    available_memory: int

    def batch_size(self) -> int:
        return self.available_memory // 4

    def is_faster(self, other) -> bool:
        return self.compute_capacity > other.compute_capacity


@dataclass
class WorkerGroup:
    workers: list[Worker]
    d2d_bandwidths: dict[tuple[Worker, Worker], int]

    def get_slowest(self):
        return min(self.workers, key=lambda d: d.compute_capacity)

    def get_fastest_with_memory(self, samples_to_transfer):
        candidates = [
            d for d in self.workers if d.available_memory >= samples_to_transfer * 4
        ]
        if not candidates:
            raise ValueError("No device has enough memory to transfer the samples")
        return max(candidates, key=lambda d: d.compute_capacity)

    def d2d_bandwidth(self, worker1: Worker, worker2: Worker):
        if (worker1, worker2) in self.d2d_bandwidths:
            return self.d2d_bandwidths[(worker1, worker2)]
        if (worker2, worker1) in self.d2d_bandwidths:
            return self.d2d_bandwidths[(worker2, worker1)]
        return None

    def min_d2d_bandwidth(self):
        return min([d2d_bandwidth for d2d_bandwidth in self.d2d_bandwidths.values()])

    def sorted(self, key):
        worker_group = copy.copy(self)
        worker_group.workers = sorted(self.workers, key=key)
        return worker_group

    def __getitem__(self, index):
        worker_group = copy.copy(self)
        worker_group.workers = self.workers[index]
        return worker_group

    def __len__(self):
        return len(self.workers)

    def __iter__(self):
        return iter(self.workers)
