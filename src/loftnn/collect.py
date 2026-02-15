import time

import torch
import torch.distributed as dist
import torch.nn as nn

from torch import Tensor

from loftnn.batch import max_batch_size
from loftnn.model_utils import LayerInputProvider
from loftnn.types import Device
from loftnn.worker_utils import compute_capacity, free_memory


class WorkerDataCollector:
    def __init__(
        self, layer_input: LayerInputProvider, device: Device, num_workers: int
    ):
        self.layer_input = layer_input
        self.device = device
        self.num_workers = num_workers

    def collect_compute_capacities(self) -> dict[int, int]:
        return self._collect_from_workers(compute_capacity(self.device))

    def collect_batch_size_limits(
        self,
        model: nn.Module,
        input_sample: Tensor,
    ) -> dict[int, int]:
        batch_size_limit = max_batch_size(
            model, self.layer_input, self.device, input_sample
        )
        return self._collect_from_workers(batch_size_limit)

    def collect_device_memory(self) -> dict[int, dict[Device, int]]:
        rank = dist.get_rank()

        cpu_memory = self._collect_from_workers(free_memory(Device.cpu))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cuda_memory = self._collect_from_workers(free_memory(Device.cuda))

        device_memory = None
        if rank == 0:
            device_memory = {}
            for i, m in cpu_memory.items():
                device_memory[i] = {Device.cpu: m}
            for i, m in cuda_memory.items():
                device_memory[i][Device.cuda] = m
        return device_memory

    def collect_d2d_bandwidths(self) -> dict[tuple[int, int], int]:
        size = 10_000_000  # 10 MB
        rank = dist.get_rank()

        bandwidths_to = torch.zeros(self.num_workers)
        for src in range(self.num_workers):
            for dst in range(self.num_workers):
                if src == dst:
                    continue

                dist.barrier()

                if rank == dst:
                    recv = torch.empty(size, dtype=torch.uint8)

                    req = dist.irecv(recv, src=src)
                    req.wait()
                elif rank == src:
                    send = torch.randint(0, 2, (size,), dtype=torch.uint8)

                    start = time.time()
                    req = dist.isend(send, dst=dst)
                    req.wait()
                    end = time.time()
                    elapsed = end - start

                    bandwidth = send.numel() / elapsed
                    bandwidths_to[dst] = bandwidth

        if rank == 0:
            gather_list = [
                torch.zeros_like(bandwidths_to) for _ in range(self.num_workers)
            ]
        else:
            gather_list = None
        dist.gather(bandwidths_to, gather_list, dst=0)

        d2d_bandwidths = None
        if rank == 0:
            d2d_bandwidths = {}
            for i in range(self.num_workers):
                for j in range(i + 1, self.num_workers):
                    d2d_bandwidths[(i, j)] = int(
                        (gather_list[i][j].item() + gather_list[j][i].item()) / 2
                    )
        return d2d_bandwidths

    def _collect_from_workers(self, data) -> dict[int, int]:
        rank = dist.get_rank()
        tensor = torch.tensor(data, device=self.device)
        if rank == 0:
            gather_list = [torch.zeros_like(tensor) for _ in range(self.num_workers)]
        else:
            gather_list = None
        dist.gather(tensor, gather_list, dst=0)
        return (
            {i: int(gather_list[i].item()) for i in range(self.num_workers)}
            if rank == 0
            else None
        )
