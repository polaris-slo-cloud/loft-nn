import time

import torch
import torch.distributed as dist
import torch.nn as nn

from torch import Tensor

from loftnn.batch import max_batch_size
from loftnn.model_utils import repeat_forward_execution
from loftnn.types import Device


class WorkerDataCollector:
    def collect_compute_capacities(
        self, device: Device, num_workers: int
    ) -> dict[int, int]:
        layer = nn.Linear(10_000, 10_000).to(device)
        execution_time = repeat_forward_execution(
            layer, device, batch_size=10, repetitions=3
        )
        compute_capacity = 1 / execution_time
        return self._collect_from_workers(compute_capacity, device, num_workers)

    def collect_available_memory(
        self, model: nn.Module, device: Device, input_sample: Tensor, num_workers: int
    ) -> dict[int, int]:
        batch_size_limit = max_batch_size(model, device, input_sample)
        return self._collect_from_workers(batch_size_limit, device, num_workers)

    def collect_d2d_bandwidths(self, num_workers: int) -> dict[tuple[int, int], int]:
        size = 10_000_000  # 10 MB
        rank = dist.get_rank()

        bandwidths_to = torch.zeros(num_workers)
        for src in range(num_workers):
            for dst in range(num_workers):
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
            gather_list = [torch.zeros_like(bandwidths_to) for _ in range(num_workers)]
        else:
            gather_list = None
        dist.gather(bandwidths_to, gather_list, dst=0)

        d2d_bandwidths = None
        if rank == 0:
            d2d_bandwidths = {}
            for i in range(num_workers):
                for j in range(i + 1, num_workers):
                    d2d_bandwidths[(i, j)] = int(
                        (gather_list[i][j].item() + gather_list[j][i].item()) / 2
                    )
        return d2d_bandwidths

    def _collect_from_workers(
        self, data, device: Device, num_workers: int
    ) -> dict[int, int]:
        rank = dist.get_rank()
        tensor = torch.tensor(data, device=device)
        if rank == 0:
            gather_list = [torch.zeros_like(tensor) for _ in range(num_workers)]
        else:
            gather_list = None
        dist.gather(tensor, gather_list, dst=0)
        return (
            {i: int(gather_list[i].item()) for i in range(num_workers)}
            if rank == 0
            else None
        )
