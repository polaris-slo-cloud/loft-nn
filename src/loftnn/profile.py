import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn import Embedding, ReLU

from loftnn.cache import CacheProvider, CacheType
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Worker, WorkerGroup
from loftnn.model_utils import (
    layer_id,
    model_to_layers,
    random_layer_input,
    repeat_backward_execution,
    repeat_forward_execution,
)


class Profiler:
    cache_provider: CacheProvider
    sample_allocator: SampleAllocator

    def __init__(
        self,
        cache_provider: CacheProvider,
        sample_allocator: SampleAllocator,
        distributed: bool = True,
    ):
        self.cache_provider = cache_provider
        self.sample_allocator = sample_allocator
        self.distributed = distributed

    def get_forward_execution_time(
        self, layer: nn.Module, worker: Worker, sample_size: int
    ) -> float:
        cache = self.cache_provider.get_cache(CacheType.forward_execution_time)
        return cache[(layer_id(layer), worker, sample_size)]

    def get_backward_execution_time(
        self, layer: nn.Module, worker: Worker, sample_size: int
    ) -> float:
        cache = self.cache_provider.get_cache(CacheType.backward_execution_time)
        return cache[(layer_id(layer), worker, sample_size)]

    def get_transmission_time(
        self,
        layer: nn.Module,
        d2d_bandwidth: int,
        forward_only=False,
        backward_only=False,
    ) -> float:
        cache = self.cache_provider.get_cache(CacheType.transmission_time)
        return cache[(layer_id(layer), d2d_bandwidth, forward_only, backward_only)]

    def profile(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        repetitions: int = 10,
    ):
        layers = model_to_layers(model)
        rank = dist.get_rank() if self.distributed else 0

        for n in range(1, len(worker_group) + 1):
            for l in range(1, len(layers) + 1):
                for np in range(n + 1):
                    for lp in range(l + 1):
                        worker_subgroup = worker_group[np:n]
                        self._profile_worker_subgroup(
                            worker_subgroup,
                            layers,
                            l,
                            lp,
                            micro_batch_size,
                            rank,
                            repetitions,
                        )

                    if l < len(layers) and rank == 0:
                        transmission_layer = (
                            layers[l]
                            if not (
                                isinstance(layers[l], nn.ReLU) and layers[l].inplace
                            )
                            else layers[l - 1]
                        )
                        mean_d2d_bandwidth = self.get_mean_d2d_bandwidth(
                            worker_group, worker_group[0:np], worker_group[np:n]
                        )
                        self._estimate_transmission_time(
                            transmission_layer,
                            mean_d2d_bandwidth,
                            forward_only=False,
                            backward_only=False,
                        )
                        self._estimate_transmission_time(
                            transmission_layer,
                            mean_d2d_bandwidth,
                            forward_only=True,
                            backward_only=False,
                        )
                        self._estimate_transmission_time(
                            transmission_layer,
                            mean_d2d_bandwidth,
                            forward_only=False,
                            backward_only=True,
                        )

        if self.distributed:
            forward_execution_time_cache = self.cache_provider.get_cache(
                CacheType.forward_execution_time
            )
            backward_execution_time_cache = self.cache_provider.get_cache(
                CacheType.backward_execution_time
            )

            if rank == 0:
                gather_list = [None for _ in range(dist.get_world_size())]
                dist.gather_object({}, gather_list, dst=0)

                for worker_cache in gather_list:
                    forward_execution_time_cache.update(worker_cache)

                gather_list = [None for _ in range(dist.get_world_size())]
                dist.gather_object({}, gather_list, dst=0)

                for worker_cache in gather_list:
                    backward_execution_time_cache.update(worker_cache)
            else:
                gather_list = None
                dist.gather_object(forward_execution_time_cache, gather_list, dst=0)
                dist.gather_object(backward_execution_time_cache, gather_list, dst=0)

    def _profile_worker_subgroup(
        self,
        worker_subgroup: WorkerGroup,
        layers: list[nn.Module],
        l: int,
        lp: int,
        micro_batch_size: int,
        rank: int,
        repetitions: int,
    ):
        self.sample_allocator.compute_allocation_of_micro_batch(
            worker_subgroup, micro_batch_size
        )

        try:
            samples_allocated = self.sample_allocator.get_allocation_of_micro_batch(
                worker_subgroup, micro_batch_size
            )
        except ValueError:
            return

        for worker in worker_subgroup:
            for la in range(lp, l):
                if worker.rank == rank or not self.distributed:
                    self._profile_forward_execution_time(
                        layers[la], worker, samples_allocated[worker], repetitions
                    )

                    self._profile_backward_execution_time(
                        layers[la], worker, samples_allocated[worker], repetitions
                    )

    def _profile_forward_execution_time(
        self, layer: nn.Module, worker: Worker, sample_size: int, repetitions: int
    ) -> float:
        cache = self.cache_provider.get_cache(CacheType.forward_execution_time)
        if (layer_id(layer), worker, sample_size) in cache:
            return

        layer.eval()
        execution_time = repeat_forward_execution(layer, sample_size, repetitions)
        layer.train()

        cache[(layer_id(layer), worker, sample_size)] = execution_time

    def _profile_backward_execution_time(
        self, layer: nn.Module, worker: Worker, sample_size: int, repetitions: int
    ) -> float:
        cache = self.cache_provider.get_cache(CacheType.backward_execution_time)
        if (layer_id(layer), worker, sample_size) in cache:
            return

        if isinstance(layer, ReLU):
            if layer.inplace:
                cache[(layer_id(layer), worker, sample_size)] = 0
                return

        if isinstance(layer, Embedding):
            cache[(layer_id(layer), worker, sample_size)] = 0
            return

        layer.eval()
        execution_time = repeat_backward_execution(layer, sample_size, repetitions)
        layer.train()

        cache[(layer_id(layer), worker, sample_size)] = execution_time

    def _estimate_transmission_time(
        self, layer: nn.Module, d2d_bandwidth: int, forward_only, backward_only
    ):
        cache = self.cache_provider.get_cache(CacheType.transmission_time)
        if (layer_id(layer), d2d_bandwidth, forward_only, backward_only) in cache:
            return

        if d2d_bandwidth < 0:
            raise ValueError("Bandwidth cannot be negative")

        if isinstance(layer, ReLU):
            if layer.inplace:
                cache[(layer_id(layer), d2d_bandwidth, forward_only, backward_only)] = 0
                return

        if isinstance(layer, Embedding):
            cache[(layer_id(layer), d2d_bandwidth, forward_only, backward_only)] = 0
            return

        size_intermediate_tensors = (
            (
                self._random_forward_output(layer, sample_size=1).shape
                if not backward_only
                else torch.empty(0)
            ),
            (
                self._random_backward_input(layer, sample_size=1).shape
                if not forward_only
                else torch.empty(0)
            ),
        )

        cache[(layer_id(layer), d2d_bandwidth, forward_only, backward_only)] = (
            (
                size_intermediate_tensors[0].numel()
                + size_intermediate_tensors[1].numel()
            )
            * 4
            / d2d_bandwidth
        )

    def _random_forward_output(self, layer: nn.Module, sample_size: int):
        x = random_layer_input(layer, sample_size)
        layer.eval()

        with torch.no_grad():
            y = layer(x)

        layer.train()

        return y

    def _random_backward_input(self, layer: nn.Module, sample_size: int):
        x = random_layer_input(layer, sample_size)
        x.requires_grad = True
        layer.eval()

        y = layer(x)
        y.mean().backward()

        layer.train()

        return x.grad

    def get_mean_d2d_bandwidth(
        self,
        worker_group_world: WorkerGroup,
        worker_group_1: WorkerGroup,
        worker_group_2: WorkerGroup,
    ) -> int:
        if worker_group_1 and not worker_group_2:
            return 1_000_000_000
        if not worker_group_1 and worker_group_2:
            return 1

        bandwidths = []
        for worker1 in worker_group_1:
            for worker2 in worker_group_2:
                bandwidth = worker_group_world.d2d_bandwidth(worker1, worker2)
                bandwidths.append(bandwidth)

        return int(sum(bandwidths) / len(bandwidths)) if bandwidths else -1
