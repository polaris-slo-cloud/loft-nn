import torch
import torch.nn as nn

from loftnn.cache import CacheProvider
from loftnn.edge_cloud_planner import EdgeCloudPlanner
from loftnn.model_utils import LayerInputProvider
from loftnn.profiler import NoopProfiler
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Device, Worker, WorkerGroup

from .sample_model import SimpleNN, BalancedNN3, get_sample_input
from .sample_worker import edge_cloud_worker_group


class TestEdgeCloudPlanner:

    def setup_method(self):
        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        self.layer_input = LayerInputProvider()
        self.profiler = NoopProfiler()
        self.planner = EdgeCloudPlanner(
            layer_input=self.layer_input,
            sample_allocator=sample_allocator,
            device=Device.cpu,
        )

    def test_plan(self):
        model = SimpleNN()

        micro_batch_size = 60
        num_micro_batches = 4
        self.layer_input.prepare(
            model, input_shape=get_sample_input(model, micro_batch_size).shape
        )

        worker_group = edge_cloud_worker_group(num_micro_batches)

        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 2
        assert split_points == [2, 3]
        assert len(device_groups) == 2
        assert device_groups == [3, 4]
        assert len(samples_allocated) == 3
        assert samples_allocated == [
            {
                worker_group.workers[0]: 20,
                worker_group.workers[1]: 20,
                worker_group.workers[2]: 20,
            },
            {worker_group.workers[3]: 60},
            {worker_group.workers[4]: 60},
        ]

    def test_plan_splits_based_on_compute_capacity(self):
        model = BalancedNN3()

        micro_batch_size = 32
        num_micro_batches = 4
        self.layer_input.prepare(
            model, input_shape=get_sample_input(model, micro_batch_size).shape
        )

        worker1 = Worker(
            rank=0, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 32
        )
        worker2 = Worker(
            rank=1, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 32
        )
        worker3 = Worker(
            rank=2, compute_capacity=1 / 2000, batch_size_limit=num_micro_batches * 32
        )
        worker4 = Worker(
            rank=3, compute_capacity=1 / 3000, batch_size_limit=num_micro_batches * 32
        )
        workers = [worker1, worker2, worker3, worker4]
        worker_group = WorkerGroup(
            workers=workers,
            d2d_bandwidths={
                (workers[i], workers[j]): 10**5
                for i in range(len(workers))
                for j in range(i + 1, len(workers))
            },
        )

        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 2
        assert split_points == [1, 2]
        assert len(device_groups) == 2
        assert device_groups == [2, 3]
        assert len(samples_allocated) == 3
        assert samples_allocated == [
            {worker1: 16, worker2: 16},
            {worker3: 32},
            {worker4: 32},
        ]

    def test_plan_avoids_split_points_with_large_intermediate_activations(self):
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

        micro_batch_size = 32
        num_micro_batches = 4
        self.layer_input.prepare(model, input_shape=torch.Size([micro_batch_size, 10]))

        worker1 = Worker(
            rank=0, compute_capacity=1 / 1000, batch_size_limit=micro_batch_size * 32
        )
        worker2 = Worker(
            rank=1, compute_capacity=1 / 1000, batch_size_limit=micro_batch_size * 32
        )
        worker3 = Worker(
            rank=2, compute_capacity=1 / 2000, batch_size_limit=micro_batch_size * 32
        )
        worker4 = Worker(
            rank=3, compute_capacity=1 / 3000, batch_size_limit=micro_batch_size * 32
        )
        workers = [worker1, worker2, worker3, worker4]
        worker_group = WorkerGroup(
            workers=workers,
            d2d_bandwidths={
                (workers[i], workers[j]): 10**5
                for i in range(len(workers))
                for j in range(i + 1, len(workers))
            },
        )

        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 2
        assert split_points == [4, 6]
        assert len(device_groups) == 2
        assert device_groups == [2, 3]
        assert len(samples_allocated) == 3
        assert samples_allocated == [
            {worker1: 16, worker2: 16},
            {worker3: 32},
            {worker4: 32},
        ]

    def test_plan_activation_checkpointing_budgets(self):
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
        )

        micro_batch_size = 32
        num_micro_batches = 4
        self.layer_input.prepare(model, input_shape=torch.Size([micro_batch_size, 100]))

        worker1 = Worker(
            rank=0,
            compute_capacity=1 / 500,
            batch_size_limit=micro_batch_size * 32,
            device_memory={Device.cpu: 200000},
        )
        worker2 = Worker(
            rank=1,
            compute_capacity=1 / 1000,
            batch_size_limit=micro_batch_size * 32,
            device_memory={Device.cpu: 110000},
        )
        worker3 = Worker(
            rank=2,
            compute_capacity=1 / 1500,
            batch_size_limit=micro_batch_size * 32,
            device_memory={Device.cpu: 95000},
        )
        workers = [worker1, worker2, worker3]
        worker_group = WorkerGroup(
            workers=workers,
            d2d_bandwidths={
                (worker1, worker2): 10**5,
                (worker2, worker3): 1,
                (worker1, worker3): 1,
            },
        )

        _, _, _, activation_checkpointing_budgets = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        activation_checkpointing_budgets = [
            round(budget, 2) for budget in activation_checkpointing_budgets
        ]
        assert len(activation_checkpointing_budgets) == 3
        assert activation_checkpointing_budgets == [1, 0.75, 0.22]
