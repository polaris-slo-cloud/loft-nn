import torch
import torch.nn as nn

from loftnn.amb_planner import AMBPlanner
from loftnn.cache import CacheProvider
from loftnn.ecs_planner import ECSPlanner
from loftnn.model_utils import LayerInputProvider
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Device, Worker, WorkerGroup


class TestAMBPlanner:

    def setup_method(self):
        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        self.layer_input = LayerInputProvider()
        self.planner = ECSPlanner(
            layer_input=self.layer_input,
            sample_allocator=sample_allocator,
            device=Device.cpu,
        )
        self.ambp = AMBPlanner(self.layer_input, Device.cpu)

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
            device_memory={Device.cpu: 300000},
        )
        worker2 = Worker(
            rank=1,
            compute_capacity=1 / 1000,
            batch_size_limit=micro_batch_size * 32,
            device_memory={Device.cpu: 200000},
        )
        worker3 = Worker(
            rank=2,
            compute_capacity=1 / 1500,
            batch_size_limit=micro_batch_size * 32,
            device_memory={Device.cpu: 150000},
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

        split_points, device_groups, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )
        activation_checkpointing_budgets = self.ambp.plan_budgets(
            model, worker_group, split_points, device_groups
        )

        activation_checkpointing_budgets = [
            round(budget, 2) for budget in activation_checkpointing_budgets
        ]
        assert len(activation_checkpointing_budgets) == 3
        assert activation_checkpointing_budgets == [1, 0.79, 0.2]
