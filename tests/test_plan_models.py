import time

import pytest

import torch

from torchvision.models import resnet50

from loftnn.cache import CacheProvider
from loftnn.exact_planner import ExactPlanner
from loftnn.exact_profiler import ExactProfiler
from loftnn.model_utils import LayerInputProvider
from loftnn.sample_allocation import SampleAllocator
from loftnn.worker_groups import simple_worker_group


class TestPlanModels:
    def setup_method(self):
        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        self.profiler = ExactProfiler(
            cache_provider=cache_provider,
            sample_allocator=sample_allocator,
            distributed=False,
        )
        self.planner = ExactPlanner(
            profiler=self.profiler, sample_allocator=sample_allocator
        )

    @pytest.mark.skip(reason="run manually")
    def test_plan_resnet(self):
        resnet_model = resnet50(weights=None)

        micro_batch_size = 240
        num_micro_batches = 4
        input = LayerInputProvider()
        input.prepare(
            resnet_model, input_shape=torch.Size([micro_batch_size, 3, 224, 224])
        )

        worker_group = simple_worker_group(
            num_workers=6, batch_size_limit=num_micro_batches * 240
        )

        print("Profiling...")
        profile_start = time.time()
        self.profiler.profile(
            resnet_model, input, worker_group, micro_batch_size, repetitions=1
        )
        profile_end = time.time()
        print(f"Profiling took {profile_end - profile_start} seconds")

        print("Planning...")
        plan_start = time.time()
        split_points, device_groups, samples_allocated = self.planner.plan(
            resnet_model, worker_group, micro_batch_size, num_micro_batches
        )
        plan_end = time.time()
        print(f"Planning took {plan_end - plan_start} seconds")

        print()
        print(f"Split points: {split_points}")
        print(f"Device groups: {device_groups}")
        print(f"Samples allocated: {samples_allocated}")
