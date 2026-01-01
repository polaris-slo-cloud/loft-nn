import time

import pytest

from loftnn.planner import Planner
from loftnn.profiler import Profiler

from .sample_worker import simple_worker_group


class BaseTestPlanner:
    profiler: Profiler
    planner: Planner

    @pytest.mark.skip(reason="run manually")
    def test_plan_resnet(self):
        from torchvision.models import resnet50

        resnet_model = resnet50(weights=None)
        worker_group = simple_worker_group(num_workers=6, available_memory=80)
        micro_batch_size = 240
        num_micro_batches = 4

        print("Profiling...")
        profile_start = time.time()
        self.profiler.profile(
            resnet_model, worker_group, micro_batch_size, repetitions=1
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
