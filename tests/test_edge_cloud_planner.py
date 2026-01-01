from loftnn.cache import CacheProvider
from loftnn.edge_cloud_planner import EdgeCloudPlanner
from loftnn.profiler import NoopProfiler
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Worker, WorkerGroup

from .base_test_planner import BaseTestPlanner
from .sample_model import SimpleNN, BalancedNN3
from .sample_worker import edge_cloud_worker_group


class TestEdgeCloudPlanner(BaseTestPlanner):

    def setup_method(self):
        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        self.profiler = NoopProfiler()
        self.planner = EdgeCloudPlanner(sample_allocator=sample_allocator)

    def test_plan(self):
        model = SimpleNN()
        worker_group = edge_cloud_worker_group()

        micro_batch_size = 60
        num_micro_batches = 4

        split_points, device_groups, samples_allocated = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 2
        assert split_points == [1, 3]
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

        worker1 = Worker(rank=0, compute_capacity=1 / 1000, available_memory=32)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=32)
        worker3 = Worker(rank=2, compute_capacity=1 / 2000, available_memory=32)
        worker4 = Worker(rank=3, compute_capacity=1 / 3000, available_memory=32)
        workers = [worker1, worker2, worker3, worker4]
        worker_group = WorkerGroup(
            workers=workers,
            d2d_bandwidths={
                (workers[i], workers[j]): 10**5
                for i in range(len(workers))
                for j in range(i + 1, len(workers))
            },
        )

        micro_batch_size = 32
        num_micro_batches = 4

        split_points, device_groups, samples_allocated = self.planner.plan(
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
