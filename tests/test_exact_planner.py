import pytest

from loftnn.cache import CacheProvider
from loftnn.exact_planner import ExactPlanner
from loftnn.exact_profiler import ExactProfiler
from loftnn.model_utils import LayerInputProvider
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Worker, WorkerGroup

from .sample_model import BalancedNN2, BalancedNN3, SimpleNN, get_sample_input


class TestExactPlanner:

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

    @pytest.mark.parametrize("parallel", [False, True])
    def test_plan(self, parallel):
        model = SimpleNN()

        micro_batch_size = 40
        num_micro_batches = 4
        input = LayerInputProvider()
        input.prepare(
            model, input_shape=get_sample_input(model, micro_batch_size).shape
        )

        worker1 = Worker(
            rank=0, compute_capacity=1 / 2000, batch_size_limit=num_micro_batches * 40
        )
        worker2 = Worker(
            rank=1, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 20
        )
        worker3 = Worker(
            rank=2, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 20
        )
        worker_group = WorkerGroup(
            workers=[worker1, worker2, worker3],
            d2d_bandwidths={
                (worker1, worker2): 1,
                (worker2, worker3): 10**100,
                (worker1, worker3): 1,
            },
        )

        self.profiler.profile(model, input, worker_group, micro_batch_size)
        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches, parallel
        )

        # expected:
        # Q = {
        #   (6,3,2): (6,3,3,2,1)
        #   (3,2,1): (3,2,0,0,0)
        # }
        # Qt = {
        #    (6,3,5,1,1): (latency, {worker2: 20, worker3: 20})
        #    (5,1,0,0,0): (latency, {worker1: 40})
        # }

        assert len(split_points) == 1
        assert split_points[0] >= 1 and split_points[0] <= 5
        assert len(device_groups) == 1
        assert device_groups == [1]
        assert len(samples_allocated) == 2
        assert samples_allocated == [{worker1: 40}, {worker2: 20, worker3: 20}]

    def test_plan_low_bandwidth_favors_pipeline_parallelism(self):
        model = BalancedNN2()

        micro_batch_size = 32
        num_micro_batches = 4
        input = LayerInputProvider()
        input.prepare(
            model, input_shape=get_sample_input(model, micro_batch_size).shape
        )

        worker1 = Worker(
            rank=0, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 32
        )
        worker2 = Worker(
            rank=1, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 32
        )
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 1}
        )

        self.profiler.profile(model, input, worker_group, micro_batch_size)
        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 1
        assert split_points == [1]
        assert len(device_groups) == 1
        assert device_groups == [1]
        assert len(samples_allocated) == 2
        assert samples_allocated == [{worker1: 32}, {worker2: 32}]

    def test_plan_high_bandwidth_favors_data_parallelism(self):
        model = BalancedNN2()

        micro_batch_size = 20_000
        num_micro_batches = 4
        input = LayerInputProvider()
        input.prepare(
            model, input_shape=get_sample_input(model, micro_batch_size).shape
        )

        worker1 = Worker(
            rank=0,
            compute_capacity=1 / 1000,
            batch_size_limit=num_micro_batches * 10_000,
        )
        worker2 = Worker(
            rank=1,
            compute_capacity=1 / 1000,
            batch_size_limit=num_micro_batches * 10_000,
        )
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 10**100}
        )

        self.profiler.profile(model, input, worker_group, micro_batch_size)
        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 1
        assert split_points == [0]
        assert len(device_groups) == 1
        assert device_groups == [0]
        assert len(samples_allocated) == 1
        assert samples_allocated == [{worker1: 10_000, worker2: 10_000}]

    def test_plan_provides_multiple_split_points(self):
        model = BalancedNN3(features=10000)

        micro_batch_size = 48
        num_micro_batches = 4
        input = LayerInputProvider()
        input.prepare(
            model, input_shape=get_sample_input(model, micro_batch_size).shape
        )

        worker1 = Worker(
            rank=0, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 48
        )
        worker2 = Worker(
            rank=1, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 48
        )
        worker3 = Worker(
            rank=2, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 48
        )
        worker_group = WorkerGroup(
            workers=[worker1, worker2, worker3],
            d2d_bandwidths={
                (worker1, worker2): 1,
                (worker2, worker3): 1,
                (worker1, worker3): 1,
            },
        )

        self.profiler.profile(
            model, input, worker_group, micro_batch_size, repetitions=3
        )
        split_points, device_groups, samples_allocated, _ = self.planner.plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 2
        assert split_points == [1, 2]
        assert len(device_groups) == 2
        assert device_groups == [1, 2]
        assert len(samples_allocated) == 3
        assert samples_allocated == [{worker1: 48}, {worker2: 48}, {worker3: 48}]
