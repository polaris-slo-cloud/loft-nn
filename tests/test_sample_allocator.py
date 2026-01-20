from loftnn.cache import CacheProvider
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Worker, WorkerGroup


class TestSampleAllocator:

    def setup_method(self):
        cache_provider = CacheProvider()
        self.sample_allocator = SampleAllocator(cache_provider=cache_provider)

    def test_memory_aware_balancing(self):
        worker1 = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 8)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(workers=[worker1, worker2], d2d_bandwidths={})
        micro_batch_size = 24

        samples_allocated = self.sample_allocator._memory_aware_balancing(
            worker_group, micro_batch_size
        )

        assert samples_allocated == {worker1: 8, worker2: 16}

    def test_memory_aware_balancing_tie_breaking(self):
        worker1 = Worker(rank=0, compute_capacity=1 / 1000, available_memory=8)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=8)
        worker_group = WorkerGroup(workers=[worker1, worker2], d2d_bandwidths={})
        micro_batch_size = 7

        samples_allocated = self.sample_allocator._memory_aware_balancing(
            worker_group, micro_batch_size
        )

        assert samples_allocated == {worker1: 4, worker2: 3}

    def test_memory_aware_balancing_remainders(self):
        worker1 = Worker(rank=0, compute_capacity=1 / 1000, available_memory=240)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=240)
        worker3 = Worker(rank=2, compute_capacity=1 / 1000, available_memory=240)
        worker4 = Worker(rank=3, compute_capacity=1 / 995, available_memory=240)
        worker5 = Worker(rank=4, compute_capacity=1 / 995, available_memory=240)
        worker6 = Worker(rank=5, compute_capacity=1 / 1000, available_memory=240)
        worker7 = Worker(rank=6, compute_capacity=1 / 1000, available_memory=240)
        worker_group = WorkerGroup(
            workers=[worker1, worker2, worker3, worker4, worker5, worker6, worker7],
            d2d_bandwidths={},
        )
        micro_batch_size = 240

        samples_allocated = self.sample_allocator._memory_aware_balancing(
            worker_group, micro_batch_size
        )

        assert samples_allocated == {
            worker1: 34,
            worker2: 34,
            worker3: 34,
            worker4: 35,
            worker5: 35,
            worker6: 34,
            worker7: 34,
        }

    def test_straggler_workload_offloading(self):
        slow_worker = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 8)
        fast_worker = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(
            workers=[slow_worker, fast_worker],
            d2d_bandwidths={},
        )
        samples_allocated = {slow_worker: 16, fast_worker: 8}

        samples_allocated = self.sample_allocator._straggler_workload_offloading(
            worker_group, samples_allocated, samples_to_transfer=4
        )

        assert samples_allocated == {slow_worker: 12, fast_worker: 12}

    def test_straggler_workload_offloading_keeps_samples_on_workers(self):
        slow_worker = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 8)
        fast_worker = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(
            workers=[slow_worker, fast_worker],
            d2d_bandwidths={},
        )
        samples_allocated = {slow_worker: 4, fast_worker: 3}

        samples_allocated = self.sample_allocator._straggler_workload_offloading(
            worker_group, samples_allocated, samples_to_transfer=4
        )

        assert samples_allocated == {slow_worker: 4, fast_worker: 3}
