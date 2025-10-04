from loftnn.exact_hybrid_pipeline_parallel_planner import (
    memory_aware_balancing,
    plan,
    straggler_workload_offloading,
)
from loftnn.types import Worker, WorkerGroup

from .sample_model import SimpleNN


class TestExactHybridPipelineParallelPlanner:

    def test_plan(self):
        model = SimpleNN()
        worker1 = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 8)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 1}
        )
        micro_batch_size = 24
        num_micro_batches = 4

        num_stages, hpp_plan = plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )
        l, n, lp, np, p, _ = hpp_plan

        assert num_stages == 2
        assert l == 5
        assert n == 1
        assert lp >= 1 and lp <= 5
        assert np == 0
        assert p == 1

    def test_memory_aware_balancing(self):
        worker1 = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 8)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 1}
        )
        micro_batch_size = 24

        samples_allocated = memory_aware_balancing(worker_group, micro_batch_size)

        assert samples_allocated == {worker1: 8, worker2: 16}

    def test_straggler_workload_offloading(self):
        slow_worker = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 8)
        fast_worker = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(
            workers=[slow_worker, fast_worker],
            d2d_bandwidths={(slow_worker, fast_worker): 1},
        )
        samples_allocated = {slow_worker: 16, fast_worker: 8}

        samples_allocated = straggler_workload_offloading(
            worker_group, samples_allocated, samples_to_transfer=4
        )

        assert samples_allocated == {slow_worker: 12, fast_worker: 12}
