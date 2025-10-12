import pytest

from loftnn.exact_hybrid_pipeline_parallel_planner import (
    memory_aware_balancing,
    plan,
    straggler_workload_offloading,
)
from loftnn.types import Worker, WorkerGroup

from .sample_model import BalancedNN, SimpleNN

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

        split_points, device_groups, samples_allocated = plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 1
        assert split_points[0] >= 1 and split_points[0] <= 5
        assert len(device_groups) == 1
        assert device_groups == [1]
        assert len(samples_allocated) == 1
        assert samples_allocated == [{worker2: 24}]
   
    def test_plan_low_bandwidth_favors_pipeline_parallelism(self):
        model = BalancedNN()
        
        worker1 = Worker(rank=0, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 16)
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 1}
        )
        micro_batch_size = 32
        num_micro_batches = 4

        split_points, device_groups, samples_allocated = plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 1
        assert split_points == [1]
        assert len(device_groups) == 1
        assert device_groups == [1]
        assert len(samples_allocated) == 1
        assert samples_allocated == [{worker2: 32}]

    def test_plan_high_bandwidth_favors_data_parallelism(self):
        model = BalancedNN()
        
        worker1 = Worker(rank=0, compute_capacity=1 / 1000, available_memory=4 * 10_000)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 10_000)
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 10**100}
        )
        micro_batch_size = 20_000
        num_micro_batches = 4

        split_points, device_groups, samples_allocated = plan(
            model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 1
        assert split_points == [0]
        assert len(device_groups) == 1
        assert device_groups == [0]
        assert len(samples_allocated) == 1
        assert samples_allocated == [{worker1: 10_000, worker2: 10_000}]
    
    @pytest.mark.skip(reason="run manually")
    def test_plan_resnet(self):
        from torchvision.models import resnet50
        resnet_model = resnet50(weights=None)
        worker1 = Worker(rank=0, compute_capacity=1 / 2000, available_memory=4 * 80)
        worker2 = Worker(rank=1, compute_capacity=1 / 1000, available_memory=4 * 160)
        worker_group = WorkerGroup(
            workers=[worker1, worker2], d2d_bandwidths={(worker1, worker2): 1}
        )
        micro_batch_size = 240
        num_micro_batches = 4

        split_points, device_groups, samples_allocated = plan(
            resnet_model, worker_group, micro_batch_size, num_micro_batches
        )

        assert len(split_points) == 1
        assert len(device_groups) == 1
        assert len(samples_allocated) == 1


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
