import argparse
import logging

import matplotlib.pyplot as plt

from torchvision.models import resnet18, resnet50

from loftnn.cache import CacheProvider
from loftnn.exact_hybrid_pipeline_parallel_planner import Planner
from loftnn.experiments import Experiment, ExperimentRunner, ExperimentVisualizer
from loftnn.profile import Profiler
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Worker, WorkerGroup

logger = logging.getLogger(__name__)


class PlanningTimeExperiment(Experiment):
    def __init__(self):
        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        self.profiler = Profiler(
            cache_provider=cache_provider,
            sample_allocator=sample_allocator,
            distributed=False,
        )
        self.planner = Planner(
            profiler=self.profiler, sample_allocator=sample_allocator
        )

    def _worker_group(self, num_workers: int) -> WorkerGroup:
        workers = [
            Worker(rank=i, compute_capacity=1 / 1000, available_memory=4 * 80)
            for i in range(num_workers)
        ]
        d2d_bandwidths = {
            (workers[i], workers[j]): 1
            for i in range(num_workers)
            for j in range(i + 1, num_workers)
        }
        worker_group = WorkerGroup(workers=workers, d2d_bandwidths=d2d_bandwidths)
        return worker_group

    def run(self):
        resnet18_model = resnet18(weights=None)
        resnet50_model = resnet50(weights=None)

        configurations = [
            ("ResNet-18", resnet18_model, self._worker_group(3), 240, 4),
            ("ResNet-50", resnet50_model, self._worker_group(6), 240, 4),
        ]
        for configuration in configurations:
            name, model, worker_group, micro_batch_size, num_micro_batches = (
                configuration
            )

            logger.info(f"Profiling for {name}...")
            profile_duration, _ = self.time(
                lambda: self.profiler.profile(
                    model, worker_group, micro_batch_size, repetitions=1
                )
            )

            logger.info(f"Sequential planning for {name}...")
            plan_duration_seq, (
                split_points_seq,
                device_groups_seq,
                samples_allocated_seq,
            ) = self.time(
                lambda: self.planner.plan(
                    model, worker_group, micro_batch_size, num_micro_batches
                )
            )

            logger.info(f"Parallel planning for {name}...")
            plan_duration_par, (
                split_points_par,
                device_groups_par,
                samples_allocated_par,
            ) = self.time(
                lambda: self.planner.plan(
                    model,
                    worker_group,
                    micro_batch_size,
                    num_micro_batches,
                    parallel=True,
                )
            )

            logger.info("Verifying...")
            assert split_points_seq == split_points_par
            assert device_groups_seq == device_groups_par
            assert samples_allocated_seq == samples_allocated_par

            yield {
                "id": name,
                "profileDuration": profile_duration,
                "sequentialPlanDuration": plan_duration_seq,
                "parallelPlanDuration": plan_duration_par,
                "splitPoints": split_points_seq,
                "deviceGroups": device_groups_seq,
                "samplesAllocated": [
                    {worker.rank: allocation}
                    for sample_allocation in samples_allocated_seq
                    for (worker, allocation) in sample_allocation.items()
                ],
            }

    def visualize(self, results):
        xs = []
        heights = []
        for result in results:
            xs.append(f"{result['id']}_seq")
            xs.append(f"{result['id']}_par")
            heights.append(result["sequentialPlanDuration"])
            heights.append(result["parallelPlanDuration"])

        fig, ax = plt.subplots()
        ax.bar(xs, heights)
        ax.set_ylabel("Duration (s)")
        ax.set_title("Planning Time")
        plt.show()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    if args.visualize:
        ExperimentVisualizer.visualize_all()
    else:
        ExperimentRunner.run_all()


if __name__ == "__main__":
    main()
