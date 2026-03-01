import argparse
import logging

import torch

from torchvision.models import resnet50

from loftnn.cache import CacheProvider
from loftnn.ecs_planner import ECSPlanner
from loftnn.exact_planner import ExactPlanner
from loftnn.experiments import Experiment, ExperimentRunner, ExperimentVisualizer
from loftnn.exact_profiler import ExactProfiler
from loftnn.model_utils import LayerInputProvider
from loftnn.sample_allocation import SampleAllocator
from loftnn.tools.plot import plot_planning_time_scaling
from loftnn.types import Device, WorkerGroup

from ..sample_worker import simple_worker_group

logger = logging.getLogger(__name__)


class PlanningTimeScalingExperiment(Experiment):
    def _worker_group(self, num_workers: int) -> WorkerGroup:
        return simple_worker_group(num_workers, batch_size_limit=240)

    def run(self, device: Device):
        layer_input = LayerInputProvider()

        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        exact_profiler = ExactProfiler(
            cache_provider=cache_provider,
            sample_allocator=sample_allocator,
            distributed=False,
        )
        exact_planner = ExactPlanner(
            profiler=exact_profiler, sample_allocator=sample_allocator
        )

        ecs_planner = ECSPlanner(
            layer_input=layer_input,
            sample_allocator=SampleAllocator(cache_provider=CacheProvider()),
            device=device,
        )

        resnet50_model = resnet50(weights=None)

        configurations = [
            (
                f"ResNet-50 with {i} workers",
                resnet50_model,
                self._worker_group(i),
                240,
                4,
                False,
            )
            for i in range(3, 10 + 1)
        ]
        for configuration in configurations:
            name, model, worker_group, micro_batch_size, num_micro_batches, parallel = (
                configuration
            )

            model = model.to(device)
            layer_input.prepare(
                model, input_shape=torch.Size([micro_batch_size, 3, 224, 224])
            )

            logger.info(f"Exact profiling for {name}...")
            exact_profile_duration, _ = self.time(
                lambda: exact_profiler.profile(
                    model,
                    layer_input,
                    worker_group,
                    micro_batch_size,
                    device=device,
                    repetitions=1,
                )
            )
            logger.info(f"Took {exact_profile_duration}s")

            logger.info(f"Exact planning for {name}...")
            exact_plan_duration, (
                split_points_exact,
                device_groups_exact,
                samples_allocated_exact,
            ) = self.time(
                lambda: exact_planner.plan(
                    model,
                    worker_group,
                    micro_batch_size,
                    num_micro_batches,
                    parallel=parallel,
                )
            )
            logger.info(f"Took {exact_plan_duration}s")

            logger.info(f"ECS planning for {name}...")
            ecs_plan_duration, (
                split_points_ecs,
                device_groups_ecs,
                samples_allocated_ecs,
            ) = self.time(
                lambda: ecs_planner.plan(
                    model, worker_group, micro_batch_size, num_micro_batches
                )
            )
            logger.info(f"Took {ecs_plan_duration}s")

            yield {
                "id": name,
                "num_workers": len(worker_group),
                "exact": {
                    "profileDuration": exact_profile_duration,
                    "planDuration": exact_plan_duration,
                    "splitPoints": split_points_exact,
                    "deviceGroups": device_groups_exact,
                    "samplesAllocated": [
                        {worker.rank: allocation}
                        for sample_allocation in samples_allocated_exact
                        for (worker, allocation) in sample_allocation.items()
                    ],
                },
                "ecs": {
                    "profileDuration": 0,
                    "planDuration": ecs_plan_duration,
                    "splitPoints": split_points_ecs,
                    "deviceGroups": device_groups_ecs,
                    "samplesAllocated": [
                        {worker.rank: allocation}
                        for sample_allocation in samples_allocated_ecs
                        for (worker, allocation) in sample_allocation.items()
                    ],
                },
            }

    def visualize(self, results):
        labels = ["Exact planner", "ECS planner"]
        worker_counts = [result["num_workers"] for result in results]
        planning_times = [
            [result["exact"]["planDuration"] for result in results],
            [result["ecs"]["planDuration"] for result in results],
        ]
        plot_planning_time_scaling(
            labels=labels,
            worker_counts=worker_counts,
            planning_times=planning_times,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--log_level", type=str, default="WARNING")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    if args.visualize:
        ExperimentVisualizer.visualize_all()
    else:
        ExperimentRunner.run_all(args.device)


if __name__ == "__main__":
    main()
