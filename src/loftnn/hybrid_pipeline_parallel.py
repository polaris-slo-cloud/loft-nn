import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.pipelining import pipeline, SplitPoint

from torch.nn.parallel import DistributedDataParallel as DDP

from loftnn import HybridPipelineConfiguration, ProcessConfiguration
from loftnn.cache import CacheProvider
from loftnn.collect import WorkerDataCollector
from loftnn.exact_hybrid_pipeline_parallel_planner import Planner
from loftnn.model_utils import model_to_layers
from loftnn.pipeline_schedule import ScheduleHybridPipelineParallel1F1B
from loftnn.profile import Profiler
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Device, Worker, WorkerGroup


class HybridPipelineParallel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        process_config: ProcessConfiguration,
        hybrid_pipeline_config: HybridPipelineConfiguration,
        device: Device,
    ):
        super(HybridPipelineParallel, self).__init__()
        self.model = model
        self.process_config = process_config
        self.hybrid_pipeline_config = hybrid_pipeline_config
        self.device = device

        self.pipeline_schedule_train = None
        self.pipeline_schedule_eval = None

        init_process_group(
            backend=process_config.backend,
            world_size=process_config.world_size,
            rank=process_config.rank,
        )

    def _create_worker_group(
        self, compute_capacities, available_memory, d2d_bandwidths
    ):
        workers = [
            Worker(
                rank=i,
                compute_capacity=compute_capacities[i],
                available_memory=available_memory[i],
            )
            for i in range(self.process_config.world_size)
        ]

        worker_d2d_bandwidths = {}
        for pair, d2d_bandwidth in d2d_bandwidths.items():
            i, j = pair
            worker_d2d_bandwidths[(workers[i], workers[j])] = d2d_bandwidth

        return WorkerGroup(workers=workers, d2d_bandwidths=worker_d2d_bandwidths)

    def _create_profiler_and_planner(self):
        cache_provider = CacheProvider()
        sample_allocator = SampleAllocator(cache_provider=cache_provider)
        profiler = Profiler(
            cache_provider=cache_provider,
            sample_allocator=sample_allocator,
            distributed=True,
        )
        planner = Planner(profiler=profiler, sample_allocator=sample_allocator)
        return profiler, planner

    def compute_plan(self):
        collector = WorkerDataCollector()
        num_workers = self.process_config.world_size
        micro_batch_size = self.hybrid_pipeline_config.microbatch_sample.shape[0]

        compute_capacities = collector.collect_compute_capacities(num_workers)
        available_memory = collector.collect_available_memory(self.device, num_workers)
        d2d_bandwidths = collector.collect_d2d_bandwidths(num_workers)

        if self.process_config.rank == 0:
            worker_group = self._create_worker_group(
                compute_capacities, available_memory, d2d_bandwidths
            )
            object_list = [worker_group]
            dist.broadcast_object_list(object_list, src=0)
        else:
            object_list = [None]
            dist.broadcast_object_list(object_list, src=0)
            worker_group = object_list[0]

        profiler, planner = self._create_profiler_and_planner()
        split_points, device_groups, samples_allocated = None, None, None

        profiler.profile(self.model, worker_group, micro_batch_size)

        if self.process_config.rank == 0:
            num_microbatches = self.hybrid_pipeline_config.num_microbatches

            split_points, device_groups, samples_allocated = planner.plan(
                self.model, worker_group, micro_batch_size, num_microbatches
            )

            object_list = [split_points, device_groups, samples_allocated]
            dist.broadcast_object_list(object_list, src=0)
        else:
            object_list = [None, None, None]
            dist.broadcast_object_list(object_list, src=0)
            split_points, device_groups, samples_allocated = object_list

        return split_points, device_groups, samples_allocated

    def _build_device_group_mask(self, device_groups):
        device_groups.append(self.process_config.world_size)

        devices_group_sizes = [device_groups[0]]
        devices_group_sizes.extend(
            [
                (device_groups[idx + 1] - device_groups[idx])
                for idx in range(len(device_groups) - 1)
            ]
        )

        device_group_mask = (
            [  # 2 = standalone device, 1 = new device group, 0 = part of device group
                element
                for size in devices_group_sizes
                for element in (([1] + [0] * (size - 1)) if size > 1 else [2])
            ]
        )
        return device_group_mask

    def _build_pipe(self, split_points, microbatch_sample):
        layers = model_to_layers(self.model, include_names=True)
        split_spec = {
            layers[split_point][0]: SplitPoint.BEGINNING
            for split_point in split_points
            if split_point > 0
        }

        pipe = pipeline(
            module=self.model,
            mb_args=(microbatch_sample,),
            split_spec=split_spec,
        )
        return pipe

    def _build_stages(
        self, pipe, pp_ranks, pp_group, ddp_ranks, ddp_group, device_group_mask
    ):
        world_rank = self.process_config.rank

        if world_rank in pp_ranks:
            pipeline_stage_train = pipe.build_stage(
                pp_group.rank(), self.device, pp_group
            )
            pipeline_stage_eval = pipe.build_stage(
                pp_group.rank(), self.device, pp_group
            )
        else:
            device_group_start = self._device_group_start(device_group_mask)
            pipeline_stage_train = pipe.build_stage(
                device_group_start, self.device, group=pp_group
            )
            pipeline_stage_eval = pipe.build_stage(
                device_group_start, self.device, group=pp_group
            )

        for pipeline_stage in [pipeline_stage_train, pipeline_stage_eval]:
            model = pipeline_stage.submod
            if world_rank in ddp_ranks and device_group_mask[world_rank] <= 1:
                ddp_model = DDP(
                    model,
                    device_ids=None,
                    process_group=ddp_group,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
                ddp_model.graph = model.graph
                pipeline_stage.submod = ddp_model

        return pipeline_stage_train, pipeline_stage_eval

    def _build_schedules(
        self,
        pipeline_stage_train,
        pipeline_stage_eval,
        ddp_ranks,
        ddp_group,
        device_group_mask,
        microbatch_split_at,
    ):
        world_rank = self.process_config.rank
        world_size = self.process_config.world_size

        prev_stage_is_ddp = device_group_mask[max(0, self.process_config.rank - 1)] == 0
        next_stage_is_ddp = (
            device_group_mask[min(world_size - 1, self.process_config.rank + 1)] == 1
        )

        next_non_ddp_stage_world_ranks = [
            i for i in range(world_rank, world_size) if device_group_mask[i] == 2
        ]
        next_non_ddp_stage_world_rank = (
            -1
            if prev_stage_is_ddp
            else (
                -1
                if not next_non_ddp_stage_world_ranks
                else next(iter(next_non_ddp_stage_world_ranks))
            )
        )

        prev_non_ddp_stage_world_ranks = [
            i for i in reversed(range(0, world_rank)) if device_group_mask[i] == 2
        ]
        prev_non_ddp_stage_world_rank = (
            -1
            if next_stage_is_ddp
            else (
                -1
                if not prev_non_ddp_stage_world_ranks
                else next(iter(prev_non_ddp_stage_world_ranks))
            )
        )

        pipeline_schedule_train = ScheduleHybridPipelineParallel1F1B(
            pipeline_stage_train,
            n_microbatches=self.hybrid_pipeline_config.num_microbatches,
            microbatch_split_at=microbatch_split_at,
            ddp_world_ranks=ddp_ranks,
            ddp_group=ddp_group,
            prev_stage_is_ddp=prev_stage_is_ddp,
            next_stage_is_ddp=next_stage_is_ddp,
            prev_non_ddp_stage_world_rank=prev_non_ddp_stage_world_rank,
            next_non_ddp_stage_world_rank=next_non_ddp_stage_world_rank,
            loss_fn=self.hybrid_pipeline_config.loss_fn,
        )

        pipeline_schedule_eval = ScheduleHybridPipelineParallel1F1B(
            pipeline_stage_eval,
            n_microbatches=self.hybrid_pipeline_config.num_microbatches,
            microbatch_split_at=microbatch_split_at,
            ddp_world_ranks=ddp_ranks,
            ddp_group=ddp_group,
            prev_stage_is_ddp=prev_stage_is_ddp,
            next_stage_is_ddp=next_stage_is_ddp,
            prev_non_ddp_stage_world_rank=prev_non_ddp_stage_world_rank,
            next_non_ddp_stage_world_rank=next_non_ddp_stage_world_rank,
        )

        return pipeline_schedule_train, pipeline_schedule_eval

    def _device_group_start(self, device_group_mask):
        return next(
            i
            for i in reversed(range(self.process_config.rank))
            if device_group_mask[i] == 1
        )

    def prepare_schedule(
        self,
        split_points,
        device_groups,
        samples_allocated,
    ):
        world_rank = self.process_config.rank
        world_size = self.process_config.world_size

        device_group_mask = self._build_device_group_mask(device_groups)

        pp_ranks = [
            i for i in range(len(device_group_mask)) if device_group_mask[i] >= 1
        ]
        pp_group = dist.new_group(ranks=pp_ranks)

        ddp_ranks = [
            i for i in range(len(device_group_mask)) if device_group_mask[i] <= 1
        ]
        ddp_group = dist.new_group(ranks=ddp_ranks)

        microbatch_sample = self.hybrid_pipeline_config.microbatch_sample

        stage_idx = (
            pp_group.rank()
            if world_rank in pp_ranks
            else (
                self._device_group_start(device_group_mask)
                if world_rank in ddp_ranks
                else -1
            )
        )
        microbatch_split = [
            samples_allocated[i][worker]
            for i in range(len(samples_allocated))
            for worker in samples_allocated[i].keys()
            if i == stage_idx
        ]
        microbatch_split_at = torch.cumsum(torch.tensor(microbatch_split), dim=0)

        if sum(microbatch_split) != microbatch_sample.shape[0]:
            raise ValueError(
                f"Mismatch between samples allocated in stage {stage_idx} and microbatch sample shape"
            )

        if world_rank in ddp_ranks:
            microbatch_sample = torch.tensor_split(
                microbatch_sample, microbatch_split_at
            )[ddp_group.rank()]

        pipe = self._build_pipe(split_points, microbatch_sample)

        pipeline_stage_train, pipeline_stage_eval = self._build_stages(
            pipe, pp_ranks, pp_group, ddp_ranks, ddp_group, device_group_mask
        )
        self.model = pipeline_stage_train.submod

        pipeline_schedule_train, pipeline_schedule_eval = self._build_schedules(
            pipeline_stage_train,
            pipeline_stage_eval,
            ddp_ranks,
            ddp_group,
            device_group_mask,
            microbatch_split_at,
        )
        self.pipeline_schedule_train = pipeline_schedule_train
        self.pipeline_schedule_eval = pipeline_schedule_eval

        self.microbatch_split_at = microbatch_split_at
        self.ddp_group = ddp_group
        self.is_first_stage_ddp = (
            sum(device_group_mask[i] for i in reversed(range(0, world_rank + 1))) <= 1
        )
        self.is_last_stage_ddp = (
            sum(device_group_mask[i] for i in range(world_rank, world_size)) <= 1
        )

        self.training = True

    def forward(self, *inputs, **kwargs):
        if self.pipeline_schedule_train is None or self.pipeline_schedule_eval is None:
            raise RuntimeError("Schedule not prepared: Call prepare_schedule")

        X = inputs[0]
        Y = inputs[1]

        logits = None
        loss = None

        pipeline_schedule = (
            self.pipeline_schedule_train
            if self.training
            else self.pipeline_schedule_eval
        )

        if self.process_config.rank == 0 or self.is_first_stage_ddp:
            pipeline_schedule.step(X)
        elif (
            self.process_config.rank == self.process_config.world_size - 1
            or self.is_last_stage_ddp
        ):
            if self.training:
                current_losses = []
                logits = pipeline_schedule.step(target=Y, losses=current_losses)
                loss = sum(current_losses) / len(current_losses)
            else:
                logits = self.pipeline_schedule_eval.step()
                if isinstance(self.model, DDP):
                    Y_mbs = list(
                        torch.tensor_split(
                            Y, self.hybrid_pipeline_config.num_microbatches
                        )
                    )
                    Y_ddp_split = [
                        torch.tensor_split(Y_mb, self.microbatch_split_at)[
                            self.ddp_group.rank()
                        ]
                        for Y_mb in Y_mbs
                    ]
                    Y_ddp = torch.cat(Y_ddp_split)
                    loss = self.hybrid_pipeline_config.loss_fn(logits, Y_ddp)
                else:
                    loss = self.hybrid_pipeline_config.loss_fn(logits, Y)
        else:
            pipeline_schedule.step()

        return logits, loss

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def cleanup(self):
        destroy_process_group()
