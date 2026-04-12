import gc
import logging
import textwrap

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed.pipelining._IR import _modify_graph_op_device

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.profiler import profile, ProfilerActivity

from loftnn import HybridPipelineConfiguration, ProcessConfiguration
from loftnn.amb_planner import AMBPlanner
from loftnn.cache import CacheProvider
from loftnn.collect import WorkerDataCollector
from loftnn.configuration import HybridPipelinePlanningAlgorithm
from loftnn.ecs_planner import ECSPlanner
from loftnn.exact_planner import ExactPlanner
from loftnn.exact_profiler import ExactProfiler
from loftnn.logging import log_execution_time
from loftnn.model_utils import LayerInputProvider, model_to_layers
from loftnn.parallel import Parallel
from loftnn.pipeline_schedule import ScheduleECS1F1B
from loftnn.profiler import NoopProfiler
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Device, Worker, WorkerGroup

logger = logging.getLogger(__name__)


class HybridPipelineParallel(Parallel):
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

        logger.info(
            textwrap.dedent(
                f"""using:
                    process_config={process_config},
                    hybrid_pipeline_config={hybrid_pipeline_config},
                    device={device}
                """
            )
        )

        init_process_group(
            backend=process_config.backend,
            world_size=process_config.world_size,
            rank=process_config.rank,
            timeout=process_config.timeout,
        )

        slow_ops_group = dist.new_group(
            ranks=list(range(process_config.world_size)), timeout=timedelta(minutes=5)
        )
        self.slow_ops_group = slow_ops_group

        logger.info("initialized process group")

        self.layer_input = LayerInputProvider()
        self.layer_input.prepare(
            self.model,
            input_shape=self.hybrid_pipeline_config.microbatch_sample.shape,
            dtype=self.hybrid_pipeline_config.microbatch_sample.dtype,
        )

    def _create_worker_group(
        self, compute_capacities, batch_size_limits, device_memory, d2d_bandwidths
    ):
        workers = [
            Worker(
                rank=i,
                compute_capacity=compute_capacities[i],
                batch_size_limit=batch_size_limits[i],
                device_memory=device_memory[i],
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

        profiler, planner = None, None
        if self.hybrid_pipeline_config.planner == HybridPipelinePlanningAlgorithm.exact:
            profiler = ExactProfiler(
                cache_provider=cache_provider,
                sample_allocator=sample_allocator,
                distributed=True,
                process_group=self.slow_ops_group,
            )
            planner = ExactPlanner(profiler=profiler, sample_allocator=sample_allocator)
        elif self.hybrid_pipeline_config.planner == HybridPipelinePlanningAlgorithm.ecs:
            profiler = NoopProfiler()
            planner = ECSPlanner(
                layer_input=self.layer_input,
                sample_allocator=sample_allocator,
                device=self.device,
            )

        return profiler, planner

    def compute_plan(self, use_ambp=True, batch_size_limits=[]):
        num_workers = self.process_config.world_size
        collector = WorkerDataCollector(
            layer_input=self.layer_input,
            device=self.device,
            num_workers=num_workers,
        )
        micro_batch_size = self.hybrid_pipeline_config.microbatch_sample.shape[0]

        device_memory = (  # collect this first, before putting load on the devices
            collector.collect_device_memory()
        )
        compute_capacities = collector.collect_compute_capacities()
        d2d_bandwidths = collector.collect_d2d_bandwidths()
        if not batch_size_limits:
            batch_size_limits = collector.collect_batch_size_limits(
                self.model,
                self.hybrid_pipeline_config.microbatch_sample[0],
            )

        if self.process_config.rank == 0:
            worker_group = self._create_worker_group(
                compute_capacities, batch_size_limits, device_memory, d2d_bandwidths
            )
            object_list = [worker_group]
            dist.broadcast_object_list(object_list, src=0)
        else:
            object_list = [None]
            dist.broadcast_object_list(object_list, src=0)
            worker_group = object_list[0]

        logger.info(f"collected {worker_group}")

        profiler, planner = self._create_profiler_and_planner()
        split_points, device_groups, samples_allocated = None, None, None

        ambp = AMBPlanner(self.layer_input, self.device)

        with log_execution_time("profiling", logger.info):
            profiler.profile(
                self.model,
                self.layer_input,
                worker_group,
                micro_batch_size,
                device=self.device,
            )

        if self.process_config.rank == 0:
            num_microbatches = self.hybrid_pipeline_config.num_microbatches

            with log_execution_time("planning", logger.info):
                (
                    split_points,
                    device_groups,
                    samples_allocated,
                ) = planner.plan(
                    self.model, worker_group, micro_batch_size, num_microbatches
                )

            if use_ambp:
                with log_execution_time("ambp", logger.info):
                    activation_checkpointing_budgets = ambp.plan_budgets(
                        self.model,
                        worker_group,
                        split_points,
                        device_groups,
                    )
            else:
                activation_checkpointing_budgets = []

            object_list = [
                split_points,
                device_groups,
                samples_allocated,
                activation_checkpointing_budgets,
            ]
            dist.broadcast_object_list(object_list, src=0, group=self.slow_ops_group)
        else:
            object_list = [None, None, None, None]
            dist.broadcast_object_list(object_list, src=0, group=self.slow_ops_group)
            (
                split_points,
                device_groups,
                samples_allocated,
                activation_checkpointing_budgets,
            ) = object_list

        logger.info(
            textwrap.dedent(
                f"""determined plan:
                    split_points={split_points},
                    device_groups={device_groups},
                    samples_allocated={samples_allocated},
                    activation_checkpointing_budgets={activation_checkpointing_budgets}
                """
            )
        )

        if self.device == Device.cuda:
            torch.cuda.empty_cache()
        gc.collect()

        return (
            split_points,
            device_groups,
            samples_allocated,
            activation_checkpointing_budgets,
        )

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

    def _split_microbatch(
        self, microbatch_sample, samples_allocated, ddp_ranks, ddp_group
    ):
        microbatch_split_at = None
        if ddp_group:
            microbatch_split = [
                samples_allocated[i][worker]
                for i in range(len(samples_allocated))
                for worker in samples_allocated[i].keys()
                if worker.rank in ddp_ranks
            ]
            microbatch_split_at = torch.cumsum(torch.tensor(microbatch_split), dim=0)

            microbatch_split_sum = sum(microbatch_split)
            microbatch_sample_shape = microbatch_sample.shape[0]
            if microbatch_split_sum != microbatch_sample_shape:
                raise ValueError(
                    f"Mismatch between samples allocated and microbatch sample shape: {microbatch_split_sum} != {microbatch_sample_shape}"
                )

            if self.process_config.rank in ddp_ranks:
                microbatch_sample = torch.tensor_split(
                    microbatch_sample, microbatch_split_at
                )[ddp_group.rank()]

        return microbatch_sample, microbatch_split_at

    def _build_pipe(self, split_points, microbatch_sample):
        layers = model_to_layers(self.model, include_names=True)
        split_spec = {
            layers[split_point][0]: SplitPoint.BEGINNING
            for split_point in split_points
            if split_point > 0
        }

        logger.debug(f"building pipe: split_spec={split_spec}")
        pipe = pipeline(
            module=self.model,
            mb_args=(microbatch_sample,),
            split_spec=split_spec,
        )
        return pipe

    def _build_stages(
        self,
        pipe,
        pp_ranks,
        pp_group,
        ddp_ranks,
        ddp_group,
        device_group_mask,
        activation_checkpointing_budgets,
    ):
        world_rank = self.process_config.rank

        stage_device = Device.cpu  # all nodes must use cpu for building the stage
        stage_index = (
            pp_group.rank()
            if world_rank in pp_ranks
            else self._device_group_start(device_group_mask)
        )

        logger.debug(f"building stage: stage_index={stage_index}")
        pipeline_stage_train = pipe.build_stage(stage_index, stage_device, pp_group)
        pipeline_stage_eval = pipe.build_stage(stage_index, stage_device, pp_group)

        parameters = sum(p.numel() for p in pipeline_stage_train.submod.parameters())
        logger.info(f"parameters of model being trained: {parameters}")

        _ = torch._dynamo
        torch._functorch.config.activation_memory_budget = (
            activation_checkpointing_budgets[stage_index]
            if stage_index < len(activation_checkpointing_budgets)
            else 1.0
        )
        torch._inductor.config.compile_threads = 1

        logger.debug(f"compiling: stage_index={stage_index}")
        for pipeline_stage in [pipeline_stage_train, pipeline_stage_eval]:
            stage_module = pipe.get_stage_module(stage_index)
            _modify_graph_op_device(stage_module, self.device)
            pipeline_stage.submod = torch.compile(pipeline_stage.submod.to(self.device))
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

            # skip initialization
            pipeline_stage._get_init_p2p_neighbors_ops = lambda: []

        logger.info("waiting for compilation to finish on all workers")
        dist.barrier(group=self.slow_ops_group)

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

        pipeline_schedule_train = ScheduleECS1F1B(
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

        pipeline_schedule_eval = ScheduleECS1F1B(
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

        logger.debug(
            textwrap.dedent(
                f"""built schedules:
                    prev_stage_is_ddp={prev_stage_is_ddp},
                    next_stage_is_ddp={next_stage_is_ddp},
                    prev_non_ddp_stage_world_rank={prev_non_ddp_stage_world_rank},
                    next_non_ddp_stage_world_rank={next_non_ddp_stage_world_rank}
                """
            )
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
        activation_checkpointing_budgets=[],
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
        if ddp_ranks:
            ddp_group = dist.new_group(ranks=ddp_ranks)
        else:
            ddp_group = None
            logger.info("using no data parallelism")

        microbatch_sample = self.hybrid_pipeline_config.microbatch_sample
        microbatch_sample, microbatch_split_at = self._split_microbatch(
            microbatch_sample, samples_allocated, ddp_ranks, ddp_group
        )

        pipe = self._build_pipe(split_points, microbatch_sample)

        pipeline_stage_train, pipeline_stage_eval = self._build_stages(
            pipe,
            pp_ranks,
            pp_group,
            ddp_ranks,
            ddp_group,
            device_group_mask,
            activation_checkpointing_budgets,
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

        logger.debug(
            textwrap.dedent(
                f"""prepared schedule for hybrid pipeline parallelism:
                    device_group_mask={device_group_mask},
                    ddp_ranks={ddp_ranks},
                    pp_ranks={pp_ranks},
                    microbatch_split_at={microbatch_split_at},
                    is_first_stage_ddp={self.is_first_stage_ddp},
                    is_last_stage_ddp={self.is_last_stage_ddp}
                """
            )
        )

    def forward_parallel(self, *inputs, **kwargs):
        if self.pipeline_schedule_train is None or self.pipeline_schedule_eval is None:
            raise RuntimeError("Schedule not prepared: Call prepare_schedule")

        pipeline_schedule = (
            self.pipeline_schedule_train
            if self.training
            else self.pipeline_schedule_eval
        )

        ctx = (
            profile(
                activities=(
                    [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    if torch.cuda.is_available()
                    else [ProfilerActivity.CPU]
                ),
                record_shapes=True,
                profile_memory=True,
            )
            if logger.isEnabledFor(logging.DEBUG)
            else nullcontext()
        )
        with ctx as prof:
            X = inputs[0]
            Y = inputs[1]

            logits, loss = self._step(pipeline_schedule, X, Y)

        if prof:
            logger.debug("completed torch profiling")
            logger.debug(
                prof.key_averages().table(
                    sort_by="cpu_time_total",
                    top_level_events_only=True,
                    max_name_column_width=1000,
                )
            )

        return logits, loss

    def _step(self, pipeline_schedule, X, Y):
        logger.debug(f"rank {dist.get_rank()}: running step with: X={X}, Y={Y}")

        logits = None
        loss = None

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
                logits = pipeline_schedule.step()
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
