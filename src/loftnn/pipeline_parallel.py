import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed.pipelining._IR import _modify_graph_op_device

from loftnn import ProcessConfiguration, PipelineConfiguration
from loftnn.model_utils import model_to_layers
from loftnn.parallel import Parallel
from loftnn.pipeline_schedule import ScheduleECS1F1B
from loftnn.types import Device


class PipelineParallel(Parallel):
    def __init__(
        self,
        model: nn.Module,
        process_config: ProcessConfiguration,
        pipeline_config: PipelineConfiguration,
        device: Device,
    ):
        super(PipelineParallel, self).__init__()
        self.process_config = process_config
        self.pipeline_config = pipeline_config

        init_process_group(
            backend=process_config.backend,
            world_size=process_config.world_size,
            rank=process_config.rank,
            timeout=process_config.timeout,
        )

        layers = model_to_layers(model, include_names=True)
        split_spec = {
            layers[split_point][0]: SplitPoint.BEGINNING
            for split_point in pipeline_config.split_points
            if split_point > 0
        }
        pipe = pipeline(
            module=model,
            mb_args=(pipeline_config.microbatch_sample,),
            split_spec=split_spec,
        )

        stage_device = Device.cpu  # all nodes must use cpu for building the stage

        pipeline_stage_train = pipe.build_stage(
            process_config.rank, stage_device, torch.distributed.group.WORLD
        )
        pipeline_stage_eval = pipe.build_stage(
            process_config.rank, stage_device, torch.distributed.group.WORLD
        )

        stage_index = dist.get_rank()
        for pipeline_stage in [pipeline_stage_train, pipeline_stage_eval]:
            stage_module = pipe.get_stage_module(stage_index)
            _modify_graph_op_device(stage_module, device)
            pipeline_stage.submod = pipeline_stage.submod.to(device)

            # skip initialization
            pipeline_stage._get_init_p2p_neighbors_ops = lambda: []

        self.pipeline_stage_train = pipeline_stage_train
        self.pipeline_stage_eval = pipeline_stage_eval

        pipeline_schedule_train = ScheduleECS1F1B(
            self.pipeline_stage_train,
            n_microbatches=pipeline_config.num_microbatches,
            microbatch_split_at=[],
            ddp_world_ranks=[],
            ddp_group=None,
            prev_stage_is_ddp=False,
            next_stage_is_ddp=False,
            prev_non_ddp_stage_world_rank=-1,
            next_non_ddp_stage_world_rank=-1,
            loss_fn=pipeline_config.loss_fn,
        )

        pipeline_schedule_eval = ScheduleECS1F1B(
            self.pipeline_stage_eval,
            n_microbatches=pipeline_config.num_microbatches,
            microbatch_split_at=[],
            ddp_world_ranks=[],
            ddp_group=None,
            prev_stage_is_ddp=False,
            next_stage_is_ddp=False,
            prev_non_ddp_stage_world_rank=-1,
            next_non_ddp_stage_world_rank=-1,
        )

        self.pipeline_schedule_train = pipeline_schedule_train
        self.pipeline_schedule_eval = pipeline_schedule_eval

        self.training = True

    def forward_parallel(self, *inputs, **kwargs):
        X = inputs[0]
        Y = inputs[1]

        logits = None
        loss = None

        pipeline_schedule = (
            self.pipeline_schedule_train
            if self.training
            else self.pipeline_schedule_eval
        )

        if self.process_config.rank == 0:
            pipeline_schedule.step(X)
        elif self.process_config.rank == self.process_config.world_size - 1:
            if self.training:
                current_losses = []
                logits = pipeline_schedule.step(target=Y, losses=current_losses)
                loss = sum(current_losses) / len(current_losses)
            else:
                logits = self.pipeline_schedule_eval.step()
                loss = self.pipeline_config.loss_fn(logits, Y)
        else:
            pipeline_schedule.step()

        return logits, loss

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        return self.pipeline_stage_train.submod.parameters()

    def named_parameters(self):
        return self.pipeline_stage_train.submod.named_parameters()

    def state_dict(self):
        return self.pipeline_stage_train.submod.state_dict()

    def cleanup(self):
        destroy_process_group()
