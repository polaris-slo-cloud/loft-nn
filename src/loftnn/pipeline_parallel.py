import torch
import torch.nn as nn

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.pipelining import (
    pipeline,
    SplitPoint,
    Schedule1F1B,
    ScheduleGPipe,
)

from loftnn import ProcessConfiguration, PipelineConfiguration
from loftnn.types import Device


class PipelineParallel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        process_config: ProcessConfiguration,
        pipeline_config: PipelineConfiguration,
    ):
        super(PipelineParallel, self).__init__()
        self.process_config = process_config
        self.pipeline_config = pipeline_config

        init_process_group(
            backend=process_config.backend,
            world_size=process_config.world_size,
            rank=process_config.rank,
        )

        split_spec = {
            split_point: SplitPoint.BEGINNING
            for split_point in pipeline_config.split_points
        }
        pipe = pipeline(
            module=model,
            mb_args=(pipeline_config.microbatch_sample,),
            split_spec=split_spec,
        )

        device = Device.cpu  # all nodes must use cpu for building the stage

        pipeline_stage_train = pipe.build_stage(
            process_config.rank, device, torch.distributed.group.WORLD
        )
        pipeline_schedule_train = Schedule1F1B(
            pipeline_stage_train,
            n_microbatches=pipeline_config.num_microbatches,
            loss_fn=pipeline_config.loss_fn,
        )

        pipeline_stage_eval = pipe.build_stage(
            process_config.rank, device, torch.distributed.group.WORLD
        )
        pipeline_schedule_eval = ScheduleGPipe(
            pipeline_stage_eval,
            n_microbatches=pipeline_config.num_microbatches,
        )

        # skip initialization
        pipeline_stage_train._get_init_p2p_neighbors_ops = lambda: []
        pipeline_stage_eval._get_init_p2p_neighbors_ops = lambda: []

        self.pipeline_stage_train = pipeline_stage_train
        self.pipeline_schedule_train = pipeline_schedule_train

        self.pipeline_stage_eval = pipeline_stage_eval
        self.pipeline_schedule_eval = pipeline_schedule_eval

        self.training = True

    def forward(self, *inputs, **kwargs):
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
