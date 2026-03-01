import logging
import statistics

import torch.nn as nn

from torchinfo import summary

from loftnn.model_utils import LayerInputProvider
from loftnn.types import Device, WorkerGroup

logger = logging.getLogger(__name__)


class AMBPlanner:
    layer_input: LayerInputProvider
    device: Device

    def __init__(
        self,
        layer_input: LayerInputProvider,
        device: Device,
    ):
        self.layer_input = layer_input
        self.device = device

    def plan_budgets(
        self,
        device_group_ranks: list[list[int]],
        worker_group: WorkerGroup,
        layers: list[nn.Module],
        split_points: list[int],
        memory_estimation_factor: float = None,
    ):
        activation_checkpointing_budgets = []

        optimizer_states_factor = 3
        if memory_estimation_factor is None:
            # account for memory fragmentation, overheads,... and have a safety margin
            memory_estimation_factor = 1.1

        num_stages = (  # used to account for activation memory in 1F1B schedule
            len(split_points) + 1
        )

        split_points_complete = [0] + split_points + [len(layers)]
        for idx, split_point in enumerate(split_points_complete):
            if (idx - 1) >= 0:
                prev_split_point = split_points_complete[idx - 1]

                fwd_bwd_bytes = 0
                total_bytes = 0
                for i, layer in enumerate(layers[prev_split_point:split_point]):
                    x = self.layer_input.random(layer, self.device)
                    layer_statistics = summary(
                        layer,
                        input_size=x.shape,
                        dtypes=[x.dtype],
                        verbose=0,
                    )

                    fwd_bwd_bytes += int(
                        layer_statistics.total_output_bytes
                        * num_stages
                        * memory_estimation_factor
                    )
                    total_bytes += int(
                        (
                            layer_statistics.total_output_bytes
                            * num_stages
                            * memory_estimation_factor
                        )
                        + (
                            layer_statistics.total_param_bytes
                            * memory_estimation_factor
                            * optimizer_states_factor
                        )
                    )

                    is_first_layer = i == 0
                    if prev_split_point == 0 and is_first_layer:
                        total_bytes += int(
                            layer_statistics.total_input
                            * num_stages
                            * memory_estimation_factor
                        )

                workers_have_device_memory = [
                    (
                        worker.device_memory is not None
                        and self.device in worker.device_memory
                    )
                    for worker in worker_group
                ]
                if all(workers_have_device_memory):
                    free_bytes = statistics.mean(
                        [
                            worker.device_memory[self.device]
                            for worker in worker_group
                            if worker.rank in device_group_ranks[idx - 1]
                        ]
                    )
                else:
                    free_bytes = -1

                logger.info(
                    f"stage {idx} - total bytes estimate: {total_bytes}, free bytes: {free_bytes}, difference: {free_bytes - total_bytes}"
                )

                if total_bytes > free_bytes and free_bytes > 0:
                    # solve: min x s.t. (total_bytes - (x * fwd_bwd_bytes) <= free_bytes)
                    #       (total_bytes - free_bytes) <= x * fwd_bwd_bytes
                    #       ((total_bytes - free_bytes) / fwd_bwd_bytes) <= x
                    #       ((total_bytes - free_bytes) / fwd_bwd_bytes) = x
                    #
                    # then obtain budget = 1 - x

                    x = (total_bytes - free_bytes) / (fwd_bwd_bytes)

                    activation_checkpointing_budgets.append(1 - x)
                else:
                    activation_checkpointing_budgets.append(1.0)

        logger.info(
            f"activation checkpointing budgets: {activation_checkpointing_budgets}"
        )

        return activation_checkpointing_budgets
