import logging
import math

import psutil

import torch
import torch.nn as nn

from torch import Tensor

from loftnn.model_utils import LayerInputProvider, repeat_backward_execution
from loftnn.types import Device

logger = logging.getLogger(__name__)


def max_batch_size(
    model: nn.Module,
    input: LayerInputProvider,
    device: Device,
    input_sample: Tensor = None,
) -> int:
    max_batch_size = 2

    is_integrated = True
    if torch.cuda.is_available():
        is_integrated = torch.cuda.get_device_properties(0).is_integrated

    if device == Device.cuda and not is_integrated:
        logger.info("profiling max_batch_size")

        while True:
            try:
                repeat_backward_execution(
                    model,
                    input,
                    device,
                    batch_size=max_batch_size,
                    repetitions=3,
                    input_sample=input_sample,
                )
                max_batch_size *= 2
            except torch.OutOfMemoryError:
                max_batch_size //= 2
                break
    else:
        logger.info("estimating max_batch_size")

        inputs = sum([t.numel() for t in input.random(model, device)])
        trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        max_batch_size = int(
            (psutil.virtual_memory().available / 4) / (inputs + trainable_parameters)
        )

        max_batch_size = 2 ** math.floor(
            math.log2(max_batch_size)
        )  # make it a power of 2

    logger.info(f"got max_batch_size={max_batch_size}")
    return max_batch_size
