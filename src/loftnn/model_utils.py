import logging
import time

import torch
import torch.nn as nn

from torch import Tensor
from torch.fx import symbolic_trace

from torchinfo import summary

from loftnn.types import Device

logger = logging.getLogger(__name__)


def layer_id(layer: nn.Module):
    return str(layer)


def model_to_layers(model: nn.Module, include_names=False):
    traced_model = symbolic_trace(model)
    layers = [
        (node.target, model.get_submodule(node.target))
        for node in traced_model.graph.nodes
        if node.op == "call_module"
    ]
    return layers if include_names else [layer for (_, layer) in layers]


class LayerInputProvider:
    def __init__(self):
        self.shapes_by_layer = {}

    def prepare(self, model, input_shape, dtype=None):
        model_statistics = summary(
            model,
            input_size=input_shape,
            dtypes=[dtype] if dtype else None,
            verbose=(1 if logger.isEnabledFor(logging.DEBUG) else 0),
        )
        self.shapes_by_layer = {
            id(layer_info.module): layer_info
            for layer_info in model_statistics.summary_list
        }

    def random(
        self,
        layer: nn.Module,
        device: Device,
        batch_size: int = None,
        input_sample: Tensor = None,
    ):
        if not self.shapes_by_layer:
            raise RuntimeError("Layer input not prepared: Call prepare")

        layer_info = self.shapes_by_layer[id(layer)]
        input_shape = layer_info.input_size

        if batch_size:
            input_shape = [batch_size] + input_shape[1:]

        use_int = isinstance(layer, nn.Embedding)
        if input_sample is not None:
            input_shape = (batch_size,) + input_sample.shape
            use_int = not torch.is_floating_point(input_sample)

        repeat_times = 1
        if isinstance(layer, nn.MultiheadAttention):
            repeat_times = 3

        return (
            [torch.randint(0, 10, input_shape, device=device)] * repeat_times
            if use_int
            else [torch.randn(input_shape, device=device)] * repeat_times
        )


def repeat_forward_execution(
    layer: nn.Module,
    input: LayerInputProvider,
    device: Device,
    batch_size: int,
    repetitions: int,
) -> float:
    execution_times = []

    for _ in range(repetitions):
        x = input.random(layer, device, batch_size)

        with torch.no_grad():
            start = time.time()
            _ = layer(*x)
            end = time.time()

        execution_time = end - start
        execution_times.append(execution_time)

    return sum(execution_times) / repetitions


def repeat_backward_execution(
    layer: nn.Module,
    input: LayerInputProvider,
    device: Device,
    batch_size: int,
    repetitions: int,
    input_sample: Tensor = None,
) -> float:
    execution_times = []

    for _ in range(repetitions):
        x = input.random(layer, device, batch_size, input_sample)

        for t in x:
            if torch.is_floating_point(t):
                t.requires_grad = True

        y = layer(*x)
        if not torch.is_tensor(y):
            y = y[0]

        start = time.time()
        y.mean().backward()
        end = time.time()

        execution_time = end - start
        execution_times.append(execution_time)

    return sum(execution_times) / repetitions
