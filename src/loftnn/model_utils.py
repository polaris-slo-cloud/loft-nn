import time

import torch
import torch.nn as nn

from torch.fx import symbolic_trace
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Embedding, MaxPool2d


def model_to_layers(model: nn.Module, include_names=False):
    traced_model = symbolic_trace(model)
    layers = [
        (node.target, model.get_submodule(node.target))
        for node in traced_model.graph.nodes
        if node.op == "call_module"
    ]
    return layers if include_names else [layer for (_, layer) in layers]


def random_layer_input(layer: nn.Module, sample_size: int):
    # Generate a random input tensor based on the layer's expected input shape
    input_shape = (sample_size, 1000)
    use_int = False

    params = list(layer.parameters())
    if len(params) >= 1:
        # Try to infer from first parameter shape (excluding batch)
        input_shape = (sample_size,) + params[0].shape[1:]

    if isinstance(layer, AdaptiveAvgPool2d):
        input_shape = (sample_size, 1000, 1000)
    elif isinstance(layer, BatchNorm2d):
        input_shape = (sample_size, layer.num_features, 10, 10)
    elif isinstance(layer, MaxPool2d):
        input_shape = (sample_size, 1000, 1000)
    elif isinstance(layer, Embedding):
        use_int = True

    return torch.randint(0, 10, input_shape) if use_int else torch.randn(input_shape)


def repeat_forward_execution(
    layer: nn.Module, sample_size: int, repetitions: int
) -> float:
    execution_times = []

    for _ in range(repetitions):
        x = random_layer_input(layer, sample_size)

        with torch.no_grad():
            start = time.time()
            _ = layer(x)
            end = time.time()

        execution_time = end - start
        execution_times.append(execution_time)

    return sum(execution_times) / repetitions


def repeat_backward_execution(
    layer: nn.Module, sample_size: int, repetitions: int
) -> float:
    execution_times = []

    for _ in range(repetitions):
        x = random_layer_input(layer, sample_size)
        x.requires_grad = True

        y = layer(x)

        start = time.time()
        y.mean().backward()
        end = time.time()

        execution_time = end - start
        execution_times.append(execution_time)

    return sum(execution_times) / repetitions
