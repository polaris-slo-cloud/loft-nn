import time

import torch
import torch.nn as nn

from loftnn.types import WorkerGroup


def random_input(layer, sample_size):
    # Generate a random input tensor based on the layer's expected input shape
    input_shape = (sample_size, 1000)

    params = list(layer.parameters())
    if len(params) >= 1:
        # Try to infer from first parameter shape (excluding batch)
        input_shape = (sample_size,) + params[0].shape[1:]

    return torch.randn(input_shape)


def random_forward_output(layer: nn.Module, sample_size: int):
    x = random_input(layer, sample_size)
    layer.eval()

    with torch.no_grad():
        y = layer(x)

    layer.train()
    return y


def random_backward_input(layer: nn.Module, sample_size: int):
    x = random_input(layer, sample_size)
    x.requires_grad = True
    layer.eval()

    y = layer(x)
    y.mean().backward()

    layer.train()
    return x.grad


def forward_execution_time(layer: nn.Module, sample_size: int) -> float:
    x = random_input(layer, sample_size)
    layer.eval()

    with torch.no_grad():
        start = time.time()
        _ = layer(x)
        end = time.time()

    layer.train()
    return end - start


def backward_execution_time(layer: nn.Module, sample_size: int) -> float:
    x = random_input(layer, sample_size)
    x.requires_grad = True
    layer.eval()

    y = layer(x)

    start = time.time()
    y.mean().backward()
    end = time.time()

    layer.train()
    return end - start


def transmission_time(layer: nn.Module, d2d_bandwidth: int):
    size_intermediate_tensors = (
        random_forward_output(layer, sample_size=1).shape,
        random_backward_input(layer, sample_size=1).shape,
    )

    return (
        (size_intermediate_tensors[0].numel() + size_intermediate_tensors[1].numel())
        * 4
        / d2d_bandwidth
    )


def mean_d2d_bandwidth(
    worker_group_world: WorkerGroup,
    worker_group_1: WorkerGroup,
    worker_group_2: WorkerGroup,
) -> int:
    bandwidths = []
    for worker1 in worker_group_1:
        for worker2 in worker_group_2:
            bandwidth = worker_group_world.d2d_bandwidth(worker1, worker2)
            bandwidths.append(bandwidth)

    return int(sum(bandwidths) / len(bandwidths)) if bandwidths else -1
