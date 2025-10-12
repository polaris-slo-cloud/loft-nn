import time

import torch
import torch.nn as nn

from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, MaxPool2d, ReLU

from loftnn.types import Worker, WorkerGroup

FORWARD_OUTPUT_CACHE = {}
BACKWARD_INPUT_CACHE = {}

FORWARD_EXECUTION_TIME_CACHE = {}
BACKWARD_EXECUTION_TIME_CACHE = {}


def random_input(layer, sample_size):
    # Generate a random input tensor based on the layer's expected input shape
    input_shape = (sample_size, 1000)

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

    return torch.randn(input_shape)


def random_forward_output(layer: nn.Module, sample_size: int):
    if layer in FORWARD_OUTPUT_CACHE:
        return FORWARD_OUTPUT_CACHE[layer]

    x = random_input(layer, sample_size)
    layer.eval()

    with torch.no_grad():
        y = layer(x)

    layer.train()

    FORWARD_OUTPUT_CACHE[layer] = y
    return y


def random_backward_input(layer: nn.Module, sample_size: int):
    if layer in BACKWARD_INPUT_CACHE:
        return BACKWARD_INPUT_CACHE[layer]

    x = random_input(layer, sample_size)
    x.requires_grad = True
    layer.eval()

    y = layer(x)
    y.mean().backward()

    layer.train()

    BACKWARD_INPUT_CACHE[layer] = x.grad
    return x.grad


def forward_execution_time(layer: nn.Module, worker: Worker, sample_size: int) -> float:
    if (layer, worker, sample_size) in FORWARD_EXECUTION_TIME_CACHE:
        return FORWARD_EXECUTION_TIME_CACHE[(layer, worker, sample_size)]

    x = random_input(layer, sample_size)
    layer.eval()

    with torch.no_grad():
        start = time.time()
        _ = layer(x)
        end = time.time()

    layer.train()

    execution_time = end - start
    FORWARD_EXECUTION_TIME_CACHE[(layer, worker, sample_size)] = execution_time
    return execution_time


def backward_execution_time(
    layer: nn.Module, worker: Worker, sample_size: int
) -> float:
    if (layer, worker, sample_size) in BACKWARD_EXECUTION_TIME_CACHE:
        return BACKWARD_EXECUTION_TIME_CACHE[(layer, worker, sample_size)]

    if isinstance(layer, ReLU):
        if layer.inplace:
            return 0

    x = random_input(layer, sample_size)
    x.requires_grad = True
    layer.eval()

    y = layer(x)

    start = time.time()
    y.mean().backward()
    end = time.time()

    layer.train()

    execution_time = end - start
    BACKWARD_EXECUTION_TIME_CACHE[(layer, worker, sample_size)] = execution_time
    return execution_time


def transmission_time(
    layer: nn.Module, d2d_bandwidth: int, forward_only=False, backward_only=False
):
    if d2d_bandwidth < 0:
        raise ValueError("Bandwidth cannot be negative")

    if isinstance(layer, ReLU):
        if layer.inplace:
            return 0

    size_intermediate_tensors = (
        (
            random_forward_output(layer, sample_size=1).shape
            if not backward_only
            else torch.empty(0)
        ),
        (
            random_backward_input(layer, sample_size=1).shape
            if not forward_only
            else torch.empty(0)
        ),
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
    if worker_group_1 and not worker_group_2:
        return 1_000_000_000
    if not worker_group_1 and worker_group_2:
        return 1

    bandwidths = []
    for worker1 in worker_group_1:
        for worker2 in worker_group_2:
            bandwidth = worker_group_world.d2d_bandwidth(worker1, worker2)
            bandwidths.append(bandwidth)

    return int(sum(bandwidths) / len(bandwidths)) if bandwidths else -1
