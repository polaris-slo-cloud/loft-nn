import torch
import torch.nn as nn

from torch.fx import symbolic_trace
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, MaxPool2d


def model_to_layers(model: nn.Module):
    traced_model = symbolic_trace(model)
    layers = [
        model.get_submodule(node.target)
        for node in traced_model.graph.nodes
        if node.op == "call_module"
    ]
    return layers


def random_layer_input(layer: nn.Module, sample_size: int):
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
