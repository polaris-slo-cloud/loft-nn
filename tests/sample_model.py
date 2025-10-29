import torch
import torch.nn as nn


def get_sample_input(model, batch_size):
    if isinstance(model, SimpleNN):
        return torch.randn(batch_size, 40, 40)
    else:
        raise ValueError("Unsupported model.")


def get_output_shape(model):
    if isinstance(model, SimpleNN):
        return 5
    else:
        raise ValueError("Unsupported model.")


def get_total_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_model = nn.Sequential(
            nn.Linear(40 * 40, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        x_flat = self.flatten(x)
        return self.linear_relu_model(x_flat)


class BalancedNN2(nn.Module):
    def __init__(self, features=100):
        super().__init__()
        self.linear_1 = nn.Linear(features, features)
        self.linear_2 = nn.Linear(features, features)

    def forward(self, x):
        return self.linear_2(self.linear_1(x))


class BalancedNN3(nn.Module):
    def __init__(self, features=100):
        super().__init__()
        self.linear_1 = nn.Linear(features, features)
        self.linear_2 = nn.Linear(features, features)
        self.linear_3 = nn.Linear(features, features)

    def forward(self, x):
        return self.linear_3(self.linear_2(self.linear_1(x)))
