from abc import ABC, abstractmethod

import torch.nn as nn


class Parallel(nn.Module, ABC):

    def __init__(self):
        super(Parallel, self).__init__()

    def forward(self, *inputs, **kwargs):
        result = self.forward_parallel(*inputs, **kwargs)
        return result

    @abstractmethod
    def forward_parallel(self, *inputs, **kwargs):
        pass
