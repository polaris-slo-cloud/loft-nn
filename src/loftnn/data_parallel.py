import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from loftnn import ProcessConfiguration
from loftnn.parallel import Parallel
from loftnn.types import Device


class DataParallel(Parallel):
    def __init__(
        self, model: nn.Module, process_config: ProcessConfiguration, device: Device
    ):
        super(DataParallel, self).__init__()
        self.model = model
        self.process_config = process_config
        self.device = device

        init_process_group(
            backend=process_config.backend,
            world_size=process_config.world_size,
            rank=process_config.rank,
            timeout=process_config.timeout,
        )

        if device == Device.cuda:
            self.ddp_model = DDP(model, device_ids=[process_config.local_rank])
        elif device == Device.cpu:
            self.ddp_model = DDP(model, device_ids=None)

    def forward_parallel(self, *inputs, **kwargs):
        return self.ddp_model(*inputs, **kwargs)

    def train(self, mode=True):
        return self.ddp_model.train(mode)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def cleanup(self):
        destroy_process_group()
