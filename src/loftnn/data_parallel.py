import os

import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from loftnn.types import Device


class DataParallel(nn.Module):
    @staticmethod
    def is_available():
        return (
            int(os.environ.get("RANK", -1)) != -1
            and int(os.environ.get("LOCAL_RANK", -1)) != -1
            and int(os.environ.get("WORLD_SIZE", -1)) != -1
            and os.environ.get("MASTER_ADDR", "-1") != "-1"
            and int(os.environ.get("MASTER_PORT", -1)) != -1
        )

    class Configuration:
        def __init__(self, rank, local_rank, world_size, backend="gloo"):
            self.rank = rank
            self.local_rank = local_rank
            self.world_size = world_size
            self.backend = backend

        @staticmethod
        def from_env():
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            return DataParallel.Configuration(
                rank=rank, local_rank=local_rank, world_size=world_size
            )

    def __init__(self, model: nn.Module, config: Configuration, device: Device):
        super(DataParallel, self).__init__()
        self.model = model
        self.config = config
        self.device = device

        init_process_group(backend=config.backend)

        if device == Device.cuda:
            self.ddp_model = DDP(model, device_ids=[config.local_rank])
        elif device == Device.cpu:
            self.ddp_model = DDP(model, device_ids=None)

    def forward(self, *inputs, **kwargs):
        return self.ddp_model(*inputs, **kwargs)

    def train(self, mode=True):
        return self.ddp_model.train(mode)

    def cleanup(self):
        destroy_process_group()
