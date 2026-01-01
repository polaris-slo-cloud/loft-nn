import psutil

import torch

from loftnn.types import Device


def free_memory(device: Device) -> int:
    return (
        psutil.virtual_memory().available
        if device == Device.cpu
        else torch.cuda.mem_get_info()[0] if device == Device.cuda else -1
    )
