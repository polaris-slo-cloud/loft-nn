import time

import psutil

import torch

from loftnn.types import Device


def compute_capacity(device: Device) -> int:
    N = 8192 if device == Device.cuda else 1024
    mat1 = torch.randn(N, N, device=device)
    mat2 = torch.randn(N, N, device=device)

    # warmup
    for _ in range(5):
        torch.mm(mat1, mat2)
    if device == Device.cuda:
        torch.cuda.synchronize()

    start = time.time()
    duration = 5.0  # s
    end = start + duration

    mm_count = 0
    while time.time() < end:
        torch.mm(mat1, mat2)
        if device == Device.cuda:
            torch.cuda.synchronize()
        mm_count += 1

    compute_capacity = mm_count
    return compute_capacity


def total_memory(device: Device) -> int:
    return (
        psutil.virtual_memory().total
        if device == Device.cpu
        else (
            torch.cuda.mem_get_info()[1]
            if (device == Device.cuda and torch.cuda.is_available())
            else -1
        )
    )


def free_memory(device: Device) -> int:
    return (
        psutil.virtual_memory().available
        if device == Device.cpu
        else (
            torch.cuda.mem_get_info()[0]
            if (device == Device.cuda and torch.cuda.is_available())
            else -1
        )
    )


_INITIALLY_FREE_MEMORY_CPU = free_memory(Device.cpu)
_INITIALLY_FREE_MEMORY_CUDA = free_memory(Device.cuda)


def initially_free_memory(device: Device) -> int:
    return (
        _INITIALLY_FREE_MEMORY_CPU
        if device == Device.cpu
        else _INITIALLY_FREE_MEMORY_CUDA if device == Device.cuda else -1
    )
