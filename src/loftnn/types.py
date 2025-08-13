from enum import Enum


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"
