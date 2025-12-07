import contextlib
import time

import torch.distributed as dist


@contextlib.contextmanager
def log_execution_time(desc, log, layers=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    layers_name = layers.__class__.__name__ if layers else ""

    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        duration = end - start

        log(f"rank {rank}: {desc} - {layers_name} took {duration:.4f} seconds")
