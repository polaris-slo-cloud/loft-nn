import random

from loftnn.types import Worker, WorkerGroup


def simple_worker_group(
    num_workers, batch_size_limit, compute_capacity=1 / 1000, bandwidth=1
):
    workers = [
        Worker(
            rank=i,
            compute_capacity=compute_capacity,
            batch_size_limit=batch_size_limit,
        )
        for i in range(num_workers)
    ]
    d2d_bandwidths = {
        (workers[i], workers[j]): bandwidth
        for i in range(num_workers)
        for j in range(i + 1, num_workers)
    }
    worker_group = WorkerGroup(workers=workers, d2d_bandwidths=d2d_bandwidths)
    return worker_group


def randomized_worker_group(num_workers, batch_size_limit):
    workers = [
        Worker(
            rank=i,
            compute_capacity=random.randint(1, 100),
            batch_size_limit=batch_size_limit,
        )
        for i in range(num_workers)
    ]
    d2d_bandwidths = {
        (workers[i], workers[j]): random.randint(1, 1000000000)
        for i in range(num_workers)
        for j in range(i + 1, num_workers)
    }
    worker_group = WorkerGroup(workers=workers, d2d_bandwidths=d2d_bandwidths)
    return worker_group
