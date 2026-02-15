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


def edge_cloud_worker_group(num_micro_batches):
    # edge
    worker1 = Worker(
        rank=0, compute_capacity=1 / 5000, batch_size_limit=num_micro_batches * 20
    )
    worker2 = Worker(
        rank=1, compute_capacity=1 / 5000, batch_size_limit=num_micro_batches * 20
    )
    worker3 = Worker(
        rank=2, compute_capacity=1 / 5000, batch_size_limit=num_micro_batches * 20
    )
    # cloud
    worker4 = Worker(
        rank=3, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 60
    )
    worker5 = Worker(
        rank=4, compute_capacity=1 / 1000, batch_size_limit=num_micro_batches * 60
    )

    workers = [worker1, worker2, worker3, worker4, worker5]
    d2d_bandwidths = {
        (workers[i], workers[j]): 1
        for i in range(len(workers))
        for j in range(i + 1, len(workers))
    }

    # edge
    d2d_bandwidths[(worker1, worker2)] = 10**5
    d2d_bandwidths[(worker2, worker3)] = 10**5
    d2d_bandwidths[(worker1, worker3)] = 10**5
    # cloud
    d2d_bandwidths[(worker4, worker5)] = 10**5

    worker_group = WorkerGroup(workers=workers, d2d_bandwidths=d2d_bandwidths)
    return worker_group
