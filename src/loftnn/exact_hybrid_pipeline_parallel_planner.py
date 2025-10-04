# This file contains an implementation of the algorithms described in:
#   Shengyuan Ye, Liekang Zeng, Xiaowen Chu, Guoliang Xing, and Xu Chen.
#   Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices.
#   In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, ACM MobiCom ’24,
#   page 312–326, New York, NY, USA, 2024. Association for Computing Machinery.

from torch.fx import symbolic_trace
import torch.nn as nn

from loftnn.types import WorkerGroup
from loftnn.profile import (
    forward_execution_time,
    backward_execution_time,
    mean_d2d_bandwidth,
    transmission_time,
)


def memory_aware_balancing_recursive(device_group, beta, samples_allocated):
    if len(device_group) == 0 and beta > 0:
        raise ValueError("No devices in the group but beta > 0")
    elif beta == 0:
        return samples_allocated

    batch_size = {}
    compute_capacity = {d: d.compute_capacity for d in device_group}
    available_memory = {d: d.available_memory for d in device_group}

    for d in device_group:
        batch_size[d] = min(d.batch_size(), beta)
        samples_allocated[d] = samples_allocated[d] + min(
            int(
                (
                    compute_capacity[d]
                    / (sum([compute_capacity[dev] for dev in device_group]))
                )
                * beta
            ),
            batch_size[d],
        )
        available_memory[d] = available_memory[d] - samples_allocated[d]

    device_group_remaining = [d for d in device_group if available_memory[d] > 0]
    beta_remaning = beta - sum([batch_size[d] for d in device_group])
    return memory_aware_balancing_recursive(
        device_group_remaining, beta_remaning, samples_allocated
    )


def memory_aware_balancing(device_group, beta):
    samples_allocated = {d: 0 for d in device_group}
    return memory_aware_balancing_recursive(device_group, beta, samples_allocated)


def straggler_workload_offloading(
    device_group, samples_allocated, samples_to_transfer=4
):
    device_group = device_group.sorted(key=lambda d: samples_allocated[d])

    done = False
    while not done:
        old_straggler = device_group.get_slowest()
        fastest_device_with_memory = device_group.get_fastest_with_memory(
            samples_to_transfer
        )
        samples_allocated[fastest_device_with_memory] += samples_to_transfer
        samples_allocated[old_straggler] -= samples_to_transfer
        device_group = device_group.sorted(key=lambda d: samples_allocated[d])
        new_straggler = device_group.get_slowest()
        if not new_straggler.is_faster(old_straggler):
            done = True

    return samples_allocated


def allocation_of_micro_batch(device_group, micro_batch_size):
    samples_allocated = memory_aware_balancing(device_group, micro_batch_size)
    samples_allocated = straggler_workload_offloading(device_group, samples_allocated)
    return samples_allocated


def plan_hpp(layers, devices, micro_batch_size: int, num_micro_batches: int):
    Q = {}
    Qt = {}
    ef = {}
    eb = {}
    dm = None
    edmf = None
    edmb = None
    for p in range(1, min(len(layers), len(devices)) + 1):
        for n in range(1, len(devices) + 1):
            for l in range(1, len(layers) + 1):
                for np in range(n):
                    for lp in range(l):
                        # n - np devices execute layers l - lp and form a new step
                        # np devices execute layers lp

                        if p == 1:
                            device_group = devices[
                                (len(devices) - 1 - n) : (len(devices) - 1)
                            ]
                            if len(device_group) == 0:
                                ef[(l, n, lp, np, p)] = 0
                                eb[(l, n, lp, np, p)] = 0
                                Qt[(l, n, lp, np, p)] = 0
                                continue

                            samples_allocated = allocation_of_micro_batch(
                                device_group, micro_batch_size
                            )
                            i = len(layers) - 1 - l
                            j = len(layers) - 1
                            esf = max(
                                [
                                    sum(
                                        [
                                            forward_execution_time(
                                                layers[l], samples_allocated[device]
                                            )
                                            for l in range(i, j)
                                        ]
                                    )
                                    for device in device_group
                                ]
                            )
                            esb = max(
                                [
                                    sum(
                                        [
                                            backward_execution_time(
                                                layers[l], samples_allocated[device]
                                            )
                                            for l in range(i, j)
                                        ]
                                    )
                                    for device in device_group
                                ]
                            )
                            ef[(l, n, lp, np, p)] = esf
                            eb[(l, n, lp, np, p)] = esb

                            dm = p
                            edmf = esf
                            edmb = esb

                            tsw = 0
                            tsa = (
                                2
                                * (len(device_group) - 1)
                                * sum(
                                    [
                                        p.numel()
                                        for l in range(i, j)
                                        for p in layers[l].parameters()
                                    ]
                                )
                                / (len(device_group) * device_group.min_d2d_bandwidth())
                            )
                            tse = num_micro_batches * (esf + esb)

                            hpp_round_latency = tsw + tsa + tse
                            Qt[(l, n, lp, np, p)] = hpp_round_latency
                            continue

                        device_group = devices[np:n]
                        samples_allocated = allocation_of_micro_batch(
                            device_group, micro_batch_size
                        )
                        i = lp
                        j = l
                        esf = max(
                            [
                                sum(
                                    [
                                        forward_execution_time(
                                            layers[l], samples_allocated[device]
                                        )
                                        for l in range(i, j)
                                    ]
                                )
                                for device in device_group
                            ]
                        )
                        esb = max(
                            [
                                sum(
                                    [
                                        backward_execution_time(
                                            layers[l], samples_allocated[device]
                                        )
                                        for l in range(i, j)
                                    ]
                                )
                                for device in device_group
                            ]
                        )
                        ef[(l, n, lp, np, p)] = esf
                        eb[(l, n, lp, np, p)] = esb

                        dm = max(
                            [
                                (
                                    dm,
                                    num_micro_batches * (edmf + edmb)
                                    + sum(
                                        [
                                            (
                                                ef[(l, n, lp, np, i)]
                                                + eb[(l, n, lp, np, i)]
                                            )
                                            for i in range(1, dm + 1)
                                        ]
                                    ),
                                ),
                                (p, num_micro_batches * (esf + esb)),
                                (
                                    p,
                                    num_micro_batches
                                    * (
                                        transmission_time(
                                            layers[j],
                                            mean_d2d_bandwidth(
                                                devices, devices[0:np], devices[np:n]
                                            ),
                                        )
                                        if j < len(layers)
                                        else 0
                                    )
                                    + (esf + esb),
                                ),
                            ],
                            key=lambda x: x[1],
                        )[0]

                        tsw = sum([ef[(l, n, lp, np, i)] for i in range(1, p)])

                        tsa = (
                            2
                            * (len(device_group) - 1)
                            * sum(
                                [
                                    p.numel()
                                    for l in range(i, j)
                                    for p in layers[l].parameters()
                                ]
                            )
                            / (len(device_group) * device_group.min_d2d_bandwidth())
                        )

                        tse = (
                            num_micro_batches * (edmf + edmb)
                            + sum(
                                [
                                    (ef[(l, n, lp, np, i)] + eb[(l, n, lp, np, i)])
                                    for i in range(p, dm)
                                ]
                            )
                            - sum(
                                [
                                    (ef[(l, n, lp, np, i)] + eb[(l, n, lp, np, i)])
                                    for i in range(dm, p)
                                ]
                            )
                        )

                if p == 1:
                    continue

                Q[(l, n, p)] = min(
                    [
                        (l, n, lp, np, p - 1, Qt[(l, n, lp, np, p - 1)])
                        for np in range(n)
                        for lp in range(l)
                    ],
                    key=lambda x: x[-1],
                )

    return min(
        [
            (p, Q[(len(layers) - 1, len(devices) - 1, p)])
            for p in range(len(devices), min(len(layers), len(devices)) + 1)
        ],
        key=lambda x: x[1][-1],
    )


def plan(
    model: nn.Module,
    worker_group: WorkerGroup,
    micro_batch_size: int,
    num_micro_batches: int,
):
    traced_model = symbolic_trace(model)
    layers = [
        model.get_submodule(node.target)
        for node in traced_model.graph.nodes
        if node.op == "call_module"
    ]
    return plan_hpp(layers, worker_group, micro_batch_size, num_micro_batches)
