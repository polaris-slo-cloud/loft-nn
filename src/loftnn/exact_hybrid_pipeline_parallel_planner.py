# This file contains an implementation of the algorithms described in:
#   Shengyuan Ye, Liekang Zeng, Xiaowen Chu, Guoliang Xing, and Xu Chen.
#   Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices.
#   In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, ACM MobiCom ’24,
#   page 312–326, New York, NY, USA, 2024. Association for Computing Machinery.

from loftnn.sample_allocation import SampleAllocator
import torch.nn as nn

from loftnn.types import WorkerGroup
from loftnn.profile import Profiler
from loftnn.model_utils import model_to_layers

INFEASIBLE_LATENCY = 10**20


class Planner:
    profiler: Profiler
    sample_allocator: SampleAllocator

    def __init__(self, profiler: Profiler, sample_allocator: SampleAllocator):
        self.profiler = profiler
        self.sample_allocator = sample_allocator

    def plan(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        num_micro_batches: int,
    ):
        layers = model_to_layers(model)

        Q, num_stages = self._plan_hpp(
            layers, worker_group, micro_batch_size, num_micro_batches
        )
        result = Q[len(layers), len(worker_group), num_stages]

        split_points, device_groups, samples_allocated = self._decode_result(
            result, Q, num_stages
        )
        return split_points, device_groups, samples_allocated

    def _plan_hpp(self, layers, devices, micro_batch_size: int, num_micro_batches: int):
        Q = {}
        Qt = {}
        ef = {}
        eb = {}
        dm = None
        edmf = None
        edmb = None

        for p in range(0, min(len(layers), len(devices)) + 1):
            for n in range(1, len(devices) + 1):
                for l in range(1, len(layers) + 1):
                    for np in range(n + 1):
                        for lp in range(l + 1):
                            # n - np devices execute layers l - lp and form a new step

                            if p == min(len(layers), len(devices)):
                                # only take minimum of previous stage
                                continue

                            if (
                                (n - np == 0 or l - lp == 0)
                                or (p == 0 and (np != 0 or lp != 0))
                                or (p != 0 and (np == 0 or lp == 0))
                                or (lp < p or np < p)
                            ):
                                ef[(l, n, lp, np, p)] = 0
                                eb[(l, n, lp, np, p)] = 0
                                Qt[(l, n, lp, np, p)] = (INFEASIBLE_LATENCY, {})
                                continue

                            device_group = devices[np:n]

                            try:
                                samples_allocated = (
                                    self.sample_allocator.get_allocation_of_micro_batch(
                                        device_group, micro_batch_size
                                    )
                                )
                            except ValueError:
                                ef[(l, n, lp, np, p)] = 0
                                eb[(l, n, lp, np, p)] = 0
                                Qt[(l, n, lp, np, p)] = (INFEASIBLE_LATENCY, {})
                                continue

                            i = lp
                            j = l
                            esf = max(
                                [
                                    sum(
                                        [
                                            (
                                                self.profiler.get_forward_execution_time(
                                                    layers[l],
                                                    device,
                                                    samples_allocated[device],
                                                )
                                                if l < len(layers)
                                                else 0
                                            )
                                            for l in range(i, j)
                                        ]
                                    )
                                    for device in device_group
                                ]
                            ) + (
                                self.profiler.get_transmission_time(
                                    (
                                        layers[j]
                                        if not (
                                            isinstance(layers[j], nn.ReLU)
                                            and layers[j].inplace
                                        )
                                        else layers[j - 1]
                                    ),
                                    self.profiler.get_mean_d2d_bandwidth(
                                        devices, devices[0:np], devices[np : n + 1]
                                    ),
                                    forward_only=True,
                                )
                                if j < len(layers)
                                else 0
                            )
                            esb = max(
                                [
                                    sum(
                                        [
                                            (
                                                self.profiler.get_backward_execution_time(
                                                    layers[l],
                                                    device,
                                                    samples_allocated[device],
                                                )
                                                if l < len(layers)
                                                else 0
                                            )
                                            for l in range(i, j)
                                        ]
                                    )
                                    for device in device_group
                                ]
                            ) + (
                                self.profiler.get_transmission_time(
                                    (
                                        layers[j]
                                        if not (
                                            isinstance(layers[j], nn.ReLU)
                                            and layers[j].inplace
                                        )
                                        else layers[j - 1]
                                    ),
                                    self.profiler.get_mean_d2d_bandwidth(
                                        devices, devices[0:np], devices[np : n + 1]
                                    ),
                                    backward_only=True,
                                )
                                if j < len(layers)
                                else 0
                            )
                            ef[(l, n, lp, np, p)] = esf
                            eb[(l, n, lp, np, p)] = esb

                            if p == 0:
                                dm = p
                                edmf = esf
                                edmb = esb
                            else:
                                prev_dm = dm
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
                                                self.profiler.get_transmission_time(
                                                    (
                                                        layers[j]
                                                        if not (
                                                            isinstance(
                                                                layers[j], nn.ReLU
                                                            )
                                                            and layers[j].inplace
                                                        )
                                                        else layers[j - 1]
                                                    ),
                                                    self.profiler.get_mean_d2d_bandwidth(
                                                        devices,
                                                        devices[0:np],
                                                        devices[np : n + 1],
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

                                if prev_dm != dm:
                                    edmf = esf
                                    edmb = esb

                            tsw = sum([ef[(l, n, lp, np, i)] for i in range(0, p)])

                            tsa = (
                                2
                                * (len(device_group) - 1)
                                * sum(
                                    [
                                        p.numel()
                                        for l in range(i, j)
                                        for p in (
                                            layers[l].parameters()
                                            if l < len(layers)
                                            else []
                                        )
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

                            hpp_round_latency = tsw + tsa + tse
                            prev_hpp_round_latency = (
                                Q[(lp, np, p - 1)][-1][0]
                                if p > 1
                                else (
                                    Qt[(lp, np, 0, 0, 0)][0]
                                    if p == 1
                                    else INFEASIBLE_LATENCY
                                )
                            )
                            if (
                                hpp_round_latency > prev_hpp_round_latency
                                and prev_hpp_round_latency < INFEASIBLE_LATENCY
                            ):
                                hpp_round_latency = prev_hpp_round_latency
                            Qt[(l, n, lp, np, p)] = (
                                hpp_round_latency,
                                samples_allocated,
                            )

                    if p == 0:
                        continue

                    prev_stage = p - 1
                    prev_stage_candidates = [
                        (l, n, lp, np, prev_stage, Qt[(l, n, lp, np, prev_stage)])
                        for np in range(n + 1 if prev_stage > 0 else 1)
                        for lp in range(l + 1 if prev_stage > 0 else 1)
                        if Qt[(l, n, lp, np, prev_stage)][0] < INFEASIBLE_LATENCY
                    ]
                    if prev_stage_candidates:
                        Q[(l, n, p)] = min(
                            prev_stage_candidates,
                            key=lambda x: x[-1][0],
                        )
                    else:
                        Q[(l, n, p)] = (
                            l,
                            n,
                            0,
                            0,
                            prev_stage,
                            (INFEASIBLE_LATENCY, {}),
                        )

        min_p = min(
            [
                (p, Q[(len(layers), len(devices), p)])
                for p in range(1, min(len(layers), len(devices)) + 1)
            ],
            key=lambda x: x[1][-1][0],
        )[0]

        return (Q, min_p)

    def _decode_result(self, result, Q, num_stages):
        l, n, lp, np, p, v = result

        split_points = [lp]
        device_groups = [np]
        samples_allocated = [v[1]]

        for s in reversed(range(1, num_stages)):
            l, n, lp, np, p, v = Q[lp, np, s]

            samples_allocated.append(v[1])

            if lp == 0 and np == 0 and s == 1:
                continue

            split_points.append(lp)
            device_groups.append(np)

        return split_points, device_groups, samples_allocated
