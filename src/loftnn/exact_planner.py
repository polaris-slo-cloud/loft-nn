# This file contains an implementation of the algorithms described in:
#   Shengyuan Ye, Liekang Zeng, Xiaowen Chu, Guoliang Xing, and Xu Chen.
#   Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices.
#   In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, ACM MobiCom ’24,
#   page 312–326, New York, NY, USA, 2024. Association for Computing Machinery.

import logging
import os
import pickle

import torch.nn as nn
from torch.multiprocessing import get_context, Manager

from loftnn.exact_profiler import ExactProfiler
from loftnn.logging import log_execution_time
from loftnn.model_utils import model_to_layers
from loftnn.planner import Planner
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import WorkerGroup

INFEASIBLE_LATENCY = 10**20

logger = logging.getLogger(__name__)


def init_process(level):
    logging.basicConfig(level=level)


class ExactPlanner(Planner):
    profiler: ExactProfiler
    sample_allocator: SampleAllocator

    def __init__(self, profiler: ExactProfiler, sample_allocator: SampleAllocator):
        self.profiler = profiler
        self.sample_allocator = sample_allocator

    def plan(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        num_micro_batches: int,
        parallel: bool = False,
    ):
        layers = model_to_layers(model)

        logger.info(
            f"planning {len(layers)} layers on {len(worker_group)} workers with micro_batch_size={micro_batch_size} and num_micro_batches={num_micro_batches}"
        )

        Q, num_stages = self._plan_hpp(
            layers, worker_group, micro_batch_size, num_micro_batches, parallel
        )
        result = Q[len(layers), len(worker_group), num_stages]

        split_points, device_groups, samples_allocated = self._decode_result(
            result, Q, num_stages
        )
        activation_checkpointing_budgets = [1.0] + [1.0 for _ in split_points]

        return (
            split_points,
            device_groups,
            samples_allocated,
            activation_checkpointing_budgets,
        )

    def _plan_hpp(
        self,
        layers,
        devices,
        micro_batch_size: int,
        num_micro_batches: int,
        parallel: bool,
    ):
        (Q, Qt, ef, eb, dm) = ({}, {}, {}, {}, {})

        state = dict()
        if parallel:
            pool = get_context("spawn").Pool(
                processes=os.cpu_count(),
                initializer=init_process,
                initargs=(logger.getEffectiveLevel(),),
            )
            manager = Manager()
            state = manager.dict()

        state.update({"Q": Q, "Qt": Qt, "ef": ef, "eb": eb, "dm": dm})

        for p in range(0, min(len(layers), len(devices))):
            nls = [
                (n, l)
                for n in range(1, len(devices) + 1)
                for l in range(1, len(layers) + 1)
            ]

            with log_execution_time("read_state", logger.info):
                Q_c, Qt_c, ef_c, eb_c, dm_c = ({}, {}, {}, {}, {})
                Q_c.update(state["Q"])
                Qt_c.update(state["Qt"])
                ef_c.update(state["ef"])
                eb_c.update(state["eb"])
                dm_c.update(state["dm"])
            state_c = (Q_c, Qt_c, ef_c, eb_c, dm_c)

            if parallel:
                with log_execution_time("plan_nl_parallel", logger.info):
                    state_updates = pool.starmap(
                        self._plan_nl,
                        [
                            (
                                layers,
                                devices,
                                micro_batch_size,
                                num_micro_batches,
                                p,
                                n,
                                l,
                                state_c,
                            )
                            for (n, l) in nls
                        ],
                    )
            else:
                with log_execution_time("plan_nl_sequential", logger.info):
                    state_updates = []
                    for n, l in nls:
                        update = self._plan_nl(
                            layers,
                            devices,
                            micro_batch_size,
                            num_micro_batches,
                            p,
                            n,
                            l,
                            state_c,
                        )
                        state_updates.append(update)

            base_update = state_updates[0]
            (Q_b, Qt_b, ef_b, eb_b, dm_b) = base_update
            for update in state_updates[1:]:
                (Q_u, Qt_u, ef_u, eb_u, dm_u) = update
                Q_b.update(Q_u)
                Qt_b.update(Qt_u)
                ef_b.update(ef_u)
                eb_b.update(eb_u)
                dm_b.update(dm_u)

            Q_b.update(Q_c)
            Qt_b.update(Qt_c)
            ef_b.update(ef_c)
            eb_b.update(eb_c)
            dm_b.update(dm_c)

            with log_execution_time("write_state", logger.info):
                state.update({"Q": Q_b, "Qt": Qt_b, "ef": ef_b, "eb": eb_b, "dm": dm_b})

        Q = state["Q"]
        min_p = min(
            [
                (p, Q[(len(layers), len(devices), p)])
                for p in range(1, min(len(layers), len(devices)) + 1)
                if Q[(len(layers), len(devices), p)][-1][0] < INFEASIBLE_LATENCY
            ],
            key=lambda x: x[1][-1][0],
        )[0]

        logger.info(f"got min_p={min_p}")

        Q_r = {}
        Q_r.update(Q)

        if logger.isEnabledFor(logging.INFO):
            with open("exact_planner_state.pickle", "wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("saved exact_planner_state.pickle")

        for state in state.values():
            state.clear()

        if parallel:
            pool.close()

        return (Q_r, min_p)

    def _plan_nl(
        self,
        layers,
        devices,
        micro_batch_size: int,
        num_micro_batches: int,
        p,
        n,
        l,
        state,
    ):
        (Q, Qt, ef, eb, dm) = state

        state_updates = []
        for np in range(n + 1):
            for lp in range(l + 1):
                update = self._plan(
                    layers,
                    devices,
                    micro_batch_size,
                    num_micro_batches,
                    p,
                    n,
                    l,
                    np,
                    lp,
                    state,
                )
                state_updates.append(update)

        base_update = state_updates[0]
        (Qt_b, ef_b, eb_b, dm_b) = base_update
        for update in state_updates[1:]:
            (Qt_u, ef_u, eb_u, dm_u) = update
            Qt_b.update(Qt_u)
            ef_b.update(ef_u)
            eb_b.update(eb_u)
            dm_b.update(dm_u)

        Qt.update(Qt_b)
        ef.update(ef_b)
        eb.update(eb_b)
        dm.update(dm_b)

        updates = (
            Q,
            Qt,
            ef,
            eb,
            dm,
        )

        # we computed Qt[(l, n, lp, np, p)], now compute Q[(l, n, p + 1)]
        prev_stage = p
        prev_stage_candidates = [
            (l, n, lp, np, prev_stage, Qt[(l, n, lp, np, prev_stage)])
            for np in range(n + 1)
            for lp in range(l + 1)
            if Qt[(l, n, lp, np, prev_stage)][0] < INFEASIBLE_LATENCY
        ]
        if prev_stage_candidates:
            Q[(l, n, p + 1)] = min(
                prev_stage_candidates,
                key=lambda x: x[-1][0],
            )
        else:
            Q[(l, n, p + 1)] = (
                l,
                n,
                0,
                0,
                prev_stage,
                (INFEASIBLE_LATENCY, {}, {}),
            )

        return updates

    def _plan(
        self,
        layers,
        devices,
        micro_batch_size: int,
        num_micro_batches: int,
        p,
        n,
        l,
        np,
        lp,
        state,
    ):
        # n - np devices execute layers l - lp and form a new step

        (Q, Qt, ef, eb, dm) = state

        (Qt_u, ef_u, eb_u, dm_u) = ({}, {}, {}, {})
        updates = (
            Qt_u,
            ef_u,
            eb_u,
            dm_u,
        )  # updates to the state are collected here

        logger.debug(f"planning {(l, n, lp, np, p)}")

        if p == min(len(layers), len(devices)):
            # only take minimum of previous stage
            return updates

        if (
            (n - np == 0 or l - lp == 0)
            or (p == 0 and (np != 0 or lp != 0))
            or (p != 0 and (np == 0 or lp == 0))
            or (lp < p or np < p)
            or (
                isinstance(layers[lp], nn.Embedding) and p > 0
            )  # cannot split at embedding
        ):
            ef_u[(l, n, lp, np, p)] = 0
            eb_u[(l, n, lp, np, p)] = 0
            dm_u[(l, n, lp, np, p)] = (0, 0, 0)
            Qt_u[(l, n, lp, np, p)] = (INFEASIBLE_LATENCY, {}, {})
            return updates

        device_group = devices[np:n]

        try:
            samples_allocated = self.sample_allocator.get_allocation_of_micro_batch(
                device_group, micro_batch_size
            )
        except ValueError:
            logger.debug(
                f"unable to obtain allocation for {(l, n, lp, np, p)}: device_group={device_group}, micro_batch_size={micro_batch_size}"
            )

            ef_u[(l, n, lp, np, p)] = 0
            eb_u[(l, n, lp, np, p)] = 0
            dm_u[(l, n, lp, np, p)] = (0, 0, 0)
            Qt_u[(l, n, lp, np, p)] = (INFEASIBLE_LATENCY, {}, {})
            return updates

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
        ) + self._get_transmission_time_for_layer(
            layers, devices, j, n, np, forward_only=True
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
        ) + self._get_transmission_time_for_layer(
            layers, devices, j, n, np, backward_only=True
        )
        ef_u[(l, n, lp, np, p)] = esf
        eb_u[(l, n, lp, np, p)] = esb

        if p == 0:
            dm_u[(l, n, lp, np, p)] = (esf, esb, p)
        else:
            prev = Q[(lp, np, p - 1)][:5] if p > 1 else (lp, np, 0, 0, 0)
            prev_edmf, prev_edmb, prev_dmp = dm[prev]
            dm_u[(l, n, lp, np, p)] = max(
                [
                    (
                        (prev_edmf, prev_edmb, prev_dmp),
                        num_micro_batches * (prev_edmf + prev_edmb)
                        + sum(
                            [
                                (ef[s] + eb[s])
                                for s in self._backtrack(lp, np, prev_dmp + 1, Q)
                            ]
                        ),
                    ),
                    (
                        (esf, esb, p),
                        num_micro_batches * (esf + esb),
                    ),
                    (
                        (esf, esb, p),
                        num_micro_batches
                        * self._get_transmission_time_for_layer(
                            layers, devices, j, n, np
                        )
                        + (esf + esb),
                    ),
                ],
                key=lambda x: x[1],
            )[0]

        edmf, edmb, dmp = dm_u[(l, n, lp, np, p)]

        tsw = sum([ef[s] for s in self._backtrack(lp, np, p, Q)])

        tsa = (
            2
            * (len(device_group) - 1)
            * sum(
                [
                    p.numel()
                    for l in range(i, j)
                    for p in (layers[l].parameters() if l < len(layers) else [])
                ]
            )
            / (len(device_group) * device_group.min_d2d_bandwidth())
        )

        tse = (
            num_micro_batches * (edmf + edmb)
            + sum(
                [(ef[s] + eb[s]) for s in self._backtrack(lp, np, p, Q) if s[-1] < dmp]
            )
            - sum(
                [(ef[s] + eb[s]) for s in self._backtrack(lp, np, p, Q) if s[-1] >= dmp]
            )
        )

        hpp_round_latency = tsw + tsa + tse

        is_max_latency = True
        prev_hpp_round_latency = Q[(lp, np, p)][-1][0] if p > 0 else INFEASIBLE_LATENCY
        if (
            hpp_round_latency < prev_hpp_round_latency
            and prev_hpp_round_latency < INFEASIBLE_LATENCY
        ):
            # keep max latency
            is_max_latency = False
            hpp_round_latency = prev_hpp_round_latency

        try:
            self._backtrack(lp, np, p, Q, check_validity=True)
        except RuntimeError:
            hpp_round_latency = INFEASIBLE_LATENCY

        info = (
            {"tsw": tsw, "tsa": tsa, "tse": tse, "is_max_latency": is_max_latency}
            if logger.isEnabledFor(logging.INFO)
            else {}
        )

        Qt_u[(l, n, lp, np, p)] = (hpp_round_latency, samples_allocated, info)

        return updates

    def _get_transmission_time_for_layer(
        self, layers, devices, lt, n, np, forward_only=False, backward_only=False
    ):
        return (
            self.profiler.get_transmission_time(
                (
                    layers[lt]
                    if not (isinstance(layers[lt], nn.ReLU) and layers[lt].inplace)
                    else layers[lt - 1]
                ),
                self.profiler.get_mean_d2d_bandwidth(
                    devices,
                    devices[0:np],
                    devices[np : n + 1],
                ),
                forward_only=forward_only,
                backward_only=backward_only,
            )
            if lt < len(layers)
            else 0
        )

    def _backtrack(self, lp, np, p, Q, check_validity=False):
        for s in reversed(range(1, p)):
            if lp == 0 or np == 0:
                if check_validity:
                    raise RuntimeError("Invalid state during backtracking")
                break
            l, n, lp, np, pp, _ = Q[lp, np, s]
            yield (l, n, lp, np, pp)

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

        split_points = list(reversed(split_points))
        device_groups = list(reversed(device_groups))
        samples_allocated = list(reversed(samples_allocated))

        return split_points, device_groups, samples_allocated
