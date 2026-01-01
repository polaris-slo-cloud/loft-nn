import logging
import statistics

from itertools import groupby

import networkx as nx
import numpy as np
import torch.nn as nn

from loftnn.model_utils import model_to_layers
from loftnn.planner import Planner
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import WorkerGroup

INVALID = np.iinfo(np.int64).max

logger = logging.getLogger(__name__)


class EdgeCloudPlanner(Planner):
    sample_allocator: SampleAllocator

    def __init__(self, sample_allocator: SampleAllocator):
        self.sample_allocator = sample_allocator

    def plan(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        num_micro_batches: int,
        compute_capacity_threshhold: float = 0.1,
        bandwidth_threshhold: float = 0.1,
    ):
        layers = model_to_layers(model)

        # Step 1: Determine device groups

        worker_groups = []

        similiar_compute_capacities = {}
        similiar_d2d_bandwidths = {}
        for worker in worker_group:
            similiar_compute_capacity = []
            similiar_d2d_bandwidth = {}

            for other_worker in worker_group:
                if worker == other_worker:
                    continue

                if (
                    worker.compute_capacity * (1 - compute_capacity_threshhold)
                    < other_worker.compute_capacity
                    < worker.compute_capacity * (1 + compute_capacity_threshhold)
                ):
                    similiar_compute_capacity.append(other_worker)
                    added = True

                added = False
                for d2d_bandwidth, workers in similiar_d2d_bandwidth.items():
                    if (
                        d2d_bandwidth * (1 - bandwidth_threshhold)
                        < worker_group.d2d_bandwidth(worker, other_worker)
                        < d2d_bandwidth * (1 + bandwidth_threshhold)
                    ):
                        workers.append(other_worker)
                        added = True
                        break

                if not added:
                    d2d_bandwidth = worker_group.d2d_bandwidth(worker, other_worker)
                    similiar_d2d_bandwidth[d2d_bandwidth] = [other_worker]

            similiar_compute_capacities[worker] = similiar_compute_capacity
            similiar_d2d_bandwidths[worker] = similiar_d2d_bandwidth

        G = nx.Graph()
        for worker in worker_group:
            G.add_node(worker.rank)

        mean_d2d_bandwidth = statistics.mean(
            {bandwidth for bandwidth in worker_group.d2d_bandwidths.values()}
        )
        for worker in worker_group:
            similiar_d2d_bandwidth = similiar_d2d_bandwidths[worker]

            max_d2d_bandwidth = max(similiar_d2d_bandwidth)
            if max_d2d_bandwidth >= mean_d2d_bandwidth:
                for other_worker in similiar_d2d_bandwidth[max_d2d_bandwidth]:
                    G.add_edge(worker.rank, other_worker.rank, bandwidth=d2d_bandwidth)

        device_group_ranks = list(nx.find_cliques(G))

        # use data parallelism only in first device group
        device_group_ranks = device_group_ranks[:1] + [
            [rank] for ranks in device_group_ranks[1:] for rank in ranks
        ]

        # remove devices with least occurring compute capacity until devices in data parallelism groups have the same compute capacity
        all_equal = False
        ranks = device_group_ranks[0] if device_group_ranks else []
        while not all_equal and ranks:
            current_worker_group = worker_group[ranks]
            compute_capacities = [
                worker.compute_capacity for worker in current_worker_group
            ]
            all_equal = len(set(compute_capacities)) <= 1
            if not all_equal:
                compute_capacity_occurrences = {
                    key: len(list(group))
                    for (key, group) in groupby(sorted(compute_capacities))
                }
                least_occurring_compute_capacity = min(
                    compute_capacity_occurrences,
                    key=compute_capacity_occurrences.get,
                )

                ranks_to_split_out = [
                    [worker.rank]
                    for worker in current_worker_group
                    if worker.compute_capacity == least_occurring_compute_capacity
                ]
                ranks_to_keep = [
                    [
                        rank
                        for rank in device_group_ranks[0]
                        if [rank] not in ranks_to_split_out
                    ]
                ]

                device_group_ranks = (
                    ranks_to_keep
                    + ranks_to_split_out
                    + [[rank] for ranks in device_group_ranks[1:] for rank in ranks]
                )

                ranks = ranks_to_keep[0]

        device_groups = [max(ranks) + 1 for ranks in device_group_ranks[:-1]]
        device_groups = device_groups if device_groups else [0]

        worker_groups = [worker_group[ranks] for ranks in device_group_ranks]

        logger.info(f"device groups: {device_groups}")
        if logger.isEnabledFor(logging.INFO):
            import matplotlib.pyplot as plt

            fig = plt.figure()
            pos = nx.spring_layout(G)
            nx.draw(G, with_labels=True, ax=fig.add_subplot())
            edge_labels = nx.get_edge_attributes(G, "bandwidth")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            fig.savefig("edge_cloud_planner_graph.png")
            logger.info("saved edge_cloud_planner_graph.png")

        # Step 2: Determine split points

        split_points = []

        params = []
        for layer in layers:
            params.append(sum(p.numel() for p in layer.parameters()))

        params_intervals = np.cumsum(params)
        params_diff = np.diff(params_intervals)
        zero_diffs = np.concatenate(
            (
                [True],  # first layer is invalid as split point
                params_diff == 0,  # layers with 0 parameters are invalid as split point
            ),
            axis=0,
        )
        params_intervals[zero_diffs] = INVALID

        total_parameters = params_intervals[-1]
        total_compute_capacity = sum(w.compute_capacity for w in worker_group)

        compute_capacity_of_groups = np.array(
            [
                sum(w.compute_capacity for w in worker_group)
                for worker_group in worker_groups
            ]
        )
        params_per_group = (
            compute_capacity_of_groups / total_compute_capacity
        ) * total_parameters
        params_per_group_intervals = np.cumsum(params_per_group)

        split_points = []
        for params_per_group_interval in params_per_group_intervals[:-1]:
            split_point = -1
            while split_point not in split_points:
                interval_diff = np.abs(params_per_group_interval - params_intervals)
                closest_interval = interval_diff.argmin()
                if closest_interval not in split_points:
                    split_point = closest_interval
                    split_points.append(split_point)
                else:
                    params_intervals[closest_interval] = INVALID
        split_points = split_points if split_points else [0]

        logger.info(f"split points: {split_points}")

        # Step 3: Determine sample allocation

        samples_allocated = [
            self.sample_allocator.allocation_of_micro_batch(
                worker_group, micro_batch_size
            )
            for worker_group in worker_groups
        ]

        logger.info(f"samples allocated: {samples_allocated}")

        return split_points, device_groups, samples_allocated
