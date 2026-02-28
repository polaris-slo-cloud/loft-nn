import logging
import statistics

import networkx as nx
import numpy as np

import torch
import torch.nn as nn

from torchinfo import summary

from loftnn.model_utils import LayerInputProvider, model_to_layers
from loftnn.planner import Planner
from loftnn.sample_allocation import SampleAllocator
from loftnn.types import Device, WorkerGroup

INVALID = np.iinfo(np.int64).max

logger = logging.getLogger(__name__)


class EdgeCloudPlanner(Planner):
    layer_input: LayerInputProvider
    sample_allocator: SampleAllocator
    device: Device

    def __init__(
        self,
        layer_input: LayerInputProvider,
        sample_allocator: SampleAllocator,
        device: Device,
    ):
        self.layer_input = layer_input
        self.sample_allocator = sample_allocator
        self.device = device

    def _group_close(self, numbers: list[int], threshhold: float):
        groups = []
        for num in sorted(numbers):
            if groups and num <= (groups[-1][-1] * (1 + threshhold)):
                groups[-1].append(num)
            else:
                groups.append([num])
        return groups

    def _disjoint(self, cliques):
        disjoint_cliques = []
        nodes_seen = set()

        cliques.sort(key=len, reverse=True)
        for clique in cliques:
            disjoint_clique = [n for n in clique if n not in nodes_seen]

            if disjoint_clique:
                disjoint_cliques.append(disjoint_clique)
                nodes_seen.update(disjoint_clique)

        return disjoint_cliques

    def plan(
        self,
        model: nn.Module,
        worker_group: WorkerGroup,
        micro_batch_size: int,
        num_micro_batches: int,
        compute_capacity_threshhold: float = 0.25,
        bandwidth_threshhold: float = 0.1,
        memory_estimation_factor: float = None,
    ):
        layers = model_to_layers(model)

        logger.info(
            f"planning {len(layers)} layers on {len(worker_group)} workers with micro_batch_size={micro_batch_size} and num_micro_batches={num_micro_batches}"
        )

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
            if max_d2d_bandwidth >= mean_d2d_bandwidth or (
                mean_d2d_bandwidth / max_d2d_bandwidth
            ) > (1 - bandwidth_threshhold):
                for other_worker in similiar_d2d_bandwidth[max_d2d_bandwidth]:
                    G.add_edge(worker.rank, other_worker.rank, bandwidth=d2d_bandwidth)

        cliques = list(nx.find_cliques(G))
        device_group_ranks = self._disjoint(cliques)

        logger.info(f"device groups from clique: {device_group_ranks}")

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
            all_equal = min(compute_capacities) / max(compute_capacities) > (
                1 - compute_capacity_threshhold
            )
            if not all_equal:
                compute_capacity_occurrences = {
                    min(group): group
                    for group in self._group_close(
                        compute_capacities, compute_capacity_threshhold
                    )
                }
                least_occurring_compute_capacity = min(
                    compute_capacity_occurrences,
                    key=lambda m: len(compute_capacity_occurrences[m]),
                )

                ranks_to_split_out = [
                    [worker.rank]
                    for worker in current_worker_group
                    if worker.compute_capacity
                    in compute_capacity_occurrences[least_occurring_compute_capacity]
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
        device_group_ranks = sorted(device_group_ranks)

        logger.info(f"device groups with stragglers removed: {device_group_ranks}")

        max_worker_rank = max([worker.rank for worker in worker_group])
        device_groups = [
            max(ranks) + 1
            for ranks in device_group_ranks
            if max(ranks) != max_worker_rank
        ]
        device_groups = device_groups if device_groups else [0]

        worker_groups = [worker_group[ranks] for ranks in device_group_ranks]

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
        intermediate_activations = []
        for layer in layers:
            params.append(sum(p.numel() for p in layer.parameters()))

            x = self.layer_input.random(layer, self.device)
            with torch.no_grad():
                y = layer(x)
                intermediate_activations.append(y.numel())

        logger.debug(f"parameters: {dict(enumerate(params))}")
        logger.debug(
            f"intermediate activations: {dict(enumerate(intermediate_activations))}"
        )

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
                statistics.mean(w.compute_capacity for w in worker_group)
                for worker_group in worker_groups
            ]
        )
        params_per_group = (
            compute_capacity_of_groups / total_compute_capacity
        ) * total_parameters
        params_per_group_intervals = np.cumsum(params_per_group)

        logger.info(f"parameters per group intervals: {params_per_group_intervals}")

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

        split_points = [
            (
                split_point
                if not isinstance(
                    layers[split_point], nn.Embedding
                )  # cannot split at embedding
                else split_point + 1
            )
            for split_point in split_points
        ]

        logger.info(
            f"intermediate activations of split points (before), absolute: {[intermediate_activations[split_point - 1] for split_point in split_points]}"
        )
        logger.info(
            f"intermediate activations of split points (before), relative to mean: {[intermediate_activations[split_point - 1] / statistics.mean(intermediate_activations) for split_point in split_points]}"
        )

        # avoid split points with large intermediate activations
        replacement = {}
        for idx, split_point in enumerate(split_points):
            prev_split_point = split_points[idx - 1] if idx != 0 else 0
            next_split_point = (
                split_points[idx + 1] if idx != (len(split_points) - 1) else len(layers)
            )

            neighbourhood_size = max(1, int(len(layers) / 20))
            neighbourhood = [
                split_point - i for i in range(1, neighbourhood_size + 1)
            ] + [split_point + i for i in range(1, neighbourhood_size + 1)]
            for neighbour in neighbourhood:
                if (
                    neighbour not in split_points
                    and neighbour > prev_split_point
                    and neighbour < next_split_point
                ):
                    if (
                        intermediate_activations[neighbour - 1]
                        / intermediate_activations[split_point - 1]
                        < 0.5
                    ):
                        replacement[split_point] = neighbour

        split_points_to_replace = replacement.keys()
        split_points = [
            (
                split_point
                if split_point not in split_points_to_replace
                else replacement[split_point]
            )
            for split_point in split_points
        ]

        logger.info(
            f"intermediate activations of split points (after), absolute: {[intermediate_activations[split_point - 1] for split_point in split_points]}"
        )
        logger.info(
            f"intermediate activations of split points (after), relative to mean: {[intermediate_activations[split_point - 1] / statistics.mean(intermediate_activations) for split_point in split_points]}"
        )

        activation_checkpointing_budgets = []

        optimizer_states_factor = 3
        if memory_estimation_factor is None:
            # account for memory fragmentation, overheads,... and have a safety margin
            memory_estimation_factor = 1.1

        num_stages = (  # used to account for activation memory in 1F1B schedule
            len(split_points) + 1
        )

        split_points_complete = [0] + split_points + [len(layers)]
        for idx, split_point in enumerate(split_points_complete):
            if (idx - 1) >= 0:
                prev_split_point = split_points_complete[idx - 1]

                fwd_bwd_bytes = 0
                total_bytes = 0
                for i, layer in enumerate(layers[prev_split_point:split_point]):
                    x = self.layer_input.random(layer, self.device)
                    layer_statistics = summary(
                        layer,
                        input_size=x.shape,
                        dtypes=[x.dtype],
                        verbose=0,
                    )

                    fwd_bwd_bytes += int(
                        layer_statistics.total_output_bytes
                        * num_stages
                        * memory_estimation_factor
                    )
                    total_bytes += int(
                        (
                            layer_statistics.total_output_bytes
                            * num_stages
                            * memory_estimation_factor
                        )
                        + (
                            layer_statistics.total_param_bytes
                            * memory_estimation_factor
                            * optimizer_states_factor
                        )
                    )

                    is_first_layer = i == 0
                    if prev_split_point == 0 and is_first_layer:
                        total_bytes += int(
                            layer_statistics.total_input
                            * num_stages
                            * memory_estimation_factor
                        )

                workers_have_device_memory = [
                    (
                        worker.device_memory is not None
                        and self.device in worker.device_memory
                    )
                    for worker in worker_group
                ]
                if all(workers_have_device_memory):
                    free_bytes = statistics.mean(
                        [
                            worker.device_memory[self.device]
                            for worker in worker_group
                            if worker.rank in device_group_ranks[idx - 1]
                        ]
                    )
                else:
                    free_bytes = -1

                logger.info(
                    f"stage {idx} - total bytes estimate: {total_bytes}, free bytes: {free_bytes}, difference: {free_bytes - total_bytes}"
                )

                if total_bytes > free_bytes and free_bytes > 0:
                    # solve: min x s.t. (total_bytes - (x * fwd_bwd_bytes) <= free_bytes)
                    #       (total_bytes - free_bytes) <= x * fwd_bwd_bytes
                    #       ((total_bytes - free_bytes) / fwd_bwd_bytes) <= x
                    #       ((total_bytes - free_bytes) / fwd_bwd_bytes) = x
                    #
                    # then obtain budget = 1 - x

                    x = (total_bytes - free_bytes) / (fwd_bwd_bytes)

                    activation_checkpointing_budgets.append(1 - x)
                else:
                    activation_checkpointing_budgets.append(1.0)

        logger.info(
            f"activation checkpointing budgets: {activation_checkpointing_budgets}"
        )

        # Step 3: Determine sample allocation

        samples_allocated = [
            self.sample_allocator.allocation_of_micro_batch(
                worker_group, micro_batch_size
            )
            for worker_group in worker_groups
        ]

        return (
            split_points,
            device_groups,
            samples_allocated,
            activation_checkpointing_budgets,
        )
