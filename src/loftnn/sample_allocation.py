# This file contains an implementation of the algorithms described in:
#   Shengyuan Ye, Liekang Zeng, Xiaowen Chu, Guoliang Xing, and Xu Chen.
#   Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices.
#   In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, ACM MobiCom ’24,
#   page 312–326, New York, NY, USA, 2024. Association for Computing Machinery.

from loftnn.cache import CacheProvider, CacheType


class SampleAllocator:
    cache_provider: CacheProvider

    def __init__(self, cache_provider: CacheProvider):
        self.cache_provider = cache_provider

    def get_allocation_of_micro_batch(self, device_group, micro_batch_size):
        cache = self.cache_provider.get_cache(CacheType.sample_allocation_cache)
        if (device_group, micro_batch_size) in cache:
            return cache[(device_group, micro_batch_size)]
        else:
            raise ValueError(
                f"Sample allocation infeasible: device_group={device_group}, micro_batch_size={micro_batch_size}"
            )

    def compute_allocation_of_micro_batch(self, device_group, micro_batch_size):
        try:
            cache = self.cache_provider.get_cache(CacheType.sample_allocation_cache)
            cache[(device_group, micro_batch_size)] = self._allocation_of_micro_batch(
                device_group, micro_batch_size
            )
        except ValueError:
            pass

    def _allocation_of_micro_batch(self, device_group, micro_batch_size):
        samples_allocated = self._memory_aware_balancing(device_group, micro_batch_size)
        samples_allocated = self._straggler_workload_offloading(
            device_group, samples_allocated
        )
        return samples_allocated

    def _memory_aware_balancing(self, device_group, beta):
        samples_allocated = {d: 0 for d in device_group}
        return self._memory_aware_balancing_recursive(
            device_group, beta, samples_allocated
        )

    def _memory_aware_balancing_recursive(self, device_group, beta, samples_allocated):
        if len(device_group) == 0 and beta > 0:
            raise ValueError(f"No devices in the group but beta={beta} > 0")
        elif beta == 0:
            return samples_allocated

        batch_size = {}
        delta = {}
        compute_capacity = {d: d.compute_capacity for d in device_group}
        available_memory = {d: d.available_memory for d in device_group}

        for idx, d in enumerate(device_group):
            batch_size[d] = min(d.batch_size(), beta)
            compute_capacity_fraction = compute_capacity[d] / (
                sum([compute_capacity[dev] for dev in device_group])
            )
            if beta % 2 == 1 and compute_capacity_fraction == 0.5:
                compute_capacity_fraction = 0.5001 if idx % 2 == 0 else 0.4999
            delta[d] = min(
                round(compute_capacity_fraction * beta),
                batch_size[d],
            )
            samples_allocated[d] = samples_allocated[d] + delta[d]
            available_memory[d] = available_memory[d] - samples_allocated[d]

        device_group_remaining = [d for d in device_group if available_memory[d] > 0]
        beta_remaining = beta - sum([delta[d] for d in device_group])
        return self._memory_aware_balancing_recursive(
            device_group_remaining, beta_remaining, samples_allocated
        )

    def _straggler_workload_offloading(
        self, device_group, samples_allocated, samples_to_transfer=4
    ):
        device_group = device_group.sorted(key=lambda d: samples_allocated[d])

        done = False
        while not done:
            old_straggler = device_group.get_slowest()

            if samples_allocated[old_straggler] - samples_to_transfer > 0:
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
