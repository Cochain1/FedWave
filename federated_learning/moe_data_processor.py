import torch
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import logging
import random
import copy
import math
from datasets import Dataset

logger = logging.getLogger(__name__)

class MoEDataProcessor:
    """
    MoE expert data processor — assigns data based on router outputs.
    """
    def __init__(
        self,
        moe_router,
        tokenizer,
        num_experts: int,
        top_k: int = 2,
        expert_names: List[str] = None,
        balance_strategy: str = "count"
    ):
        """
        Initialize the MoE data processor.

        Args:
            moe_router: MoE router instance.
            tokenizer: Model tokenizer.
            num_experts: Number of experts.
            top_k: Number of top experts to select.
            expert_names: Optional list of expert names.
            balance_strategy: Balancing strategy for expert selection
                ('none', 'count', 'weight').
        """
        self.moe_router = moe_router
        self.tokenizer = tokenizer
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_names = expert_names or [f"Expert{i}" for i in range(num_experts)]
        self.balance_strategy = balance_strategy

        # Expert assignment counters
        self.expert_counts = [0] * num_experts
        # Expert assignment weights (for balancing)
        self.expert_weights = [1.0] * num_experts

        # Cache for routing results
        self.routing_cache = {}

    def route_text_batch(self, text_batch: List[str]) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        Route a batch of texts to experts.

        Args:
            text_batch: List of input texts.

        Returns:
            - List of expert index lists per sample [batch_size, top_k]
            - List of per-sample expert-weight tensors [batch_size, num_experts]
        """
        # Check cache first
        cached_indices = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(text_batch):
            cache_key = hash(text)  # use text hash as cache key
            if cache_key in self.routing_cache:
                cached_indices.append((i, self.routing_cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Compute routing for uncached texts
        if uncached_texts:
            expert_indices = self.moe_router.route_text_input(uncached_texts, self.tokenizer)

            # Apply balancing if requested
            if self.balance_strategy != "none":
                expert_indices = self._balance_expert_selection(expert_indices)

            # Update cache
            for i, idx in enumerate(uncached_indices):
                cache_key = hash(text_batch[idx])
                self.routing_cache[cache_key] = expert_indices[i]

        # Merge cached and newly computed results
        all_indices = [None] * len(text_batch)
        for i, cached in cached_indices:
            all_indices[i] = cached
        for i, idx in enumerate(uncached_indices):
            all_indices[idx] = expert_indices[i]

        # Build weight tensors — equal weights for selected experts
        weights = []
        for indices in all_indices:
            weight = torch.zeros(self.num_experts)
            weight[indices] = 1.0 / len(indices)
            weights.append(weight)

        return all_indices, weights

    def _balance_expert_selection(self, expert_indices: List[List[int]]) -> List[List[int]]:
        """
        Balance expert selection to avoid overloading some experts while others are idle.

        Args:
            expert_indices: Original expert indices [batch_size, top_k]

        Returns:
            Balanced expert indices.
        """
        # Count selections per expert within this batch
        batch_counts = [0] * self.num_experts
        for indices in expert_indices:
            for idx in indices:
                batch_counts[idx] += 1

        # Update running totals
        for i in range(self.num_experts):
            self.expert_counts[i] += batch_counts[i]

        if self.balance_strategy == "count":
            # Rebalance — prefer experts with lower cumulative counts
            balanced_indices = []

            for indices in expert_indices:
                # Keep part of the router's original top-k as-is
                keep_count = max(1, int(self.top_k * 0.5))
                # Preserve original order priority (lower position = higher priority)
                sorted_indices = sorted(indices, key=lambda x: -indices.index(x))
                keep_indices = sorted_indices[:keep_count]

                # Fill remaining slots with least-used experts
                remaining_slots = self.top_k - keep_count
                if remaining_slots > 0:
                    all_experts = list(range(self.num_experts))
                    all_experts.sort(key=lambda x: self.expert_counts[x])

                    balance_indices = []
                    for expert in all_experts:
                        if expert not in keep_indices and len(balance_indices) < remaining_slots:
                            balance_indices.append(expert)

                    final_indices = keep_indices + balance_indices
                else:
                    final_indices = keep_indices

                # Update counts for chosen experts
                for idx in final_indices:
                    self.expert_counts[idx] += 1

                balanced_indices.append(final_indices)

            return balanced_indices

        # Default: return original selections
        return expert_indices

    def partition_dataset_by_expert(
        self,
        dataset,
        local_datasets=None
    ) -> Tuple[List[Any], Dict[int, List[int]]]:
        """
        Partition the dataset by MoE routing results to experts.

        Args:
            dataset: Original dataset.
            local_datasets: Optional pre-assigned local datasets (if any).

        Returns:
            - List of expert-specific datasets.
            - Mapping from expert id to sample indices.
        """
        expert_datasets = [[] for _ in range(self.num_experts)]
        expert_indices = {i: [] for i in range(self.num_experts)}

        # Ensure local_datasets has the right shape
        if local_datasets is None:
            local_datasets = [[] for _ in range(self.num_experts)]

        # Extract texts for routing
        all_texts = []
        for i in range(len(dataset)):
            text = ""
            if 'instruction' in dataset[i]:
                text += dataset[i]['instruction'] + " "
            if 'input' in dataset[i] and dataset[i]['input']:
                text += dataset[i]['input']
            all_texts.append(text)

        # Batch routing
        batch_size = 32
        all_expert_indices = []

        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_indices, _ = self.route_text_batch(batch_texts)
            all_expert_indices.extend(batch_indices)

        # Assign samples based on routing
        for i, expert_idx_list in enumerate(all_expert_indices):
            for expert_idx in expert_idx_list:
                expert_datasets[expert_idx].append(dataset[i])
                expert_indices[expert_idx].append(i)

        # Logging stats
        total_assignments = sum(len(indices) for indices in expert_indices.values())
        logger.info(f"MoE routing stats: total samples: {len(dataset)}, total assignments: {total_assignments}")

        for i in range(self.num_experts):
            logger.info(f"Expert {i} ({self.expert_names[i]}): {len(expert_datasets[i])} samples")

        return expert_datasets, expert_indices

    def get_client_batch_for_sample(
        self,
        sample_idx: int,
        all_expert_indices: List[List[int]]
    ) -> List[int]:
        """
        Get the expert indices (client batch) for a given sample.
        """
        if sample_idx < len(all_expert_indices):
            return all_expert_indices[sample_idx]
        else:
            return random.sample(range(self.num_experts), min(self.top_k, self.num_experts))
