# federated_learning/moe_improved_aggregation.py
"""
Improved federated aggregation module.
Implements joint aggregation of LoRA weights and MoE router weights.
"""

import torch
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def aggregate_with_router(
    fed_args,
    global_dict: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    global_router_dict: Dict[str, torch.Tensor],
    local_router_dict_list: List[Dict[str, torch.Tensor]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    round_idx: int,
    router_aggregation_weight: float = 0.5,
    momentum: float = 0.9,
    router_lr: float = 0.1,
    use_adaptive_weights: bool = True
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Aggregate LoRA weights and MoE router weights.

    Args:
        fed_args: Federated learning args.
        global_dict: Global LoRA state dict.
        local_dict_list: Per-client LoRA state dicts.
        global_router_dict: Global router state dict.
        local_router_dict_list: Per-client router state dicts.
        sample_num_list: Sample counts for each client.
        clients_this_round: Indices of participating clients this round.
        round_idx: Current round index.
        router_aggregation_weight: Base coefficient for router aggregation.
        momentum: Momentum factor for LoRA update.
        router_lr: Router learning rate for EMA-style update.
        use_adaptive_weights: Whether to use adaptive client weights.

    Returns:
        (updated global LoRA dict, updated router dict)
    """
    # Total samples this round
    total_samples = sum([sample_num_list[client] for client in clients_this_round])

    if total_samples == 0:
        logger.warning(f"Round {round_idx}: no samples participated in aggregation")
        return global_dict, global_router_dict

    logger.info(
        f"Round {round_idx}: start aggregation - clients: {clients_this_round}, "
        f"total samples: {total_samples}"
    )

    # ========== 1) Aggregate LoRA weights ==========
    aggregated_lora = aggregate_lora_weights(
        global_dict=global_dict,
        local_dict_list=local_dict_list,
        sample_num_list=sample_num_list,
        clients_this_round=clients_this_round,
        total_samples=total_samples,
        momentum=momentum,
        round_idx=round_idx
    )

    # ========== 2) Aggregate MoE router weights ==========
    aggregated_router = None
    if global_router_dict is not None and local_router_dict_list is not None:
        aggregated_router = aggregate_router_weights(
            global_router_dict=global_router_dict,
            local_router_dict_list=local_router_dict_list,
            sample_num_list=sample_num_list,
            clients_this_round=clients_this_round,
            total_samples=total_samples,
            router_aggregation_weight=router_aggregation_weight,
            router_lr=router_lr,
            round_idx=round_idx,
            fed_args=fed_args,
            use_adaptive_weights=use_adaptive_weights
        )

    return aggregated_lora, aggregated_router


def aggregate_lora_weights(
    global_dict: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    total_samples: int,
    momentum: float,
    round_idx: int
) -> Dict[str, torch.Tensor]:
    """
    Aggregate LoRA weights using FedAvg with optional momentum.
    """
    logger.info(f"Aggregating LoRA weights - param count: {len(global_dict.keys())}")

    aggregated_dict = {}

    for key in global_dict.keys():
        # Initialize to zeros
        aggregated_param = torch.zeros_like(global_dict[key])

        # Weighted sum
        valid_clients = 0
        for client in clients_this_round:
            if client < len(local_dict_list) and key in local_dict_list[client]:
                # Sample-proportional weight
                weight = sample_num_list[client] / total_samples
                aggregated_param += local_dict_list[client][key] * weight
                valid_clients += 1

        if valid_clients == 0:
            logger.warning(f"Parameter {key} not updated by any client this round")
            aggregated_dict[key] = global_dict[key]
        else:
            # Momentum update if configured
            if 0 < momentum < 1:
                # new = momentum * old + (1 - momentum) * aggregated
                aggregated_dict[key] = momentum * global_dict[key] + (1 - momentum) * aggregated_param
            else:
                aggregated_dict[key] = aggregated_param

    # Stats
    param_changes = {}
    for key in aggregated_dict.keys():
        if key in global_dict:
            change = torch.norm(aggregated_dict[key] - global_dict[key]).item()
            param_changes[key] = change

    if param_changes:
        avg_change = np.mean(list(param_changes.values()))
        max_change = max(param_changes.values())
        logger.info(f"LoRA avg param change: {avg_change:.6f}, max change: {max_change:.6f}")

    return aggregated_dict


def aggregate_router_weights(
    global_router_dict: Dict[str, torch.Tensor],
    local_router_dict_list: List[Dict[str, torch.Tensor]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    total_samples: int,
    router_aggregation_weight: float,
    router_lr: float,
    round_idx: int,
    fed_args,
    use_adaptive_weights: bool
) -> Dict[str, torch.Tensor]:
    """
    Aggregate MoE router weights with a specialized strategy that considers:
    1) sample counts, 2) client position (curriculum-style weighting), 3) performance proxy.
    """
    logger.info(f"Aggregating MoE router weights - param count: {len(global_router_dict.keys())}")

    num_clients = fed_args.num_clients
    aggregated_router = {}

    for key in global_router_dict.keys():
        aggregated_param = torch.zeros_like(global_router_dict[key])
        total_weight = 0.0

        for client in clients_this_round:
            if client < len(local_router_dict_list) and local_router_dict_list[client] is not None:
                if key in local_router_dict_list[client]:
                    # 1) Base weight: proportional to samples
                    base_weight = (
                        sample_num_list[client] / total_samples
                        if total_samples > 0 else 1.0 / len(clients_this_round)
                    )

                    # 2) Optional adaptive factors
                    if use_adaptive_weights:
                        # Progress in [0, 1]
                        progress = min(1.0, round_idx / max(fed_args.num_rounds, 1))

                        position_weight = calculate_position_weight(
                            client_id=client,
                            num_clients=num_clients,
                            progress=progress
                        )

                        performance_weight = calculate_performance_weight(
                            client_id=client,
                            round_idx=round_idx
                        )

                        final_weight = base_weight * position_weight * performance_weight * router_aggregation_weight
                    else:
                        final_weight = base_weight * router_aggregation_weight

                    aggregated_param += local_router_dict_list[client][key] * final_weight
                    total_weight += final_weight

        # Normalize
        if total_weight > 0:
            aggregated_param = aggregated_param / total_weight

        # EMA-style cautious update for router
        if key in global_router_dict:
            aggregated_router[key] = (1 - router_lr) * global_router_dict[key] + router_lr * aggregated_param
        else:
            aggregated_router[key] = aggregated_param

    # Stats
    router_changes = {}
    for key in aggregated_router.keys():
        if key in global_router_dict:
            change = torch.norm(aggregated_router[key] - global_router_dict[key]).item()
            router_changes[key] = change

    if router_changes:
        avg_change = np.mean(list(router_changes.values()))
        max_change = max(router_changes.values())
        logger.info(f"Router avg param change: {avg_change:.6f}, max change: {max_change:.6f}")

    return aggregated_router


def calculate_position_weight(
    client_id: int,
    num_clients: int,
    progress: float
) -> float:
    """
    Position-based weight.

    Early training emphasizes lower-index clients; later training emphasizes higher-index clients.
    """
    # Normalize position to [0, 1]
    normalized_position = client_id / max(num_clients - 1, 1)

    if progress < 0.3:
        # Early: emphasize front clients
        weight = 1.0 - 0.5 * normalized_position
    elif progress < 0.7:
        # Mid: balanced
        weight = 0.8 + 0.2 * np.sin(np.pi * normalized_position)
    else:
        # Late: emphasize tail clients
        weight = 0.5 + 0.5 * normalized_position

    return max(0.1, weight)  # ensure a minimum weight


def calculate_performance_weight(
    client_id: int,
    round_idx: int,
    performance_history: Optional[Dict[int, float]] = None
) -> float:
    """
    Performance-based weight.

    If performance history is available, use it; otherwise apply a mild, round-dependent variation.
    """
    if performance_history and client_id in performance_history:
        performance = performance_history[client_id]
        # Map to [0.5, 1.5]
        weight = 0.5 + performance
    else:
        base_weight = 1.0
        variation = 0.1 * np.sin(2 * np.pi * client_id / 10 + round_idx * 0.1)
        weight = base_weight + variation

    return max(0.1, min(2.0, weight))  # clamp to a reasonable range


def adaptive_aggregation(
    fed_args,
    global_dict: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    global_router_dict: Optional[Dict[str, torch.Tensor]],
    local_router_dict_list: Optional[List[Dict[str, torch.Tensor]]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    round_idx: int,
    aggregation_strategy: str = "weighted_avg",
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Adaptive aggregation strategies.

    aggregation_strategy:
        - "weighted_avg": FedAvg
        - "weighted_geo": geometric mean (not implemented here)
        - "trimmed_mean": trimmed mean (remove extremes)
        - "median": median aggregation
        - "krum": Krum (Byzantine-robust)
    """
    logger.info(f"Using aggregation strategy: {aggregation_strategy}")

    if aggregation_strategy == "weighted_avg":
        return aggregate_with_router(
            fed_args, global_dict, local_dict_list,
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )

    elif aggregation_strategy == "trimmed_mean":
        return trimmed_mean_aggregation(
            fed_args, global_dict, local_dict_list,
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            trim_ratio=0.1, **kwargs
        )

    elif aggregation_strategy == "median":
        return median_aggregation(
            fed_args, global_dict, local_dict_list,
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )

    elif aggregation_strategy == "krum":
        return krum_aggregation(
            fed_args, global_dict, local_dict_list,
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )

    else:
        logger.warning(f"Unknown aggregation strategy {aggregation_strategy}; falling back to weighted_avg")
        return aggregate_with_router(
            fed_args, global_dict, local_dict_list,
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )


def trimmed_mean_aggregation(
    fed_args,
    global_dict: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    global_router_dict: Optional[Dict[str, torch.Tensor]],
    local_router_dict_list: Optional[List[Dict[str, torch.Tensor]]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    round_idx: int,
    trim_ratio: float = 0.1,
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Trimmed-mean aggregation: drop extremes before averaging.
    """
    logger.info(f"Trimmed-mean aggregation - trim_ratio={trim_ratio}")

    num_trim = max(1, int(len(clients_this_round) * trim_ratio))
    aggregated_dict = {}

    for key in global_dict.keys():
        params = []
        for client in clients_this_round:
            if client < len(local_dict_list) and key in local_dict_list[client]:
                params.append(local_dict_list[client][key])

        if len(params) > 2 * num_trim:
            stacked = torch.stack(params)
            norms = torch.norm(stacked.view(len(params), -1), dim=1)
            sorted_indices = torch.argsort(norms)
            trimmed_indices = sorted_indices[num_trim:-num_trim]
            trimmed_params = stacked[trimmed_indices]
            aggregated_dict[key] = torch.mean(trimmed_params, dim=0)
        elif len(params) > 0:
            aggregated_dict[key] = torch.mean(torch.stack(params), dim=0)
        else:
            aggregated_dict[key] = global_dict[key]

    # Router aggregation (if present): reuse standard routine
    aggregated_router = None
    if global_router_dict is not None:
        _, aggregated_router = aggregate_with_router(
            fed_args, {}, [],
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )

    return aggregated_dict, aggregated_router


def median_aggregation(
    fed_args,
    global_dict: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    global_router_dict: Optional[Dict[str, torch.Tensor]],
    local_router_dict_list: Optional[List[Dict[str, torch.Tensor]]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    round_idx: int,
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Median aggregation across client parameters.
    """
    logger.info("Median aggregation")

    aggregated_dict = {}

    for key in global_dict.keys():
        params = []
        for client in clients_this_round:
            if client < len(local_dict_list) and key in local_dict_list[client]:
                params.append(local_dict_list[client][key])

        if len(params) > 0:
            stacked = torch.stack(params)
            aggregated_dict[key] = torch.median(stacked, dim=0)[0]
        else:
            aggregated_dict[key] = global_dict[key]

    aggregated_router = None
    if global_router_dict is not None:
        _, aggregated_router = aggregate_with_router(
            fed_args, {}, [],
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )

    return aggregated_dict, aggregated_router


def krum_aggregation(
    fed_args,
    global_dict: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    global_router_dict: Optional[Dict[str, torch.Tensor]],
    local_router_dict_list: Optional[List[Dict[str, torch.Tensor]]],
    sample_num_list: List[int],
    clients_this_round: List[int],
    round_idx: int,
    num_byzantine: int = 1,
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Krum aggregation: select the client update closest to others (robust to Byzantine faults).
    """
    logger.info(f"Krum aggregation - assumed Byzantine count={num_byzantine}")

    n = len(clients_this_round)
    m = n - num_byzantine - 2  # Krum parameter

    if m <= 0:
        logger.warning("Not enough clients for Krum; falling back to weighted_avg")
        return aggregate_with_router(
            fed_args, global_dict, local_dict_list,
            global_router_dict, local_router_dict_list,
            sample_num_list, clients_this_round, round_idx,
            **kwargs
        )

    # Pairwise distances
    distances = np.zeros((n, n))

    for i, client_i in enumerate(clients_this_round):
        for j, client_j in enumerate(clients_this_round):
            if i != j:
                dist = 0
                count = 0
                for key in global_dict.keys():
                    if key in local_dict_list[client_i] and key in local_dict_list[client_j]:
                        diff = local_dict_list[client_i][key] - local_dict_list[client_j][key]
                        dist += torch.norm(diff).item() ** 2
                        count += 1
                distances[i, j] = dist / max(count, 1)

    # Krum scores
    scores = []
    for i in range(n):
        dists = distances[i, :]
        sorted_indices = np.argsort(dists)
        closest_m = sorted_indices[1:m+1]  # exclude self at index 0
        score = sum(distances[i, j] for j in closest_m)
        scores.append(score)

    best_client_idx = np.argmin(scores)
    best_client = clients_this_round[best_client_idx]

    logger.info(f"Krum chose client {best_client} (score={scores[best_client_idx]:.4f})")

    aggregated_dict = copy.deepcopy(local_dict_list[best_client])

    aggregated_router = None
    if global_router_dict is not None and local_router_dict_list[best_client] is not None:
        aggregated_router = copy.deepcopy(local_router_dict_list[best_client])

    return aggregated_dict, aggregated_router


def analyze_aggregation_impact(
    global_dict_before: Dict[str, torch.Tensor],
    global_dict_after: Dict[str, torch.Tensor],
    local_dict_list: List[Dict[str, torch.Tensor]],
    clients_this_round: List[int]
) -> Dict[str, Any]:
    """
    Analyze the impact of aggregation.

    Returns:
        A dict of summary statistics.
    """
    stats = {
        'total_change': 0.0,
        'max_change': 0.0,
        'min_change': float('inf'),
        'param_changes': {},
        'client_contributions': defaultdict(float)
    }

    # Per-parameter changes
    for key in global_dict_after.keys():
        if key in global_dict_before:
            change = torch.norm(global_dict_after[key] - global_dict_before[key]).item()
            stats['param_changes'][key] = change
            stats['total_change'] += change
            stats['max_change'] = max(stats['max_change'], change)
            stats['min_change'] = min(stats['min_change'], change)

            # Per-client contributions (distance to pre-aggregation global)
            for client in clients_this_round:
                if client < len(local_dict_list) and key in local_dict_list[client]:
                    contrib = torch.norm(
                        local_dict_list[client][key] - global_dict_before[key]
                    ).item()
                    stats['client_contributions'][client] += contrib

    num_params = len(stats['param_changes'])
    stats['avg_change'] = stats['total_change'] / num_params if num_params > 0 else 0.0

    logger.info("Aggregation impact analysis:")
    logger.info(f"  total change: {stats['total_change']:.6f}")
    logger.info(f"  avg change: {stats['avg_change']:.6f}")
    logger.info(f"  max/min change: {stats['max_change']:.6f} / {stats['min_change']:.6f}")

    return stats
