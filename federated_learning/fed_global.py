import random
import torch
import numpy as np
import logging
from .moe_improved_aggregation import aggregate_with_router

logger = logging.getLogger(__name__)

def get_clients_this_round(fed_args, round_idx):
    """
    Determine which clients participate in this round based on federated learning settings.
    """
    clients_this_round = []

    if fed_args.fed_alg == 'local':
        # Exact "local" algorithm
        if fed_args.num_clients == 1:
            # Single-client scenario (e.g., single-machine DPO): choose client 0 by default
            clients_this_round = [0]
            logger.debug("fed_alg is 'local' with num_clients=1. Selecting client 0.")
        else:
            logger.warning(
                f"fed_alg is 'local' with num_clients={fed_args.num_clients}. "
                "For this DPO-like setup, defaulting to client 0. "
                "If this is a federated scenario, review client selection for 'local' algorithm."
            )
            clients_this_round = [0]  # Or list(range(fed_args.num_clients)) if intent differs

    elif fed_args.fed_alg.startswith('local') and len(fed_args.fed_alg) > 5 and fed_args.fed_alg[5:].isdigit():
        client_id_str = fed_args.fed_alg[5:]  # part after "local"
        try:
            client_id = int(client_id_str)
            if 0 <= client_id < fed_args.num_clients:
                clients_this_round = [client_id]
                logger.debug(f"fed_alg is '{fed_args.fed_alg}'. Selecting client {client_id}.")
            else:
                logger.error(
                    f"Parsed client_id {client_id} from fed_alg '{fed_args.fed_alg}' "
                    f"is out of range for num_clients {fed_args.num_clients}. "
                    "Defaulting to all clients for safety."
                )
                clients_this_round = list(range(fed_args.num_clients))
        except ValueError:
            logger.error(
                f"Could not parse client_id from fed_alg '{fed_args.fed_alg}'. "
                "Defaulting to all clients."
            )
            clients_this_round = list(range(fed_args.num_clients))

    elif fed_args.fed_alg.startswith('fedavg') or fed_args.fed_alg.startswith('fedsgd'):
        num_select = int(fed_args.num_clients * fed_args.sample_clients)
        # Ensure num_select is within [1, num_clients] (if num_clients > 0)
        num_select = max(1, min(num_select, fed_args.num_clients)) if fed_args.num_clients > 0 else 0

        if fed_args.sample_clients < 1.0 and fed_args.num_clients > num_select and num_select > 0:
            clients_this_round = random.sample(range(fed_args.num_clients), num_select)
            logger.debug(f"fed_alg is '{fed_args.fed_alg}'. Sampling {num_select} clients.")
        else:
            clients_this_round = list(range(fed_args.num_clients))  # select all clients
            logger.debug(
                f"fed_alg is '{fed_args.fed_alg}'. Selecting all {fed_args.num_clients} clients "
                "(sampling not applicable or sample_clients >= 1.0)."
            )

    elif (
        fed_args.fed_alg.startswith('fedprox') or
        fed_args.fed_alg.startswith('scaffold') or
        fed_args.fed_alg.startswith('fednova') or
        fed_args.fed_alg.startswith('fedadam') or
        fed_args.fed_alg.startswith('fedyogi')
    ):
        num_select = int(fed_args.num_clients * fed_args.sample_clients)
        num_select = max(1, min(num_select, fed_args.num_clients)) if fed_args.num_clients > 0 else 0

        if fed_args.sample_clients < 1.0 and fed_args.num_clients > num_select and num_select > 0:
            clients_this_round = random.sample(range(fed_args.num_clients), num_select)
            logger.debug(f"fed_alg is '{fed_args.fed_alg}'. Sampling {num_select} clients.")
        else:
            clients_this_round = list(range(fed_args.num_clients))
            logger.debug(
                f"fed_alg is '{fed_args.fed_alg}'. Selecting all {fed_args.num_clients} clients "
                "(sampling not applicable or sample_clients >= 1.0)."
            )
    else:
        logger.warning(
            f"fed_alg '{fed_args.fed_alg}' not specifically handled for client selection. "
            "Defaulting to all clients."
        )
        clients_this_round = list(range(fed_args.num_clients))

    logger.info(f"Round {round_idx}: selected clients: {clients_this_round}")
    return clients_this_round


def global_aggregate(
    fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round,
    round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, vc_args=None,
    global_router_dict=None, local_router_dict_list=None
):
    """
    Perform global aggregation based on the specified federated algorithm. Supports router-aware
    aggregation, value-chain-aware aggregation, and several FedOpt variants.
    """
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if global_router_dict is not None and local_router_dict_list is not None:
        return aggregate_with_router(
            fed_args=fed_args,
            global_dict=global_dict,
            local_dict_list=local_dict_list,
            global_router_dict=global_router_dict,
            local_router_dict_list=local_router_dict_list,
            sample_num_list=sample_num_list,
            clients_this_round=clients_this_round,
            round_idx=round_idx,
            router_aggregation_weight=0.5
        )

    # Value-chain-aware aggregation
    elif hasattr(vc_args, 'use_value_chain') and vc_args.use_value_chain:
        return value_chain_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list,
            clients_this_round, round_idx, vc_args
        )

    elif fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([
                local_dict_list[client][key] * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round])
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients

    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ])
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key]) for client in clients_this_round
            ]) / len(clients_this_round)
            # In "Adaptive Federated Optimization", momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau
            )

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key]) for client in clients_this_round
            ]) / len(clients_this_round)
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1 - fed_args.fedopt_beta2) * delta_square * torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau
            )

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key]) for client in clients_this_round
            ]) / len(clients_this_round)
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            opt_proxy_dict[key] = fed_args.fedopt_beta2 * param + (1 - fed_args.fedopt_beta2) * torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau
            )

    else:  # Standard sample-size-weighted aggregation
        for key in global_dict.keys():
            global_dict[key] = sum([
                local_dict_list[client][key] * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ])

    return global_dict, global_auxiliary


def value_chain_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, vc_args):
    """
    Value-chain-aware model aggregation.

    Args:
        fed_args: Federated learning args.
        global_dict: Global model parameters.
        local_dict_list: List of local model parameters.
        sample_num_list: List of sample counts per client.
        clients_this_round: Participating clients this round.
        round_idx: Current round index.
        vc_args: Value-chain args.

    Returns:
        (updated global model parameters, auxiliary info)
    """
    # Compute position-based weights
    position_weights = {}
    num_clients = fed_args.num_clients

    # Dynamic weight coefficient
    if vc_args.dynamic_weight_adjust:
        # Adjust with training progress
        progress = min(1.0, (round_idx + 1) / fed_args.num_rounds)

        # Early phase: front stages weigh more; late phase: back stages weigh more
        if progress < 0.3:  # early training
            alpha = 1.0 - progress * 2  # from 1.0 down to 0.4
        elif progress < 0.7:  # mid training: balanced
            alpha = 0.4
        else:  # late training
            alpha = 0.4 - (progress - 0.7) * 1.3  # from 0.4 down to 0

        # Per-client weights
        for client in range(num_clients):
            # Earlier position => higher early weight; later position => higher late weight
            position_factor = (alpha * (num_clients - client) + (1 - alpha) * (client + 1)) / num_clients
            position_weights[client] = position_factor
    else:
        # Static weights: later positions weigh more
        for client in range(num_clients):
            position_weights[client] = (client + 1) / num_clients

    # Normalize over participating clients
    weight_sum = sum([position_weights[client] for client in clients_this_round])
    for client in clients_this_round:
        position_weights[client] /= weight_sum

    # Periodic logging
    if round_idx % 10 == 0:  # every 10 rounds
        print(
            f"Round {round_idx} - Value Chain Weights: "
            + ", ".join([f"Client {c}: {position_weights[c]:.3f}" for c in clients_this_round])
        )

    # Choose aggregation style
    if vc_args.chain_type == "strict_chain":
        # Strict chain: sequential aggregation by value-chain order
        return strict_chain_aggregate(
            global_dict, local_dict_list, sample_num_list, clients_this_round, position_weights
        )
    else:
        # Relaxed chain: weighted average across all clients
        return relaxed_chain_aggregate(
            global_dict, local_dict_list, sample_num_list, clients_this_round, position_weights
        )


def strict_chain_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round, position_weights):
    """Strict chain aggregation: sequentially aggregate by value-chain order."""
    # Sort clients by position (ascending)
    sorted_clients = sorted(clients_this_round)

    # Initialize result with copies of global params
    result_dict = {key: global_dict[key].clone() for key in global_dict.keys()}

    # Keys common to all participating local models
    common_keys = set(global_dict.keys())
    for client in sorted_clients:
        client_local_keys = set(local_dict_list[client].keys())
        common_keys = common_keys.intersection(client_local_keys)

    print(f"Aggregating: using {len(common_keys)} common keys across models (total keys: {len(global_dict.keys())})")

    # Sequential aggregation over common keys
    for i, client in enumerate(sorted_clients):
        client_weight = position_weights[client]
        for key in common_keys:
            if i == 0:  # first client: mix with global
                result_dict[key] = (1 - client_weight) * global_dict[key] + client_weight * local_dict_list[client][key]
            else:  # subsequent clients: mix with current result
                result_dict[key] = (1 - client_weight) * result_dict[key] + client_weight * local_dict_list[client][key]

    return result_dict, None


def relaxed_chain_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round, position_weights):
    """
    Relaxed chain aggregation: weighted averaging across all clients.
    """
    # Initialize result
    result_dict = {key: torch.zeros_like(global_dict[key]) for key in global_dict.keys()}

    # Weighted sum
    for client in clients_this_round:
        client_weight = position_weights[client]
        for key in global_dict.keys():
            result_dict[key] += client_weight * local_dict_list[client][key]

    return result_dict, None
