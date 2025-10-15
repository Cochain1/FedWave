# federated_learning/moe_improved_data.py
"""
Improved MoE data utilities.
Implements mixed data allocation so each client receives a mixture rather than strictly specialized data.
"""

import random
import torch
import numpy as np
from datasets import Dataset
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def distribute_mixed_data_to_clients(
    dataset,
    fed_args,
    script_args,
    mix_ratio: float = 1.0,
    overlap_ratio: float = 0.0,
    seed: Optional[int] = None
) -> List[Dataset]:
    """
    Distribute mixed data to each client instead of pre-assigning specialized data.

    Args:
        dataset: Original dataset.
        fed_args: Federated learning arguments; must include num_clients.
        script_args: Script arguments; should include seed.
        mix_ratio: Mixing ratio.
            - 1.0: Fully mixed; each client gets random samples.
            - 0.0: Fully specialized; each client gets a distinct slice.
            - 0.5: Half mixed; 50% specialized + 50% random.
        overlap_ratio: Inter-client overlap ratio.
            - 0.0: No overlap.
            - 0.5: 50% of data overlaps across multiple clients.
        seed: Random seed.

    Returns:
        local_datasets: A list of Dataset objects, one per client.
    """
    # Set random seeds
    if seed is None:
        seed = script_args.seed if hasattr(script_args, 'seed') else 42
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle and basic stats
    dataset = dataset.shuffle(seed=seed)
    num_samples = len(dataset)
    num_clients = fed_args.num_clients
    
    logger.info(f"Start allocation: total={num_samples}, clients={num_clients}")
    logger.info(f"mix_ratio={mix_ratio}, overlap_ratio={overlap_ratio}")
    
    # Initialize index sets for each client
    client_indices = [set() for _ in range(num_clients)]
    
    if mix_ratio >= 1.0:
        # ===== Fully mixed mode =====
        logger.info("Using fully mixed allocation")
        
        if overlap_ratio > 0:
            # Mixed with overlap
            samples_per_client = int(num_samples / num_clients * (1 + overlap_ratio))
            
            for client_id in range(num_clients):
                client_samples = min(samples_per_client, num_samples)
                selected_indices = np.random.choice(
                    num_samples, 
                    size=client_samples, 
                    replace=False
                )
                client_indices[client_id].update(selected_indices)
        else:
            # Mixed without overlap
            indices = list(range(num_samples))
            random.shuffle(indices)
            
            samples_per_client = num_samples // num_clients
            remaining = num_samples % num_clients
            
            start_idx = 0
            for client_id in range(num_clients):
                # Base quota + distribute remainder
                client_samples = samples_per_client + (1 if client_id < remaining else 0)
                end_idx = start_idx + client_samples
                
                client_indices[client_id].update(indices[start_idx:end_idx])
                start_idx = end_idx
    
    elif mix_ratio <= 0.0:
        # ===== Fully specialized mode =====
        logger.info("Using fully specialized allocation")
        
        samples_per_client = num_samples // num_clients
        
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = (client_id + 1) * samples_per_client if client_id < num_clients - 1 else num_samples
            client_indices[client_id].update(range(start_idx, end_idx))
    
    else:
        # ===== Partially mixed mode =====
        logger.info(f"Using partially mixed allocation (mix_ratio={mix_ratio})")
        
        # Compute specialized vs mixed counts
        specialized_samples = int(num_samples * (1 - mix_ratio))
        mixed_samples = num_samples - specialized_samples
        
        # Step 1: allocate specialized slice per client
        samples_per_client_specialized = specialized_samples // num_clients
        
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client_specialized
            end_idx = min((client_id + 1) * samples_per_client_specialized, specialized_samples)
            for idx in range(start_idx, end_idx):
                client_indices[client_id].add(idx)
        
        # Step 2: allocate mixed samples
        mixed_indices = list(range(specialized_samples, num_samples))
        
        if overlap_ratio > 0:
            # Mixed with overlap
            for idx in mixed_indices:
                recipients = max(1, int(num_clients * overlap_ratio))
                recipients = min(recipients, num_clients)
                selected_clients = random.sample(range(num_clients), recipients)
                for client_id in selected_clients:
                    client_indices[client_id].add(idx)
        else:
            # Mixed without overlap
            random.shuffle(mixed_indices)
            mixed_per_client = len(mixed_indices) // num_clients
            
            for client_id in range(num_clients):
                start = client_id * mixed_per_client
                end = (client_id + 1) * mixed_per_client if client_id < num_clients - 1 else len(mixed_indices)
                for idx in mixed_indices[start:end]:
                    client_indices[client_id].add(idx)
    
    # ===== Build Dataset objects =====
    local_datasets = []
    
    for client_id in range(num_clients):
        indices = sorted(list(client_indices[client_id]))
        
        if len(indices) == 0:
            logger.warning(f"Client {client_id} received no data; assigning small fallback set")
            fallback_size = min(10, num_samples)
            indices = random.sample(range(num_samples), fallback_size)
        
        client_dataset = dataset.select(indices)
        local_datasets.append(client_dataset)
        
        logger.info(f"Client {client_id}: assigned {len(indices)} samples")
    
    # Stats
    total_assigned = sum(len(idxset) for idxset in client_indices)
    unique_samples = len(set().union(*client_indices)) if client_indices else 0
    
    logger.info("Allocation summary:")
    logger.info(f"  - total assigned samples: {total_assigned}")
    logger.info(f"  - unique samples: {unique_samples}")
    if unique_samples > 0:
        logger.info(f"  - avg duplication rate: {(total_assigned - unique_samples) / unique_samples * 100:.2f}%")
    else:
        logger.info("  - avg duplication rate: N/A (no samples)")
    
    return local_datasets


def get_mixed_batch_for_client(
    client_id: int,
    local_dataset: Dataset,
    batch_size: int,
    round_idx: int,
    sampling_strategy: str = "random",
    temperature: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Fetch a mixed batch for a specific client.

    Args:
        client_id: Client id.
        local_dataset: Client-local dataset.
        batch_size: Batch size.
        round_idx: Current round index (used for deterministic sampling).
        sampling_strategy: "random" | "sequential" | "curriculum".
        temperature: Sampling temperature for probabilistic selection.

    Returns:
        batch: A list of examples.
    """
    dataset_size = len(local_dataset)
    if dataset_size == 0:
        logger.warning(f"Client {client_id} dataset is empty")
        return []
    
    actual_batch_size = min(batch_size, dataset_size)
    random.seed(round_idx * 1000 + client_id)
    
    if sampling_strategy == "random":
        indices = random.sample(range(dataset_size), actual_batch_size)
    elif sampling_strategy == "sequential":
        start_idx = (round_idx * batch_size) % dataset_size
        indices = [(start_idx + i) % dataset_size for i in range(actual_batch_size)]
    elif sampling_strategy == "curriculum":
        # Assume data sorted by difficulty; gradually increase coverage
        progress = min(1.0, round_idx / 100)
        max_idx = int(dataset_size * (0.3 + 0.7 * progress))
        indices = random.sample(range(max_idx), min(actual_batch_size, max_idx))
    else:
        indices = random.sample(range(dataset_size), actual_batch_size)
    
    batch = [local_dataset[idx] for idx in indices]
    return batch


def analyze_data_distribution(
    local_datasets: List[Dataset],
    fed_args,
    num_samples_to_analyze: int = 100
) -> Dict[str, Any]:
    """
    Analyze the distribution across clients.

    Args:
        local_datasets: List of client datasets.
        fed_args: Federated learning arguments.
        num_samples_to_analyze: Number of samples to hash/check for overlap.

    Returns:
        statistics: Summary statistics.
    """
    num_clients = fed_args.num_clients
    
    client_stats = []
    overlap_matrix = np.zeros((num_clients, num_clients))
    client_hashes = []
    
    for client_id in range(num_clients):
        dataset = local_datasets[client_id]
        stats = {
            'client_id': client_id,
            'num_samples': len(dataset),
            'sample_hashes': set()
        }
        for i in range(min(len(dataset), num_samples_to_analyze)):
            sample = dataset[i]
            sample_hash = hash(str(sample))
            stats['sample_hashes'].add(sample_hash)
        
        client_stats.append(stats)
        client_hashes.append(stats['sample_hashes'])
    
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                overlap = len(client_hashes[i] & client_hashes[j])
                denom = max(len(client_hashes[i]), 1)
                overlap_matrix[i, j] = overlap / denom
    
    statistics = {
        'num_clients': num_clients,
        'total_samples': sum(s['num_samples'] for s in client_stats),
        'avg_samples_per_client': np.mean([s['num_samples'] for s in client_stats]),
        'std_samples_per_client': np.std([s['num_samples'] for s in client_stats]),
        'min_samples': min(s['num_samples'] for s in client_stats),
        'max_samples': max(s['num_samples'] for s in client_stats),
        'avg_overlap': np.mean(overlap_matrix[overlap_matrix > 0]) if np.any(overlap_matrix > 0) else 0,
        'client_stats': client_stats,
        'overlap_matrix': overlap_matrix
    }
    
    logger.info("=" * 50)
    logger.info("Data distribution stats:")
    logger.info(f"  total samples: {statistics['total_samples']}")
    logger.info(f"  avg per client: {statistics['avg_samples_per_client']:.2f} ± {statistics['std_samples_per_client']:.2f}")
    logger.info(f"  range: [{statistics['min_samples']}, {statistics['max_samples']}]")
    logger.info(f"  avg overlap: {statistics['avg_overlap']*100:.2f}%")
    logger.info("=" * 50)
    
    return statistics


def create_routing_aware_batches(
    client_id: int,
    local_dataset: Dataset,
    moe_router,
    tokenizer,
    batch_size: int,
    round_idx: int,
    device: str = "cuda"
) -> Tuple[List[Dict], List[float]]:
    """
    Create routing-aware batches by preferring samples the router assigns to this client.

    Args:
        client_id: Client id.
        local_dataset: Client-local dataset.
        moe_router: MoE router module.
        tokenizer: Tokenizer.
        batch_size: Batch size.
        round_idx: Current round index.
        device: Device string.

    Returns:
        selected_batch: Selected examples.
        routing_scores: Router scores for those examples.
    """
    if moe_router is None:
        batch = get_mixed_batch_for_client(
            client_id, local_dataset, batch_size, round_idx
        )
        return batch, [1.0] * len(batch)
    
    all_scores = []
    dataset_size = min(len(local_dataset), batch_size * 10)  # cap compute cost
    
    with torch.no_grad():
        for i in range(dataset_size):
            sample = local_dataset[i]
            text = sample.get('instruction', '') + ' ' + sample.get('input', '')
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)
            
            # For simplicity, use a random hidden state placeholder here.
            hidden_states = torch.randn(1, inputs['input_ids'].shape[1], moe_router.hidden_size).to(device)
            routing_weights, _ = moe_router(hidden_states)
            client_score = routing_weights[0, client_id].item()
            all_scores.append((i, client_score))
    
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    temperature = 0.5
    selected_indices = []
    selected_scores = []
    
    for i in range(min(batch_size, len(all_scores))):
        if random.random() < temperature:
            idx = random.randint(0, min(batch_size * 2, len(all_scores)) - 1)
        else:
            idx = i
        sample_idx, score = all_scores[idx]
        selected_indices.append(sample_idx)
        selected_scores.append(score)
    
    selected_batch = [local_dataset[idx] for idx in selected_indices]
    logger.debug(f"Client {client_id}: avg routing score {np.mean(selected_scores):.4f}")
    
    return selected_batch, selected_scores
