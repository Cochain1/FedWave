import random
import torch
import numpy as np
import logging
from datasets import Dataset
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def split_dataset_with_moe(
    fed_args, 
    script_args, 
    dataset, 
    moe_args=None, 
    moe_router=None, 
    tokenizer=None
):
    """
    Dataset splitting function based on an MoE router.

    Args:
        fed_args: Federated learning arguments.
        script_args: Script arguments.
        dataset: The dataset to split.
        moe_args: MoE arguments (if any).
        moe_router: MoE router (if any).
        tokenizer: Tokenizer for processing text.

    Returns:
        local_datasets: A list of per-client local datasets.
    """
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=script_args.seed)
    
    # Whether to use MoE routing
    use_moe = moe_args is not None and hasattr(moe_args, 'use_moe') and moe_args.use_moe
    
    # Initialize local dataset containers
    local_datasets = []
    for _ in range(fed_args.num_clients):
        local_datasets.append([])
    
    # Use the MoE data processor if routing is enabled
    if use_moe and moe_router is not None:
        try:
            from federated_learning.moe_data_processor import MoEDataProcessor
            
            # Create the processor
            processor = MoEDataProcessor(
                moe_router=moe_router,
                tokenizer=tokenizer,
                num_experts=fed_args.num_clients,
                top_k=moe_args.moe_top_k,
                expert_names=moe_args.expert_names,
                balance_strategy=moe_args.routing_balance_strategy
            )
            
            # Partition the dataset according to the MoE router
            logger.info("Using the MoE router to assign samples to experts...")
            expert_datasets, expert_indices = processor.partition_dataset_by_expert(
                dataset, local_datasets
            )
            
            # Update local_datasets
            for i in range(fed_args.num_clients):
                local_datasets[i] = expert_datasets[i]
                
            # Log sizes
            for i, ds in enumerate(local_datasets):
                logger.info(f"Expert {i}: {len(ds)} samples")
                
            # Convert lists to Dataset objects
            for i in range(fed_args.num_clients):
                if isinstance(local_datasets[i], list):
                    local_datasets[i] = Dataset.from_list(local_datasets[i])
                
            # Return the split result
            return local_datasets
            
        except Exception as e:
            logger.error(f"MoE-based dataset split failed: {e}")
            logger.warning("Falling back to standard splitting method")
    
    # Value-chain split, if enabled on fed_args
    if hasattr(fed_args, 'use_value_chain') and fed_args.use_value_chain:
        # Value-chain split — even contiguous chunks
        num_samples = len(dataset)
        samples_per_client = num_samples // fed_args.num_clients
        
        for i in range(fed_args.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < fed_args.num_clients - 1 else num_samples
            local_datasets[i] = dataset.select(range(start_idx, end_idx))
            logger.info(f"Value-chain stage {i} (client {i}): {len(local_datasets[i])} samples, index range [{start_idx}:{end_idx}]")
    
    elif fed_args.split_strategy == "iid":
        # Standard IID split
        for i in range(fed_args.num_clients):
            local_datasets[i] = dataset.shard(fed_args.num_clients, i)
            logger.info(f"IID split — client {i}: {len(local_datasets[i])} samples")
    
    elif fed_args.split_strategy == "sequential":
        # Sequential split — contiguous chunks in original order
        num_samples = len(dataset)
        samples_per_client = num_samples // fed_args.num_clients
        
        for i in range(fed_args.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < fed_args.num_clients - 1 else num_samples
            local_datasets[i] = dataset.select(range(start_idx, end_idx))
            logger.info(f"Sequential split — client {i}: {len(local_datasets[i])} samples, index range [{start_idx}:{end_idx}]")
    
    return local_datasets

def get_dataset_this_round_with_moe(
    dataset, 
    round, 
    fed_args, 
    script_args, 
    tokenizer=None, 
    client_id=None, 
    moe_args=None, 
    moe_router=None
):
    """
    Get the dataset samples for the current round, with optional MoE routing.

    Args:
        dataset: The full dataset.
        round: Current federated round.
        fed_args: Federated learning arguments.
        script_args: Script arguments.
        tokenizer: Tokenizer for processing text.
        client_id: Current client ID.
        moe_args: MoE arguments (if any).
        moe_router: MoE router (if any).

    Returns:
        The dataset for the current round.
    """
    use_moe = moe_args is not None and hasattr(moe_args, 'use_moe') and moe_args.use_moe
    
    # Number of samples to draw this round
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    
    # If MoE routing is enabled and we're in client-side training
    if use_moe and moe_router is not None and client_id is not None:
        try:
            # If a tokenizer is provided, convert to SimpleDataset
            if tokenizer is not None:
                from federated_learning.custom_dataset import convert_to_simple_dataset
                
                # Uncertainty/importance sampling — prefer samples with high routing score for this expert
                all_texts = []
                for i in range(len(dataset)):
                    text = ""
                    if 'instruction' in dataset[i]:
                        text += dataset[i]['instruction'] + " "
                    if 'input' in dataset[i] and dataset[i]['input']:
                        text += dataset[i]['input']
                    all_texts.append(text)
                
                # Batch-compute routing scores
                device = next(moe_router.parameters()).device
                routing_weights = []
                batch_size = 32
                
                # Compute routing results in batches
                for i in range(0, len(all_texts), batch_size):
                    batch_texts = all_texts[i:i+batch_size]
                    with torch.no_grad():
                        result = moe_router.route_text_input(batch_texts)
                        if isinstance(result, tuple) and len(result) > 1:
                            batch_indices = result[0]  # assume the first return is what we need
                        else:
                            batch_indices = result
                        for indices in batch_indices:
                            # Score for the current client in routing results
                            weight = 1.0 if client_id in indices else 0.0
                            routing_weights.append(weight)
                
                # Sample based on weights plus some exploration
                random.seed(round)
                weighted_indices = []
                
                # Select sample if it's routed to this client, or with 20% chance for exploration
                for i, weight in enumerate(routing_weights):
                    if weight > 0 or random.random() < 0.2:
                        weighted_indices.append(i)
                
                # Draw samples
                if len(weighted_indices) > num2sample:
                    random_idx = random.sample(weighted_indices, num2sample)
                else:
                    # Not enough weighted samples — fill from the rest
                    remaining = num2sample - len(weighted_indices)
                    all_indices = list(range(len(dataset)))
                    for idx in weighted_indices:
                        if idx in all_indices:
                            all_indices.remove(idx)
                    additional = random.sample(all_indices, min(remaining, len(all_indices)))
                    random_idx = weighted_indices + additional
                
                # Convert to SimpleDataset
                dataset_this_round = dataset.select(random_idx)
                return convert_to_simple_dataset(
                    dataset_this_round, 
                    round, 
                    script_args.batch_size, 
                    script_args.max_steps, 
                    script_args.gradient_accumulation_steps,
                    tokenizer,
                    script_args.seq_length
                )
            
            # No tokenizer — simple random sampling
            else:
                random.seed(round)
                random_idx = random.sample(range(0, len(dataset)), num2sample)
                dataset_this_round = dataset.select(random_idx)
                return dataset_this_round
                
        except Exception as e:
            logger.error(f"MoE round sampling failed: {e}")
            # Fallback to standard sampling
            logger.warning("Falling back to standard sampling method")
    
    # If a tokenizer is provided, use the custom converter
    if tokenizer is not None:
        from federated_learning.custom_dataset import convert_to_simple_dataset
        return convert_to_simple_dataset(
            dataset, 
            round, 
            script_args.batch_size, 
            script_args.max_steps, 
            script_args.gradient_accumulation_steps,
            tokenizer,
            script_args.seq_length
        )
    
    # Standard random sampling
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)
    return dataset_this_round
