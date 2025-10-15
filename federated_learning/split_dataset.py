import random
import json
import os
from datasets import Dataset

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)
    local_datasets = []
    
    if hasattr(fed_args, 'use_value_chain') and fed_args.use_value_chain:
        num_samples = len(dataset)
        samples_per_client = num_samples // fed_args.num_clients
        
        for i in range(fed_args.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < fed_args.num_clients - 1 else num_samples
            local_datasets.append(dataset.select(range(start_idx, end_idx)))
            print(f"Value-chain stage {i} (client {i}): {len(local_datasets[i])} samples, index range [{start_idx}:{end_idx}]")
    
    elif fed_args.split_strategy == "iid":
        # Standard IID split
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    elif fed_args.split_strategy == "sequential":
        # Sequential split — assign contiguous chunks of the dataset to each client
        num_samples = len(dataset)
        samples_per_client = num_samples // fed_args.num_clients
        
        for i in range(fed_args.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < fed_args.num_clients - 1 else num_samples
            local_datasets.append(dataset.select(range(start_idx, end_idx)))
    
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args, tokenizer=None):
    """Get the dataset samples for the current federated round."""
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    
    # If a tokenizer is provided, use the custom SimpleDataset converter
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
    
    # Original random sampling
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)
    return dataset_this_round

def load_custom_json_dataset(file_path, script_args):
    """
    Load a custom JSON-format dataset.

    Args:
        file_path: Path to the JSON file.
        script_args: Script arguments.

    Returns:
        A processed HuggingFace Dataset object.
    """
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract and process fields
    instructions = []
    inputs = []
    responses = []
    
    for item in data:
        inst = item.get('instruction', '')
        inp = item.get('input', '')
        output = item.get('output', '')
        
        # Merge instruction and input
        if inp:
            instructions.append(f"{inst}\n{inp}")
        else:
            instructions.append(inst)
        
        inputs.append(inp)  # Keep original input in case it is needed
        responses.append(output)
    
    # Create a Dataset object
    dataset_dict = {
        'instruction': instructions,
        'input': inputs,
        'response': responses
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Print dataset info
    print(f"Loaded custom dataset with {len(dataset)} samples")
    print("Example sample:")
    print(dataset[0])
    
    return dataset
