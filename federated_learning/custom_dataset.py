import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import random

class SimpleDataset(Dataset):
    
    def __init__(self, original_dataset, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        

        if isinstance(original_dataset, HFDataset):
            all_items = [original_dataset[i] for i in range(len(original_dataset))]
        else:
            all_items = original_dataset
        
        for item in all_items:
            instruction = item.get('instruction', '')
            inp = item.get('input', '')
            output = item.get('response', item.get('output', ''))
            if inp:
                full_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{inp}\n\n### Response: {output}"
            else:
                full_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: {output}"
            
            try:
                encodings = tokenizer(
                    full_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                if 'input_ids' not in encodings or encodings['input_ids'].size(0) == 0:
                    continue
                    
                encodings['labels'] = encodings['input_ids'].clone()
                
                self.examples.append({
                    'input_ids': encodings['input_ids'][0],
                    'attention_mask': encodings['attention_mask'][0],
                    'labels': encodings['labels'][0]
                })
                
            except Exception as e:
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def convert_to_simple_dataset(dataset, round_idx, batch_size, max_steps, grad_accum_steps, tokenizer, max_length=512):
    if len(dataset) == 0:
        return dataset
    
    num_samples = batch_size * grad_accum_steps * max_steps
    num_samples = min(num_samples, len(dataset))
    
    random.seed(round_idx)
    indices = random.sample(range(len(dataset)), num_samples)
    sampled_dataset = [dataset[i] for i in indices]
    
    return SimpleDataset(sampled_dataset, tokenizer, max_length)