import copy
import os
import json
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from federated_learning.value_chain_model import add_value_chain_to_model, save_value_chain_layer, get_peft_state_from_value_chain, set_peft_state_to_value_chain, detach_value_chain, attach_value_chain
from federated_learning.custom_trainer import ValueChainTrainer
from federated_learning.custom_dataset import SimpleDataset
import torch
# torch.cuda.set_device(1)

def custom_data_collator(features):
    if not features:
        return {}
    batch = {}
    for key in features[0].keys():
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = torch.stack([f[key] for f in features])
    return batch

def get_consistent_model_dict(model, vc_args):
    if vc_args.use_value_chain and hasattr(model, 'value_chain_layer'):
        base_model, vc_state = detach_value_chain(model)
        state_dict = get_peft_model_state_dict(base_model)
        return state_dict, vc_state
    else:
        return get_peft_model_state_dict(model), None

@dataclass
class ValueChainDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        batch = {}
        for key in features[0].keys():
            if key in ['input_ids', 'attention_mask', 'labels']:
                batch[key] = torch.stack([f[key] for f in features])
        if 'stage_id' not in batch:
            batch['stage_id'] = None
        return batch

# ===== Define the arguments =====
script_args, fed_args, peft_config, vc_args = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args, vc_args)
print(script_args, fed_args)
if vc_args.use_value_chain:
    print("use value_chain:", vc_args)

# ===== Load the dataset =====
if hasattr(script_args, 'custom_dataset_path') and script_args.custom_dataset_path:
    dataset = load_custom_json_dataset(script_args.custom_dataset_path, script_args)
else:
    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# model.config.use_cache = False

vc_state = None
if vc_args.use_value_chain:
    model = add_value_chain_to_model(
        model,
        num_stages=fed_args.num_clients,
        chain_type=vc_args.chain_type,
        cross_head_communication=vc_args.cross_head_communication
    )
    model.config.use_cache = False

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict, _ = get_consistent_model_dict(model, vc_args)
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # following Vicuna

# ===== Define the formatting function (for TRL SFTTrainer) =====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # e.g., for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)
    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            training_loss[client].append(-1)
            continue
        
        if vc_args.use_value_chain:
            base_model, vc_state = detach_value_chain(model)
            set_peft_model_state_dict(base_model, global_dict)
            model = attach_value_chain(base_model, vc_state)
            model.current_stage_id = client
        else:
            set_peft_model_state_dict(model, global_dict)
        
        raw_sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args, tokenizer=tokenizer)
        if len(sub_dataset) == 0:
            training_loss[client].append(-1)
            continue
        if len(sub_dataset) > 0:
            try:
                first_sample = sub_dataset[0]
                print("\n===== Debug Info: First sample of dataset =====")
                print(f"Fields in sample: {list(first_sample.keys())}")
                for key, value in first_sample.items():
                    print(f"Field '{key}' type: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                print("==============================================\n")
                
                # Try to manually create a mini-batch
                batch = [sub_dataset[i] for i in range(min(len(sub_dataset), 2))]
                print("\n===== Debug Info: Manually created mini-batch =====")
                for i, sample in enumerate(batch):
                    print(f"Batch sample {i} fields: {list(sample.keys())}")
                print("==============================================\n")
                
            except Exception as e:
                print(f"Error during debugging: {e}")
        print(f"Client {client} dataset size: {len(sub_dataset)}")
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        training_args = get_training_args(script_args, new_lr)

        # Training
        if vc_args.use_value_chain:
            trainer = ValueChainTrainer(
                client_id=client,
                vc_args=vc_args,
                model=model,
                args=training_args,
                train_dataset=sub_dataset,
                tokenizer=tokenizer,
                data_collator=custom_data_collator,  # Use custom collator
            )
        else:
            # Use default trainer
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
        
        # Train and collect results
        try:
            results = trainer.train()
            training_loss[client].append(results.training_loss)
        except Exception as e:
            print(f"Client {client} training failed: {e}")
            training_loss[client].append(-1)  # mark as failed
            continue  # skip this client
        
        # Handle special algorithms
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        
        # Get local dict
        local_dict, temp_vc_state = get_consistent_model_dict(model, vc_args)
        local_dict_list[client] = copy.deepcopy(local_dict)

        # If using value chain, make sure to re-attach
        if vc_args.use_value_chain and temp_vc_state is not None:
            # Ensure model has value chain layer
            if not hasattr(model, 'value_chain_layer'):
                base_model = model
                model = attach_value_chain(base_model, temp_vc_state)
    
    # Server aggregates local models
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list,
        clients_this_round, round, proxy_dict=proxy_dict,
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
        vc_args=vc_args if vc_args.use_value_chain else None
    )
    
    if vc_args.use_value_chain and hasattr(model, 'value_chain_layer'):
        # Detach value chain, update base model, then re-attach
        base_model, vc_state = detach_value_chain(model)
        
        # Ensure keys in global_dict are compatible with base_model
        compatible_dict = {}
        base_state = get_peft_model_state_dict(base_model)
        for key in base_state.keys():
            if key in global_dict:
                compatible_dict[key] = global_dict[key]
            else:
                print(f"Warning: key '{key}' missing in global dict, using original model value")
                compatible_dict[key] = base_state[key]
        
        set_peft_model_state_dict(base_model, compatible_dict)
        model = attach_value_chain(base_model, vc_state)
    else:
        set_peft_model_state_dict(model, global_dict)
    
    # Save model
    if (round+1) % fed_args.save_model_freq == 0:
        # Bring in custom save utility
        try:
            from model_save_utils import save_model_with_lora_and_value_chain
        except ImportError:
            try:
                from federated_learning.model_save_utils import save_model_with_lora_and_value_chain
            except ImportError:
                # If import fails, inline the function
                print("Failed to import model_save_utils, using inline function")
                exec("""
import os
import json
import torch
from dataclasses import asdict
from peft import get_peft_model_state_dict

def save_model_with_lora_and_value_chain(model, save_dir, tokenizer=None, vc_args=None):
    os.makedirs(save_dir, exist_ok=True)
    try:
        has_vc = hasattr(model, 'value_chain_layer')
        has_peft = hasattr(model, 'peft_config') or (hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'))
        
        print(f"Saving model to {save_dir} ...")
        print(f"Detected value chain layer: {has_vc}")
        print(f"Detected LoRA/PEFT: {has_peft}")
        
        # Step 1: Save value chain layers (if present)
        if has_vc:
            base_model = model.base_model
            value_chain_state = {
                'value_chain_layer': model.value_chain_layer.state_dict(),
                'output_layer': model.output_layer.state_dict()
            }
            vc_path = os.path.join(save_dir, "value_chain_layer.pt")
            torch.save(value_chain_state, vc_path)
            print(f"Value chain layers saved to {vc_path}")
            
            if vc_args is not None:
                vc_config_path = os.path.join(save_dir, "value_chain_config.json")
                with open(vc_config_path, 'w') as f:
                    json.dump(asdict(vc_args), f, indent=4)
        else:
            base_model = model
        
        # Step 2: Save LoRA adapter and config
        if has_peft:
            peft_model = None
            if hasattr(model, 'peft_config'):
                peft_model = model
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
                peft_model = model.base_model
            elif hasattr(base_model, 'peft_config'):
                peft_model = base_model
            
            if peft_model is not None:
                active_adapter = getattr(peft_model, 'active_adapter', None)
                peft_model.save_pretrained(save_dir)
                print(f"LoRA adapter and config saved to {save_dir}")
                
                adapter_config_path = os.path.join(save_dir, "adapter_config.json")
                if not os.path.exists(adapter_config_path):
                    print(f"Warning: adapter_config.json not found, saving manually")
                    if hasattr(peft_model, 'peft_config'):
                        config_dict = {}
                        for k, v in peft_model.peft_config.items():
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                                config_dict[k] = v
                        with open(adapter_config_path, 'w') as f:
                            json.dump(config_dict, f, indent=4)
        else:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(save_dir)
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        
        # Step 3: Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        
        return True
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")
        import traceback
        print(traceback.format_exc())
        return False
""")

        # Define save path
        save_dir = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
        
        # Use the dedicated save function
        save_model_with_lora_and_value_chain(
            model=model,
            save_dir=save_dir,
            tokenizer=tokenizer,
            vc_args=vc_args if vc_args.use_value_chain else None
        )
        
        print(f"Round {round+1} model has been saved to {save_dir}")
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
