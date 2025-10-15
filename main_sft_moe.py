import copy
import os
import json
from tqdm.auto import tqdm
import colorama
from colorama import Fore, Style
import time
import numpy as np
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

logger = logging.getLogger(__name__)

try:
    # --- Import from project root utils package ---
    from utils import *

    # --- Import required components precisely from federated_learning package ---
    from federated_learning.fed_global import get_clients_this_round, global_aggregate
    from federated_learning.fed_utils import get_proxy_dict, get_auxiliary_dict
    from federated_learning.split_dataset import split_dataset, get_dataset_this_round, load_custom_json_dataset
    from federated_learning.custom_dataset import SimpleDataset, convert_to_simple_dataset
    from federated_learning.custom_trainer import ValueChainTrainer
    from config import get_config, save_config, get_model_config, get_training_args

    logger.info("Successfully imported all required modules precisely.")

except ImportError as e:
    logger.error(f"Failed to import modules precisely: {e}")
    raise

# Add MoE-related modules
try:
    from federated_learning.moe_config import MoEArguments, update_config_with_moe_args, fix_moe_value_chain_compatibility
    from federated_learning.moe_router import create_automotive_moe_router, MoERouter
    from federated_learning.moe_valuechain_integration import create_moe_value_chain_model
    from federated_learning.moe_model_utils import save_moe_value_chain_model, load_moe_value_chain_model
    from federated_learning.moe_data_processor import MoEDataProcessor
    from federated_learning.moe_split_dataset import split_dataset_with_moe, get_dataset_this_round_with_moe
    
    # Import improved modules
    from federated_learning.moe_improved_data import distribute_mixed_data_to_clients
    from federated_learning.moe_improved_model import ImprovedMoEValueChainModel
    from federated_learning.moe_improved_aggregation import aggregate_with_router
    
    HAS_MOE_SUPPORT = True
    logger.info("Loaded MoE support modules and improved modules.")
except ImportError as e:
    logger.warning(f"Failed to import MoE support modules: {e}")
    HAS_MOE_SUPPORT = False

class ColorfulProgressBar:
    """Custom colorful progress bar with nesting and dynamic descriptions."""
    
    def __init__(self, total, desc="", position=0, color=Fore.BLUE):
        self.total = total
        self.desc = desc
        self.position = position
        self.color = color
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.bar_length = 30
        self.update_interval = 0.2
        self.print_progress()
    
    def print_progress(self):
        """Print current progress."""
        elapsed = time.time() - self.start_time
        
        percent = self.current / self.total if self.total > 0 else 0
        filled_length = int(self.bar_length * percent)
        
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
        
        if percent > 0:
            remaining = elapsed / percent - elapsed
        else:
            remaining = 0
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds // 60
                seconds = seconds % 60
                return f"{int(minutes)}m {int(seconds)}s"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{int(hours)}h {int(minutes)}m"
        
        elapsed_str = format_time(elapsed)
        remaining_str = format_time(remaining)
        
        progress_str = f"{self.color}[{self.desc}] |{bar}| {self.current}/{self.total} [{elapsed_str}<{remaining_str}]{Style.RESET_ALL}"
        
        print(f"\033[{self.position};0H\033[K{progress_str}")
    
    def update(self, n=1):
        """Update progress."""
        self.current += n
        current_time = time.time()
        
        if current_time - self.last_update_time > self.update_interval:
            self.print_progress()
            self.last_update_time = current_time
    
    def set_description(self, desc):
        """Set description text."""
        self.desc = desc
        self.print_progress()
    
    def set_postfix(self, **kwargs):
        """Set postfix text."""
        postfix = ' '.join(f"{k}={v}" for k, v in kwargs.items())
        self.desc = f"{self.desc.split(' |')[0]} | {postfix}"
        self.print_progress()
    
    def close(self):
        """Close the progress bar."""
        print()

def custom_data_collator(features):
    """Ensure correct handling of SimpleDataset outputs."""
    if not features:
        return {}
    
    batch = {}
    for key in features[0].keys():
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = torch.stack([f[key] for f in features])
    
    return batch

def get_consistent_model_dict(model, use_moe=False, vc_args=None):
    """Get a consistent model state dict, handling Value Chain and MoE special cases."""
    if use_moe and hasattr(model, 'moe_router'):
        base_model = model.base_model
        state_dict = get_peft_model_state_dict(base_model)
        return state_dict
    elif vc_args and hasattr(vc_args, 'use_value_chain') and vc_args.use_value_chain and hasattr(model, 'value_chain_layer'):
        try:
            from federated_learning.value_chain_model import detach_value_chain
            base_model, vc_state = detach_value_chain(model)
            state_dict = get_peft_model_state_dict(base_model)
            return state_dict
        except Exception as e:
            logger.error(f"Failed to detach value chain layer: {e}")
            return get_peft_model_state_dict(model)
    else:
        return get_peft_model_state_dict(model)

def save_router_weights(router_dict, save_path, moe_args=None):
    """
    Save router weights in the expected format.
    
    Args:
        router_dict: state_dict of the router
        save_path: path to save
        moe_args: MoE arguments (optional)
    """
    # Wrap into the expected format
    router_save_dict = {
        'state_dict': router_dict,  # Important: include 'state_dict' key
        'num_experts': moe_args.moe_num_experts if moe_args else 4,
        'top_k': moe_args.moe_top_k if moe_args else 2,
        'hidden_size': moe_args.moe_hidden_size if moe_args and hasattr(moe_args, 'moe_hidden_size') else 768,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(router_save_dict, save_path)
    logger.info(f"Router weights saved to {save_path} (with 'state_dict' format)")

def create_improved_moe_value_chain_model(
    base_model,
    moe_args,
    vc_args,
    tokenizer
):
    """Create an improved MoE + Value Chain model."""
    from federated_learning.value_chain_model import ValueChainLayer
    
    # Get hidden size
    if hasattr(base_model, 'config'):
        hidden_size = base_model.config.hidden_size
    elif hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'config'):
        hidden_size = base_model.base_model.config.hidden_size
    else:
        hidden_size = 768
        logger.warning(f"Could not infer hidden_size from model; using default {hidden_size}")
    
    # Create MoE router (neural network based, not keywords)
    moe_router = None
    if moe_args and moe_args.use_moe:
        moe_router = MoERouter(
            hidden_size=hidden_size,
            num_experts=moe_args.moe_num_experts,
            top_k=moe_args.moe_top_k,
            use_keywords=False  # Key: use neural routing
        )
        logger.info(f"Created MoE router: {moe_args.moe_num_experts} experts, top-k={moe_args.moe_top_k}")
    
    # Create value chain layer
    value_chain_layer = None
    if vc_args and vc_args.use_value_chain:
        value_chain_layer = ValueChainLayer(
            hidden_size=hidden_size,
            num_stages=moe_args.moe_num_experts if moe_args else 4,
            chain_type=vc_args.chain_type,
            cross_head_communication=vc_args.cross_head_communication
        )
        logger.info(f"Created value chain layer: {vc_args.chain_type}")
    
    # Create improved integrated model
    model = ImprovedMoEValueChainModel(
        base_model=base_model,
        value_chain_layer=value_chain_layer,
        moe_router=moe_router,
        num_stages=moe_args.moe_num_experts if moe_args else 4,
        top_k=moe_args.moe_top_k if moe_args else 2,
        router_loss_weight=0.1,
        load_balancing_loss_weight=0.01
    )
    
    logger.info("Improved MoE + Value Chain model created successfully.")
    return model

# ===== Main entry point =====
def main():
    # ===== Configure arguments =====
    if HAS_MOE_SUPPORT:
        update_config_with_moe_args()
    
    # Get config
    script_args, fed_args, peft_config, vc_args, moe_args = get_config()
    
    # Add MoE arguments
    if HAS_MOE_SUPPORT:
        moe_args = MoEArguments(
            use_moe=True,
            moe_top_k=2,
            moe_num_experts=fed_args.num_clients,
            moe_router_type="neural",  # Use neural routing
            use_improved_routing=True  # Enable improved routing
        )
        
        # Ensure compatibility between MoE and Value Chain
        moe_args, vc_args, fed_args = fix_moe_value_chain_compatibility(moe_args, vc_args, fed_args)
    else:
        moe_args = None
    
    # Save config
    training_args = get_training_args(script_args, script_args.learning_rate)
    save_config(script_args, fed_args, vc_args)
    
    # Log current configs
    logger.info(f"Script args: {script_args}")
    logger.info(f"Federated args: {fed_args}")
    if vc_args and hasattr(vc_args, 'use_value_chain') and vc_args.use_value_chain:
        logger.info(f"Value chain args: {vc_args}")
    if moe_args and moe_args.use_moe:
        logger.info(f"MoE args: {moe_args}")
    
    # ===== Load dataset =====
    if hasattr(script_args, 'custom_dataset_path') and script_args.custom_dataset_path:
        dataset = load_custom_json_dataset(script_args.custom_dataset_path, script_args)
    else:
        dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
        dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
    
    # ===== Load model =====
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
    
    # ===== Fix MoE params based on actual base model =====
    if HAS_MOE_SUPPORT and moe_args is not None and moe_args.use_moe:
        actual_hf_base_model_config = None
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
            actual_hf_base_model_config = model.base_model.config
        elif hasattr(model, 'config'):
            actual_hf_base_model_config = model.config

        if actual_hf_base_model_config and hasattr(actual_hf_base_model_config, 'hidden_size'):
            base_model_actual_hidden_size = actual_hf_base_model_config.hidden_size
            if not hasattr(moe_args, 'moe_hidden_size') or moe_args.moe_hidden_size != base_model_actual_hidden_size:
                logger.info(f"Set MoE router hidden size to {base_model_actual_hidden_size}")
                moe_args.moe_hidden_size = base_model_actual_hidden_size
    
    # ===== Create improved MoE + Value Chain model if needed =====
    use_moe = moe_args is not None and moe_args.use_moe
    use_improved = moe_args is not None and hasattr(moe_args, 'use_improved_routing') and moe_args.use_improved_routing
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, 
        use_fast=False, 
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_moe or (vc_args and vc_args.use_value_chain):
        if use_improved:
            logger.info("Creating improved integrated model...")
            model = create_improved_moe_value_chain_model(
                base_model=model,
                moe_args=moe_args,
                vc_args=vc_args,
                tokenizer=tokenizer
            )
        else:
            logger.info("Creating standard integrated model...")
            model = create_moe_value_chain_model(
                base_model=model,
                moe_args=moe_args if use_moe else None,
                vc_args=vc_args,
                tokenizer=tokenizer
            )
        
        logger.info("Integrated model created!")
        model.to(training_args.device)
        logger.info(f"Moved model to device: {training_args.device}")
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # Disable cache to avoid CUDA OOM
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    # ===== Define global and local models =====
    global_dict = get_consistent_model_dict(model, use_moe, vc_args)
    local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    
    # Initialize router weights dict
    if use_improved and hasattr(model, 'get_router_state_dict'):
        global_router_dict = model.get_router_state_dict()
        local_router_dict_list = [copy.deepcopy(global_router_dict) for _ in range(fed_args.num_clients)]
        logger.info("Initialized router weight dictionaries.")
    else:
        global_router_dict = None
        local_router_dict_list = None
    
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
    
    # ===== Define data collator =====
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    # ===== Split dataset =====
    if use_improved:
        # Use mixed data distribution
        logger.info("Using improved mixed data distribution strategy.")
        local_datasets = distribute_mixed_data_to_clients(
            dataset=dataset,
            fed_args=fed_args,
            script_args=script_args,
            mix_ratio=1.0  # fully mixed
        )
    elif use_moe:
        # Use MoE routing to split dataset
        logger.info("Using MoE routing-based dataset splitting strategy.")
        local_datasets = split_dataset_with_moe(
            fed_args, script_args, dataset, 
            moe_args=moe_args, 
            moe_router=model.moe_router if hasattr(model, 'moe_router') else None,
            tokenizer=tokenizer
        )
    else:
        # Use standard splitting
        logger.info("Using standard dataset splitting strategy.")
        local_datasets = split_dataset(fed_args, script_args, dataset)
    
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    
    # ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]
    
    # Pretty title
    print("\n" + "="*80)
    print(f"{Fore.GREEN}[MoE + Value Chain Federated Learning]{Style.RESET_ALL} Start training - {fed_args.num_rounds} rounds, {fed_args.num_clients} experts")
    print("="*80 + "\n")
    
    # Reserve a few lines for progress bars
    print("\n"*(fed_args.num_clients + 2))
    
    # Main progress bar
    main_pbar = ColorfulProgressBar(fed_args.num_rounds, desc=f"Federated training progress", position=1, color=Fore.CYAN)
    
    # Client progress bars
    client_pbars = {}
    
    # Expose to global for trainer access
    import builtins
    builtins.client_pbars = client_pbars
    
    for round in range(fed_args.num_rounds):
        clients_this_round = get_clients_this_round(fed_args, round)
        
        # Update round info
        main_pbar.set_description(f"Round {round+1}/{fed_args.num_rounds} - Active experts: {clients_this_round}")
        
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)
                continue
            
            # Create or get client progress bar
            if client not in client_pbars:
                expert_name = moe_args.expert_names[client] if hasattr(moe_args, 'expert_names') and moe_args and moe_args.use_moe else f'Client {client}'
                client_pbars[client] = ColorfulProgressBar(
                    script_args.max_steps, 
                    desc=f"Expert {client} ({expert_name})", 
                    position=3+client,
                    color=Fore.YELLOW if client % 2 == 0 else Fore.MAGENTA
                )
            
            # Sync global model
            if hasattr(model, 'base_model'):
                set_peft_model_state_dict(model.base_model, global_dict)
            else:
                set_peft_model_state_dict(model, global_dict)
            
            # Sync router weights
            if global_router_dict is not None and hasattr(model, 'set_router_state_dict'):
                model.set_router_state_dict(global_router_dict)

            # Set current client id
            if hasattr(model, 'current_stage_id'):
                model.current_stage_id = client
            
            # Prepare dataset
            if use_moe:
                sub_dataset = get_dataset_this_round_with_moe(
                    local_datasets[client], round, fed_args, script_args,
                    tokenizer=tokenizer, client_id=client, 
                    moe_args=moe_args, moe_router=model.moe_router if hasattr(model, 'moe_router') else None
                )
            else:
                sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args, tokenizer=tokenizer)
            
            # Validate dataset
            if len(sub_dataset) == 0:
                logger.warning(f"Client {client} dataset is empty, skipping.")
                training_loss[client].append(-1)
                continue
            
            # Set LR
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
            training_args = get_training_args(script_args, new_lr)
            
            # Update progress bar description
            client_pbars[client].set_description(
                f"Expert {client} - size: {len(sub_dataset)} - LR: {new_lr:.6f}"
            )
            
            # Reset progress counter
            client_pbars[client].current = 0
            
            # Initialize trainer
            if use_improved:
                # Improved trainer
                class ImprovedValueChainTrainer(ValueChainTrainer):
                    def __init__(self, client_id, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.client_id = client_id
                    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                        # Inject client id
                        inputs['stage_id'] = self.client_id
                        outputs = model(**inputs)
                        # Log routing
                        if self.state.global_step % 100 == 0 and hasattr(model, 'last_routing_scores'):
                            if model.last_routing_scores is not None:
                                scores = model.last_routing_scores
                                current_score = scores[:, self.client_id].mean().item()
                                logger.info(f"Step {self.state.global_step}: client {self.client_id} routing score = {current_score:.3f}")
                        return (outputs.loss, outputs) if return_outputs else outputs.loss
                
                trainer = ImprovedValueChainTrainer(
                    client_id=client,
                    full_model=model,
                    model=model.base_model,
                    args=training_args,
                    train_dataset=sub_dataset,
                    tokenizer=tokenizer,
                    data_collator=custom_data_collator,
                )
            elif use_moe or (vc_args and vc_args.use_value_chain):
                # Standard value chain trainer
                trainer = ValueChainTrainer(
                    full_model=model,
                    model=model.base_model,
                    args=training_args,
                    train_dataset=sub_dataset,
                    tokenizer=tokenizer,
                    data_collator=custom_data_collator,
                )
            else:
                # Standard trainer
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
            
            # Train
            try:
                results = trainer.train()
                training_loss[client].append(results.training_loss)
                logger.info(f"Client {client} training finished, loss: {results.training_loss:.4f}")
                
                # Update progress bar
                client_pbars[client].current = script_args.max_steps
                client_pbars[client].set_postfix(loss=f"{results.training_loss:.4f}")
            except Exception as e:
                logger.error(f"Client {client} training failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                training_loss[client].append(-1)
                continue
            
            # Special algorithms
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
            
            # Local model state
            local_dict = get_consistent_model_dict(model, use_moe, vc_args)
            local_dict_list[client] = copy.deepcopy(local_dict)
            
            # Save router weights
            if use_improved and hasattr(model, 'get_router_state_dict'):
                local_router_dict_list[client] = model.get_router_state_dict()
        
        # Server aggregation
        if use_improved and global_router_dict is not None:
            # Improved aggregation (LoRA + router)
            logger.info("Using improved aggregation strategy.")
            global_dict, global_router_dict = aggregate_with_router(
                fed_args=fed_args,
                global_dict=global_dict,
                local_dict_list=local_dict_list,
                global_router_dict=global_router_dict,
                local_router_dict_list=local_router_dict_list,
                sample_num_list=sample_num_list,
                clients_this_round=clients_this_round,
                round_idx=round,
                router_aggregation_weight=0.5,
                momentum=0.9,
                router_lr=0.1
            )
            global_auxiliary = None  # aggregate_with_router does not handle auxiliary
        else:
            # Standard aggregation
            logger.info("Using standard aggregation strategy.")
            global_dict, global_auxiliary = global_aggregate(
                fed_args, global_dict, local_dict_list, sample_num_list,
                clients_this_round, round, proxy_dict=proxy_dict,
                opt_proxy_dict=opt_proxy_dict, 
                auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
                vc_args=vc_args if vc_args and vc_args.use_value_chain else None
            )
        
        # Update global model
        if hasattr(model, 'base_model'):
            set_peft_model_state_dict(model.base_model, global_dict)
        else:
            set_peft_model_state_dict(model, global_dict)
        
        # Update global router
        if global_router_dict is not None and hasattr(model, 'set_router_state_dict'):
            model.set_router_state_dict(global_router_dict)
        
        # Update main progress bar
        main_pbar.update(1)
        
        # Save checkpoint
        if (round+1) % fed_args.save_model_freq == 0:
            # Create save dir
            save_dir = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            if use_moe:
                # Use MoE-specific saver
                success = save_moe_value_chain_model(
                    model=model,
                    save_dir=save_dir,
                    tokenizer=tokenizer,
                    moe_args=moe_args,
                    vc_args=vc_args if vc_args and vc_args.use_value_chain else None
                )
            else:
                # Try value chain saver
                try:
                    from federated_learning.model_save_utils import save_model_with_lora_and_value_chain
                    success = save_model_with_lora_and_value_chain(
                        model=model,
                        save_dir=save_dir,
                        tokenizer=tokenizer,
                        vc_args=vc_args if vc_args and vc_args.use_value_chain else None
                    )
                except ImportError:
                    # Fallback simple save
                    logger.warning("model_save_utils not found; using basic save method.")
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    success = True
            
            # Save router weights with the correct format
            if use_improved and global_router_dict is not None:
                router_save_path = os.path.join(save_dir, "moe_router.pt")
                save_router_weights(global_router_dict, router_save_path, moe_args)
            
            if success:
                logger.info(f"Checkpoint for round {round+1} saved to {save_dir}")
            else:
                logger.error(f"Failed to save checkpoint for round {round+1}")
        
        # Save training loss
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    
    # Close progress bars
    main_pbar.close()
    for pbar in client_pbars.values():
        pbar.close()
    
    # Save final model
    final_dir = os.path.join(script_args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    if use_moe:
        # MoE-specific saver
        save_moe_value_chain_model(
            model=model,
            save_dir=final_dir,
            tokenizer=tokenizer,
            moe_args=moe_args,
            vc_args=vc_args if vc_args and vc_args.use_value_chain else None
        )
    else:
        # Try value chain saver
        try:
            from federated_learning.model_save_utils import save_model_with_lora_and_value_chain
            save_model_with_lora_and_value_chain(
                model=model,
                save_dir=final_dir,
                tokenizer=tokenizer,
                vc_args=vc_args if vc_args and vc_args.use_value_chain else None
            )
        except ImportError:
            # Simple save
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
    
    # Save final router weights in correct format
    if use_improved and global_router_dict is not None:
        router_save_path = os.path.join(final_dir, "moe_router.pt")
        save_router_weights(global_router_dict, router_save_path, moe_args)
        logger.info(f"Final global router weights saved to {router_save_path} (correct format).")
    
    print("\n" + "="*80)
    print(f"{Fore.GREEN}[Training Completed]{Style.RESET_ALL} Models saved to: {script_args.output_dir}")
    print("="*80)
    
    logger.info(f"Final model saved to {final_dir}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
