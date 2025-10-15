import os
import json
import torch
import logging
from typing import Dict, Any, Optional, Union, Tuple
import copy

logger = logging.getLogger(__name__)

def save_moe_value_chain_model(
    model, 
    save_dir, 
    tokenizer=None, 
    moe_args=None, 
    vc_args=None
):
    """
    Save a model that may include an MoE router and a Value-Chain layer.

    Args:
        model: The model to save.
        save_dir: Output directory.
        tokenizer: Tokenizer (optional).
        moe_args: MoE arguments (optional).
        vc_args: Value-chain arguments (optional).

    Returns:
        bool: Whether saving succeeded.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Detect model composition
        has_moe = hasattr(model, 'moe_router') and model.moe_router is not None
        has_vc = hasattr(model, 'value_chain_layer') and model.value_chain_layer is not None
        has_peft = hasattr(model, 'peft_config') or (hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'))
        
        logger.info(f"Saving model to {save_dir} ...")
        logger.info(f"Detected MoE router: {has_moe}")
        logger.info(f"Detected value-chain layer: {has_vc}")
        logger.info(f"Detected LoRA/PEFT: {has_peft}")
        
        # Step 1: Save MoE router (if present)
        if has_moe:
            moe_router_state = {
                'state_dict': model.moe_router.state_dict(),
                'hidden_size': getattr(model.moe_router, 'hidden_size', 768),
                'num_experts': getattr(model.moe_router, 'num_experts', 4),
                'top_k': getattr(model.moe_router, 'top_k', 2),
                'expert_names': getattr(model.moe_router, 'expert_names', None),
                'expert_descriptions': getattr(model.moe_router, 'expert_descriptions', None),
                'use_keywords': getattr(model.moe_router, 'use_keywords', True),
                'routing_stats': getattr(model.moe_router, 'routing_stats', {})
            }
            
            moe_path = os.path.join(save_dir, "moe_router.pt")
            torch.save(moe_router_state, moe_path)
            logger.info(f"MoE router saved to {moe_path}")
            
            # Save MoE config
            if moe_args is not None:
                from dataclasses import asdict
                moe_config_path = os.path.join(save_dir, "moe_config.json")
                with open(moe_config_path, 'w') as f:
                    json.dump(asdict(moe_args), f, indent=4)
                logger.info(f"MoE config saved to {moe_config_path}")
        
        # Step 2: Save value-chain layer (if present)
        if has_vc:
            value_chain_state = {
                'value_chain_layer': model.value_chain_layer.state_dict(),
                'output_layer': model.output_layer.state_dict() if hasattr(model, 'output_layer') else {}
            }
            
            vc_path = os.path.join(save_dir, "value_chain_layer.pt")
            torch.save(value_chain_state, vc_path)
            logger.info(f"Value-chain layer saved to {vc_path}")
            
            # Save value-chain config
            if vc_args is not None:
                from dataclasses import asdict
                vc_config_path = os.path.join(save_dir, "value_chain_config.json")
                with open(vc_config_path, 'w') as f:
                    json.dump(asdict(vc_args), f, indent=4)
                logger.info(f"Value-chain config saved to {vc_config_path}")
        
        # Step 3: Save base model and LoRA adapters
        base_model = model.base_model if (has_moe or has_vc) else model
        
        if has_peft:
            try:
                peft_model = None
                
                if hasattr(model, 'peft_config'):
                    peft_model = model
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
                    peft_model = model.base_model
                elif hasattr(base_model, 'peft_config'):
                    peft_model = base_model
                
                if peft_model is not None:
                    active_adapter = getattr(peft_model, 'active_adapter', None)
                    logger.info(f"Saving PEFT adapter: {active_adapter}")
                    
                    peft_model.save_pretrained(save_dir)
                    logger.info(f"LoRA adapter and config saved to {save_dir}")
                    
                    adapter_config_path = os.path.join(save_dir, "adapter_config.json")
                    if not os.path.exists(adapter_config_path):
                        logger.warning("adapter_config.json not found, saving it manually")
                        if hasattr(peft_model, 'peft_config'):
                            config_dict = {}
                            for k, v in peft_model.peft_config.items():
                                if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                                    config_dict[k] = v
                            with open(adapter_config_path, 'w') as f:
                                json.dump(config_dict, f, indent=4)
                            logger.info("Manually saved adapter_config.json")
                else:
                    logger.warning("Could not find a model instance with peft_config; LoRA config may not be saved correctly")
                    # Fallback: save entire model
                    if has_moe or has_vc:
                        base_model.save_pretrained(save_dir)
                    else:
                        model.save_pretrained(save_dir)
            except Exception as e:
                logger.error(f"Error while saving PEFT model: {e}")
                if hasattr(base_model, 'save_pretrained'):
                    base_model.save_pretrained(save_dir)
                    logger.info(f"Base model saved to {save_dir}")
        else:
            # No LoRA: save the full model
            if hasattr(base_model, 'save_pretrained'):
                base_model.save_pretrained(save_dir)
                logger.info(f"Full model saved to {save_dir}")
            else:
                torch.save(base_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                logger.info(f"Model state dict saved to {save_dir}/pytorch_model.bin")
        
        # Step 4: Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
            logger.info(f"Tokenizer saved to {save_dir}")
        
        # Save integration info
        integration_info = {
            "has_moe": has_moe,
            "has_vc": has_vc,
            "has_peft": has_peft,
            "model_type": "MoEValueChainModel" if (has_moe or has_vc) else "Standard"
        }
        with open(os.path.join(save_dir, "integration_info.json"), 'w') as f:
            json.dump(integration_info, f, indent=4)
        
        logger.info(f"Model successfully saved to {save_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error while saving model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_moe_value_chain_model(
    model_path, 
    base_model_name=None, 
    device_map="auto", 
    torch_dtype=None,
    use_moe=True,
    use_vc=True
):
    """
    Load a model that may include an MoE router and a Value-Chain layer.

    Args:
        model_path: Path to the saved model directory.
        base_model_name: Base model name/path (if not provided, will try reading from adapter_config.json).
        device_map: Device map for loading.
        torch_dtype: Torch dtype for the model.
        use_moe: Whether to load/enable MoE.
        use_vc: Whether to load/enable Value-Chain.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    # Integration info
    integration_path = os.path.join(model_path, "integration_info.json")
    has_moe = False
    has_vc = False
    has_peft = False
    
    if os.path.exists(integration_path):
        with open(integration_path, 'r') as f:
            info = json.load(f)
            has_moe = info.get("has_moe", False) and use_moe
            has_vc = info.get("has_vc", False) and use_vc
            has_peft = info.get("has_peft", False)
    
    # Infer features from files if needed
    if not has_moe:
        has_moe = os.path.exists(os.path.join(model_path, "moe_router.pt")) and use_moe
    if not has_vc:
        has_vc = os.path.exists(os.path.join(model_path, "value_chain_layer.pt")) and use_vc
    if not has_peft:
        has_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    # 1) Determine base model
    if has_peft and base_model_name is None:
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
                logger.info(f"Base model from adapter config: {base_model_name}")
    
    if base_model_name is None:
        logger.warning("Base model name not provided and not found in config; trying to load the model directly")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    # 2) Load base model and LoRA adapter
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    if has_peft:
        logger.info(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model
    
    # 3) Create MoE + Value-Chain integration if needed
    if has_moe or has_vc:
        try:
            # Load MoE config (if any)
            moe_args = None
            if has_moe:
                moe_config_path = os.path.join(model_path, "moe_config.json")
                if os.path.exists(moe_config_path):
                    from federated_learning.moe_config import MoEArguments
                    with open(moe_config_path, 'r') as f:
                        moe_dict = json.load(f)
                        moe_args = MoEArguments()
                        for k, v in moe_dict.items():
                            if hasattr(moe_args, k):
                                setattr(moe_args, k, v)
                else:
                    from federated_learning.moe_config import MoEArguments
                    logger.warning(f"moe_config.json not found in {model_path}. Initializing default MoEArguments.")
                    moe_args = MoEArguments()
                
                # Ensure hidden_size matches base model
                actual_hf_base_model = model
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    actual_hf_base_model = model.base_model.model
                elif hasattr(model, 'model'):
                    actual_hf_base_model = model.model

                if hasattr(actual_hf_base_model, 'config') and hasattr(actual_hf_base_model.config, 'hidden_size'):
                    base_model_hidden_size = actual_hf_base_model.config.hidden_size
                    if moe_args is None:
                        from federated_learning.moe_config import MoEArguments
                        moe_args = MoEArguments()
                        logger.info("Initialized MoEArguments because previous config was missing.")
                    if moe_args.moe_hidden_size != base_model_hidden_size:
                        logger.warning(
                            "MoE router hidden_size in config/default "
                            f"({moe_args.moe_hidden_size}) != base model hidden_size ({base_model_hidden_size}); "
                            f"overriding to {base_model_hidden_size}."
                        )
                        moe_args.moe_hidden_size = base_model_hidden_size
                    else:
                        logger.info(f"MoE router hidden_size matches base model ({base_model_hidden_size}).")
                else:
                    logger.error("Unable to determine hidden_size from base model config; MoE router may be misconfigured.")
            
            # Load VC config (if any)
            vc_args = None
            if has_vc:
                vc_config_path = os.path.join(model_path, "value_chain_config.json")
                if os.path.exists(vc_config_path):
                    from dataclasses import dataclass, field
                    @dataclass
                    class ValueChainArguments:
                        use_value_chain: bool = True
                        chain_type: str = "strict_chain"
                        cross_head_communication: bool = False
                        position_loss_weight: float = 0.1
                        continuity_loss_weight: float = 0.1
                        consistency_loss_weight: float = 0.1
                        collaborative_coef: float = 0.5
                        dynamic_weight_adjust: bool = True
                    with open(vc_config_path, 'r') as f:
                        vc_dict = json.load(f)
                        vc_args = ValueChainArguments()
                        for k, v in vc_dict.items():
                            if hasattr(vc_args, k):
                                setattr(vc_args, k, v)
            
            # Build integrated model
            from federated_learning.moe_valuechain_integration import create_moe_value_chain_model
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_model_name,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = create_moe_value_chain_model(
                base_model=model,
                moe_args=moe_args,
                vc_args=vc_args,
                tokenizer=tokenizer
            )
            
            # Load MoE router weights (if present)
            if has_moe and hasattr(model, 'moe_router') and model.moe_router is not None:
                moe_path = os.path.join(model_path, "moe_router.pt")
                if os.path.exists(moe_path):
                    logger.info(
                        "Found moe_router.pt; loading into current model.moe_router "
                        f"(hidden_size={getattr(model.moe_router, 'hidden_size', 'N/A')})"
                    )
                    moe_state_saved = torch.load(moe_path, map_location="cpu")
                    
                    saved_h = moe_state_saved.get('hidden_size')
                    current_h = getattr(model.moe_router, 'hidden_size')
                    if saved_h is not None and saved_h != current_h:
                        logger.warning(
                            f"Saved router hidden_size ({saved_h}) != current router hidden_size ({current_h}). "
                            "Skipping state_dict load to avoid dim mismatch; using freshly initialized router."
                        )
                    elif 'state_dict' in moe_state_saved:
                        logger.info("Loading state_dict into model.moe_router ...")
                        model.moe_router.load_state_dict(moe_state_saved['state_dict'])
                        if 'routing_stats' in moe_state_saved:
                            model.moe_router.routing_stats = moe_state_saved['routing_stats']
                        logger.info("MoE router weights loaded.")
                    else:
                        logger.warning("No 'state_dict' found in moe_router.pt; cannot load router weights.")
                else:
                    logger.warning(f"moe_router.pt not found at {moe_path}; using initialized router weights.")
            elif has_moe:
                logger.error("Expected to load MoE router (has_moe=True), but model.moe_router is None.")
            
            # Load value-chain layer weights (if present)
            if has_vc:
                vc_path = os.path.join(model_path, "value_chain_layer.pt")
                if os.path.exists(vc_path):
                    logger.info(f"Loading value-chain layer from {vc_path}")
                    device = next(model.parameters()).device
                    state_dict = torch.load(vc_path, map_location=device)
                    
                    if hasattr(model, 'value_chain_layer'):
                        model.value_chain_layer.load_state_dict(state_dict['value_chain_layer'])
                    if hasattr(model, 'output_layer') and 'output_layer' in state_dict:
                        model.output_layer.load_state_dict(state_dict['output_layer'])
                    
                    logger.info("Value-chain layer loaded.")
            
            target_device = device  # default to previously inferred device
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'parameters'):
                try:
                    target_device = next(model.base_model.parameters()).device
                except StopIteration:
                    logger.warning("Could not infer device from base_model; using previously inferred device.")
            model.to(target_device)
            logger.info(f"Moved integrated MoEValueChainModel to device: {target_device}")
            model.eval()
            return model, tokenizer
                
        except Exception as e:
            logger.error(f"Error while creating integrated model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to basic model loading.")
    
    # Fallback: return base or directly loaded model with tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer
