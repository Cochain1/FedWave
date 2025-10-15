import os
import json
import torch
from dataclasses import asdict
from peft import get_peft_model_state_dict, PeftModel, PeftConfig

def save_model_with_lora_and_value_chain(model, save_dir, tokenizer=None, vc_args=None):
    """
    Comprehensive save function that handles LoRA and Value Chain cases.

    Args:
        model: The model to save
        save_dir: Directory to save into
        tokenizer: Tokenizer (optional)
        vc_args: Value chain arguments (optional)

    Returns:
        bool: Whether saving succeeded
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        has_vc = hasattr(model, 'value_chain_layer')
        has_peft = hasattr(model, 'peft_config') or (hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'))
        
        print(f"Saving model to {save_dir} ...")
        print(f"Detected value chain layer: {has_vc}")
        print(f"Detected LoRA/PEFT: {has_peft}")
        
        # Step 1: Save value chain layer (if present)
        if has_vc:
            # Get base_model and value chain state from ValueChainModel
            base_model = model.base_model
            
            # Save value chain layers
            value_chain_state = {
                'value_chain_layer': model.value_chain_layer.state_dict(),
                'output_layer': model.output_layer.state_dict()
            }
            vc_path = os.path.join(save_dir, "value_chain_layer.pt")
            torch.save(value_chain_state, vc_path)
            print(f"Value chain layers saved to {vc_path}")
            
            # Save value chain config
            if vc_args is not None:
                vc_config_path = os.path.join(save_dir, "value_chain_config.json")
                with open(vc_config_path, 'w') as f:
                    json.dump(asdict(vc_args), f, indent=4)
                print(f"Value chain config saved to {vc_config_path}")
        else:
            base_model = model
        
        # Step 2: Save LoRA adapter and config
        if has_peft:
            # Find the actual model instance that contains peft_config
            peft_model = None
            if hasattr(model, 'peft_config'):
                peft_model = model
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
                peft_model = model.base_model
            elif hasattr(base_model, 'peft_config'):
                peft_model = base_model
            
            if peft_model is not None:
                # Use PEFT's save_pretrained to save adapters and config
                active_adapter = getattr(peft_model, 'active_adapter', None)
                if active_adapter:
                    print(f"Saving PEFT adapter: {active_adapter}")
                    
                # Save adapter
                peft_model.save_pretrained(save_dir)
                print(f"LoRA adapter and config saved to {save_dir}")
                
                # Ensure adapter_config.json exists; if not, save it manually
                adapter_config_path = os.path.join(save_dir, "adapter_config.json")
                if not os.path.exists(adapter_config_path):
                    print(f"Warning: adapter_config.json not found, saving manually")
                    if hasattr(peft_model, 'peft_config'):
                        # Manually save adapter_config.json
                        config_dict = {}
                        for k, v in peft_model.peft_config.items():
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                                config_dict[k] = v
                        
                        with open(adapter_config_path, 'w') as f:
                            json.dump(config_dict, f, indent=4)
                        print(f"adapter_config.json saved manually")
            else:
                print("Warning: Could not find a model instance containing peft_config; LoRA config may not be saved correctly")
                # Fallback: try saving the whole model
                if has_vc:
                    base_model.save_pretrained(save_dir)
                else:
                    model.save_pretrained(save_dir)
        else:
            # No LoRA; save full model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(save_dir)
                print(f"Full model saved to {save_dir}")
            else:
                # Fallback: save state_dict
                torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                print(f"Model state saved to {save_dir}/pytorch_model.bin")
        
        # Step 3: Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
            print(f"Tokenizer saved to {save_dir}")
        
        print(f"Model saved successfully to {save_dir}")
        return True
    
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def load_model_with_lora_and_value_chain(model_path, base_model_name=None, device_map="auto", torch_dtype=None):
    """
    Load a model that may include LoRA configuration and value chain layers.

    Args:
        model_path: Path where the model is saved
        base_model_name: Base model name (if not provided, read from adapter_config.json)
        device_map: Device mapping
        torch_dtype: Torch dtype for loading

    Returns:
        The loaded model, or (model, tokenizer) if tokenizer files are present
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    # 1) Check for LoRA config
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    value_chain_path = os.path.join(model_path, "value_chain_layer.pt")
    
    # 2) Determine base model
    if os.path.exists(adapter_config_path):
        # Read base model name from adapter_config.json
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
            if base_model_name is None:
                base_model_name = config.get("base_model_name_or_path")
        
        print(f"Using base model: {base_model_name}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load full model directly
        print(f"No LoRA config found; loading full model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    
    # 3) Load value chain layers (if present)
    if os.path.exists(value_chain_path):
        print(f"Loading value chain layers from: {value_chain_path}")
        from federated_learning.value_chain_model import add_value_chain_to_model, load_value_chain_layer
        
        # Read value chain config
        vc_config_path = os.path.join(model_path, "value_chain_config.json")
        if os.path.exists(vc_config_path):
            with open(vc_config_path, 'r') as f:
                vc_config = json.load(f)
                num_stages = vc_config.get('num_stages', 4)
                chain_type = vc_config.get('chain_type', "strict_chain")
                cross_head_communication = vc_config.get('cross_head_communication', False)
        else:
            # Defaults
            num_stages = 4
            chain_type = "strict_chain"
            cross_head_communication = False
        
        # Add value chain layers
        model = add_value_chain_to_model(
            model, 
            num_stages=num_stages,
            chain_type=chain_type, 
            cross_head_communication=cross_head_communication
        )
        
        # Load value chain state
        state_dict = torch.load(value_chain_path, map_location="cpu")
        model.value_chain_layer.load_state_dict(state_dict['value_chain_layer'])
        model.output_layer.load_state_dict(state_dict['output_layer'])
        print("Value chain layers loaded successfully")
    
    # 4) Load tokenizer if available
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_path) or os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    return model
