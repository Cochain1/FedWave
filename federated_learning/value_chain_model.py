import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from peft import PeftModel
import copy
import logging  # add logging
import types
import functools

logger = logging.getLogger(__name__)  # add logger for this module

class ValueChainLayer(nn.Module):
    """Value-chain-aware layer: adds stage-specific multi-head self-attention mechanisms."""
    def __init__(self, hidden_size, num_stages=4, num_heads=4, dropout=0.1, 
                 chain_type="strict_chain", cross_head_communication=False):
        super(ValueChainLayer, self).__init__()

        print("DEBUG: Initializing ValueChainLayer. Checking for 'vc_q_proj' and 'q_proj'.")
        self.hidden_size = hidden_size
        self.num_stages = num_stages
        self.num_heads = num_heads
        
        if hidden_size % num_stages != 0:
            logger.warning(
                f"ValueChainLayer: hidden_size ({hidden_size}) is not perfectly divisible by num_stages ({num_stages}). "
                f"stage_dim will be floor({hidden_size}/{num_stages}) = {hidden_size // num_stages}. "
                f"This might lead to unused dimensions if not handled carefully in concatenation."
            )
        self.stage_dim = hidden_size // num_stages  # feature dim processed by each stage

        if self.stage_dim == 0:
            raise ValueError(
                f"ValueChainLayer: stage_dim calculated to 0. "
                f"hidden_size ({hidden_size}) must be >= num_stages ({num_stages})."
            )
        
        if self.stage_dim % num_heads != 0:
            logger.warning(
                f"ValueChainLayer: stage_dim ({self.stage_dim}) is not perfectly divisible by num_heads ({num_heads}). "
                f"head_dim will be floor({self.stage_dim}/{num_heads}) = {self.stage_dim // num_heads}."
            )
        self.head_dim = self.stage_dim // num_heads  # feature dim per head
        if self.head_dim == 0:
            raise ValueError(
                f"ValueChainLayer: head_dim calculated to 0. "
                f"stage_dim ({self.stage_dim}) must be >= num_heads ({num_heads})."
            )

        self.chain_type = chain_type
        self.cross_head_communication = cross_head_communication
        
        # --- Change: rename q_proj, k_proj, v_proj here ---
        self.vc_q_proj = nn.ModuleList([nn.Linear(hidden_size, self.stage_dim) for _ in range(num_stages)])
        self.vc_k_proj = nn.ModuleList([nn.Linear(hidden_size, self.stage_dim) for _ in range(num_stages)])
        self.vc_v_proj = nn.ModuleList([nn.Linear(hidden_size, self.stage_dim) for _ in range(num_stages)])
        # --- End change ---
        if hasattr(self, 'vc_q_proj'):
            print("DEBUG: ValueChainLayer has 'vc_q_proj'.")
        if hasattr(self, 'q_proj'):
            print("DEBUG: ValueChainLayer STILL HAS 'q_proj'. THIS IS LIKELY THE ISSUE.")
        # Keep output projection unchanged
        self.o_proj = nn.Linear(hidden_size, hidden_size)  # input dim equals concatenated stage_dim = hidden_size
        
        if cross_head_communication:
            self.cross_comm = nn.Parameter(torch.ones(num_stages, num_stages) / num_stages)
        
        self.dropout = nn.Dropout(dropout)
        self.stage_embeddings = nn.Parameter(torch.randn(num_stages, hidden_size))

    def forward(self, hidden_states, stage_id, attention_mask=None):
        # --- Debug: simplest "pass-through" logic ---
        # Only add a learnable stage-specific bias, then pass through the final projection.
        # This verifies that the whole computation graph from input to output is sound.

        # Add a stage-specific offset (a learnable parameter)
        output_with_bias = hidden_states + self.stage_embeddings[stage_id]

        # Pass through the final output projection layer
        final_output = self.o_proj(output_with_bias)

        return final_output

    '''
    def forward(self, hidden_states, stage_id, attention_mask=None):
        batch_size, seq_len, current_hidden_size = hidden_states.size()

        if current_hidden_size != self.hidden_size:
            # Input hidden_states do not match expected hidden_size.
            # Could happen if this layer receives partial outputs from a custom previous layer
            # or misconfigured hidden_size. For now, assume it's correct.
            pass

        stage_mask = torch.zeros(self.num_stages, device=hidden_states.device)
        if self.chain_type == "strict_chain":
            stage_mask[stage_id] = 1.0
            if stage_id > 0:
                stage_mask[stage_id-1] = 1.0  # also consider the immediately preceding stage
        elif self.chain_type == "relaxed_chain":
            for i in range(stage_id + 1):
                stage_mask[i] = 1.0
        else:  # default: only current stage
            stage_mask[stage_id] = 1.0
        
        hidden_states_with_emb = hidden_states + self.stage_embeddings[stage_id].unsqueeze(0).unsqueeze(0)
        
        outputs_for_stages = []
        total_concat_dim = 0
        for i in range(self.num_stages):
            if stage_mask[i] > 0:
                # --- Use renamed attributes ---
                q = self.vc_q_proj[i](hidden_states_with_emb)
                k = self.vc_k_proj[i](hidden_states_with_emb)
                v = self.vc_v_proj[i](hidden_states_with_emb)
                # --- end ---
                
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
                k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
                v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                
                if attention_mask is not None:
                    # Ensure broadcastable mask
                    scores = scores + attention_mask 
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                context = torch.matmul(attn_weights, v)
                context = context.permute(0, 2, 1, 3).contiguous()
                context = context.view(batch_size, seq_len, -1)
                
                outputs_for_stages.append(context)
                total_concat_dim += self.stage_dim
            else:
                # For inactive stages, append zeros of correct stage_dim
                outputs_for_stages.append(torch.zeros(batch_size, seq_len, self.stage_dim, device=hidden_states.device))
                total_concat_dim += self.stage_dim
        
        if self.cross_head_communication:
            mixed_outputs = []
            for i in range(self.num_stages):
                weighted_sum = sum(self.cross_comm[i, j] * outputs_for_stages[j] for j in range(self.num_stages))
                mixed_outputs.append(weighted_sum)
            outputs_for_stages = mixed_outputs
        
        # Concatenate stage outputs; expect num_stages * stage_dim == hidden_size
        if total_concat_dim != self.hidden_size:
            # Might happen if hidden_size not divisible by num_stages
            pass

        combined_output = torch.cat(outputs_for_stages, dim=-1)

        # Ensure o_proj input matches combined_output last dim
        if self.o_proj.in_features != combined_output.shape[-1]:
            # Handle mismatch by padding or truncation
            if combined_output.shape[-1] < self.o_proj.in_features:
                padding_size = self.o_proj.in_features - combined_output.shape[-1]
                padding = torch.zeros(batch_size, seq_len, padding_size, device=hidden_states.device)
                combined_output = torch.cat([combined_output, padding], dim=-1)
            elif combined_output.shape[-1] > self.o_proj.in_features:
                combined_output = combined_output[:, :, :self.o_proj.in_features]
        
        output = self.o_proj(combined_output)
        return output
    '''

    def get_position_loss(self, stage_id):
        position_embeddings = self.stage_embeddings
        position_loss = 0
        for i in range(self.num_stages - 1):
            similarity = F.cosine_similarity(position_embeddings[i:i+1], position_embeddings[i+1:i+2], dim=1)
            target_similarity = 1.0 - (1.0 / self.num_stages) * (i + 1)  # example target
            position_loss += F.mse_loss(similarity, torch.tensor([target_similarity], device=similarity.device))
        return position_loss
    
    def get_continuity_loss(self):
        continuity_loss = 0
        for i in range(self.num_stages):
            for j in range(i+1, self.num_stages):
                # Using weights of the renamed attributes
                q_similarity = F.cosine_similarity(
                    self.vc_q_proj[i].weight.view(1, -1), 
                    self.vc_q_proj[j].weight.view(1, -1), dim=1)
                target_similarity = 0.5 if j == i+1 else 0.2  # example target
                continuity_loss += F.mse_loss(q_similarity, torch.tensor([target_similarity], device=q_similarity.device))
        return continuity_loss
    
    def get_consistency_loss(self, hidden_states, stage_id):
        if stage_id == 0:
            return torch.tensor(0.0, device=hidden_states.device)
        
        current_output = self.forward(hidden_states, stage_id)  # uses the simplified forward above
        previous_output = self.forward(hidden_states, stage_id - 1)
        
        similarity = F.cosine_similarity(current_output.mean(dim=1), previous_output.mean(dim=1))  # mean for stability
        target_similarity = torch.tensor([0.6], device=similarity.device)  # example target
        consistency_loss = F.mse_loss(similarity, target_similarity)
        return consistency_loss


# --- The rest of value_chain_model.py (ValueChainModel class, add_value_chain_to_model, etc.) ---
# --- remains unchanged in logic; comments/strings translated to English. ---

'''
class ValueChainModel(nn.Module):
    """
    Wrapper that adds a value-chain-aware layer on top of the base model.
    """
    def __init__(self, base_model, num_stages=4, chain_type="strict_chain", 
                 cross_head_communication=False, hidden_size=None):
        super(ValueChainModel, self).__init__()
        self.base_model = base_model
        
        if hasattr(base_model, 'config'):
            self.config = base_model.config
        
        if hasattr(base_model, 'peft_config'):
            self.peft_config = base_model.peft_config
        
        peft_attrs = [
            'active_adapter', 'base_model_torch_dtype', 'save_pretrained', 
            'modules_to_save', 'current_device'
        ]
        for peft_attr in peft_attrs:
            if hasattr(base_model, peft_attr):
                setattr(self, peft_attr, getattr(base_model, peft_attr))
        
        if hasattr(base_model, 'modules_to_save'):
            self.modules_to_save = base_model.modules_to_save
        
        if hidden_size is None:
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            else:  # Try to get from base_model.base_model if it's a PeftModel
                try:
                    hidden_size = base_model.base_model.config.hidden_size
                except AttributeError:
                    raise ValueError("Hidden size must be specified or available in base_model.config or base_model.base_model.config")

        self.num_stages = num_stages
        self.chain_type = chain_type
        self.current_stage_id = num_stages - 1 
        
        # Instantiate ValueChainLayer
        self.value_chain_layer = ValueChainLayer(
            hidden_size=hidden_size,
            num_stages=num_stages,
            chain_type=chain_type,
            cross_head_communication=cross_head_communication
        )
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self._vc_call_count = 0
        self._original_methods = {}
        self._patch_base_model()
    
    def _patch_base_model(self):
        # Helper to unwrap from possible multiple wrappers to find the actual HF model
        def _get_actual_hf_model(model_wrapper):
            current_model = model_wrapper
            # Prefer PEFT's get_base_model
            if hasattr(current_model, "get_base_model"):
                return current_model.get_base_model()
            # Fallback: manually unwrap
            while hasattr(current_model, 'model'):
                current_model = current_model.model
            return current_model

        target_model_to_patch = _get_actual_hf_model(self.base_model)
        
        # Ensure we have something like LlamaForCausalLM
        if not hasattr(target_model_to_patch, 'lm_head'):
            logger.error(f"Could not find 'lm_head' in {type(target_model_to_patch)}. Patch failed.")
            return

        if hasattr(target_model_to_patch, 'forward'):
            original_forward_func = target_model_to_patch.forward
            
            # Avoid double-patching
            if hasattr(original_forward_func, '__wrapped__'):
                logger.info("Model 'forward' already patched, skipping.")
                return

            value_chain_model_self_ref = self  # capture outer instance

            @functools.wraps(original_forward_func)
            def patched_forward(hf_model_self_arg, *patch_args, **patch_kwargs):
                # Step 1: call original forward
                outputs = original_forward_func(*patch_args, **patch_kwargs)
                
                # Step 2: get hidden states
                hidden_states = outputs.last_hidden_state
                
                # Step 3: apply custom layer
                vc_self = value_chain_model_self_ref
                if vc_self.value_chain_layer is None:
                    return outputs

                modified_hidden_states = vc_self.value_chain_layer(hidden_states, vc_self.current_stage_id)
                modified_hidden_states = vc_self.output_layer(modified_hidden_states)
                
                # Step 4: recompute logits using new hidden states
                if hasattr(hf_model_self_arg, 'lm_head'):
                    new_logits = hf_model_self_arg.lm_head(modified_hidden_states)
                    outputs.logits = new_logits
                else:
                    logger.warning("No 'lm_head' found in patched_forward; cannot recompute logits.")
                
                outputs.last_hidden_state = modified_hidden_states
                return outputs

            # Apply patch
            target_model_to_patch.forward = types.MethodType(patched_forward, target_model_to_patch)
            logger.info(f"Successfully patched 'forward' for {type(target_model_to_patch)}.")
        else:
            logger.error(f"Target model {type(target_model_to_patch)} has no 'forward'; cannot patch.")

    def _unpatch_base_model(self):
        target_model_to_unpatch = self.base_model
        if hasattr(self.base_model, 'model') and isinstance(self.base_model.model, PreTrainedModel):
            target_model_to_unpatch = self.base_model.model

        if 'forward' in self._original_methods and hasattr(target_model_to_unpatch, 'forward'):
            target_model_to_unpatch.forward = self._original_methods['forward']
    
    def forward(self, stage_id, **kwargs):
        self.current_stage_id = stage_id
        kwargs['output_hidden_states'] = True  # ensure base model returns hidden states
        self._vc_call_count = 0
        return self.base_model(**kwargs) 
    
    def generate(self, *args, **kwargs):
        # base_model's forward is already patched, so generate uses it.
        return self.base_model.generate(*args, **kwargs)
    
    def get_value_chain_loss(self, stage_id, hidden_states=None, **kwargs):
        if self.value_chain_layer is None:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        if hidden_states is None:
            outputs = self.forward(stage_id, **kwargs)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            else:
                raise ValueError("Cannot extract hidden states from model outputs for VC loss.")
        
        position_loss = self.value_chain_layer.get_position_loss(stage_id)
        continuity_loss = self.value_chain_layer.get_continuity_loss()
        consistency_loss = self.value_chain_layer.get_consistency_loss(hidden_states, stage_id)
        
        return position_loss, continuity_loss, consistency_loss
    
    def enable_input_require_grads(self):
        if hasattr(self.base_model, 'enable_input_require_grads'):
            return self.base_model.enable_input_require_grads()
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'enable_input_require_grads'):
            return self.base_model.model.enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs is not None:
                return self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            return self.base_model.gradient_checkpointing_enable()
        # ... similarly for self.base_model.model ...
            
    def print_trainable_parameters(self):
        if hasattr(self.base_model, 'print_trainable_parameters'):
            self.base_model.print_trainable_parameters()
        else:
            total_params = 0
            trainable_params = 0
            for _, param in self.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            logger.info(f"MoEValueChainModel: trainable params: {trainable_params} || all params: {total_params} || trainable%: {100 * trainable_params / total_params if total_params > 0 else 0}")

    def get_vc_call_count(self):
        return self._vc_call_count
    
    def __del__(self):
        self._unpatch_base_model()
'''

def add_value_chain_to_model(model, num_stages=4, chain_type="strict_chain", 
                             cross_head_communication=False):
    # Determine hidden_size from the provided model (could be PeftModel or base HF model)
    config_to_check = model.config if hasattr(model, 'config') else None
    if config_to_check is None and hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):  # PeftModel case
        config_to_check = model.base_model.config
    
    if config_to_check and hasattr(config_to_check, 'hidden_size'):
        hidden_size = config_to_check.hidden_size
    else:
        raise ValueError("Cannot determine hidden_size from the provided model for ValueChainModel.")

    value_chain_model_instance = ValueChainModel(
        base_model=model,  # model here is likely PeftModel(Llama) or Llama
        num_stages=num_stages,
        chain_type=chain_type,
        cross_head_communication=cross_head_communication,
        hidden_size=hidden_size  # pass determined hidden_size
    )
    return value_chain_model_instance

'''
# Functions like save_value_chain_layer, load_value_chain_layer, 
# get_peft_state_from_value_chain, set_peft_state_to_value_chain,
# detach_value_chain, attach_value_chain remain largely the same in logic,
# but comments/logs translated to English.
'''

def save_value_chain_layer(model, output_path):
    if hasattr(model, 'value_chain_layer') and model.value_chain_layer is not None:
        state_dict = {
            'value_chain_layer': model.value_chain_layer.state_dict(),
            'output_layer': model.output_layer.state_dict() if hasattr(model, 'output_layer') else {}
        }
        torch.save(state_dict, output_path)
    else:
        logger.warning("Model does not have 'value_chain_layer' or it is None. Cannot save VC layer.")

def load_value_chain_layer(model, input_path):
    if hasattr(model, 'value_chain_layer') and model.value_chain_layer is not None:
        if not os.path.exists(input_path):
            logger.warning(f"Value chain layer state file not found at {input_path}. Skipping load.")
            return
        state_dict = torch.load(input_path, map_location=lambda storage, loc: storage)  # load to CPU first
        model.value_chain_layer.load_state_dict(state_dict['value_chain_layer'])
        if hasattr(model, 'output_layer') and model.output_layer is not None and 'output_layer' in state_dict:
            model.output_layer.load_state_dict(state_dict['output_layer'])
        logger.info(f"Value chain layer state loaded from {input_path}")
    else:
        logger.warning("Model does not have 'value_chain_layer' or it is None. Cannot load VC layer.")

def get_peft_state_from_value_chain(model):
    """Get PEFT state dict from the value-chain model."""
    if hasattr(model, 'base_model'):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(model.base_model)
    else:
        raise ValueError("ValueChainModel cannot get PEFT state")

def set_peft_state_to_value_chain(model, state_dict):
    """Set PEFT state dict into the value-chain model's base model."""
    if hasattr(model, 'base_model'):
        from peft import set_peft_model_state_dict
        return set_peft_model_state_dict(model.base_model, state_dict)
    else:
        raise ValueError("ValueChainModel cannot set PEFT state")

def detach_value_chain(model):
    """Detach value-chain layer; return base model and VC state."""
    if not hasattr(model, 'value_chain_layer'):
        return model, None
    
    # Save VC state
    vc_state = {
        'value_chain_layer': model.value_chain_layer.state_dict(),
        'output_layer': model.output_layer.state_dict(),
        'num_stages': model.num_stages,
        'chain_type': getattr(model, 'chain_type', "strict_chain"),
        'cross_head_communication': getattr(model.value_chain_layer, 'cross_head_communication', False)
    }
    
    # Return base model and state
    return model.base_model, vc_state

def attach_value_chain(base_model, vc_state):
    """Re-attach value-chain layer to a base model."""
    if vc_state is None:
        return base_model
    
    # Create a new VC model
    model = add_value_chain_to_model(
        base_model, 
        num_stages=vc_state['num_stages'],
        chain_type=vc_state['chain_type'],
        cross_head_communication=vc_state['cross_head_communication']
    )
    
    # Load saved state
    model.value_chain_layer.load_state_dict(vc_state['value_chain_layer'])
    model.output_layer.load_state_dict(vc_state['output_layer'])
    
    return model
