import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import os
import json
import copy
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import types
import functools

logger = logging.getLogger(__name__)

def _get_actual_hf_model_from_wrapper(model_wrapper):
    """ Helper: unwrap (possibly multi-layer) wrappers to obtain the actual Hugging Face PreTrainedModel. """
    current_model = model_wrapper
    # Prefer PEFT's get_base_model if available
    if hasattr(current_model, "get_base_model"):
        try:
            base_internal = current_model.get_base_model()
            # get_base_model might still return a PeftModel (if stacked adapters)
            # or directly an HF model
            if isinstance(base_internal, PreTrainedModel):
                return base_internal
            # If get_base_model returns something that has .model and that is a PreTrainedModel
            elif hasattr(base_internal, 'model') and isinstance(base_internal.model, PreTrainedModel):
                return base_internal.model
            else:  # Otherwise, keep unwrapping
                current_model = base_internal
        except Exception as e_get_base:
            logger.warning(f"Error calling get_base_model: {e_get_base}. Trying alternative unwrapping methods.")
            pass  # Continue with fallback unwrapping

    # Generic unwrapping loop (fallback)
    # PeftModel.model -> underlying model (could be another PeftModel or the HF model)
    # PeftModel.base_model.model -> HF model if PeftModel(HFModel)
    # PeftModel.base_model.base_model.model -> HF model if PeftModel(PeftModel(HFModel))
    
    # Try unwrapping via .model chain
    temp_model = model_wrapper
    visited_models = set()
    while hasattr(temp_model, 'model') and id(temp_model.model) not in visited_models:
        visited_models.add(id(temp_model.model))
        if temp_model.model is temp_model:  # prevent infinite loops
            break
        temp_model = temp_model.model
        if isinstance(temp_model, PreTrainedModel):
            return temp_model
            
    # If not found via .model, try .base_model.model (common with layered PEFT)
    if hasattr(model_wrapper, 'base_model'):
        base_of_wrapper = model_wrapper.base_model
        visited_models_base = set()
        while hasattr(base_of_wrapper, 'model') and id(base_of_wrapper.model) not in visited_models_base:
            visited_models_base.add(id(base_of_wrapper.model))
            if base_of_wrapper.model is base_of_wrapper:
                break
            base_of_wrapper = base_of_wrapper.model
            if isinstance(base_of_wrapper, PreTrainedModel):
                return base_of_wrapper
        # If base_of_wrapper itself is a PreTrainedModel
        if isinstance(base_of_wrapper, PreTrainedModel):
            return base_of_wrapper

    # If the original input is already a PreTrainedModel
    if isinstance(model_wrapper, PreTrainedModel):
        return model_wrapper

    logger.warning(f"_get_actual_hf_model_from_wrapper: Failed to unwrap {type(model_wrapper)} to a PreTrainedModel. Returning original wrapper.")
    return model_wrapper

class MoEValueChainModel(nn.Module):
    def __init__(self, base_model, value_chain_layer=None, moe_router=None, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.value_chain_layer = value_chain_layer
        self.moe_router = moe_router

        # Proxy essential attributes from the base model
        self.config = base_model.config
        self.current_stage_id = 0
        attrs_to_proxy = ['peft_config']
        methods_to_proxy = [
            'print_trainable_parameters',
            'generate',
            'enable_input_require_grads',   # added missing method
            'gradient_checkpointing_enable' # future-proofing
        ]
        for attr in attrs_to_proxy:
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

        for method in methods_to_proxy:
            if hasattr(base_model, method):
                setattr(self, method, getattr(base_model, method))
    
        for method_name in methods_to_proxy:
            if hasattr(base_model, method_name):
                setattr(self, method_name, getattr(base_model, method_name))
        # Proxy PEFT model methods so Trainer can call them
        if hasattr(base_model, 'print_trainable_parameters'):
            self.print_trainable_parameters = base_model.print_trainable_parameters
        if hasattr(base_model, 'generate'):
            self.generate = base_model.generate

    def forward(self, input_ids, attention_mask=None, labels=None, stage_id=0, **kwargs):
        # 1) Call the base model to get its raw outputs (we let it compute its loss; we'll override it)
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,  # ensure hidden_states are returned
        )

        # 2) Extract the last hidden state
        hidden_states = base_model_output.hidden_states[-1]

        # 3) Apply our custom value-chain layer (currently simplified version)
        modified_hidden_states = self.value_chain_layer(hidden_states, stage_id)

        # 4) Recompute logits using the base model's lm_head
        #    First unwrap to the actual HF model (e.g., LlamaForCausalLM)
        underlying_hf_model_with_head = self.base_model.get_base_model()

        final_logits = underlying_hf_model_with_head.lm_head(modified_hidden_states)

        # 5) Compute the final loss from the new logits
        loss = None
        if labels is not None:
            # Standard cross-entropy
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            # In the future, value-chain losses could be added here

        # 6) Return an output object matching HF's CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=base_model_output.past_key_values,
            hidden_states=base_model_output.hidden_states,  # pass through original or modified as needed
        )

    def __getattr__(self, name: str):
        """
        Automatically forward unknown attributes to the underlying base_model.
        This technique helps avoid many 'has no attribute' issues in wrappers.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

def create_moe_value_chain_model(
    base_model,
    moe_args=None,
    vc_args=None,
    tokenizer=None
):
    use_moe = moe_args is not None and hasattr(moe_args, 'use_moe') and moe_args.use_moe
    use_vc = vc_args is not None and hasattr(vc_args, 'use_value_chain') and vc_args.use_value_chain

    if not use_moe and not use_vc:
        logger.info("Both MoE and VC are disabled. Returning the base model directly.")
        return base_model

    num_stages_for_init = 4
    top_k_for_init = 2
    chain_type_for_init = "strict_chain"
    cross_head_communication_for_init = False
    model_hidden_size = 768

    _actual_hf_model_for_config = _get_actual_hf_model_from_wrapper(base_model)
    if hasattr(_actual_hf_model_for_config, 'config') and hasattr(_actual_hf_model_for_config.config, 'hidden_size'):
        model_hidden_size = _actual_hf_model_for_config.config.hidden_size
    else:
        logger.warning("Could not determine hidden_size from the base model config; will try moe_args or vc_args.")
        if use_moe and moe_args and hasattr(moe_args, 'moe_hidden_size'):
            model_hidden_size = moe_args.moe_hidden_size
        elif use_vc and vc_args and hasattr(vc_args, 'hidden_size'):
            model_hidden_size = vc_args.hidden_size
        else:
            logger.warning(f"Falling back to default hidden_size {model_hidden_size}.")

    moe_router_instance = None
    if use_moe:
        from federated_learning.moe_router import create_automotive_moe_router
        if not all(hasattr(moe_args, attr) for attr in ['moe_num_experts', 'moe_top_k']):
            raise ValueError("moe_args is missing 'moe_num_experts' or 'moe_top_k'.")
        
        current_moe_hidden_size = model_hidden_size
        if hasattr(moe_args, 'moe_hidden_size') and moe_args.moe_hidden_size != current_moe_hidden_size:
            logger.warning(
                f"moe_args.moe_hidden_size ({moe_args.moe_hidden_size}) != model hidden_size ({current_moe_hidden_size}). "
                f"Using the model hidden_size ({current_moe_hidden_size})."
            )
        
        moe_router_instance = create_automotive_moe_router(
            hidden_size=current_moe_hidden_size,
            num_experts=moe_args.moe_num_experts,
            top_k=moe_args.moe_top_k
        )
        num_stages_for_init = moe_args.moe_num_experts
        top_k_for_init = moe_args.moe_top_k
        logger.info(f"MoE router created: experts={num_stages_for_init}, top-k={top_k_for_init}, hidden_size={current_moe_hidden_size}")

    value_chain_layer_instance = None
    if use_vc:
        from federated_learning.value_chain_model import ValueChainLayer
        vc_hidden_size_to_use = model_hidden_size

        num_stages_for_vc = getattr(vc_args, 'num_stages', num_stages_for_init)

        value_chain_layer_instance = ValueChainLayer(
            hidden_size=vc_hidden_size_to_use,
            num_stages=num_stages_for_vc,
            chain_type=vc_args.chain_type,
            cross_head_communication=vc_args.cross_head_communication
        )
        chain_type_for_init = vc_args.chain_type
        cross_head_communication_for_init = vc_args.cross_head_communication
        if num_stages_for_init != num_stages_for_vc and use_moe:
            logger.warning(f"MoE expert count ({num_stages_for_init}) != VC stage count ({num_stages_for_vc}).")
        elif not use_moe:
            num_stages_for_init = num_stages_for_vc

        logger.info(f"Value-chain layer created: type={chain_type_for_init}, stages={num_stages_for_vc}, hidden_size={vc_hidden_size_to_use}")

    model_instance = MoEValueChainModel(
        base_model=base_model,
        value_chain_layer=value_chain_layer_instance,
        moe_router=moe_router_instance,
        num_stages=num_stages_for_init,
        top_k=top_k_for_init,
        chain_type=chain_type_for_init,
        cross_head_communication=cross_head_communication_for_init
    )
    if tokenizer is not None:
        model_instance.tokenizer = tokenizer

    return model_instance
