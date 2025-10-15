import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

class ImprovedMoEValueChainModel(nn.Module):
    def __init__(
        self,
        base_model,
        value_chain_layer=None,
        moe_router=None,
        num_stages=4,
        top_k=2,
        chain_type="strict_chain",
        cross_head_communication=False,
        router_loss_weight=0.1,
        load_balancing_loss_weight=0.01,
    ):
        super().__init__()
        self.base_model = base_model
        self.value_chain_layer = value_chain_layer
        self.moe_router = moe_router
        self.num_stages = num_stages
        self.top_k = top_k
        self.current_stage_id = 0
        self.router_loss_weight = router_loss_weight
        self.load_balancing_loss_weight = load_balancing_loss_weight
        
        self.config = base_model.config if hasattr(base_model, 'config') else None
        
        self.last_routing_scores = None
        self.last_expert_indices = None
        
        if hasattr(base_model, 'peft_config'):
            self.peft_config = base_model.peft_config
            
        methods_to_proxy = [
            'print_trainable_parameters',
            'generate', 
            'enable_input_require_grads',
            'gradient_checkpointing_enable',
        ]
        for method in methods_to_proxy:
            if hasattr(base_model, method):
                setattr(self, method, getattr(base_model, method))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage_id: Optional[int] = None,
        return_router_logits: bool = False,
        **kwargs
    ) -> CausalLMOutputWithPast:
        if stage_id is not None:
            self.current_stage_id = stage_id
        
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = base_outputs.hidden_states[-1]
        
        router_logits = None
        routing_scores = None
        if self.moe_router is not None:

            pooled_hidden = hidden_states.mean(dim=1)

            routing_weights, expert_indices = self.moe_router(hidden_states)
            routing_scores = routing_weights

            self.last_routing_scores = routing_scores
            self.last_expert_indices = expert_indices
            
            router_logits = routing_scores
        
        if self.value_chain_layer is not None:
            hidden_states = self.value_chain_layer(hidden_states, self.current_stage_id)
        
        actual_hf_model = self._get_actual_hf_model()
        if hasattr(actual_hf_model, 'lm_head'):
            logits = actual_hf_model.lm_head(hidden_states)
        else:
            logits = base_outputs.logits
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            batch_size = shift_logits.size(0)
            seq_len = shift_logits.size(1)
            token_losses = token_losses.view(batch_size, seq_len)
            
            sample_losses = token_losses.mean(dim=1)  # [batch_size]
            
            if routing_scores is not None and self.current_stage_id < routing_scores.size(1):
                current_expert_scores = routing_scores[:, self.current_stage_id]  # [batch_size]
                
                weighted_losses = sample_losses * current_expert_scores
                
                main_loss = weighted_losses.mean()
                
                if self.load_balancing_loss_weight > 0:
                    expert_usage = routing_scores.mean(dim=0)  # [num_experts]
                    load_balance_loss = expert_usage.var() * self.load_balancing_loss_weight
                    main_loss = main_loss + load_balance_loss
                
                if self.router_loss_weight > 0:
                    routing_probs = F.softmax(routing_scores, dim=-1)
                    entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1)
                    router_loss = entropy.mean() * self.router_loss_weight
                    main_loss = main_loss + router_loss
                
                loss = main_loss
            else:
                loss = sample_losses.mean()
            
            if self.value_chain_layer is not None:
                vc_losses = self.get_value_chain_loss(
                    self.current_stage_id,
                    hidden_states,
                    **kwargs
                )
                vc_total_loss = sum(vc_losses) * 0.1
                loss = loss + vc_total_loss
        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values if hasattr(base_outputs, 'past_key_values') else None,
            hidden_states=base_outputs.hidden_states if hasattr(base_outputs, 'hidden_states') else None,
            attentions=base_outputs.attentions if hasattr(base_outputs, 'attentions') else None,
        )
        
        if return_router_logits and router_logits is not None:
            output.router_logits = router_logits
        
        return output
    
    def _get_actual_hf_model(self):
        model = self.base_model
        while hasattr(model, 'model'):
            model = model.model
        if hasattr(model, 'get_base_model'):
            try:
                model = model.get_base_model()
            except:
                pass
        return model
    
    def get_value_chain_loss(self, stage_id, hidden_states=None, **kwargs):
        device = next(self.parameters()).device
        
        if self.value_chain_layer is None:
            return (torch.tensor(0.0, device=device),) * 3
        
        position_loss = torch.tensor(0.0, device=device)
        continuity_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)
        
        return position_loss, continuity_loss, consistency_loss
    
    def get_router_state_dict(self):
        if self.moe_router is not None:
            return self.moe_router.state_dict()
        return {}
    
    def set_router_state_dict(self, state_dict):
        if self.moe_router is not None:
            self.moe_router.load_state_dict(state_dict)
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)