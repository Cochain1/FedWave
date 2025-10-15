import torch
import copy
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from .value_chain_model import save_value_chain_layer, load_value_chain_layer, get_peft_state_from_value_chain, set_peft_state_to_value_chain
import types
class ValueChainCallback(TrainerCallback):
    def __init__(self, stage_id, position_loss_weight, continuity_loss_weight, 
                 consistency_loss_weight, collaborative_coef):
        self.stage_id = stage_id
        self.position_loss_weight = position_loss_weight
        self.continuity_loss_weight = continuity_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.collaborative_coef = collaborative_coef
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, 'value_chain_layer'):
            if not hasattr(model, 'current_stage_id'):
                model.current_stage_id = self.stage_id


def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, 
                              formatting_prompts_func, data_collator, global_dict, local_auxiliary, 
                              global_auxiliary, vc_args=None, client_id=None):
    
    if hasattr(vc_args, 'use_value_chain') and vc_args.use_value_chain and client_id is not None:
        if hasattr(model, 'forward'):
            original_forward = model.forward
            
            def new_forward(self, *args, **kwargs):
                return original_forward(client_id, *args, **kwargs)
            
            import types
            model.forward = types.MethodType(new_forward, model)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
        
        trainer.add_callback(ValueChainCallback(
            stage_id=client_id,
            position_loss_weight=vc_args.position_loss_weight,
            continuity_loss_weight=vc_args.continuity_loss_weight,
            consistency_loss_weight=vc_args.consistency_loss_weight,
            collaborative_coef=vc_args.collaborative_coef
        ))
        
        original_compute_loss = trainer.compute_loss
        
        def value_chain_compute_loss(self, model, inputs, return_outputs=False):
            if return_outputs:
                loss, outputs = original_compute_loss(model, inputs, return_outputs=True)
            else:
                loss = original_compute_loss(model, inputs, return_outputs=False)
                outputs = None
            
            if hasattr(model, 'get_value_chain_loss'):
                hidden_states = None
                if outputs is not None and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                
                try:
                    position_loss, continuity_loss, consistency_loss = model.get_value_chain_loss(
                        client_id, hidden_states=hidden_states, **inputs
                    )
                    
                    vc_loss = (
                        self.callback_handler.callbacks[0].position_loss_weight * position_loss + 
                        self.callback_handler.callbacks[0].continuity_loss_weight * continuity_loss + 
                        self.callback_handler.callbacks[0].consistency_loss_weight * consistency_loss
                    )
                    
                    total_loss = loss + self.callback_handler.callbacks[0].collaborative_coef * vc_loss
                    
                    if self.state.global_step % 10 == 0:
                        print(f"Step {self.state.global_step} - Stage {client_id} - "
                              f"Original Loss: {loss.item():.4f}, VC Loss: {vc_loss.item():.4f}, "
                              f"Total Loss: {total_loss.item():.4f}")
                    
                    return (total_loss, outputs) if return_outputs else total_loss
                except Exception as e:
                    print(f"Error computing value chain loss: {e}")
            
            return (loss, outputs) if return_outputs else loss
        
        trainer.compute_loss = types.MethodType(value_chain_compute_loss, trainer)
        
    elif fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

class SFTTrainerValueChain(SFTTrainer):
    def __init__(self, model, global_state, stage_id, position_loss_weight=0.1, 
                 continuity_loss_weight=0.1, consistency_loss_weight=0.1,
                 collaborative_coef=0.5, **kwargs):
        self.global_state = global_state
        self.stage_id = stage_id
        self.position_loss_weight = position_loss_weight
        self.continuity_loss_weight = continuity_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.collaborative_coef = collaborative_coef
        if 'model' in kwargs:
            del kwargs['model']
        super().__init__(model=model, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super(SFTTrainerValueChain, self).compute_loss(model, inputs, return_outputs=return_outputs)
        
        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values
            outputs = None
        
        if hasattr(model, 'get_value_chain_loss'):
            hidden_states = None
            if outputs is not None and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            
            position_loss, continuity_loss, consistency_loss = model.get_value_chain_loss(
                self.stage_id, hidden_states=hidden_states, **inputs
            )
            
            vc_loss = (
                self.position_loss_weight * position_loss + 
                self.continuity_loss_weight * continuity_loss + 
                self.consistency_loss_weight * consistency_loss
            )
            
            total_loss = loss + self.collaborative_coef * vc_loss
            
            if self.state.global_step % 10 == 0:
                print(f"Step {self.state.global_step} - Stage {self.stage_id} - "
                      f"Original Loss: {loss.item():.4f}, VC Loss: {vc_loss.item():.4f}, "
                      f"Total Loss: {total_loss.item():.4f}")
            
            return (total_loss, outputs) if return_outputs else total_loss
        else:
            return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        super(SFTTrainerValueChain, self).save_model(output_dir, _internal_call)
        
        if hasattr(self.model, 'value_chain_layer'):
            vc_path = f"{output_dir}/value_chain_layer.pt"
            save_value_chain_layer(self.model, vc_path)
            
            with open(f"{output_dir}/value_chain_info.txt", 'w') as f:
                f.write(f"stage_id: {self.stage_id}\n")
                f.write(f"position_loss_weight: {self.position_loss_weight}\n")
                f.write(f"continuity_loss_weight: {self.continuity_loss_weight}\n")
                f.write(f"consistency_loss_weight: {self.consistency_loss_weight}\n")
                f.write(f"collaborative_coef: {self.collaborative_coef}\n")

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)