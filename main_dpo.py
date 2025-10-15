import sys
import os
import json
import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, PreTrainedModel
from peft import get_peft_model, PeftModel
from trl import DPOTrainer
from tqdm.auto import tqdm

# --- Import project modules ---
from config import get_config
from federated_learning.fed_global import get_clients_this_round
from federated_learning.moe_model_utils import load_moe_value_chain_model, save_moe_value_chain_model

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)

# --- Custom argument class (unchanged) ---
@dataclass
class DPOTrainingArguments(TrainingArguments):
    # --- Core DPO parameters ---
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the reference model."})
    loss_type: str = field(default="sigmoid", metadata={"help": "The loss type to use. Options: 'sigmoid', 'hinge', 'ipo', 'bco_pair', etc."})
    label_smoothing: float = field(default=0.0, metadata={"help": "Label smoothing for DPO loss."})

    # --- Model & PEFT-related parameters ---
    model_init_kwargs: Optional[dict] = field(default=None)
    ref_model_init_kwargs: Optional[dict] = field(default=None)
    model_adapter_name: Optional[str] = field(default=None)
    ref_adapter_name: Optional[str] = field(default=None)
    reference_free: bool = field(default=False, metadata={"help": "If True, use the policy model as both policy and reference."})
    disable_dropout: bool = field(default=True, metadata={"help": "Whether to disable dropout in the model."})
    sync_ref_model: bool = field(default=False, metadata={"help": "Whether to sync the reference model with the policy model during training."})

    # --- Dataset and length control ---
    max_prompt_length: Optional[int] = field(default=None)
    max_length: Optional[int] = field(default=None)
    max_completion_length: Optional[int] = field(default=None)
    padding_value: Optional[int] = field(default=None)
    truncation_mode: str = field(default="keep_end")
    label_pad_token_id: int = field(default=-100)
    dataset_num_proc: Optional[int] = field(default=None)

    # --- Training behavior control ---
    precompute_ref_log_probs: bool = field(default=False, metadata={"help": "Whether to precompute reference model log probabilities."})
    generate_during_eval: bool = field(default=False)
    use_logits_to_keep: Optional[bool] = field(default=None)
    padding_free: bool = field(default=False, metadata={"help": "Whether to use padding-free training."})
    use_liger_loss: bool = field(default=False, metadata={"help": "Whether to use the LIGER fused kernel for faster training."})
    use_weighting: bool = field(default=False, metadata={"help": "Whether to use weighting for the loss."})
    loss_weights: bool = field(default=False, metadata={"help": "Whether to use weights for the loss. (Added for compatibility with new TRL version)"})
    # --- f-divergence parameters ---
    f_divergence_type: Optional[str] = field(default=None)
    f_alpha_divergence_coef: Optional[float] = field(default=None)

    # --- Other custom parameters your code may use ---
    tools: Optional[Any] = field(default=None)
    rpo_alpha: Optional[float] = field(default=None, metadata={"help": "The alpha parameter for the RPO loss."})
    ld_alpha: Optional[float] = field(default=None, metadata={"help": "The alpha parameter for the LD loss."})

# ====================================================================================
# Core change: refactor CustomDPOTrainer.__init__
# ====================================================================================
class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model,
        ref_model,
        args,
        train_dataset,
        tokenizer,  # we accept tokenizer here
        eval_dataset=None,
        data_collator=None,
        # capture extra legacy kwargs to avoid passing to super()
        beta: Optional[float] = None,
        loss_type: Optional[str] = None,
        max_prompt_length: Optional[int] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        # Important fix: do NOT pass tokenizer to super().__init__
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )
        
        # After parent init, attach tokenizer to the instance
        # This ensures self.tokenizer exists, since parent didn't receive it
        self.tokenizer = tokenizer
        
        logger.info(f"CustomDPOTrainer: loss_type from args is {self.args.loss_type}")
        logger.info(f"CustomDPOTrainer: beta from args is {self.args.beta}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # This method remains the same, reading configs from self.args
        policy_outputs = self.concatenated_forward(model, inputs)
        policy_chosen_logps = policy_outputs["chosen_logps"]
        policy_rejected_logps = policy_outputs["rejected_logps"]

        with torch.no_grad():
            ref_outputs = self.concatenated_forward(self.ref_model or self.model, inputs)
            ref_chosen_logps = ref_outputs["chosen_logps"]
            ref_rejected_logps = ref_outputs["rejected_logps"]

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.args.beta * logits)
        elif self.args.loss_type == "hinge":
            losses = torch.relu(1 - self.args.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.args.beta)) ** 2
        else:
            raise ValueError(f"Unsupported loss_type: {self.args.loss_type}")
        
        loss = losses.mean()
        
        with torch.no_grad():
            chosen_rewards = self.args.beta * (policy_chosen_logps - ref_chosen_logps).detach()
            rejected_rewards = self.args.beta * (policy_rejected_logps - ref_rejected_logps).detach()
            metrics = {"rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().cpu().item()}
        
        if self.is_in_train:
            self.store_metrics(metrics, train_eval="train")
            
        if return_outputs:
            return loss, metrics
            
        return loss

'''
    def get_batch_samples(self, epoch_iterator, num_batches, device=None):
        """
        Override to be compatible with newer transformers.
        New transformers may pass a 'device' arg, while the TRL DPOTrainer
        in our env does not accept it. We swallow it and call the parent.
        """
        return super().get_batch_samples(epoch_iterator, num_batches)
'''

def main():
    script_args, fed_args, peft_config_from_file, _, _ = get_config()

    dpo_training_args = DPOTrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=getattr(script_args, 'batch_size', 1),
        gradient_accumulation_steps=getattr(script_args, 'gradient_accumulation_steps', 1),
        learning_rate=getattr(script_args, 'learning_rate', 5e-5),
        num_train_epochs=getattr(script_args, 'num_train_epochs', 1),
        max_steps=getattr(script_args, 'max_steps', -1),
        logging_steps=getattr(script_args, 'logging_steps', 10),
        save_steps=getattr(script_args, 'save_steps', 500),
        remove_unused_columns=False,
        report_to=[],
        bf16=getattr(script_args, 'bf16', False),
        fp16=getattr(script_args, 'fp16', False),
        gradient_checkpointing=False,
        seed=getattr(script_args, 'seed', 42),
        save_total_limit=getattr(script_args, 'save_total_limit', 1),
        beta=getattr(script_args, 'dpo_beta', 0.2),
        loss_type="sigmoid",
        max_length=getattr(script_args, 'seq_length', 2048),
        max_prompt_length=getattr(script_args, 'seq_length', 3072) - 256,
        label_pad_token_id=-100,
    )

    logger.info("="*50)
    logger.info("!!! DEBUGGING: Final Training Arguments !!!")
    logger.info(f"num_train_epochs used: {dpo_training_args.num_train_epochs}")
    logger.info(f"max_steps used: {dpo_training_args.max_steps}")
    logger.info("="*50)

    train_dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")
    local_datasets = {0: train_dataset}
    
    model_dtype = torch.float16 if dpo_training_args.fp16 else (torch.bfloat16 if dpo_training_args.bf16 else torch.float32)

    model_for_dpo_training, tokenizer = load_moe_value_chain_model(
        model_path=script_args.model_name_or_path,
        base_model_name=script_args.base_model_name,
        device_map=str(dpo_training_args.device), 
        torch_dtype=model_dtype,
        use_moe=True, 
        use_vc=True  
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dpo_training_args.padding_value = tokenizer.pad_token_id

    if script_args.use_peft and peft_config_from_file:
        model_for_dpo_training.base_model.add_adapter("dpo_adapter", peft_config_from_file)
        model_for_dpo_training.base_model.set_adapter("dpo_adapter")
    
    model_for_dpo_training.train()

    for round_num in tqdm(range(fed_args.num_rounds), desc="Federated Rounds"):
        clients_this_round = get_clients_this_round(fed_args, round_num)
        for client_idx in clients_this_round:
            sub_dataset = local_datasets[client_idx]
            
            dpo_trainer = CustomDPOTrainer(
                model=model_for_dpo_training,
                ref_model=None,
                args=dpo_training_args,
                train_dataset=sub_dataset,
                tokenizer=tokenizer,
            )

            logger.info(f"Client {client_idx}: Starting DPO training...")
            train_results = dpo_trainer.train()
            logger.info(f"Client {client_idx}: DPO training finished. Loss: {getattr(train_results, 'training_loss', 'N/A')}")
            
    final_save_path = os.path.join(dpo_training_args.output_dir, "final_dpo_model")
    logger.info(f"All DPO training completed. Saving final model to: {final_save_path}")
    dpo_trainer.save_model(final_save_path)


if __name__ == "__main__":
    main()
