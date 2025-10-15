from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, TaskType
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta

# ===== Define and parse arguments =====
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=10, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})

@dataclass
class ValueChainArguments:
    use_value_chain: Optional[bool] = field(default=True, metadata={"help": "Whether to use value chain mechanism"})
    chain_type: Optional[str] = field(default="strict_chain", metadata={"help": "Chain type: strict_chain or relaxed_chain"})
    cross_head_communication: Optional[bool] = field(default=False, metadata={"help": "Whether to allow cross-head communication"})
    position_loss_weight: Optional[float] = field(default=0.001, metadata={"help": "Weight for position loss"})
    continuity_loss_weight: Optional[float] = field(default=0.001, metadata={"help": "Weight for continuity loss"})
    consistency_loss_weight: Optional[float] = field(default=0.001, metadata={"help": "Weight for consistency loss"})
    collaborative_coef: Optional[float] = field(default=0.5, metadata={"help": "Coefficient for collaborative loss"})
    dynamic_weight_adjust: Optional[bool] = field(default=True, metadata={"help": "Whether to dynamically adjust weights"})

@dataclass
class MoEArguments:
    """
    MoE (Mixture of Experts) configuration parameters
    """
    use_moe: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use MoE (Mixture of Experts) for expert selection"}
    )
    moe_top_k: Optional[int] = field(
        default=2,
        metadata={"help": "Number of top experts selected during MoE routing"}
    )
    moe_num_experts: Optional[int] = field(
        default=4,
        metadata={"help": "Total number of MoE experts, usually equal to the number of clients in federated learning"}
    )
    moe_hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Hidden size of the MoE router"}
    )
    moe_router_type: Optional[str] = field(
        default="keyword",
        metadata={"help": "Type of MoE router: 'keyword' or 'model'"}
    )
    expert_names: Optional[List[str]] = field(
        default_factory=lambda: [
            "Automotive Design Expert",
            "Automotive Manufacturing Expert",
            "Automotive Supply Chain Expert",
            "Automotive Quality Control Expert"
        ],
        metadata={"help": "List of expert names"}
    )
    expert_descriptions: Optional[List[str]] = field(
        default_factory=lambda: [
            "Responsible for vehicle exterior/interior design, focusing on aesthetics and user experience",
            "Responsible for automotive manufacturing processes, emphasizing efficiency and high-quality assembly",
            "Responsible for component supply chain management, ensuring steady material flow",
            "Responsible for quality inspection and fault analysis, ensuring safety and reliability"
        ],
        metadata={"help": "List of expert descriptions"}
    )
    moe_router_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained checkpoint path for the MoE router (if available)"}
    )
    default_expert_indices: Optional[List[int]] = field(
        default_factory=lambda: [0, 1],
        metadata={"help": "Default expert indices when router is unavailable"}
    )
    log_routing_stats: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to log MoE routing statistics"}
    )
    routing_balance_strategy: Optional[str] = field(
        default="count",
        metadata={"help": "Expert selection balancing strategy: 'none', 'count', or 'weight'"}
    )

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "Base model name or path if using PEFT adapters"})
    dataset_name: Optional[str] = field(default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 precision"})
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 precision"})
    custom_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to your custom JSON dataset"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8-bit precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4-bit precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Whether to use PEFT for adapter training"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable trust_remote_code"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "Number of updates steps before saving checkpoints"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints"})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "Local data directory for downloaded data"})

# Update parser to include MoE parameters
parser = HfArgumentParser((ScriptArguments, FedArguments, ValueChainArguments, MoEArguments))
script_args, fed_args, vc_args, moe_args = parser.parse_args_into_dataclasses()

# Ensure MoE and value chain parameters are compatible
if moe_args.use_moe and vc_args.use_value_chain:
    moe_args.moe_num_experts = fed_args.num_clients
    while len(moe_args.expert_names) < moe_args.moe_num_experts:
        index = len(moe_args.expert_names)
        moe_args.expert_names.append(f"Expert {index}")
        moe_args.expert_descriptions.append(f"Specialization {index}")
    moe_args.moe_top_k = min(moe_args.moe_top_k, moe_args.moe_num_experts)

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )
else:
    peft_config = None

def get_config():
    return script_args, fed_args, peft_config, vc_args, moe_args

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, quantization_config, torch_dtype

def save_config(script_args, fed_args, vc_args=None, moe_args=None):
    now_time = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_name_split = os.path.basename(script_args.dataset_name) if script_args.dataset_name else "custom_dataset"
    output_dir = (
        f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_"
        f"c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_"
        f"b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_"
        f"l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    )

    if moe_args and moe_args.use_moe:
        output_dir = f"{output_dir}_moe{moe_args.moe_top_k}"
    if vc_args and vc_args.use_value_chain:
        output_dir = f"{output_dir}_vc{vc_args.chain_type}"

    while True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{now_time}"

    script_args.output_dir = output_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        if vc_args is not None:
            combined_dict["vc_args"] = asdict(vc_args)
        if moe_args is not None:
            combined_dict["moe_args"] = asdict(moe_args)
        json.dump(combined_dict, f, indent=4)
