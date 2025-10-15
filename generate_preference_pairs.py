import argparse
import json
import torch
import logging
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import random

try:
    from OpenFedLLM_moe_now.federated_learning.moe_model_utils import load_moe_value_chain_model
except ImportError:
    try:
        from federated_learning.moe_model_utils import load_moe_value_chain_model
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure federated_learning.moe_model_utils.load_moe_value_chain_model can be imported correctly.")
        print("You may need to adjust PYTHONPATH or the location of this script.")
        exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_hidden_states_for_router(model, tokenizer, prompt: str, device):
    """
    Get hidden states from the base model for the MoE router given an input prompt.
    Optimized for decoder-only models like Llama-2.
    """
    logger.debug(f"Preparing hidden states for prompt: '{prompt[:50]}...'")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    actual_base_hf_model = None
    if hasattr(model, 'base_model'):
        peft_model_wrapper = model.base_model
        if hasattr(peft_model_wrapper, 'base_model') and hasattr(peft_model_wrapper.base_model, 'model'):
            actual_base_hf_model = peft_model_wrapper.base_model.model
        elif hasattr(peft_model_wrapper, 'model'):
            actual_base_hf_model = peft_model_wrapper.model
        else:
            actual_base_hf_model = peft_model_wrapper
    else:
        actual_base_hf_model = model

    if actual_base_hf_model is None:
        raise ValueError("Could not find the core Hugging Face model (e.g., LlamaForCausalLM) within the provided model structure.")

    logger.debug(f"Core HF model used to extract hidden_states: {type(actual_base_hf_model)}")

    with torch.no_grad():
        outputs = actual_base_hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Failed to extract hidden_states. Neither 'outputs.hidden_states' nor 'outputs.last_hidden_state' found in outputs.")
            
    logger.debug(f"Extracted hidden_states shape (for router): {hidden_states.shape}")
    return hidden_states


def get_expert_routing_info(model, tokenizer, prompt: str, device):
    if not hasattr(model, 'moe_router') or model.moe_router is None:
        raise ValueError("Model does not have attribute 'moe_router' or it is None.")

    try:
        hidden_states_for_router = get_hidden_states_for_router(model, tokenizer, prompt, device)
    except Exception as e:
        logger.error(f"Failed to get router hidden_states for prompt '{prompt[:50]}...': {e}", exc_info=True)
        return None, None

    with torch.no_grad():
        routing_weights, _ = model.moe_router(hidden_states_for_router)
    
    scores_softmax = routing_weights.squeeze(0)
    sorted_scores, sorted_expert_indices = torch.sort(scores_softmax, descending=True)
    
    return sorted_scores.cpu().tolist(), sorted_expert_indices.cpu().tolist()


def generate_response_for_expert(model, tokenizer, prompt: str, expert_id: int, generation_config: dict, device):
    logger.debug(f"Generating response with expert {expert_id} ...")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    if not hasattr(model, 'current_stage_id'):
        raise AttributeError("Model does not have attribute 'current_stage_id'. Cannot force expert selection.")
    
    original_stage_id = model.current_stage_id
    model.current_stage_id = expert_id
    logger.debug(f"Set model.current_stage_id to {expert_id} for generation.")

    target_model_for_generation = model
    if hasattr(model, 'base_model'):
        target_model_for_generation = model.base_model

    try:
        with torch.no_grad():
            outputs = target_model_for_generation.generate(**inputs, **generation_config)
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Error during generation with expert {expert_id}: {e}", exc_info=True)
        response = f"Failed to generate response with expert {expert_id}"
    finally:
        model.current_stage_id = original_stage_id
        logger.debug(f"Restored model.current_stage_id to {original_stage_id}.")
        
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate preference pairs from an MoE model for DPO training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SFT MoE training model checkpoint directory.")
    parser.add_argument("--base_model_name", type=str, help="Base model name or path, if needed when loading the MoE model (e.g., with LoRA adapters).")
    parser.add_argument("--tokenizer_name_or_path", type=str, help="Path to the tokenizer. If None, use model_path.")
    parser.add_argument("--input_prompts_file", type=str, required=True, help="JSON file containing prompts (a JSON array of objects).")
    parser.add_argument("--output_preference_file", type=str, required=True, help="Output JSONL file to save generated preference pairs.")
    parser.add_argument("--num_experts", type=int, help="Number of experts in the model. If not provided, attempt to infer.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--top_k_sampling", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (e.g., 'cuda:0', 'cpu').")

    args = parser.parse_args()

    logger.info(f"Loading tokenizer from {args.tokenizer_name_or_path or args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path or args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    logger.info(f"Loading MoE model from {args.model_path} with base model {args.base_model_name}")
    model, _ = load_moe_value_chain_model(
        model_path=args.model_path,
        base_model_name=args.base_model_name, 
        device_map=args.device, 
        use_moe=True, 
        use_vc=True 
    )
    model.eval()
    logger.info("Model loaded successfully.")

    num_experts = args.num_experts
    if num_experts is None:
        if hasattr(model, 'moe_router') and model.moe_router is not None and hasattr(model.moe_router, 'num_experts'):
            num_experts = model.moe_router.num_experts
            logger.info(f"Inferred number of experts from moe_router: {num_experts}")
        elif hasattr(model, 'num_stages'):
            num_experts = model.num_stages
            logger.info(f"Inferred number of experts from model.num_stages: {num_experts}")
        else:
            logger.error("Could not infer number of experts. Please provide via --num_experts.")
            return

    if num_experts <= 1:
        logger.error(f"Number of experts is {num_experts}. Cannot generate distinct preference pairs. Exiting.")
        return

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k_sampling,
        "num_beams": args.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True if args.temperature > 0 and args.temperature < 1.0 else False,
    }
    if args.num_beams > 1:
        generation_config["do_sample"] = False
        
    logger.info(f"Generation config: {generation_config}")

    preference_data_count = 0
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_preference_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read JSON array input
    try:
        with open(args.input_prompts_file, 'r', encoding='utf-8') as f_in:
            prompts_data_list = json.load(f_in)
        if not isinstance(prompts_data_list, list):
            logger.error(f"Input file {args.input_prompts_file} is not a JSON array. Please check file format.")
            return
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {args.input_prompts_file}: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading input file {args.input_prompts_file}: {e}")
        return

    with open(args.output_preference_file, 'w', encoding='utf-8') as f_out:
        for data_item in tqdm(prompts_data_list, desc="Generating preference pairs"):
            try:
                # Extract prompt from your data structure (expects "instruction" and optional "input")
                instruction = data_item.get("instruction", "")
                input_text = data_item.get("input", "")
                
                if not instruction:
                    logger.warning(f"Skipped an item missing 'instruction': {data_item}")
                    continue
                
                # Build the final prompt
                prompt_parts = [instruction]
                if input_text and input_text.strip():
                    prompt_parts.append(input_text)
                prompt = "\n".join(prompt_parts).strip()

            except Exception as e:
                logger.warning(f"Skipped invalid item or error extracting prompt: {data_item}, error: {e}")
                continue

            logger.info(f"\nProcessing prompt: {prompt[:100]}...")

            expert_scores, sorted_expert_indices = get_expert_routing_info(model, tokenizer, prompt, args.device)

            if expert_scores is None or not sorted_expert_indices:
                logger.warning(f"Could not get routing info for prompt: {prompt[:100]}... Skipping.")
                continue
            
            logger.info(f"  Router scores (softmaxed): {['%.3f' % s for s in expert_scores[:num_experts]]}")
            logger.info(f"  Sorted expert indices: {sorted_expert_indices[:num_experts]}")

            chosen_expert_id = sorted_expert_indices[0]
            
            rejected_expert_id = None
            if len(sorted_expert_indices) > 1:
                rejected_expert_id = sorted_expert_indices[1]
            else:
                logger.warning(f"  Only one expert selected ({chosen_expert_id}). Cannot form a distinct rejected pair.")
                if num_experts > 1:
                    all_possible_experts = list(range(num_experts))
                    potential_rejected = [idx for idx in all_possible_experts if idx != chosen_expert_id]
                    if potential_rejected:
                        rejected_expert_id = random.choice(potential_rejected)
                        logger.info(f"  Since routing returned a single expert, randomly chose a rejected expert: {rejected_expert_id}")
                    else:
                        logger.warning(f"  Could not choose a different rejected expert for chosen_expert_id={chosen_expert_id}.")
                        continue
                else:
                    logger.warning(f"  Model has only one expert; cannot choose a rejected expert.")
                    continue

            logger.info(f"  Generating y_w (chosen) with expert {chosen_expert_id} ...")
            response_chosen = generate_response_for_expert(model, tokenizer, prompt, chosen_expert_id, generation_config, args.device)
            logger.info(f"  y_w (expert {chosen_expert_id}): {response_chosen[:100]}...")

            response_rejected = None
            if rejected_expert_id is not None and rejected_expert_id != chosen_expert_id:
                logger.info(f"  Generating y_l (rejected) with expert {rejected_expert_id} ...")
                response_rejected = generate_response_for_expert(model, tokenizer, prompt, rejected_expert_id, generation_config, args.device)
                logger.info(f"  y_l (expert {rejected_expert_id}): {response_rejected[:100]}...")
            elif rejected_expert_id is None and num_experts > 1:
                logger.warning(f"  Could not determine a rejected expert (chosen={chosen_expert_id}, only one routing result and random selection failed).")
            elif rejected_expert_id is not None and rejected_expert_id == chosen_expert_id:
                logger.warning(f"  Chosen and rejected experts are the same ({chosen_expert_id}). This should not happen if routing returns >1 result or random selection works.")

            if response_chosen and response_rejected and \
               response_chosen.strip() and response_rejected.strip() and \
               response_chosen.strip() != response_rejected.strip():
                preference_entry = {
                    "prompt": prompt,  # Save the actual combined prompt used
                    "original_instruction": instruction,
                    "original_input": input_text,
                    "chosen": response_chosen,
                    "rejected": response_rejected,
                    "chosen_expert_id": int(chosen_expert_id),
                    "rejected_expert_id": int(rejected_expert_id) if rejected_expert_id is not None else None,
                    "router_scores": dict(zip(map(str, sorted_expert_indices), map(lambda x: float('%.4f' % x), expert_scores)))
                }
                f_out.write(json.dumps(preference_entry, ensure_ascii=False) + "\n")
                preference_data_count += 1
            else:
                logger.warning(f"  Skipping DPO pair for prompt '{prompt[:50]}...' due to missing/identical responses.")
                if not response_chosen or not response_chosen.strip():
                    logger.warning("    y_w (chosen) is empty or invalid.")
                if not response_rejected or not response_rejected.strip():
                    logger.warning("    y_l (rejected) is empty or invalid.")
                if response_chosen and response_rejected and response_chosen.strip() == response_rejected.strip():
                    logger.warning("    y_w and y_l are identical.")

    logger.info(f"Successfully generated {preference_data_count} preference pairs.")
    logger.info(f"Output saved to: {args.output_preference_file}")

if __name__ == "__main__":
    main()
