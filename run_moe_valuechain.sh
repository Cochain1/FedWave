#!/bin/bash

# Run the training script integrating MoE and Value Chain
# Usage: ./run_moe_valuechain.sh [train|test|help]

# Default parameters
#MODEL_PATH="./LLM/Qwen2-7b-instruct"
MODEL_PATH="./LLM/Llama3-8b-Instruct"
DATASET_PATH="./OpenFedLLM-moe_now/data/train/business_workflow.json"
OUTPUT_DIR="./evaluation_results/moe_vc_test"
NUM_CLIENTS=4
SAMPLE_CLIENTS=4
#MAX_STEPS=50
MAX_STEPS=10
NUM_ROUNDS=20
BATCH_SIZE=4
SEQ_LENGTH=2048
LORA_R=32
LORA_ALPHA=64
TOP_K=2
GPU_ID=1
COMMAND="train"

# Help function
show_help() {
    echo "Usage: ./run_moe_valuechain.sh [train|test|help] [options]"
    echo ""
    echo "Commands:"
    echo "  train    Run training with integrated MoE and Value Chain (default)"
    echo "  test     Evaluate the MoE and Value Chain model"
    echo "  help     Show this help message"
    echo ""
    echo "Options:"
    echo "  --model          Base model path (default: ${MODEL_PATH})"
    echo "  --dataset        Dataset path (default: ${DATASET_PATH})"
    echo "  --output         Output directory (default: ${OUTPUT_DIR})"
    echo "  --num_clients    Number of clients (default: ${NUM_CLIENTS})"
    echo "  --sample_clients Number of sampled clients per round (default: ${SAMPLE_CLIENTS})"
    echo "  --max_steps      Max training steps per client (default: ${MAX_STEPS})"
    echo "  --num_rounds     Number of federated learning rounds (default: ${NUM_ROUNDS})"
    echo "  --batch_size     Batch size (default: ${BATCH_SIZE})"
    echo "  --seq_length     Sequence length (default: ${SEQ_LENGTH})"
    echo "  --lora_r         Rank of LoRA adapter (default: ${LORA_R})"
    echo "  --lora_alpha     Alpha of LoRA adapter (default: ${LORA_ALPHA})"
    echo "  --top_k          Number of experts selected by MoE routing (default: ${TOP_K})"
    echo "  --gpu            GPU ID (default: ${GPU_ID})"
    echo "  --moe_only       Use only MoE (disable Value Chain)"
    echo "  --vc_only        Use only Value Chain (disable MoE)"
    echo "  --neither        Disable both MoE and Value Chain"
    echo ""
    echo "Examples:"
    echo "  ./run_moe_valuechain.sh train --model ./models/model --dataset ./data/data.json --output ./output/test1 --num_rounds 5"
    echo "  ./run_moe_valuechain.sh test --model ./output/test1/final_model --dataset ./data/test.json --output ./output/test1/eval"
}

# Parse command-line arguments
if [ $# -ge 1 ]; then
    case $1 in
        train|test|help)
            COMMAND=$1
            shift
            ;;
    esac
fi

# Show help and exit
if [ "$COMMAND" = "help" ]; then
    show_help
    exit 0
fi

# Mode flags
USE_MOE=true
USE_VC=false

# Parse remaining arguments
while [ "$1" != "" ]; do
    case $1 in
        --model)
            shift
            MODEL_PATH=$1
            ;;
        --dataset)
            shift
            DATASET_PATH=$1
            ;;
        --output)
            shift
            OUTPUT_DIR=$1
            ;;
        --num_clients)
            shift
            NUM_CLIENTS=$1
            ;;
        --sample_clients)
            shift
            SAMPLE_CLIENTS=$1
            ;;
        --max_steps)
            shift
            MAX_STEPS=$1
            ;;
        --num_rounds)
            shift
            NUM_ROUNDS=$1
            ;;
        --batch_size)
            shift
            BATCH_SIZE=$1
            ;;
        --seq_length)
            shift
            SEQ_LENGTH=$1
            ;;
        --lora_r)
            shift
            LORA_R=$1
            ;;
        --lora_alpha)
            shift
            LORA_ALPHA=$1
            ;;
        --top_k)
            shift
            TOP_K=$1
            ;;
        --gpu)
            shift
            GPU_ID=$1
            ;;
        --moe_only)
            USE_MOE=true
            USE_VC=false
            ;;
        --vc_only)
            USE_MOE=false
            USE_VC=true
            ;;
        --neither)
            USE_MOE=false
            USE_VC=false
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Create output directory
mkdir -p ${OUTPUT_DIR}
# algorithms=("fedavg" "fedprox" "scaffold" "fedavgm" "fedadagrad" "fedyogi" "fedadam")

# Generate run command
if [ "$COMMAND" = "train" ]; then
    # Training command
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python main_sft_moe.py \
        --model_name_or_path \"${MODEL_PATH}\" \
        --custom_dataset_path \"${DATASET_PATH}\" \
        --dataset_sample 20000 \
        --fed_alg \"fedavg\" \
        --num_clients ${NUM_CLIENTS} \
        --sample_clients ${SAMPLE_CLIENTS} \
        --max_steps ${MAX_STEPS} \
        --num_rounds ${NUM_ROUNDS} \
        --batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --seq_length ${SEQ_LENGTH} \
        --peft_lora_r ${LORA_R} \
        --peft_lora_alpha ${LORA_ALPHA} \
        --load_in_8bit \
        --use_peft \
        --output_dir \"${OUTPUT_DIR}\" \
        --template \"alpaca\" "
    
    # Add MoE and Value Chain parameters
    if [ "$USE_MOE" = true ]; then
        CMD="${CMD} --use_moe \
        --moe_top_k ${TOP_K} \
        --moe_router_type \"keyword\" \
        --log_routing_stats"
    fi
    
    if [ "$USE_VC" = true ]; then
        CMD="${CMD} --use_value_chain \
        --chain_type \"relaxed_chain\" \
        --cross_head_communication True \
        --position_loss_weight 0.1 \
        --continuity_loss_weight 0.1 \
        --consistency_loss_weight 0.2 \
        --collaborative_coef 0.5 \
        --dynamic_weight_adjust True"
    fi
    
elif [ "$COMMAND" = "test" ]; then
    # Evaluation command
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate_moe_model.py \
        --model_path \"${MODEL_PATH}\" \
        --test_file \"${DATASET_PATH}\" \
        --output_dir \"${OUTPUT_DIR}\" \
        --max_samples 10"
    
    # Add MoE and Value Chain parameters
    if [ "$USE_MOE" = true ]; then
        CMD="${CMD} --use_moe"
    else
        CMD="${CMD} --use_moe False"
    fi
    
    if [ "$USE_VC" = true ]; then
        CMD="${CMD} --use_vc"
    else
        CMD="${CMD} --use_vc False"
    fi
fi

# Print and execute command
echo "Running command:"
echo $CMD
echo ""
echo "Starting..."

eval $CMD
