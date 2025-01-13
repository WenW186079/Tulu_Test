#!/bin/bash

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_gpus> <config_file>"
    echo "Example: $0 2 path/to/config.yaml"
    exit 1
fi

NUM_GPUS="$1"
CONFIG_FILE="$2"

# Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

echo "Number of GPUs: $NUM_GPUS"
echo "Using config file: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file qlora/stage2_lora.conf \
    qlora/dpo_tune_qlora.py \
    "$2"

python open_instruct/merge_lora.py \
    --config_file $CONFIG_FILE \
    --push_to_hub \
    --save_tokenizer

# python open_instruct/merge_lora.py \
#     --lora_model_name_or_path "WenWW/Sparse8B_TO_DPO_Qlora" \
#     --base_model_name_or_path "neuralmagic/Sparse-Llama-3.1-8B-2of4" \
#     --tokenizer_name_or_path "neuralmagic/Sparse-Llama-3.1-8B-2of4" \
#     --output_dir output/dpo_8b_Qlora_merged \
#     --pad_to_multiple_of 8 \
#     --push_to_hub \
#     --save_tokenizer \
#     --hf_repo_id "WenWW/Sparse8B_TO_DPO_Qlora"
#     #--config_file $CONFIG_FILE \

