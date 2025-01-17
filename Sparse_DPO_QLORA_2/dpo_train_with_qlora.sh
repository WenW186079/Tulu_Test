#!/bin/bash

# example usage
# sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/default.yaml
# sh scripts/finetune_with_accelerate_config.sh 8 configs/train_configs/sft/olmo_17_sft.yaml

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
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2.conf \
    open_instruct/dpo_tune.py \
    "$2"

python open_instruct/merge_lora.py \
    --config_file $CONFIG_FILE \
    --push_to_hub \
    --save_tokenizer 
