#!/bin/bash

# Example:
# sh qlora/dpo_qlora_merge.sh


python open_instruct/merge_lora.py \
    --lora_model_name_or_path "WenWW/Sparse8B_TO_DPO_Qlora" \
    --base_model_name_or_path "neuralmagic/Sparse-Llama-3.1-8B-2of4" \
    --tokenizer_name_or_path "neuralmagic/Sparse-Llama-3.1-8B-2of4" \
    --output_dir output/dpo_8b_Qlora_merged \
    --pad_to_multiple_of 8 \
    --push_to_hub \
    --save_tokenizer \
    --hf_repo_id "WenWW/Sparse8B_TO_DPO_Qlora_merged"


