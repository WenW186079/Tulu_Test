# Tulu_Test

# 0.0 Set an environment

Example:
```
python -m venv env 
source env/bin/activate
```

# 0.1 Install requirements
```
git clone https://github.com/allenai/open-instruct.git &&
cd open-instruct &&
pip install --upgrade pip "setuptools<70.0.0" wheel  &&
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 &&
pip install packaging && 
pip install flash-attn==2.6.3 --no-build-isolation &&
pip install -r requirements.txt &&
python -m nltk.downloader punkt && 
pip install -e .
```

# 0.2 Login hugggingface and wandb
```
huggingface-cli login

wandb login
```

# Step 1. SFT
Example:
- Test: 
```
sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/mini.yaml
```

- meta-llama/Llama-3.1-8B :
  - GPU: 3 * A100 PCIe
  - High memory pressure
  - Run with loss
```
sh scripts/finetune_with_accelerate_config.sh 3 configs/train_configs/tulu3/tulu3_sft.yaml
```

Hyper parameters:
  - Note that '#####' is used here to mark changes from the original version
```
model_name_or_path: meta-llama/Llama-3.1-8B
model_revision: main
use_flash_attn: true
tokenizer_name: meta-llama/Llama-3.1-8B
use_slow_tokenizer: true
dataset_mixer:
    allenai/tulu-3-sft-mixture: 1.0
preprocessing_num_workers: 128
per_device_train_batch_size: 1  # note, this is set up for 8 GPUs
gradient_accumulation_steps: 2  # effective batch size 128 with 1 node
learning_rate: 5.0e-06 # best LR so far
max_seq_length: 2048  ##### original : 4096
lr_scheduler_type: linear
warmup_ratio: 0.03
weight_decay: 0.0
num_train_epochs: 2
output_dir: output/dpo_8b
with_tracking: true
report_to:
  - wandb
logging_steps: 1
checkpointing_steps: epoch
dataset_mix_dir: output/dpo_8b1
gradient_checkpointing : true ###### add to avoid cuda oom
reduce_loss : "sum" ###### default is 'mean'

```
# Step 2. DPO
Example:
- Test: 
```sh scripts/dpo_train_with_accelerate_config.sh 1 configs/train_configs/dpo/mini.yaml ```

- 8B :
  - GPU: 3 * A100 PCIe
  - High memory pressure
  - Run with loss
```sh scripts/dpo_train_with_accelerate_config.sh 3 configs/train_configs/tulu3/tulu3_dpo_8b.yaml```

Hyper parameters:
  - Same as the original version
```
model_name_or_path: allenai/Llama-3.1-Tulu-3-8B-SFT
model_revision: main
use_flash_attn: true
gradient_checkpointing: true
dataset_mixer:
    allenai/llama-3.1-tulu-3-8b-preference-mixture: 1.0
tokenizer_name: allenai/Llama-3.1-Tulu-3-8B-SFT
use_slow_tokenizer: true
max_seq_length: 2048
preprocessing_num_workers: 16  
per_device_train_batch_size: 1
gradient_accumulation_steps: 16 # designed for 8 GPUs, so batch size 128
learning_rate: 5.0e-7
lr_scheduler_type: linear
warmup_ratio: 0.1
weight_decay: 0.0
num_train_epochs: 1
output_dir: output/dpo_8b
with_tracking: true
report_to:
  - wandb
logging_steps: 1
use_lora: false
dpo_loss_type: dpo_norm
dpo_beta: 5
checkpointing_steps: 1000
```
# Step3: RLVR
- Test:
  
```
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"ai2-adapt-dev/gsm8k_math_ifeval_ground_truth_mixed": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_math_ground_truth": 1.0}' \
    --dataset_eval_splits test \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path HuggingFaceTB/SmolLM2-360M-Instruct \
    --reward_model_path HuggingFaceTB/SmolLM2-360M-Instruct \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --num_epochs 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_1b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
```

- 8B
  - When I use 6 * A100 PCIe, shows high memory pressure, subsequent cache flushes can slow down the training
  - GPU: 7 * A100 PCIe, no cuda oom, shows no bug, but run really slow, 20mins didn't see the loss
    - because "train_batch_size": 224

      
Example:(to save the time, I use samll batch size and reponse length)
```
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"ai2-adapt-dev/gsm8k_math_ifeval_ground_truth_mixed": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_math_ground_truth": 1.0}' \
    --dataset_eval_splits test \
    --max_token_length 1024 \
    --max_prompt_token_length 1024 \
    --response_length 1024 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \ 
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 1 \ 
    --local_rollout_batch_size 1 \ 
    --actor_num_gpus_per_node 7 \ 
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_8b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking

```



Some bugs will happen if you use lower version GPU, for example:
- `Bfloat16 is only supported on GPUs with compute capability of at least 8.0. `
- Stuck somewhere without showing bugs, ususally before loss shows

For 'Evaluation responses not received'
  - As I see from their code, it's not a bug. They update only few times, so most of them are all 'Evaluation responses not received'.

Cache problem:
```
(PolicyTrainerRayProcess pid=9930) Invalidate trace cache @ step 423: expected module 454, but got module 1
(PolicyTrainerRayProcess pid=9930) Invalidate trace cache @ step 421: expected module 908, but got module 455
```
--> 
Set the TRITON_CACHE_DIR path to a NFS
```
export TRITON_CACHE_DIR=/tmp/triton/autotune
echo $TRITON_CACHE_DIR
```

```
pip install --upgrade transformers deepspeed vllm
```
The package as in [rlvr_requirements.txt](rlvr_requirements.txt)

Might be a problem:

1. cache, sometimes it will show
```
Invalidate trace cache @ step 422 and module 0: cache has only 422 modules
```
2. [repeated 7x across cluster], no idea

```
(PolicyTrainerRayProcess pid=72803) Applying ground truth reward 🤗 [repeated 7x across cluster]
(PolicyTrainerRayProcess pid=72776) Applying ground truth reward 🤗 [repeated 5x across cluster]
(PolicyTrainerRayProcess pid=72776) Applying ground truth reward 🤗 [repeated 8x across cluster]
```
