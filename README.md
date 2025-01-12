# Tulu_Test

# 0.0 Set an environment
```
python -m venv tulu_env &&
source tulu_env/bin/activate
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

# 0.2 Set dictionary
```
mkdir -p /tmp/triton/autotune &&
mkdir -p /root/.triton/autotune &&
ln -s /tmp/triton/autotune /root/.triton/autotune &&

export HF_HOME="/workspace" &&
export TRITON_CACHE_DIR="/tmp/triton/autotune"

```
# 0.3 Clone these files and move to relative dictionary
```
cd .. &
git clone https://github.com/WenW186079/Tulu_Test.git &&
mv Tulu_Test/Mask Tulu_Test/qlora open-instruct/ &&
cd open-instruct

```
# 0.4 Login hugggingface and wandb
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

With QLORA version-SFT
```
sh scripts/finetune_qlora_with_accelerate_config.sh 2 configs/train_configs/sft/mini.yaml
sh scripts/finetune_qlora_with_accelerate_config.sh 7 configs/train_configs/tulu3/tulu3_sft.yaml
```

# Step 2. DPO
Example:
- Test: 
```
sh scripts/dpo_train_with_accelerate_config.sh 1 configs/train_configs/dpo/mini.yaml
sh scripts/dpo_train_with_accelerate_config.sh 3 configs/train_configs/tulu3/tulu3_dpo_8b.yaml
```

- 8B :
  - GPU: 3 * A100 PCIe
  - High memory pressure
  - Run with loss

    
With MASK version-DPO
```
# Here 4 is gpu number
sh Mask/dpo_train_mask.sh 4 qlora/8b_dpo_mask.yaml
```

With QLORA version-DPO
```
sh qlora/dpo_train_with_qlora.sh 4 qlora/8b_dpo_qlora.yaml
```

# Step3: RLVR

- 8B model + 8B reward model
  - GPU: 8 x RTX A6000, cuda oom
  - When I use 6 * A100 PCIe, shows high memory pressure, subsequent cache flushes can slow down the training
  - GPU: 7 * A100 PCIe, no cuda oom, shows no bug, but run really slow, 20mins didn't see the loss
    - because "train_batch_size": 224

With MASK version-RLVR
```
python Mask/ppo_vllm_thread_ray_gtrl_mask.py Mask/rlvr_mask.yaml
```

With QLORA version-RLVR
```
python qlora/ppo_vllm_thread_ray_gtrl_qlora.py qlora/rlvr_qlora.yaml
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

Might be a problem [leave it or to be solved]:

1. cache, sometimes it will show
```
Invalidate trace cache @ step 422 and module 0: cache has only 422 modules
```
2. [repeated 7x across cluster], no idea if it's a problem

```
(PolicyTrainerRayProcess pid=72803) Applying ground truth reward 🤗 [repeated 7x across cluster]
(PolicyTrainerRayProcess pid=72776) Applying ground truth reward 🤗 [repeated 5x across cluster]
(PolicyTrainerRayProcess pid=72776) Applying ground truth reward 🤗 [repeated 8x across cluster]
```
