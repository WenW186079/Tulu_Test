import torch
from torch.nn import Linear
from peft import PeftModel
from transformers import AutoModelForCausalLM
import pandas as pd

def check_model_sparsity(model):
    
    layer_data = []

    total_zeros = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, Linear) or hasattr(module, "weight"):
            if module.weight is not None:
                weight = module.weight.detach().cpu()  
                num_params = weight.numel()
                num_zeros = (weight == 0).sum().item()

                sparsity = num_zeros / num_params * 100 if num_params > 0 else 0
                non_zero_params = num_params - num_zeros

                total_zeros += num_zeros
                total_params += num_params

                layer_data.append({
                    "Layer Name": name,
                    "Non-Zero Params": non_zero_params,
                    "Zero Params": num_zeros,
                    "Sparsity %": f"{sparsity:.2f}%"
                })

    overall_sparsity = total_zeros / total_params * 100 if total_params > 0 else 0
    layer_data.append({
        "Layer Name": "Overall",
        "Non-Zero Params": total_params - total_zeros,
        "Zero Params": total_zeros,
        "Sparsity %": f"{overall_sparsity:.2f}%"
    })

    
    df = pd.DataFrame(layer_data)
    print(df.to_string(index=False))
    return df

base_model = AutoModelForCausalLM.from_pretrained("neuralmagic/Sparse-Llama-3.1-8B-2of4")
model = PeftModel.from_pretrained(base_model, "WenWW/Sparse8B_TO_DPO_Qlora",ignore_mismatched_sizes=True)

sparsity_df = check_model_sparsity(model)
