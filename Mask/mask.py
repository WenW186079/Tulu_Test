import torch 
import torch.distributed as dist  
from torch.nn import Linear  
import deepspeed 

@torch.no_grad()
def mask_weights(module, verify=False, log=False):
    print(' ========= apply mask_weights ==========')
    """Apply mask to sharded weights directly with verification."""
    if isinstance(module, torch.nn.Linear) and hasattr(module, 'mask'):
        if module.weight.numel() == 0:
            return
            
        # Store stats before masking
        pre_zeros = (module.weight == 0).sum().item()
        pre_total = module.weight.numel()
        
        # Apply mask
        module.weight *= module.mask
        
        # Verify after masking
        post_zeros = (module.weight == 0).sum().item()
        sparsity = post_zeros / pre_total
        
        if verify:
            # Verify mask was applied correctly
            mismatched = (module.weight != 0) & (module.mask == 0)
            mask_violated = mismatched.any().item()
            
            if mask_violated:
                print(f"WARNING: Mask violation in {module.__class__.__name__}")
                print(f"Number of violations: {mismatched.sum().item()}")
            
            if abs(sparsity - 0.5) > 1e-3:  # Check for 50% sparsity
                print(f"WARNING: Incorrect sparsity {sparsity:.1%} in {module.__class__.__name__}")
        
        if log:
            print(f"Module: {module.__class__.__name__}")
            print(f"Pre-mask zeros: {pre_zeros}, Post-mask zeros: {post_zeros}")
            print(f"Current sparsity: {sparsity:.1%}")


def verify_sparsity(weight_tensor, target_sparsity=0.5, tolerance=1e-3):
    """
    Verify if the weight tensor has the expected sparsity ratio.
    Handles empty tensors and distributed training cases.
    
    Args:
        weight_tensor: The weight tensor to check
        target_sparsity: Expected ratio of zero weights (default: 0.5 for 2:4 sparsity)
        tolerance: Acceptable deviation from target sparsity
    
    Returns:
        tuple: (bool, float) - (whether sparsity is correct, actual sparsity ratio)
    """
    numel = weight_tensor.numel()
    if numel == 0:  # Handle empty tensors
        return True, 0.0
        
    zero_count = (weight_tensor == 0).sum().item()
    zero_ratio = zero_count / numel
    return abs(zero_ratio - target_sparsity) <= tolerance, zero_ratio


    
def attach_masks(model, to_layer, rank):
    print('=========== apply attach_masks ===========')
    total_layers = 0
    correct_sparsity_layers = 0
    skipped_empty_layers = 0
    total_params = 0
    masked_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, to_layer):
            # Skip embedding, layer norm, and lm_head
            if any(x in name for x in ['embed_tokens', 'ln', 'norm', 'lm_head']):
                continue
            
            # Gather parameters to check sparsity
            with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=None):
                params = module.weight.numel()
                total_params += params
                
                if params == 0:
                    skipped_empty_layers += 1
                    print(f"[Rank {rank}] Skipping empty sharded layer: {name}")
                    continue
                    
                total_layers += 1
                is_correct, sparsity = verify_sparsity(module.weight)
                
                if is_correct:
                    correct_sparsity_layers += 1
                    zero_params = (module.weight == 0).sum().item()
                    masked_params += zero_params
                    
                    # Create mask from gathered weights
                    local_mask = torch.where(module.weight == 0,
                                        torch.tensor(0, dtype=torch.uint8, device=module.weight.device),
                                        torch.tensor(1, dtype=torch.uint8, device=module.weight.device))
                    
                    # Register the mask
                    module.register_buffer("mask", local_mask, persistent=False)
                    
                    print(f"[Rank {rank}] Attached mask to {name}:")
                    print(f"  - Parameters: {params:,}")
                    print(f"  - Masked (zero) params: {zero_params:,}")
                    print(f"  - Sparsity: {sparsity:.1%}")
                else:
                    print(f"[Rank {rank}] WARNING: Skipping {name} - Incorrect sparsity: {sparsity:.1%}")

    # Gather stats from all ranks
    stats = torch.tensor([total_params, masked_params, total_layers, skipped_empty_layers, correct_sparsity_layers], 
                        device=module.weight.device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_params_all, masked_params_all, total_layers_all, skipped_layers_all, correct_layers_all = stats.tolist()

    print(f"\n[Rank {rank}] Local Stats:")
    print(f"Total parameters: {total_params:,}")
    print(f"Masked parameters: {masked_params:,}")
    print(f"Sparsity: {(masked_params/total_params*100 if total_params > 0 else 0):.1f}%")
    print(f"Non-empty layers: {total_layers}")
    print(f"Empty layers: {skipped_empty_layers}")

    if rank == 0:
        print(f"\nGlobal Stats (All Ranks):")
        print(f"Total parameters across model: {total_params_all:,}")
        print(f"Total masked parameters: {masked_params_all:,}")
        if total_params_all > 0:
            print(f"Global sparsity: {(masked_params_all/total_params_all*100):.1f}%")
        print(f"Total non-empty layers: {total_layers_all}")
        print(f"Total empty/skipped layers: {skipped_layers_all}")
        print(f"Layers with correct sparsity: {correct_layers_all}")
