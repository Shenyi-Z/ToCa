from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
#from .token_merge import token_merge
import torch
def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']

    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)

    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio, device = tokens.device), min=0, max=1)
    # Generate the index tensor for fresh tokens
    score = score_evaluate(cache_dic, tokens, current) # s1, s2, s3 mentioned in the paper
    #score = local_selection_with_space_time_bonus(cache_dic, score, 0.3, 2, time_mean=False) # s4 mentioned in the paper.
    indices = score.argsort(dim=-1, descending=True)
    topk = int(fresh_ratio * score.shape[1])
    fresh_indices = indices[:, :topk]
    stale_indices = indices[:, topk:]
    # (B, fresh_ratio *N)

    # Updating the Cache Frequency Score s3 counter mentioned in the paper
    # stale tokens index + 1 in each ***module***, fresh tokens index = 0
    cache_dic['cache_index'][current['flag']][layer][module] += 1
    cache_dic['cache_index'][current['flag']][layer][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    cache_dic['cache_index']['layer_index'][module] += 1
    cache_dic['cache_index']['layer_index'][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    # select the fresh tokens out
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])

    if module in ['mlp', 'attn', 'cross-attn']:
         
        fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)

        return fresh_indices, fresh_tokens
    else:
        raise ValueError("Unrecognized module?", module)
    
import torch
from einops import rearrange

def local_selection_with_space_time_bonus(cache_dic, score, bonus_ratio, grid_size=2, time_mean = False):
    # Get the shape of the tensor from cache_dic
    B, T, H, W = cache_dic['dynamic_size']
    
    # Reshape the score to [B, T, H, W]
    score = rearrange(score, "B (T H W) -> B T H W", T=T, H=H, W=W)
    
    # Calculate the padding size to make H and W divisible by grid_size
    pad_h = (grid_size - H % grid_size) % grid_size  # Number of zeros to pad in H dimension
    pad_w = (grid_size - W % grid_size) % grid_size  # Number of zeros to pad in W dimension
    
    # Pad the H and W dimensions with zeros
    if pad_h > 0 or pad_w > 0:
        score = torch.nn.functional.pad(score, (0, pad_w, 0, pad_h))  # (pad width left/right, pad height top/bottom)

    # Update H and W after padding
    H_padded, W_padded = score.shape[2], score.shape[3]
    
    # Step 1: Normalize along the H*W dimension so that information from different time steps has equal weight
    score = score.view(B, T, -1)  # Merge H and W into one dimension [B, T, H*W]
    score = torch.nn.functional.softmax(score, dim=-1)  # Normalize along H*W dimension
    score = score.view(B, T, H_padded, W_padded)  # Restore to [B, T, H_padded, W_padded] shape

    # Step 2: Perform block-wise operation on each spatial slice (each T time step)
    block_size = grid_size * grid_size
    assert (H_padded * W_padded) % block_size == 0, f"H_padded * W_padded must be divisible by block size, shape: {B},{T},{H_padded},{W_padded}; block:{grid_size}*{grid_size};" 

    # Reshape the score into block-wise grouped shape
    score_reshaped = score.view(B, T, H_padded // grid_size, grid_size, W_padded // grid_size, grid_size)
    score_reshaped = score_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, T, H//grid_size, W//grid_size, grid_size, grid_size]
    score_reshaped = score_reshaped.view(B, T, -1, block_size)  # [B, T, num_blocks, block_size]

    # Step 3: Find the maximum score in each block
    max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)  # [B, T, num_blocks, 1]
    
    # Step 4: Create a mask to identify the token with the maximum score
    mask = torch.zeros_like(score_reshaped)
    mask.scatter_(-1, max_indices, 1)  # Set the mask to 1 at the index of the maximum score
    
    # Step 5: Apply the bonus only to the token with the maximum score
    score_reshaped = score_reshaped + (mask * max_scores * bonus_ratio)  # Apply bonus only to the maximum score
    
    # Step 6: Restore the score to its original shape
    score_modified = score_reshaped.view(B, T, H_padded // grid_size, W_padded // grid_size, grid_size, grid_size)
    score_modified = score_modified.permute(0, 1, 2, 4, 3, 5).contiguous()
    score_modified = score_modified.view(B, T, H_padded, W_padded)

    # Step 7: Remove the padded zeros
    if pad_h > 0 or pad_w > 0:
        score_modified = score_modified[:, :, :H, :W]  # Remove the padded zeros

    if time_mean:
        score_modified = score_modified.mean(dim = 1)
        score_modified = score_modified.unsqueeze(1).expand(B, T, H, W)
        
    # Finally, reshape the score back to the original shape [B, (T H W)]
    score_modified = rearrange(score_modified, "B T H W -> B (T H W)")
    
    return score_modified
