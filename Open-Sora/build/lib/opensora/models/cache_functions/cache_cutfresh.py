from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
#from .token_merge import token_merge
import torch
def cache_cutfresh(cache_dic, tokens, current):
    """
    indices: (B, N), the index tensor for the fresh tokens, tell where the 1st, 2nd, 3rd... tokens are
    fresh_indices: (B, fresh_ratio * N), top fresh_ratio cut for indices
    fresh_tokens: (B, fresh_ratio * N, D), the fresh tokens
    """
    tick1 = torch.cuda.Event(enable_timing=True)
    tick2 = torch.cuda.Event(enable_timing=True)
    #tick3 = torch.cuda.Event(enable_timing=True)
    #tick4 = torch.cuda.Event(enable_timing=True)

    step = current['step']
    layer = current['layer']
    module = current['module']

    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)

    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio, device = tokens.device), min=0, max=1) # 0.03ms
    # Generate the index tensor for fresh tokens
    #tick1.record()
    score = score_evaluate(cache_dic, tokens, current) # 0.26ms
    #tick2.record()
    #score = local_selection_with_space_time_bonus(cache_dic, score, 0.3, 2, time_mean=False)
    indices = score.argsort(dim=-1, descending=True) # 0.12ms
    #indices = cache_dic['indices_cache'][current['flag']][current['layer']]
    topk = int(fresh_ratio * score.shape[1])
    #topk = int(fresh_ratio * cache_dic['dynamic_size'][2] * cache_dic['dynamic_size'][3]) * cache_dic['dynamic_size'][1]
    fresh_indices = indices[:, :topk] #前fresh_ratio的token的index
    stale_indices = indices[:, topk:] #后1-fresh_ratio的token的index
    # (B, fresh_ratio *N)

    # stale tokens index + 1 in each ***module***, fresh tokens index = 0
    cache_dic['cache_index'][current['flag']][layer][module] += 1
    cache_dic['cache_index'][current['flag']][layer][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    cache_dic['cache_index']['layer_index'][module] += 1
    cache_dic['cache_index']['layer_index'][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    # 0.08ms
    # select the fresh tokens out
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    #stale_indices_expand = stale_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    #if cache_dic['merge_weight'] != 0:
    #    token_merge(cache_dic, tokens, current, fresh_indices, stale_indices)        
    
    if module in ['mlp', 'attn', 'cross-attn']:
         
        fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)
        # 0.10ms
        #torch.cuda.synchronize()
        #print(tick1.elapsed_time(tick2))
        return fresh_indices, fresh_tokens
    else:
        raise ValueError("Unrecognized module?", module)
    
import torch
from einops import rearrange

def local_selection_with_space_time_bonus(cache_dic, score, bonus_ratio, grid_size=2, time_mean = False):
    # 从 cache_dic 中获取张量的形状
    B, T, H, W = cache_dic['dynamic_size']
    
    # 对 score 进行变形，将其重塑为 [B, T, H, W] 的形状
    score = rearrange(score, "B (T H W) -> B T H W", T=T, H=H, W=W)
    
    # 计算补 0 的尺寸，使得 H 和 W 都能被 grid_size 整除
    pad_h = (grid_size - H % grid_size) % grid_size  # H 维度需要补充的 0 的数量
    pad_w = (grid_size - W % grid_size) % grid_size  # W 维度需要补充的 0 的数量
    
    # 对 H 和 W 维度进行补 0
    if pad_h > 0 or pad_w > 0:
        score = torch.nn.functional.pad(score, (0, pad_w, 0, pad_h))  # (W 左右补 pad_w, H 上下补 pad_h)

    # 更新补 0 后的 H 和 W
    H_padded, W_padded = score.shape[2], score.shape[3]
    
    # Step 1: 在 H*W 维度上进行归一化，使得不同时间步的信息权重相同
    score = score.view(B, T, -1)  # 将 H 和 W 合并为一个维度 [B, T, H*W]
    score = torch.nn.functional.softmax(score, dim=-1)  # 在 H*W 维度上进行归一化
    score = score.view(B, T, H_padded, W_padded)  # 恢复到 [B, T, H_padded, W_padded] 形状

    # Step 2: 在每个空间切片（即每个 T 时间步内）进行分块操作
    block_size = grid_size * grid_size
    assert (H_padded * W_padded) % block_size == 0, f"H_padded * W_padded 必须能被块大小整除, shape: {B},{T},{H_padded},{W_padded}; block:{grid_size}*{grid_size};" 

    # 将 score 重塑为按块分组的形状
    score_reshaped = score.view(B, T, H_padded // grid_size, grid_size, W_padded // grid_size, grid_size)
    score_reshaped = score_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, T, H//grid_size, W//grid_size, grid_size, grid_size]
    score_reshaped = score_reshaped.view(B, T, -1, block_size)  # [B, T, num_blocks, block_size]

    # Step 3: 找到每个块中的最大分数
    max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)  # [B, T, num_blocks, 1]
    
    # Step 4: 创建掩码以标识最大分数的 token
    mask = torch.zeros_like(score_reshaped)
    mask.scatter_(-1, max_indices, 1)  # 将掩码在最大分数的索引位置设置为 1
    
    # Step 5: 仅对最大分数的 token 应用加成
    score_reshaped = score_reshaped + (mask * max_scores * bonus_ratio)  # 仅对最大分数应用加成
    
    # Step 6: 将 score 还原为原始的形状
    score_modified = score_reshaped.view(B, T, H_padded // grid_size, W_padded // grid_size, grid_size, grid_size)
    score_modified = score_modified.permute(0, 1, 2, 4, 3, 5).contiguous()
    score_modified = score_modified.view(B, T, H_padded, W_padded)

    # Step 7: 去除补 0 的部分
    if pad_h > 0 or pad_w > 0:
        score_modified = score_modified[:, :, :H, :W]  # 移除补的 0

    if time_mean:
        score_modified = score_modified.mean(dim = 1)
        score_modified = score_modified.unsqueeze(1).expand(B, T, H, W)
    # 最后将 score 变回原始的形状 [B, (T H W)]
    score_modified = rearrange(score_modified, "B T H W -> B (T H W)")
    
    return score_modified
