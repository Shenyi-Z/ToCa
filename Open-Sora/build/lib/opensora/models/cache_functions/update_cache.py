import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    """
    Update the cache with fresh tokens based on the given index.
    
    Args:
    indices (torch.Tensor): The index tensor for tokens. 从权重高到底的index
    fresh_tokens (torch.Tensor): The fresh tokens to update the cache with.
    cach_dic (dict): The cache dictionary containing cache data and indices.
    current (dict): Dictionary containing the current step, layer, and module information.
    fresh_attn_map (torch.Tensor): The attention map for the fresh tokens. attn模块里已经排好序了,直接盖上去就行
    """
    step = current['step']
    layer = current['layer']
    module = current['module']
    # Update the cached tokens at the positions
    if module == 'attn':
        indices = fresh_indices#.sort(dim=1, descending=False)[0]
        cache_dic['attn_map'][current['flag']][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'cross-attn':
        indices = fresh_indices#.sort(dim=1, descending=False)[0]
        cache_dic['cross_attn_map'][current['flag']][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'mlp':
        indices = fresh_indices

    #if (indices.shape[1] != 0):
    #    to_be_updated_fresh_tokens = torch.gather(input = cache_dic['cache'][current['flag']][layer][module], dim = 1, index = indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]))
    #    residual_token = (fresh_tokens - to_be_updated_fresh_tokens).mean(dim=1)
    #    cache_dic['cache'][current['flag']][layer][module] = cache_dic['cache'][current['flag']][layer][module] + 0.0 * residual_token.unsqueeze(1)
    
    cache_dic['cache'][current['flag']][layer][module].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)


        
        