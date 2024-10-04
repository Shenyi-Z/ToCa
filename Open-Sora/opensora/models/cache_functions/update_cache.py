import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
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

    cache_dic['cache'][current['flag']][layer][module].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)


        
        