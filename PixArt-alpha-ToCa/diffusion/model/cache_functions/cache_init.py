def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['cross_attn_map'] = {}
    cache_dic['cross_attn_map'][-1] = {}

    for j in range(28):
        cache[-1][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1][j] = {}
        cache_dic['cross_attn_map'][-1][j] = {}

    cache_dic['cache_type'] = model_kwargs['cache_type']
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = model_kwargs['ratio_scheduler']
    cache_dic['fresh_ratio'] = model_kwargs['fresh_ratio']
    cache_dic['fresh_threshold'] = model_kwargs['fresh_threshold']
    cache_dic['force_fresh'] = model_kwargs['force_fresh']
    cache_dic['soft_fresh_weight'] = model_kwargs['soft_fresh_weight']
    #cache_dic['merge_weight'] = merge_weight
    current = {}
    current['num_steps'] = num_steps
    return cache_dic, current
    