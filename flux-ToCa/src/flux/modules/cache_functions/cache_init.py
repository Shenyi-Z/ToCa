def cache_init(timesteps, model_kwargs=None):   
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
    cache_dic['attn_map'][-1]['double_stream'] = {}
    cache_dic['attn_map'][-1]['single_stream'] = {}

    cache_dic['k-norm'] = {}
    cache_dic['k-norm'][-1] = {}
    cache_dic['k-norm'][-1]['double_stream'] = {}
    cache_dic['k-norm'][-1]['single_stream'] = {}

    cache_dic['v-norm'] = {}
    cache_dic['v-norm'][-1] = {}
    cache_dic['v-norm'][-1]['double_stream'] = {}
    cache_dic['v-norm'][-1]['single_stream'] = {}

    cache_dic['cross_attn_map'] = {}
    cache_dic['cross_attn_map'][-1] = {}
    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    for j in range(19):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}
        
        cache_dic['k-norm'][-1]['double_stream'][j] = {}
        cache_dic['k-norm'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['k-norm'][-1]['double_stream'][j]['img_mlp'] = {}

        cache_dic['v-norm'][-1]['double_stream'][j] = {}
        cache_dic['v-norm'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['v-norm'][-1]['double_stream'][j]['img_mlp'] = {}

    for j in range(38):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

        cache_dic['k-norm'][-1]['single_stream'][j] = {}
        cache_dic['k-norm'][-1]['single_stream'][j]['total'] = {}

        cache_dic['v-norm'][-1]['single_stream'][j] = {}
        cache_dic['v-norm'][-1]['single_stream'][j]['total'] = {}

    mode = 'ToCa'
    if mode == 'original':
        cache_dic['cache_type'] = 'random'              # model_kwargs['cache_type'] # no use
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'      # model_kwargs['ratio_scheduler']
        cache_dic['fresh_ratio'] = 0.0                  # model_kwargs['fresh_ratio']
        cache_dic['fresh_threshold'] = 1                # model_kwargs['fresh_threshold']
        cache_dic['force_fresh'] = 'global'             # model_kwargs['force_fresh']
        cache_dic['soft_fresh_weight'] = 0.0            # model_kwargs['soft_fresh_weight']
    
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'attention'           # Attention cache type for ToCa, use Self-Attention Weight to evaluate the importance of each token
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.1
        cache_dic['fresh_threshold'] = 4
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.25
        
    current = {}
    current['final_time'] = timesteps[-2]
    return cache_dic, current
