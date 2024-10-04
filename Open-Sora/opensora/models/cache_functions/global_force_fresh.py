from .force_scheduler import force_scheduler
def global_force_fresh(cache_dic, current):
    '''
    Return whether to force fresh tokens globally.
    '''
    is_force_fresh = {}
    fresh_thresholds = {}
    first_step = (current['step'] == 0)
    first_3steps = (current['step'] <= 2) # Note the fact that for OpenSora series models, the first 3 steps is with great importance!!!
    last_step = current['step'] == current['num_steps'] - 1
    force_fresh = cache_dic['force_fresh']
    if not first_step:
        fresh_thresholds['spat-attn']  = cache_dic['cal_threshold']['spat-attn']
        fresh_thresholds['temp-attn']  = cache_dic['cal_threshold']['temp-attn']
        fresh_thresholds['cross-attn'] = cache_dic['cal_threshold']['cross-attn']
        fresh_thresholds['mlp']        = cache_dic['cal_threshold']['mlp']
    else:
        fresh_thresholds['spat-attn']  = cache_dic['fresh_threshold']
        fresh_thresholds['temp-attn']  = cache_dic['fresh_threshold']
        fresh_thresholds['cross-attn'] = cache_dic['fresh_threshold']
        fresh_thresholds['mlp']        = cache_dic['fresh_threshold']

    if force_fresh == 'global':
        if current['flag'] == -1:
            is_force_fresh['attn'] =   (first_3steps or (current['step']% fresh_thresholds['temp-attn'] == 0))
        else:
            is_force_fresh['attn'] =   (first_3steps or (current['step']% fresh_thresholds['spat-attn'] == 0))

        is_force_fresh['cross-attn'] = (first_3steps or (current['step']% fresh_thresholds['cross-attn'] == 0))
        is_force_fresh['mlp'] =        (first_3steps or (current['step']% fresh_thresholds['mlp'] == 0))

        return is_force_fresh
    elif force_fresh == 'local':
        return first_step
    elif force_fresh == 'none':
        return first_step
    else:
        raise ValueError("unrecognized force fresh strategy", force_fresh)