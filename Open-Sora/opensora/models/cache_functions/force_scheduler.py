import torch
def force_scheduler(cache_dic, current):
    thresholds = {}
    if cache_dic['fresh_ratio'] == 0:
        # FORA
        linear_step_weight = 0.0
    else: 
        # TokenCache
        linear_step_weight = 0.0 #N=6 0.2 #N=4 0.3
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current['step'] / current['num_steps'])
    threshold = torch.round(cache_dic['fresh_threshold'] / step_factor)

    # Here we set force activation cycles for different modules separately.
    thresholds = {
        'spat-attn' : 3,
        'temp-attn' : 3,
       'cross-attn' : 6,
              'mlp' : 3   }
    
    #thresholds = {
    #    'spat-attn' : 2,
    #    'temp-attn' : 2,
    #   'cross-attn' : 2,
    #          'mlp' : 2   }

    cache_dic['cal_threshold'] = thresholds
    #return threshold