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
    #threshold = torch.round(4 / step_factor)
    key_point = 2
    if current['step'] in range(0,key_point):
        threshold = 1
    #thresholds = {
    #    'spat-attn' : 3,
    #    'temp-attn' : 3,
    #   'cross-attn' : 6,
    #          'mlp' : 3   }
    thresholds = {
        'spat-attn' : 1,
        'temp-attn' : 1,
       'cross-attn' : 1,
              'mlp' : 1   }
    #if current['step'] in range(150,175):
    #    threshold = 4
    #elif current['step'] in list(range(0,25)) + list(range(75,100)) + list(range(175,200)) + list(range(225,250)):
    #    threshold = 3
    #elif current['step'] in list(range(100,125)) + list(range(150,175)) + list(range(200,225)):
    #    threshold = 4
    #elif current['step'] in range(100,175):
    #    threshold = 5
    #elif current['step'] in range(200,225):
    #    threshold = 5
    #step_weight = 0.25
    #if current['step'] >= 0.5 * (1 - step_weight) * current['num_steps']:
    #    threshold =  int(cache_dic['fresh_threshold'] * (1 + step_weight))
    #elif current['step'] <= 0.5 * (1 - step_weight) * current['num_steps']:
    #    threshold = int(cache_dic['fresh_threshold'] * (1 - step_weight))
    cache_dic['cal_threshold'] = thresholds
    #return threshold