import torch
def force_scheduler(cache_dic, current):
    '''
    Force Activation Cycle Scheduler
    '''
    if cache_dic['fresh_ratio'] == 0:
        # FORA
        linear_step_weight = 0.0
    else: 
        # ToCa
        linear_step_weight = 0.4
    step_factor = torch.tensor(1 + linear_step_weight - 2 * linear_step_weight * current['step'] / current['num_steps'])
    threshold = torch.round(cache_dic['fresh_threshold'] / step_factor)

    if (current['step'] in range(50,100)) and (cache_dic['fresh_ratio'] == 0):
        # We find that in 50~100 steps, the model is extremely sensitive for cache, i.e. worse temporal redundancy.
        threshold = 2

    cache_dic['cal_threshold'] = threshold
    #return threshold