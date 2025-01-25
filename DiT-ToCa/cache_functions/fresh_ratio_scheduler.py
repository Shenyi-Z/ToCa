import torch
def fresh_ratio_scheduler(cache_dic, current):
    '''
    Return the fresh ratio for the current step.
    '''
    fresh_ratio = cache_dic['fresh_ratio']
    fresh_ratio_schedule = cache_dic['fresh_ratio_schedule']
    step = current['step']
    num_steps = current['num_steps']
    threshold = cache_dic['fresh_threshold']
    weight = 0.9
    if fresh_ratio_schedule == 'constant':
        return fresh_ratio
    elif fresh_ratio_schedule == 'linear':
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps)
    elif fresh_ratio_schedule == 'exp':
        #return 0.5 * (0.052 ** (step/num_steps))
        return fresh_ratio * (weight ** (step / num_steps))
    elif fresh_ratio_schedule == 'linear-mode':
        mode = (step % threshold)/threshold - 0.5
        mode_weight = 0.1
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps + mode_weight * mode)
    elif fresh_ratio_schedule == 'layerwise':
        return fresh_ratio * (1 + weight - 2 * weight * current['layer'] / 27)
    elif fresh_ratio_schedule == 'linear-layerwise':
        step_weight = 0.4 
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

        layer_weight = 0.8
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        module_weight = 2.5
        module_time_weight = 0.6
        module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='attn' else (1 + module_time_weight * module_weight)
        
        return fresh_ratio * layer_factor * step_factor * module_factor
    
###### Recommended Configurations ######

    elif fresh_ratio_schedule == 'ToCa-ddim50':
        # Proposed scheduling method in toca.

        # step wise scheduling, we find there is little differece if change the weight of step factor, so this is not a key factor. 
        step_weight = 2.0 #0.4 #0.0 # 2.0
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

        # layer wise scheduling, important. Meaning caculate more in the front layers, less in the back layers.
        layer_weight = -0.2#0.8 #0.0 # -0.2
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        # Module wise scheduling, important. Meaning caculate more in the mlp module, less in the attn module.
        module_weight = 2.5 # no calculations for attn module (2.5 * 0.4 = 1.0), compuation is transformed to mlp module.
        module_time_weight = 0.6 # estimated from the time and flops of mlp and attn module, may change in different situations.
        module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='attn' else (1 + module_time_weight * module_weight)
        
        return fresh_ratio * layer_factor * step_factor * module_factor
    
    elif fresh_ratio_schedule == 'ToCa-ddpm250':
        # Proposed scheduling method in toca.

        # step wise scheduling, we find there is little differece if change the weight of step factor, so this is not a key factor. 
        step_weight = 0.4 #0.0 # 2.0
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

        # layer wise scheduling, important. Meaning caculate more in the front layers, less in the back layers.
        layer_weight = 0.8 #0.0 # -0.2
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        # Module wise scheduling, important. Meaning caculate more in the mlp module, less in the attn module.
        module_weight = 2.5 # no calculations for attn module (2.5 * 0.4 = 1.0), compuation is transformed to mlp module.
        module_time_weight = 0.6 # estimated from the time and flops of mlp and attn module, may change in different situations.
        module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='attn' else (1 + module_time_weight * module_weight)
        return fresh_ratio * layer_factor * step_factor * module_factor

    else:
        raise ValueError("unrecognized fresh ratio schedule", fresh_ratio_schedule)
