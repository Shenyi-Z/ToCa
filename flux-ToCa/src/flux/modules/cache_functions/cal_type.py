from .force_scheduler import force_scheduler

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if cache_dic['fresh_ratio'] == 0.0:
        # FORA: Uniform
        first_step = (current['step'] == 0)
    else:
        # ToCa: First 3 steps enhanced
        first_step = (current['step'] <= 2)
    
    force_fresh = cache_dic['force_fresh']
    if not first_step:
        fresh_interval = cache_dic['cal_threshold']
    else:
        fresh_interval = cache_dic['fresh_threshold']

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        force_scheduler(cache_dic, current)
    
    # ToCa
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'ToCa'

######################################################################
    #if (current['step'] in [3,2,1,0]):
    #    current['type'] = 'full'