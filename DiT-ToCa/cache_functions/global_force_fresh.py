from .force_scheduler import force_scheduler
def global_force_fresh(cache_dic, current):
    '''
    Return whether to force fresh tokens globally.
    '''
    last_steps = (current['step'] <= 2)
    first_step = (current['step'] == (current['num_steps'] - 1))
    force_fresh = cache_dic['force_fresh']
    if not first_step:
        fresh_threshold = cache_dic['cal_threshold']
    else:
        fresh_threshold = cache_dic['fresh_threshold']

    if force_fresh == 'global':
    # global force fresh means force activate all tokens in this step.
        return (first_step or (current['step']% fresh_threshold == 0))
    
    elif force_fresh == 'local':
    # fresh locally cause much worse results, for the misalignment of cache and computed tokens.
        return first_step
    elif force_fresh == 'none':
        return first_step
    else:
        raise ValueError("unrecognized force fresh strategy", force_fresh)