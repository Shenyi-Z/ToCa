import torch
from .force_scheduler import force_scheduler
def force_init(cache_dic, current, tokens):
    '''
    Initialization for Force Activation step.
    '''
    # reset the cache index to 0
    cache_dic['cache_index'][-1][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)
    force_scheduler(cache_dic, current)
    if current['layer'] == 0:
        cache_dic['cache_index']['layer_index'][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)