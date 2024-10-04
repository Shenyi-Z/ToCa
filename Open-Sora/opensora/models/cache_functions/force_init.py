import torch
from .force_scheduler import force_scheduler
def force_init(cache_dic, current, tokens):
    cache_dic['cache_index'][current['flag']][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)
    force_scheduler(cache_dic, current)
    if current['layer'] == 0:
        cache_dic['cache_index']['layer_index'][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)