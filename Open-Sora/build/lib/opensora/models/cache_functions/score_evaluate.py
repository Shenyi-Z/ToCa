import torch
import torch.nn as nn
from .scores import attn_score, similarity_score, norm_score
def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens.
    '''
    #这里用match case 来做可读性更好，但是考虑到match case是3.10版本才有的,而且其加速性能未验证，先用if else
    #fresh_ratio = cache_dic['fresh_ratio']
    #cache_index = cache_dic['cache_index']
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    #start.record()
    if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')):
        # 0.4ms extra on 4090
        # 从cache_index中找出达到cache_step达到fresh_threshold的tokens
        force_fresh_mask = torch.as_tensor((cache_dic['cache_index'][current['flag']][current['layer']][current['module']] >= 2 * cache_dic['fresh_threshold']), dtype = int) # 2 because the threshold is for step, not module
        force_len = force_fresh_mask.sum(dim=1)
        force_indices = force_fresh_mask.argsort(dim = -1, descending = True)[:, :force_len.min()]
        #在维度-1随机重排
        force_indices = force_indices[:, torch.randperm(force_indices.shape[1])]

    if cache_dic['cache_type'] == 'random':
        score = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1], device=tokens.device)
        score = torch.cat([score, score], dim=0).to(tokens.device)

    elif cache_dic['cache_type'] == 'straight':
        score = torch.ones(tokens.shape[0], tokens.shape[1]).to(tokens.device)
    
    elif cache_dic['cache_type'] == 'attention':
        # cache_dic['attn_map'][step][layer] (B, N, N), the last dimention has get softmaxed
        score = attn_score(cache_dic, current)
        #score = score + 0.0 * torch.rand_like(score, device= score.device)
    
    elif cache_dic['cache_type'] == 'similarity':
        score = similarity_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'norm':
        score = norm_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'compress':
        score1 = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1])
        score1 = torch.cat([score1, score1], dim=0).to(tokens.device)
        score2 = cache_dic['attn_map'][current['flag']][current['layer']].sum(dim=1)#.mean(dim=0) # (B, N)
        # normalize
        score2 = score2 / score2.max(dim=1, keepdim=True)[0]
        score = 0.5 * score1 + 0.5 * score2
    #end.record()
    #torch.cuda.synchronize()
    #print(f"Time for score evaluation: {start.elapsed_time(end)} ms")
    if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')): # current['is_force_fresh'] is False, cause when it is True, no cut and fresh are needed
            #print(torch.ones_like(force_indices, dtype=float, device=force_indices.device).dtype)
        score.scatter_(dim=1, index=force_indices, src=torch.ones_like(force_indices, dtype=torch.float32, 
                                                                           device=force_indices.device))
    
    if (True and (cache_dic['force_fresh'] == 'global')):
        soft_step_score = cache_dic['cache_index'][current['flag']][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
        soft_layer_score = cache_dic['cache_index']['layer_index'][current['module']].float() / (27)
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score
    
    return score.to(tokens.device)