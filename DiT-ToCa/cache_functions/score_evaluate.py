import torch
import torch.nn as nn
from .scores import attn_score, similarity_score, norm_score, kv_norm_score
def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens. Mainly include s1, (s2,) s3 mentioned in the paper.
    '''

    #if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')):
    ## abandoned branch, if you want to explore the local force fresh strategy, this may help.
    #    force_fresh_mask = torch.as_tensor((cache_dic['cache_index'][-1][current['layer']][current['module']] >= 2 * cache_dic['fresh_threshold']), dtype = int) # 2 because the threshold is for step, not module
    #    force_len = force_fresh_mask.sum(dim=1)
    #    force_indices = force_fresh_mask.argsort(dim = -1, descending = True)[:, :force_len.min()]
    #
    #    force_indices = force_indices[:, torch.randperm(force_indices.shape[1])]

    if cache_dic['cache_type'] == 'random':
        # select tokens randomly, but remember to keep the same for cfg and no cfg.
        score = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1], device=tokens.device)
        score = torch.cat([score, score], dim=0).to(tokens.device)

    elif cache_dic['cache_type'] == 'straight':
        # abandon the cache, just return 1 hhh, obviously no use.
        score = torch.ones(tokens.shape[0], tokens.shape[1]).to(tokens.device)
    
    elif cache_dic['cache_type'] == 'attention':
        # Recommended selection method in the paper.

        # cache_dic['attn_map'][step][layer] (B, N, N), the last dimention has get softmaxed

        # calculate the attention score, for DiT, there is no cross-attention, so just self-attention score s1 applied.
        score = attn_score(cache_dic, current)

        # if you'd like to add some randomness to the score as SiTo does to avoid tokens been over cached. This works, but we have another elegant way.
        #score = score + 0.0 * torch.rand_like(score, device= score.device)
    elif cache_dic['cache_type'] == 'kv-norm':
        score = kv_norm_score(cache_dic, current)

    elif cache_dic['cache_type'] == 'similarity':
        # why don't we calculate similarity score? 
        # This is natural but we find it cost **TOO MUCH TIME**, cause in DiT series models, you can calculate similarity for scoring every where.
        score = similarity_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'norm':
        # an interesting exploration, but not used in the final version.
        # use norm as the selectioon method is probably because of the norm of the tokens may indicate the importance of the token. but it is not the case.
        score = norm_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'compress':
        # if you want to combine any of the methods mentioned, we have not tried this yet hhh.
        score1 = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1])
        score1 = torch.cat([score1, score1], dim=0).to(tokens.device)
        score2 = cache_dic['attn_map'][-1][current['layer']].sum(dim=1)#.mean(dim=0) # (B, N)
        # normalize
        score2 = score2 / score2.max(dim=1, keepdim=True)[0]
        score = 0.5 * score1 + 0.5 * score2

    # abandon the branch, if you want to explore the local force fresh strategy, this may help.
    #if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')): # current['is_force_fresh'] is False, cause when it is True, no cut and fresh are needed
    #        #print(torch.ones_like(force_indices, dtype=float, device=force_indices.device).dtype)
    #    score.scatter_(dim=1, index=force_indices, src=torch.ones_like(force_indices, dtype=torch.float32, 
    #                                                                       device=force_indices.device))
    
    if (True and (cache_dic['force_fresh'] == 'global')):
        # apply s3 mentioned in the paper, the "True" above is for a switch to turn on/off the s3.
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])

        # layer wise s3, not used in the final version. seems it is not necessary to add if step wise is applied.
        #soft_layer_score = cache_dic['cache_index']['layer_index'][current['module']].float() / (27)
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score
    
    #cfg_score, no_cfg_score = torch.split(score, len(score)//2, dim = 0)
    #score = 0.5* cfg_score + 0.5* no_cfg_score
    #score = torch.cat([score,score], dim=0)

    return score.to(tokens.device)