import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_score(cache_dic, current):
    #self_attn_score = 1- cache_dic['attn_map'][-1][current['layer']].diagonal(dim1=1, dim2=2)
    #self_attn_score = F.normalize(self_attn_score, dim=1, p=2)
    #attention_score = F.normalize(cache_dic['attn_map'][-1][current['layer']].sum(dim=1), dim=1, p=2)
    #cross_attn_map = F.threshold(cache_dic['cross_attn_map'][-1][current['layer']],threshold=0.0, value=0.0)
    #cross_attention_score = F.normalize(cross_attn_map.sum(dim=-1), dim=-1, p=2)

    # Note: It is important to give a same selection method for cfg and no cfg.
    # Because the influence of **Cross-Attention** in text-contidional models makes cfg and no cfg a BIG difference.

    # Same selection for cfg and no cfg
    cond_cmap, uncond_cmap = torch.split(cache_dic['cross_attn_map'][-1][current['layer']], len(cache_dic['cross_attn_map'][-1][current['layer']]) // 2, dim=0)
    cond_weight = 0.5
    cmap = cond_weight * cond_cmap + (1 - cond_weight) * uncond_cmap

    # Entropy score
    cross_attention_entropy = -torch.sum(cmap * torch.log(cmap + 1e-7), dim=-1)
    cross_attention_score   = F.normalize(1 + cross_attention_entropy, dim=1, p=2) # Note here "1" does not influence the sorted sequence, but provie stability.
    score = cross_attention_score.repeat(2, 1)

    # In PixArt, the cross_attention_score (s2) is used as the score, for a better text-image alignment.

    # You can try conbining the self_attention_score (s1) and cross_attention_score (s2) as the final score, there exists a balance.
    #cross_weight = 0.0
    #score =  (1-cross_weight) * attention_score + cross_weight * cross_attention_score
    return score

def similarity_score(cache_dic, current, tokens):
    cosine_sim = F.cosine_similarity(tokens, cache_dic['cache'][-1][current['layer']][current['module']], dim=-1)

    return F.normalize(1- cosine_sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)
