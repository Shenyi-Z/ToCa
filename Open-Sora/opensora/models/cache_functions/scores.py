import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_score(cache_dic, current):
    #self_attn_score = 1- cache_dic['attn_map'][current['flag']][current['layer']].diagonal(dim1=1, dim2=2)
    #self_attn_score = F.normalize(self_attn_score, dim=1, p=2)
    #attention_score = F.normalize(cache_dic['attn_map'][current['flag']][current['layer']].sum(dim=1), dim=1, p=2)
    #cross_attn_map = F.threshold(cache_dic['cross_attn_map'][current['flag']][current['layer']],threshold=0.0, value=0.0)
    #cross_attention_score = F.normalize(cross_attn_map.sum(dim=-1), dim=-1, p=2)
    
    cond_cmap, uncond_cmap = torch.split(cache_dic['cross_attn_map'][current['flag']][current['layer']], len(cache_dic['cross_attn_map'][current['flag']][current['layer']]) // 2, dim=0)
    cond_weight = 0.5
    cmap = cond_weight * cond_cmap + (1 - cond_weight) * uncond_cmap
    cross_attention_entropy = -torch.sum(cmap * torch.log(cmap + 1e-7), dim=-1)
    cross_attention_score   = F.normalize(1 + cross_attention_entropy, dim=1, p=2)
    #score = self_attn_score
    #score = attention_score
    score = cross_attention_score.repeat(2, 1)
    #cross_weight = 0.0
    #score =  (1-cross_weight) * attention_score + cross_weight * cross_attention_score
    return score

def similarity_score(cache_dic, current, tokens):
    cosine_sim = F.cosine_similarity(tokens, cache_dic['cache'][current['flag']][current['layer']][current['module']], dim=-1)

    return F.normalize(1- cosine_sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)
