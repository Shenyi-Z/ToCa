import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_score(cache_dic, current):
    '''
    Attention Score s1 (s2, but dit doesn't contain cross-attention for s2)
    '''
    #self_attn_score = 1- cache_dic['attn_map'][-1][current['layer']].diagonal(dim1=1, dim2=2)
    #self_attn_score = F.normalize(self_attn_score, dim=1, p=2)

    attention_score = F.normalize(cache_dic['attn_map'][-1][current['layer']].sum(dim=1), dim=1, p=2)

    #score = self_attn_score
    score = attention_score
    return score

def similarity_score(cache_dic, current, tokens):
    cosine_sim = F.cosine_similarity(tokens, cache_dic['cache'][-1][current['layer']][current['module']], dim=-1)

    return F.normalize(1- cosine_sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)

def kv_norm_score(cache_dic, current):
    # (B, num_heads, N)
    #k_norm = cache_dic['cache'][-1][current['layer']]['k_norm']
    v_norm = cache_dic['cache'][-1][current['layer']]['v_norm']
    kv_norm = 1- v_norm 


    return F.normalize(kv_norm.sum(dim = -2), p=2)