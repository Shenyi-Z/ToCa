import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, **kwargs) -> Tensor:
    
    cache_dic = kwargs.get('cache_dic', None)
    current = kwargs.get('current', None)     

    q, k = apply_rope(q, k, pe)
    
    if cache_dic is None:
        x, score = dot_product_attention(q, k, v)
        #x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    elif cache_dic['cache_type'] == 'attention':
        x, score = dot_product_attention(q, k, v)
        cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'] = score
    else:
        #x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x, score = dot_product_attention(q, k, v) # if you are testing the FLOPs, should change to dot_product_attention
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

############################################################################################################

import math

def dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor | torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    #attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.matmul(query, key.transpose(-2, -1))* scale_factor
    attn_weight += attn_bias
    
    #attn_weight = torch.softmax(attn_weight, dim=-1)
    #attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#
    #return torch.matmul(attn_weight, value)

    attn_map = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_map, dropout_p, train=True)
    #return attn_weight @ value, attn_map.mean(dim=1).mean(dim=1) 
    return torch.matmul(attn_weight, value), attn_map.mean(dim=1).mean(dim=1) 