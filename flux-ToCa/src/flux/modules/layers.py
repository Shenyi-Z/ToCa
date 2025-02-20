import math
from dataclasses import dataclass
from typing import Optional
import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope

from flux.modules.cache_functions import force_init, cache_cutfresh, update_cache

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        
        cache_dic = kwargs.get('cache_dic', None)
        current = kwargs.get('current', None)        
        
        if cache_dic is None:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

            # prepare image for attention
            img_modulated = self.img_norm1(img)
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

            # prepare txt for attention
            txt_modulated = self.txt_norm1(txt)
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

            # run actual attention
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)

            attn = attention(q, k, v, pe=pe)
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

            # calculate the img bloks
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

            # calculate the txt bloks
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        
        else:
            current['stream'] = 'double_stream'

            if current['type'] == 'full':    
                img_mod1, img_mod2 = self.img_mod(vec)
                txt_mod1, txt_mod2 = self.txt_mod(vec)

                # prepare image for attention
                img_modulated = self.img_norm1(img)
                img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
                img_qkv = self.img_attn.qkv(img_modulated)
                img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
                
                if cache_dic['cache_type'] == 'k-norm':
                    img_k_norm = img_k.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['k-norm'][-1][current['stream']][current['layer']]['img_mlp'] = img_k_norm
                elif cache_dic['cache_type'] == 'v-norm':
                    img_v_norm = img_v.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['v-norm'][-1][current['stream']][current['layer']]['img_mlp'] = img_v_norm
                
                img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

                # prepare txt for attention
                txt_modulated = self.txt_norm1(txt)
                txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
                txt_qkv = self.txt_attn.qkv(txt_modulated)
                txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

                if cache_dic['cache_type'] == 'k-norm':
                    txt_k_norm = txt_k.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['k-norm'][-1][current['stream']][current['layer']]['txt_mlp'] = txt_k_norm
                elif cache_dic['cache_type'] == 'v-norm':
                    txt_v_norm = txt_v.norm(dim=-1, p=2).mean(dim=1)
                    cache_dic['v-norm'][-1][current['stream']][current['layer']]['txt_mlp'] = txt_v_norm
                
                txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

                # run actual attention
                q = torch.cat((txt_q, img_q), dim=2)
                k = torch.cat((txt_k, img_k), dim=2)
                v = torch.cat((txt_v, img_v), dim=2)

                attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)
                cache_dic['cache'][-1]['double_stream'][current['layer']]['attn'] = attn

                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
                cache_dic['txt_shape'] = txt.shape[1]
                
                if cache_dic['cache_type'] == 'attention':
                    cache_dic['attn_map'][-1][current['stream']][current['layer']]['txt_mlp'] = cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'][:, : txt.shape[1]]
                    cache_dic['attn_map'][-1][current['stream']][current['layer']]['img_mlp'] = cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'][:, txt.shape[1] :]

                current['module'] = 'img_mlp'
                force_init(cache_dic=cache_dic, current=current, tokens=img)
                # calculate the img bloks
                img = img + img_mod1.gate * self.img_attn.proj(img_attn)
                cache_dic['cache'][-1]['double_stream'][current['layer']]['img_mlp'] = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
                img = img + img_mod2.gate * cache_dic['cache'][-1]['double_stream'][current['layer']]['img_mlp']

                current['module'] = 'txt_mlp'
                force_init(cache_dic=cache_dic, current=current, tokens=txt)
                # calculate the txt bloks
                txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
                cache_dic['cache'][-1]['double_stream'][current['layer']]['txt_mlp'] = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
                txt = txt + txt_mod2.gate * cache_dic['cache'][-1]['double_stream'][current['layer']]['txt_mlp']

            elif current['type'] == 'ToCa':
                img_mod1, img_mod2 = self.img_mod(vec)
                txt_mod1, txt_mod2 = self.txt_mod(vec)

                attn = cache_dic['cache'][-1]['double_stream'][current['layer']]['attn']
                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

                current['module'] = 'img_mlp'
                # calculate the img bloks
                img = img + img_mod1.gate * self.img_attn.proj(img_attn)
                fresh_indices, fresh_tokens_img = cache_cutfresh(cache_dic=cache_dic, tokens=img, current=current)
                fresh_tokens_img = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(fresh_tokens_img) + img_mod2.shift)
                update_cache(fresh_indices=fresh_indices, fresh_tokens=fresh_tokens_img, cache_dic=cache_dic, current=current)
                cache_dic['cache'][-1]['double_stream'][current['layer']]['img_mlp']
                img = img + img_mod2.gate * cache_dic['cache'][-1]['double_stream'][current['layer']]['img_mlp']

                current['module'] = 'txt_mlp'
                # calculate the txt bloks
                txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
                fresh_indices, fresh_tokens_txt = cache_cutfresh(cache_dic=cache_dic, tokens=txt, current=current)
                fresh_tokens_txt = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(fresh_tokens_txt) + txt_mod2.shift)
                update_cache(fresh_indices=fresh_indices, fresh_tokens=fresh_tokens_txt, cache_dic=cache_dic, current=current)
                txt = txt + txt_mod2.gate * cache_dic['cache'][-1]['double_stream'][current['layer']]['txt_mlp']
            
            elif current['type'] == 'FORA':
                img_mod1, img_mod2 = self.img_mod(vec)
                txt_mod1, txt_mod2 = self.txt_mod(vec)
                img = img + img_mod2.gate * cache_dic['cache'][-1]['double_stream'][current['layer']]['img_mlp']
                txt = txt + txt_mod2.gate * cache_dic['cache'][-1]['double_stream'][current['layer']]['txt_mlp']
            elif current['type'] == 'aggressive':
                current['module'] = 'skipped'
            else:
                raise ValueError("Unknown cache type.")
            
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        # mlp_in
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim)

    def load_mlp_in_weights(self, linear1_weight: torch.Tensor, linear1_bias: Optional[torch.Tensor] = None):
        """
        Split and load the weights of the original `linear1` layer, keeping only the MLP hidden layer part.

        Parameters:
          - linear1_weight: Tensor, with shape (hidden_size * 3 + mlp_hidden_dim, hidden_size)
          - linear1_bias: Tensor, with shape (hidden_size * 3 + mlp_hidden_dim,) or None

        """
        hidden_size = self.hidden_size
        mlp_hidden_dim = self.mlp_hidden_dim
        device = self.linear1.weight.device  # target device

        self.mlp_in.weight = torch.nn.Parameter(linear1_weight[hidden_size * 3:, :].to(device))

        if linear1_bias is not None:

            self.mlp_in.bias = torch.nn.Parameter(linear1_bias[hidden_size * 3:].to(device))

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:

        cache_dic = kwargs.get('cache_dic', None)
        current = kwargs.get('current', None)

        mod, _ = self.modulation(vec)
        
        if cache_dic is None:
            x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k = self.norm(q, k, v)

            # compute attention
            attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)
            # compute activation in mlp stream, cat again and run second linear layer
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        
        else:
            current['stream'] = 'single_stream'

            if current['type'] == 'full':
                #if (current['layer'] == 0):
                #    print(current['step'])
                x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
                qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
                cache_dic['cache'][-1]['single_stream'][current['layer']]['mlp'] = mlp
                current['module'] = 'attn'
                q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

                if cache_dic['cache_type'] == 'k-norm':
                    cache_dic['k-norm'][-1][current['stream']][current['layer']]['total'] = k.norm(dim=-1, p=2).mean(dim=1)
                elif cache_dic['cache_type'] == 'v-norm':
                    cache_dic['v-norm'][-1][current['stream']][current['layer']]['total'] = v.norm(dim=-1, p=2).mean(dim=1)
                
                q, k = self.norm(q, k, v)

                # compute attention
                attn = attention(q, k, v, pe=pe, cache_dic=cache_dic, current=current)
                force_init(cache_dic=cache_dic, current=current, tokens=attn)
                cache_dic['cache'][-1]['single_stream'][current['layer']]['attn'] = attn
                # compute activation in mlp stream, cat again and run second linear layer
                current['module'] = 'mlp'
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                force_init(cache_dic=cache_dic, current=current, tokens=output)
                current['module'] = 'total'
                cache_dic['cache'][-1]['single_stream'][current['layer']]['total'] = output

            elif current['type'] == 'ToCa':
                self.load_mlp_in_weights(self.linear1.weight, self.linear1.bias)
                current['module'] = 'mlp'
                fresh_indices, fresh_tokens_mlp = cache_cutfresh(cache_dic=cache_dic, tokens=x, current=current)
                x_mod = (1 + mod.scale) * self.pre_norm(fresh_tokens_mlp) + mod.shift
                #cache_dic['cache'][-1]['single_stream'][current['layer']]['mlp']
                mlp_fresh = self.mlp_in(x_mod)
                #_, mlp_fresh1 = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
                update_cache(fresh_indices=fresh_indices, fresh_tokens=mlp_fresh, cache_dic=cache_dic, current=current)
                # compute attention
                fake_fresh_attn = torch.gather(input = cache_dic['cache'][-1]['single_stream'][current['layer']]['attn'], dim = 1, 
                                               index = fresh_indices.unsqueeze(-1).expand(-1, -1, cache_dic['cache'][-1]['single_stream'][current['layer']]['attn'].shape[-1]))
                
                current['module'] = 'total'
                fresh_tokens_output = self.linear2(torch.cat((fake_fresh_attn, self.mlp_act(mlp_fresh)), 2))
                update_cache(fresh_indices=fresh_indices, fresh_tokens=fresh_tokens_output, cache_dic=cache_dic, current=current)
                #attn = cache_dic['cache'][-1]['single_stream'][current['layer']]['attn']
                #mlp  = cache_dic['cache'][-1]['single_stream'][current['layer']]['mlp']
                # compute activation in mlp stream, cat again and run second linear layer
                #output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                output = cache_dic['cache'][-1]['single_stream'][current['layer']]['total']
            
            elif current['type'] == 'FORA':
                output = cache_dic['cache'][-1]['single_stream'][current['layer']]['total']
                
            elif current['type'] == 'aggressive':
                current['module'] = 'skipped'
                if current['layer'] == 37:
                    x = cache_dic['cache'][-1]['aggressive_output']
                return x
            else:
                raise ValueError("Unknown cache type.")
            
            if current['layer'] == 37:
                cache_dic['cache'][-1]['aggressive_output'] = x
            
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
