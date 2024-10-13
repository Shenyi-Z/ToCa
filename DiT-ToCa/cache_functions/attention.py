# Besides, re-arrange the attention module
from torch.jit import Final
from timm.layers import use_fused_attn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn()
        self.fused_attn = False
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cache_dic, current, fresh_indices=None) -> torch.Tensor:
    # 0.4ms extra cost on A800, mainly tensor operations
        """
        fresh_indices: (B, fresh_ratio*N), the index tensor for the fresh tokens
        """
        #timetick0 = torch.cuda.Event(enable_timing=True)
        #timetick1 = torch.cuda.Event(enable_timing=True)
        #timetick2 = torch.cuda.Event(enable_timing=True)
        #timetick3 = torch.cuda.Event(enable_timing=True)
        #timetick4 = torch.cuda.Event(enable_timing=True)
        #timetick5 = torch.cuda.Event(enable_timing=True)
        #timetick6 = torch.cuda.Event(enable_timing=True)
        #timetick7 = torch.cuda.Event(enable_timing=True)
        #timetick8 = torch.cuda.Event(enable_timing=True)
        ##timetick9 = torch.cuda.Event(enable_timing=True)
        ##timetick10 = torch.cuda.Event(enable_timing=True)
        ##timetick11 = torch.cuda.Event(enable_timing=True)
        ##endtick = torch.cuda.Event(enable_timing=True)

        B, N, C = x.shape
        
        if fresh_indices is not None:
            #timetick0.record()
            N= fresh_indices.shape[1]
            sorted_indices_tokens = fresh_indices.argsort(dim=-1, descending=False)
            x = torch.gather(input = x, dim = 1, index = sorted_indices_tokens.unsqueeze(-1).expand(-1, -1, x.shape[-1]) )  #(B, fresh_ratio*N, hidden_size)
            #timetick1.record()
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            #timetick2.record()
            q, k, v = qkv.unbind(0)   #q, k, v: (B, num_heads, fresh_ratio*N, head_dim)
            #timetick3.record()
            sorted_indices_qkv_expanded = fresh_indices.sort(dim=-1, descending=False)[0].unsqueeze(1).unsqueeze(-1).expand(-1, k.shape[1], -1, k.shape[-1])
            cache_dic['cache'][-1][current['layer']]['k'].scatter_(dim=2, index=sorted_indices_qkv_expanded, src=k)
            k =  cache_dic['cache'][-1][current['layer']]['k']

            cache_dic['cache'][-1][current['layer']]['v'].scatter_(dim=2, index=sorted_indices_qkv_expanded, src=v)
            v =  cache_dic['cache'][-1][current['layer']]['v']

            q, k = self.q_norm(q), self.k_norm(k)

        else:
            #timetick0.record()
            #timetick1.record()
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            #timetick2.record()
            q, k, v = qkv.unbind(0)   #q: (B, num_heads, N, head_dim)
            #timetick3.record()
            #cache_dic['cache'][-1][current['layer']]['k'] = k
            #cache_dic['cache'][-1][current['layer']]['v'] = v
            q, k = self.q_norm(q), self.k_norm(k)
        #timetick4.record()
        #q: (B, num_heads, N-M, head_dim), k: (B, num_heads, N, head_dim), v: (B, num_heads, N, head_dim)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_map= attn.softmax(dim=-1) #extra cost for attn
            attn = self.attn_drop(attn_map)
            x = attn @ v
        #timetick5.record()
        x = x.transpose(1, 2).reshape(B, N, C)
        attn_map = attn_map.mean(dim=1) #head mean
        x = self.proj(x)
        x = self.proj_drop(x) 
        #timetick6.record()
        #torch.cuda.synchronize()
        #print(f"total attn: {format(timetick0.elapsed_time(timetick6),'.2f')} ms. pre: {format(timetick0.elapsed_time(timetick1),'.2f')} ms. qkv: {format(timetick1.elapsed_time(timetick2),'.2f')} ms. qkv2qkv: {format(timetick2.elapsed_time(timetick3),'.2f')} ms. qk_norm_and_fresh: {format(timetick3.elapsed_time(timetick4),'.2f')} ms. attn: {format(timetick4.elapsed_time(timetick5),'.2f')} ms. proj: {format(timetick5.elapsed_time(timetick6),'.2f')} ms")
        return x, attn_map # x: (B, N-M, C), attn_map: (B, N-M, N)
