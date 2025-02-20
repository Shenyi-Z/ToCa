import torch
from ..model import Flux
from .split_utils import split, merge
from torch import Tensor
from ..modules.cache_functions import cache_init

def denoise_cache(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):  
    # init cache
    cache_dic, current = cache_init(timesteps)
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current['step']=0
    current['num_steps'] = len(timesteps)-1
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        current['t'] = t_curr
        #print(t_curr)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic = cache_dic,
            current = current,
            guidance=guidance_vec,
        )
        #print(img.shape)
        img = img + (t_prev - t_curr) * pred
        current['step'] += 1

    return img
