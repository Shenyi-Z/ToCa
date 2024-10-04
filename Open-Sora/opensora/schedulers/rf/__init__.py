import torch
from tqdm import tqdm

from opensora.registry import SCHEDULERS
from calflops import calculate_flops
from .rectified_flow import RFlowScheduler, timestep_transform
from ...models.cache_functions import cache_init
import re

@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        
        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        #flops_cal=True,
    ):  
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        
        cache_dic_cal_flops, current_cal_flops = cache_init(model_kwargs=model_args, num_steps=self.num_sampling_steps)
        cache_dic, current = cache_init(model_kwargs=model_args, num_steps=self.num_sampling_steps)
        flops_sum = 0
        cal_flops = False
        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            current['step'] = i
            current_cal_flops['step'] = i
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            if cal_flops:
                flop_kwargs = model_args.copy()
                flop_kwargs['x'] = z_in.clone()
                flop_kwargs['timestep'] = t.clone()
                flop_kwargs['cache_dic'] = cache_dic_cal_flops
                flop_kwargs['current'] = current_cal_flops
                flops, macs, params = calculate_flops(model=model,
                                          kwargs = flop_kwargs,
                                          print_results=False)
                # 将字符串转换为浮点数
                #flops = float(re.findall(r"[-+]?\d*\.\d+|\d+", flops)[0])
                match = re.findall(r"([-+]?\d*\.\d+|\d+)\s*([GMTP]?)FLOPS", flops)
                flops_value = float(match[0][0])  # 提取数值部分
                unit = match[0][1]  # 提取量级部分，如 G 或 T
                if unit == 'G':
                    flops = flops_value * 0.001
                else:
                    flops = flops_value
                flops_sum += flops
                
            else:
                pred = model(z_in, t, cache_dic=cache_dic, current=current, **model_args).chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # update z
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]

                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
        if cal_flops:
            print("FLOPs:", flops_sum, "TFLOPs")
        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)
