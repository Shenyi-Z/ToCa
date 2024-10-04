import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_256_TEST, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--t5_path', default='../autodl-tmp/pretrained_models/t5_ckpts', type=str) # change to your t5 path
    parser.add_argument('--tokenizer_path', default='../autodl-tmp/pretrained_models/sd-vae-ft-ema', type=str) # change to your tokenizer path
    parser.add_argument('--txt_file', default='asset/samples.txt', type=str) # change to your txt prompt file
    parser.add_argument('--model_path', default='../autodl-tmp/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument("--fresh_ratio", type=float, default=0.30)
    parser.add_argument("--cache_type", type=str, choices=['random', 'attention', 'similarity', 'norm', 'compress'], default='attention')
    parser.add_argument("--ratio_scheduler", type=str, default='ToCa', choices=['linear', 'cosine', 'exp', 'constant', 'linear-mode', 'layerwise', 'ToCa'])
    parser.add_argument("--force_fresh", type=str, choices=['global', 'local'], default='global')
    parser.add_argument("--fresh_threshold", type=int, default=3)
    parser.add_argument("--soft_fresh_weight", type=float, default=0.25)
    return parser.parse_args()


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def set_env(seed=0, local_rank=None):
    global_seed = seed + local_rank
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    #torch.cuda.manual_seed_all(global_seed)
    torch.set_grad_enabled(False)
    return torch.device(f'cuda:{local_rank}')



@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale, device):
    sampler = DistributedSampler(items, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    data_loader = DataLoader(items, batch_size=bs, sampler=sampler, drop_last=False)
    
    pbar = tqdm(data_loader, unit='batch') if dist.get_rank() == 0 else data_loader
    for chunk in pbar:
        prompts = []
        if bs == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size


        null_y = model.module.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        with torch.no_grad():
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:, None]
            #print('finish embedding')

            if args.sampling_algo == 'iddpm':
                # we have not tested this part, there may bugsss.
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks,
                                    cache_type=args.cache_type,
                                    fresh_ratio=args.fresh_ratio,
                                    fresh_threshold=args.fresh_threshold,
                                    force_fresh=args.force_fresh,
                                    ratio_scheduler=args.ratio_scheduler,
                                    soft_fresh_weight=args.soft_fresh_weight)
                diffusion = IDDPM(str(sample_steps))
                samples = diffusion.p_sample_loop(
                    model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)

            elif args.sampling_algo == 'dpm-solver':
                # Main srategy, we have tested and make sure it works.
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks,
                                    cache_type=args.cache_type,
                                    fresh_ratio=args.fresh_ratio,
                                    fresh_threshold=args.fresh_threshold,
                                    force_fresh=args.force_fresh,
                                    ratio_scheduler=args.ratio_scheduler,
                                    soft_fresh_weight=args.soft_fresh_weight)
                dpm_solver = DPMS(model.module.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                    model_kwargs=model_kwargs,
                    rank = dist.get_rank()
                )
            # not supported now
            elif args.sampling_algo == 'sa-solver':
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks,
                                    cache_type=args.cache_type,
                                    fresh_ratio=args.fresh_ratio,
                                    fresh_threshold=args.fresh_threshold,
                                    force_fresh=args.force_fresh,
                                    ratio_scheduler=args.ratio_scheduler,
                                    soft_fresh_weight=args.soft_fresh_weight)
                sa_solver = SASolverSampler(model.module.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]

        samples = vae.decode(samples / 0.18215).sample
        torch.cuda.empty_cache()

        dist.barrier()
        #if dist.get_rank() == 0:
        os.umask(0o000)
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            #print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()
    
    # Setup DDP
    local_rank = setup_ddp()
    
    # Setup environment
    device = set_env(args.seed, local_rank)
    
    # only support fixed latent size currently
    latent_size = args.image_size // 8
    lewei_scale = {256: 1, 512: 1, 1024: 2}
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting
    if args.image_size in [256, 512]:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)

    print(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model.module.eval()
    model.module.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    work_dir = os.path.join(*args.model_path.split('/')[:-2])
    work_dir = f'/{work_dir}' if args.model_path[0] == '/' else work_dir

    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    epoch_name = re.search(r'.*epoch_(\d+).*.pth', args.model_path).group(1) if re.search(r'.*epoch_(\d+).*.pth', args.model_path) else 'unknown'
    step_name = re.search(r'.*step_(\d+).*.pth', args.model_path).group(1) if re.search(r'.*step_(\d+).*.pth', args.model_path) else 'unknown'
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{args.seed}")
    os.makedirs(save_root, exist_ok=True)

    visualize(items, args.bs, sample_steps, args.cfg_scale, device)
    
    cleanup_ddp()
