<div align=center>
  
# **[ICLR 2025]** *ToCa*: Accelerating Diffusion Transformers with *To*ken-wise Feature *Ca*ching

<p>
<a href='https://arxiv.org/abs/2410.05317'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://toca2024.github.io/ToCa/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>

## 🔥 News

* `2025/01/22` 💥💥 ToCa is honored to be accepted by ICLR 2025!
* `2024/12/29` 🚀🚀 We release our work [DuCa](https://arxiv.org/abs/2412.18911) about accelerating diffusion transformers for FREE, which achieves nearly lossless acceleration of **2.50×** on [OpenSora](https://github.com/hpcaitech/Open-Sora)! 🎉 **DuCa also overcomes the limitation of ToCa by fully supporting FlashAttention, enabling broader compatibility and efficiency improvements.**
* `2024/12/24` 🤗🤗 We release an open-sourse repo "[Awesome-Token-Reduction-for-Model-Compression](https://github.com/xuyang-liu16/Awesome-Token-Reduction-for-Model-Compression)", which collects recent awesome token reduction papers! Feel free to contribute your suggestions!
* `2024/12/20` 💥💥 Our ToCa has achieved nearly lossless acceleration of **1.51×** on [FLUX](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell), feel free to check the latest version of our [paper](https://arxiv.org/pdf/2410.05317#page=19)!
* `2024/10/16` 🤗🤗 Users with autodl accounts can now quickly experience [OpenSora-ToCa](https://www.codewithgpu.com/i/Shenyi-Z/ToCa/OpenSora-ToCa) by directly using our publicly available image!
* `2024/10/12` 🚀🚀 We release our work [ToCa](https://arxiv.org/abs/2410.05317) about accelerating diffusion transformers for FREE, which achieves nearly lossless acceleration of **2.36×** on [OpenSora](https://github.com/hpcaitech/Open-Sora)!
* `2024/07/15` 🤗🤗 We release an open-sourse repo "[Awesome-Generation-Acceleration](https://github.com/xuyang-liu16/Awesome-Generation-Acceleration)", which collects recent awesome generation accleration papers! Feel free to contribute your suggestions!

##  Dependencies
``` cmd
Python>=3.9
CUDA>=11.8
```

## 🛠 Installation

``` cmd
git clone https://github.com/Shenyi-Z/ToCa.git
```

### Environment Settings

#### Original Models (recommended)

We evaluated our model under the same environments as the original models.
So you may set the environments through following the requirements of the mentioned original models.

Links: 

| Original  Models  |                     urls                     |
| :---------------: | :------------------------------------------: |
|        DiT        |   https://github.com/facebookresearch/DiT    |
|     PixArt-α      | https://github.com/PixArt-alpha/PixArt-alpha |
|     OpenSora      |    https://github.com/hpcaitech/Open-Sora    |


#### From our environment.yaml

Besides, we provide a replica for our environment here

##### DiT

```bash
cd DiT-ToCa
conda env create -f environment-dit.yml
```

##### PixArt-α

```bash
cd PixArt-alpha-ToCa
conda env create -f environment-pixart.yml
```

##### OpenSora

```bash
cd Open-Sora
conda env create -f environment-opensora.yml
pip install -v . # for development mode, `pip install -v -e .`
```

## 🚀 Run and evaluation

### Run DiT-ToCa

sample images for **visualization**

```bash
cd DiT-ToCa
python sample.py --image-size 256 --num-sampling-steps 250 --cache-type attention --fresh-threshold 4 --fresh-ratio 0.07 --ratio-scheduler ToCa  --force-fresh global --soft-fresh-weight 0.25
```

sample images for **evaluation** (e.g 50k)

```bash
cd DiT-ToCa
torchrun --nnodes=1 --nproc_per_node=6 sample_ddp.py --model DiT-XL/2 --per-proc-batch-size 150 --image-size 256 --cfg-scale 1.5 --num-sampling-steps 250 --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa --force-fresh global --fresh-threshold 4 --soft-fresh-weight 0.25 --num-fid-samples 50000
```

### Run PixArt-α-ToCa

sample images for **visualization**

```bash
cd PixArt-alpha-ToCa
python scripts/inference.py --model_path /root/autodl-tmp/pretrained_models/PixArt-XL-2-256x256.pth --image_size 256 --bs 100 --txt_file /root/autodl-tmp/test.txt --fresh_threshold 3 --fresh_ratio 0.30 --cache_type attention --force_fresh global --soft_fresh_weight 0.25 --ratio_scheduler ToCa
```

sample images for **evaluation** (e.g 30k for COCO, 1.6k for PartiPrompts)

```bash
cd PixArt-alpha-ToCa
torchrun --nproc_per_node=6 scripts/inference_ddp.py --model_path /root/autodl-tmp/pretrained_models/PixArt-XL-2-256x256.pth --image_size 256 --bs 100 --txt_file /root/autodl-tmp/COCO/COCO_caption_prompts_30k.txt --fresh_threshold 3 --fresh_ratio 0.30 --cache_type attention --force_fresh global --soft_fresh_weight 0.25 --ratio_scheduler ToCa
```

（Besides, if you need our npz file: https://drive.google.com/file/d/1vUdoSgdIvtXo1cAS_aOFCJ1-XC_i1KEQ/view?usp=sharing)

### Run OpenSora-ToCa

sample video for **visualizaiton**

```bash
cd Open-Sora
python scripts/inference.py configs/opensora-v1-2/inference/sample.py   --num-frames 2s --resolution 480p --aspect-ratio 9:16   --prompt "a beautiful waterfall"
```

sample video for **VBench evaluation**

```bash
cd Open-Sora
bash eval/vbench/launch.sh /root/autodl-tmp/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors 51 opensora-ToCa 480p 9:16
```

( remember replacing  "/root/autodl-tmp/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors" with your own path!)

## 👍 Acknowledgements
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their great work and codebase upon which we build DiT-ToCa.
- Thanks to [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) for their great work and codebase upon which we build PixArt-α-ToCa.
- Thanks to [OpenSora](https://github.com/hpcaitech/Open-Sora) for their great work and codebase upon which we build OpenSora-ToCa.

## 📌 Citation
```bibtex
@article{zou2024accelerating,
  title={Accelerating Diffusion Transformers with Token-wise Feature Caching},
  author={Zou, Chang and Liu, Xuyang and Liu, Ting and Huang, Siteng and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2410.05317},
  year={2024}
}
```

## :e-mail: Contact
If you have any questions, please email [`shenyizou@outlook.com`](mailto:shenyizou@outlook.com).
