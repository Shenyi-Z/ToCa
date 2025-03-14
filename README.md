<div align=center>
  
# **[ICLR 2025]** *ToCa*: Accelerating Diffusion Transformers with *To*ken-wise Feature *Ca*ching

<p>
<a href='https://arxiv.org/abs/2410.05317'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://toca2024.github.io/ToCa/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>

## 🔥 News

* `2025/03/10` 🚀🚀 Our latest work "From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers" is released! Codes are available at [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)! TaylorSeer supports lossless compression at a rate of 4.99x on FLUX.1-dev (with a latency speedup of 3.53x) and high-quality acceleration at a compression rate of 5.00x on HunyuanVideo (with a latency speedup of 4.65x)! We hope *TaylorSeer* can move the paradigm of feature caching methods from reusing to forecasting.For more details, please refer to our latest research [paper](https://arxiv.org/abs/2503.06923).
* `2025/02/19` 🚀🚀 ToCa solution for **FLUX** has been officially released after adjustments, now achieving up to **3.14× lossless acceleration**!
* `2025/01/22` 💥💥 ToCa is honored to be accepted by ICLR 2025!
* `2024/12/29` 🚀🚀 We release our work [DuCa](https://arxiv.org/abs/2412.18911) about accelerating diffusion transformers for FREE, which achieves nearly lossless acceleration of **2.50×** on [OpenSora](https://github.com/hpcaitech/Open-Sora)! 🎉 **DuCa also overcomes the limitation of ToCa by fully supporting FlashAttention, enabling broader compatibility and efficiency improvements.**
* `2024/12/24` 🤗🤗 We release an open-sourse repo "[Awesome-Token-Reduction-for-Model-Compression](https://github.com/xuyang-liu16/Awesome-Token-Reduction-for-Model-Compression)", which collects recent awesome token reduction papers! Feel free to contribute your suggestions!
* `2024/12/20` 💥💥 Our ToCa has achieved nearly lossless acceleration of **1.51×** on [FLUX](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell), feel free to check the latest version of our [paper](https://arxiv.org/pdf/2410.05317#page=19)!
* `2024/12/10` 💥💥 Our team's recent work, **SiTo** (https://github.com/EvelynZhang-epiclab/SiTo), has been accepted to **AAAI 2025**. It accelerates diffusion models through adaptive **Token Pruning**.
* `2024/10/16` 🤗🤗 Users with autodl accounts can now quickly experience [OpenSora-ToCa](https://www.codewithgpu.com/i/Shenyi-Z/ToCa/OpenSora-ToCa) by directly using our publicly available image!
* `2024/10/12` 🚀🚀 We release our work [ToCa](https://arxiv.org/abs/2410.05317) about accelerating diffusion transformers for FREE, which achieves nearly lossless acceleration of **2.36×** on [OpenSora](https://github.com/hpcaitech/Open-Sora)!
* `2024/07/15` 🤗🤗 We release an open-sourse repo "[Awesome-Generation-Acceleration](https://github.com/xuyang-liu16/Awesome-Generation-Acceleration)", which collects recent awesome generation accleration papers! Feel free to contribute your suggestions!

## TODO:

- [x] Support for FLOPs calculation
- [x] Add the FLUX version of ToCa
- [ ] Further optimize the code logic to reduce the time consumption of tensor operations


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

| Original  Models |                     urls                     |
| :--------------: | :------------------------------------------: |
|       DiT        |   https://github.com/facebookresearch/DiT    |
|     PixArt-α     | https://github.com/PixArt-alpha/PixArt-alpha |
|     OpenSora     |    https://github.com/hpcaitech/Open-Sora    |
|       FLUX       |  https://github.com/black-forest-labs/flux   |

Besides, we provide a replica for our environment here:

<details>
<summary>From our environment.yaml</summary>

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

</details>

## 🚀 Run and evaluation

### Run DiT-ToCa

#### DDPM-250 Steps

sample images for **visualization**

```bash
cd DiT-ToCa
python sample.py --image-size 256 --num-sampling-steps 250 --cache-type attention --fresh-threshold 4 --fresh-ratio 0.07 --ratio-scheduler ToCa-ddpm250  --force-fresh global --soft-fresh-weight 0.25
```

sample images for **evaluation** (e.g 50k)

```bash
cd DiT-ToCa
torchrun --nnodes=1 --nproc_per_node=6 sample_ddp.py --model DiT-XL/2 --per-proc-batch-size 150 --image-size 256 --cfg-scale 1.5 --num-sampling-steps 250 --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa-ddpm250 --force-fresh global --fresh-threshold 4 --soft-fresh-weight 0.25 --num-fid-samples 50000
```

#### DDIM-50 Steps

sample images for **visualization**

```bash
cd DiT-ToCa
python sample.py --image-size 256 --num-sampling-steps 50 --cache-type attention --fresh-threshold 3 --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50  --force-fresh global --soft-fresh-weight 0.25 --ddim-sample
```

sample images for **evaluation** (e.g 50k)

```bash
cd DiT-ToCa
torchrun --nnodes=1 --nproc_per_node=6 sample_ddp.py --model DiT-XL/2 --per-proc-batch-size 150 --image-size 256 --cfg-scale 1.5 --num-sampling-steps 50 --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50 --force-fresh global --fresh-threshold 3 --soft-fresh-weight 0.25 --num-fid-samples 50000 --ddim-sample
```

#### test FLOPs

Just add --test-FLOPs, here an example: 

```bash
cd DiT-ToCa
python sample.py --image-size 256 --num-sampling-steps 50 --cache-type attention --fresh-threshold 3 --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50  --force-fresh global --soft-fresh-weight 0.25 --ddim-sample --test-FLOPs
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

sample video for **visualization**

```bash
cd Open-Sora
python scripts/inference.py configs/opensora-v1-2/inference/sample.py   --num-frames 2s --resolution 480p --aspect-ratio 9:16   --prompt "a beautiful waterfall"
```

sample video for **VBench evaluation**

```bash
cd Open-Sora
bash eval/vbench/launch.sh /root/autodl-tmp/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors 51 opensora-ToCa 480p 9:16
```

(remember replacing  "/root/autodl-tmp/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors" with your own path!)

### Run FLUX-ToCa

First, you need to enter the environment adapted for FLUX. While the official documentation uses `venv` to build the environment, you can also set it up using `conda`, which you might be more familiar with.

<details>
<summary>How to build a conda environment for FLUX?</summary>

```bash
cd flux-ToCa
conda create -n flux python=3.10
pip install -e ".[all]"
```

</details>

For interactive sampling run

```bash
python -m flux --name <name> --loop
```

Or to generate a single sample run

```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

Typically, `<name>` should be set to `flux-dev`.

Generate image samples with a txt file

```bash
python src/sample.py --prompt_file </path/to/your/prompt.txt> --width 1024 --height 1024 --model_name flux-dev --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> --num_steps 50
```

The `--add_sampling_metadata` parameter is used to control whether the prompt is added to the image's EXIF metadata.
We also provide function for FLOPs testing, but **in this mode, no generated samples are given**.

```bash
python src/sample.py --prompt_file </path/to/your/test/prompt.txt> --width 1024 --height 1024 --model_name flux-dev --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> --num_steps 50 --test_FLOPs
```

Use the framework of Geneval for evaluation


```bash
python src/geneval_flux.py /root/geneval/prompts/evaluation_metadata.jsonl --model_name flux-dev --n_samples 4 --steps 50 --width 1024 --height 1024 --seed 42 --output_dir /root/autodl-tmp/samples/flux-ToCa
```

<details>
<summary>How to prepare environment for geneval?</summary>

The environment required for Geneval's metric computation is somewhat specific. As of February 2025, it is not yet possible to set up the environment directly using the default method provided in the project. However, we can follow the guidance in this Geneval issue [https://github.com/djghosh13/geneval/issues/12](https://github.com/djghosh13/geneval/issues/12) to set up the environment. The instructions are very detailed.

</details>

#### Awesome acceleration results for the Latest Version of ToCa on FLUX


| Method       | Geneval $\uparrow$<br />overall score | ImageRewrd $\uparrow$<br />DrawBench200 | FLOPs $\downarrow$ | Latency $\downarrow$ | Compress Ratio $\uparrow$ | Speed Up $\uparrow$ |
| ------------ | :-----------------------------------: | :-------------------------------------: | :----------------: | :------------------: | :-----------------------: | :-----------------: |
| **original** |                0.6752                 |                 0.9898                  |      3719.50       |        33.87s        |           1.00            |        1.00         |
| 60% steps    |                0.6700                 |                 0.9739                  |      2231.70       |        20.49s        |           1.67            |        1.65         |
| 50% steps    |                0.6656                 |                 0.9429                  |      1859.75       |        17.12s        |           2.00            |        1.98         |
| 40% steps    |                0.6606                 |                 0.9317                  |      1487.80       |        13.77s        |           2.62            |        2.45         |
| **FORA3**    |                0.6594                 |                 0.9227                  |      1320.07       |        12.98s        |           2.82            |        2.61         |
| **ToCa4-01** |                0.6748                 |               **0.9798**                |      1263.22       |        11.91s        |           2.94            |        2.84         |
| **ToCa5-01** |              **0.6750**               |                 0.9731                  |      1126.76       |        10.80s        |           3.30            |        3.14         |
| **ToCa6-01** |                0.6653                 |                 0.9493                  |       990.30       |        9.48s         |           3.76            |        3.57         |


<details>
<summary>Explanation of the Improved ToCa</summary>

The **acceleration effect has significantly improved while maintaining generation quality** compared with the previous version. This is because, in the current version of the code, we have further optimized ToCa and adopted more reliable metrics (Image Reward on DrawBench200, Geneval).

</details>

## 👍 Acknowledgements

- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their great work and codebase upon which we build DiT-ToCa.
- Thanks to [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) for their great work and codebase upon which we build PixArt-α-ToCa.
- Thanks to [OpenSora](https://github.com/hpcaitech/Open-Sora) for their great work and codebase upon which we build OpenSora-ToCa.
- Thanks to [FLUX](https://github.com/black-forest-labs/flux) for their great work and codebase upon which we build FLUX-ToCa.

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
