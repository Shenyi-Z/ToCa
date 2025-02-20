import argparse
import json
import os

import torch
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

# --- Imports related to FLUX module ---
from flux.sampling import (
    denoise_test_FLOPs,
    get_noise,
    get_schedule,
    prepare,
    unpack,
)
from flux.ideas import denoise_cache
from flux.util import (
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from transformers import pipeline

# NSFW threshold (adjustable as needed)
NSFW_THRESHOLD = 0.85


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using the FLUX model within the Geneval framework")
    # Required: input JSONL metadata file, each line must contain at least the "prompt" key
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing metadata for each prompt, each line is a JSON object"
    )
    # FLUX model related parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell"],
        help="FLUX model name"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of sampling steps (if not specified: 4 for flux-schnell, 50 for flux-dev)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1360,
        help="Width of the generated image (pixels)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Height of the generated image (pixels)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Conditional guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples per batch during image generation"
    )
    # Output related parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory to save the generated results"
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip saving the overall grid image"
    )
    # Other options
    parser.add_argument(
        "--add_sampling_metadata",
        action="store_true",
        help="Add the prompt text to the metadata of the generated images"
    )
    parser.add_argument(
        "--use_nsfw_filter",
        action="store_true",
        help="Enable NSFW content filtering (requires downloading the relevant model)"
    )
    parser.add_argument(
        "--test_FLOPs",
        action="store_true",
        help="Test inference FLOPs only (no images will be generated)"
    )
    return parser.parse_args()


def main(args):
    # Read the metadata file, each line is a JSON object (must contain at least the "prompt" field)
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp if line.strip()]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If NSFW filtering is enabled, load the corresponding classifier (please modify the model path or name accordingly)
    if args.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="/path/to/your/nsfw_model",  # Please replace with the actual NSFW model path
            device=0 if torch.cuda.is_available() else -1
        )
    else:
        nsfw_classifier = None

    # If sampling steps are not specified, set default steps based on the model name
    if args.steps is None:
        args.steps = 4 if args.model_name == "flux-schnell" else 50

    # Ensure the image width and height are multiples of 16 (required by FLUX)
    args.width = 16 * (args.width // 16)
    args.height = 16 * (args.height // 16)

    # Load FLUX model components onto the device (T5, CLIP, Flow model, autoencoder)
    t5 = load_t5(device, max_length=256 if args.model_name == "flux-schnell" else 512)
    clip = load_clip(device)
    model = load_flow_model(args.model_name, device=device)
    ae = load_ae(args.model_name, device=device)

    # Generate results for each prompt:
    # Each prompt corresponds to a subfolder (e.g., outputs/00000/), inside which samples and (optionally) a grid image grid.png are saved,
    # along with the prompt's metadata saved in a metadata.jsonl file.
    for idx, metadata in enumerate(metadatas):
        prompt = metadata.get("prompt", "")
        print(f"Processing prompt {idx + 1}/{len(metadatas)}: '{prompt}'")

        # Define output directory and samples directory
        outpath = os.path.join(args.output_dir, f"{idx:05d}")
        sample_path = os.path.join(outpath, "samples")

        # If the output directory already exists, check the number of PNG files already in the samples folder
        existing_samples = []
        sample_count = 0
        if os.path.exists(sample_path):
            files = sorted(
                fname for fname in os.listdir(sample_path)
                if fname.endswith(".png") and fname != "grid.png"
            )
            sample_count = len(files)
            # Load existing images (to be used later for generating the grid image)
            for fname in files:
                full_path = os.path.join(sample_path, fname)
                try:
                    img = Image.open(full_path).convert("RGB")
                    existing_samples.append(ToTensor()(img))
                except Exception as e:
                    print(f"Failed to read existing image {full_path}: {e}")

        # If the number of generated images is sufficient, skip generation
        if sample_count >= args.n_samples:
            print(f"Samples for prompt {idx + 1} already exist ({sample_count} images), skipping generation.")
            continue

        # Create output directory and samples subdirectory
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)
        # Save the current prompt's metadata to metadata.jsonl
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        # Initialize: use the number of existing images as the starting count, and copy existing samples for later grid generation
        local_index = sample_count
        all_samples = existing_samples.copy()
        # The initial value of the progress bar is the number of existing samples
        pbar = tqdm(total=args.n_samples, initial=sample_count, desc="Sampling")

        # For the current prompt, only generate the missing images
        while local_index < args.n_samples:
            current_bs = min(args.batch_size, args.n_samples - local_index)
            # Set seed for the current batch (using the number of images already present in the prompt as offset)
            seed = args.seed + local_index
            # Generate random noise
            x = get_noise(current_bs, args.height, args.width, device=device, dtype=torch.bfloat16, seed=seed)
            prompt_list = [prompt] * current_bs
            # Prepare input (prompt encoding, initial image noise, etc.)
            inp = prepare(t5, clip, x, prompt=prompt_list)
            # Compute denoising schedule based on the input shape (note: the second parameter is the number of latent channels)
            timesteps = get_schedule(args.steps, inp["img"].shape[1], shift=(args.model_name != "flux-schnell"))

            with torch.no_grad():
                if args.test_FLOPs:
                    latent = denoise_test_FLOPs(model, **inp, timesteps=timesteps, guidance=args.guidance)
                else:
                    latent = denoise_cache(model, **inp, timesteps=timesteps, guidance=args.guidance)
                # Unpack latent to a shape suitable for the decoder input
                latent = unpack(latent.float(), args.height, args.width)
                # Decode to image with automatic mixed precision
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    decoded = ae.decode(latent)

            # Post-processing: clamp, embed watermark, and rearrange to [B, H, W, C] format
            decoded = decoded.clamp(-1, 1)
            decoded = embed_watermark(decoded.float())
            images_tensor = rearrange(decoded, "b c h w -> b h w c")

            # Iterate over each generated image in the current batch
            for i in range(current_bs):
                img_array = (127.5 * (images_tensor[i] + 1.0)).cpu().numpy().astype(np.uint8)
                img = Image.fromarray(img_array)
                # NSFW filtering (if enabled)
                if nsfw_classifier is not None:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                else:
                    nsfw_score = 0.0

                if nsfw_score < NSFW_THRESHOLD:
                    # Add sampling metadata (EXIF info); note: PNG format may not fully support EXIF
                    if args.add_sampling_metadata:
                        exif_data = Image.Exif()
                        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                        exif_data[ExifTags.Base.Model] = args.model_name
                        exif_data[ExifTags.Base.ImageDescription] = prompt
                    else:
                        exif_data = None

                    sample_fname = os.path.join(sample_path, f"{local_index:05d}.png")
                    if exif_data is not None:
                        img.save(sample_fname, exif=exif_data)
                    else:
                        img.save(sample_fname)
                    all_samples.append(ToTensor()(img))
                else:
                    print("The generated image may contain inappropriate content and has been skipped.")
                local_index += 1
                pbar.update(1)
            # end for current batch
        pbar.close()

        # If grid generation is not skipped and there is at least one sample, create and save a grid image (consistent with Geneval format)
        if not args.skip_grid and len(all_samples) > 0:
            grid_tensor = torch.stack(all_samples, 0)
            grid = make_grid(grid_tensor, nrow=args.batch_size)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            grid_img = Image.fromarray(grid.astype(np.uint8))
            grid_img.save(os.path.join(outpath, "grid.png"))
    # end for each prompt

    print("Generation completed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

'''
python src/geneval_flux.py /root/geneval/prompts/evaluation_metadata.jsonl --model_name flux-dev --n_samples 4 --steps 50 --width 1024 --height 1024 --seed 42 --output_dir /root/autodl-tmp/samples/geneval_original --batch_size 1
'''
