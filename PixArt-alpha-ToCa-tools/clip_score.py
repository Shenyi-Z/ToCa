import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
import torch.multiprocessing as mp

# Load prompts file
def load_prompts(txt_file):
    with open(txt_file, "r") as f:
        prompts = f.read().splitlines()
    return prompts

# Find matching image file: first, directly use the prompt as the filename, 
# and if not found, match using a prefix
def find_image_file(image_folder, prompt):
    img_filename = prompt + ".jpg"  # Assume filename is {prompt}.jpg
    img_path = os.path.join(image_folder, img_filename)
    
    if os.path.exists(img_path):
        return img_path

    # If direct match fails, use prefix matching
    for file in os.listdir(image_folder):
        if file.startswith(prompt[:20]):  # Use the first 20 characters as a prefix for matching
            return os.path.join(image_folder, file)

    return None

# Load a batch of images and convert them to Tensors
def load_images(image_folder, prompts_batch):
    images = []
    valid_prompts = []
    
    for prompt in prompts_batch:
        img_path = find_image_file(image_folder, prompt)
        
        if img_path:
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = ToTensor()(image).unsqueeze(0)  # Shape (1, C, H, W)
                images.append(image_tensor)
                valid_prompts.append(prompt)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        else:
            print(f"No image found for prompt: {prompt}")
    
    if len(images) > 0:
        images_tensor = torch.cat(images, dim=0)  # Combine into a single batch (N, C, H, W)
        return images_tensor, valid_prompts
    else:
        return None, None

# Single task: process a batch of prompts and corresponding images, and calculate CLIP Score
def process_batch(prompts_batch, image_folder, model_path, device):
    clip_score_metric = CLIPScore(model_name_or_path=model_path).to(device)
    
    # Load image batch
    images_tensor, valid_prompts = load_images(image_folder, prompts_batch)
    if images_tensor is not None:
        images_tensor = images_tensor.to(device)
        
        with torch.no_grad():  # Avoid building computation graph, reducing memory consumption
            # Calculate CLIP Score for each image and prompt
            for i, prompt in enumerate(valid_prompts):
                clip_score_metric.update(images_tensor[i].unsqueeze(0).float(), prompt)
        
        # Release memory
        del images_tensor
        torch.cuda.empty_cache()

        return clip_score_metric.compute().item()
    else:
        return None

# Split data into batches
def chunked(iterable, batch_size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# Main processing function
def main_worker(rank, prompts, image_folder, model_path, device, batch_size, queue):
    # Split into batches
    prompts_batches = list(chunked(prompts, batch_size))
    
    clip_scores = []
    for batch in prompts_batches:
        score = process_batch(batch, image_folder, model_path, device)
        if score is not None:
            clip_scores.append(score)
        # After processing each batch, send information to the main process
        queue.put(1)  # Send signal indicating one batch is processed
    
    queue.put(clip_scores)  # Put final result into the queue for the main process

def main(prompt_file="prompts.txt", image_folder="images", batch_size=64, num_workers=4):
    # Load prompts
    prompts = load_prompts(prompt_file)
    model_path = "/root/autodl-tmp/pretrained_models/clip-vit-large-patch14"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create multiprocessing queue
    queue = mp.Queue()

    # Start multiple processes
    processes = []
    chunk_size = len(prompts) // num_workers
    total_batches = (len(prompts) + batch_size - 1) // batch_size  # Calculate total batch count
    for rank in range(num_workers):
        worker_prompts = prompts[rank * chunk_size: (rank + 1) * chunk_size]
        p = mp.Process(target=main_worker, args=(rank, worker_prompts, image_folder, model_path, device, batch_size, queue))
        p.start()
        processes.append(p)

    # Use tqdm to create a progress bar
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        all_scores = []
        finished_batches = 0

        # Get results or progress from the queue
        while finished_batches < total_batches:
            result = queue.get()
            if isinstance(result, list):  # If it's a list, it means final scores
                all_scores.extend(result)
            else:
                pbar.update(1)  # Update progress bar
                finished_batches += 1

    # Wait for subprocesses to end
    for p in processes:
        p.join()

    # Calculate final result
    if all_scores:
        final_clip_score = sum(all_scores) / len(all_scores)
        print(f"Final averaged CLIP Score for folder '{image_folder}': {final_clip_score}")
    else:
        print(f"No valid images found in folder '{image_folder}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate CLIP Score for images and prompts with batch parallel processing.")
    parser.add_argument("--prompt_file", type=str, default="/root/autodl-tmp/COCO/COCO_caption_prompts_30k.txt", help="Path to the prompts text file.")
    parser.add_argument("--image_folder", type=str, default="/root/autodl-tmp/vis/2024-09-04_custom_epochunknown_stepunknown_scale4.5_step20_size256_bs100_sampdpm-solver_seed0", help="Path to the folder containing images.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to process in each batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()
    
    # Set multiprocessing start method to 'spawn', suitable for CUDA
    mp.set_start_method('spawn', force=True)

    main(prompt_file=args.prompt_file, image_folder=args.image_folder, batch_size=args.batch_size, num_workers=args.num_workers)
