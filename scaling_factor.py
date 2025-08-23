import argparse
import os
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from diffusers import AutoencoderKL

# --- Helper Functions and Classes (from train_vae.py) ---

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff',
              '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.TIF', '.TIFF'}

def is_image_file(name):
    return any(name.endswith(ext) for ext in IMAGE_EXTS)

def round4(x):
    r = int(round(x / 4.0)) * 4
    return max(4, r)

def adjust_size_to_div4(size):
    w, h = size
    return (round4(w), round4(h))

def find_images_recursive(root):
    """Recursively finds all image files in a directory."""
    files = []
    print(f"Scanning for images in {root}...")
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if is_image_file(fn):
                files.append(os.path.join(dirpath, fn))
    print(f"Found {len(files)} total image files.")
    return files

def group_images_by_adjusted_size(files):
    """Groups image file paths by their adjusted (divisible by 4) size."""
    groups = {}
    print("Grouping images by resolution...")
    for p in tqdm(files, desc="Scanning image sizes"):
        try:
            with Image.open(p) as im:
                w, h = im.size
                adj_size = adjust_size_to_div4((w, h))
                if adj_size not in groups:
                    groups[adj_size] = []
                groups[adj_size].append(p)
        except Exception:
            continue
    print(f"Found {len(groups)} unique resolution groups.")
    return groups

class RecImageDataset(Dataset):
    """Dataset from train_vae.py, adapted for inference (no augmentation)."""
    def __init__(self, files, target_size):
        self.files = files
        self.target_size = target_size
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(self.target_size, resample=Image.BICUBIC)
                return self.transform(img)
        except Exception as e:
            print(f"Warning: Skipping file {path} due to error: {e}")
            return None

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def build_vae():
    """Constructs the VAE with a fixed latent_channels of 4."""
    return AutoencoderKL(
        sample_size=128, in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256), latent_channels=4,
    )

# --- Main Script Logic ---

def calculate_scaling_factor():
    parser = argparse.ArgumentParser(
        description="Calculate VAE scaling factor using the same data grouping logic as train_vae.py."
    )
    parser.add_argument('--data', required=True, help='Path to the root directory containing all image subfolders.')
    parser.add_argument('--vae_checkpoint', required=True, help='Path to the trained VAE model checkpoint (.pth, .ckpt).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing images.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the DataLoader.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build and load the VAE model
    print("Building VAE architecture (latent_channels=4)...")
    vae = build_vae()
    
    print(f"Loading VAE checkpoint from {args.vae_checkpoint}...")
    state_dict = torch.load(args.vae_checkpoint, map_location="cpu")
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    vae.load_state_dict(state_dict, strict=True)
    vae = vae.to(device)
    vae.eval()
    print("VAE model loaded successfully.")

    # 2. Find, group, and filter datasets, then create dataloaders
    all_files = find_images_recursive(args.data)
    if not all_files:
        print("No images found. Exiting.")
        return
    
    size_groups = group_images_by_adjusted_size(all_files)

    # 修复 1: 过滤掉样本数小于等于20的分辨率组
    filtered_groups = {}
    skipped_groups = {}
    for size, file_list in size_groups.items():
        if len(file_list) > 20:
            filtered_groups[size] = file_list
        else:
            skipped_groups[size] = len(file_list)
    
    if skipped_groups:
        print(f"\nSkipped {len(skipped_groups)} groups with <= 20 images: {skipped_groups}")
    print(f"Using {len(filtered_groups)} groups for statistics calculation.")

    if not filtered_groups:
        print("No resolution groups with > 20 images found. Cannot calculate statistics. Exiting.")
        return

    data_loaders = []
    for size, file_list in filtered_groups.items():
        dataset = RecImageDataset(file_list, target_size=size)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none
        )
        data_loaders.append(loader)
        print(f"Created DataLoader for group {size} with {len(file_list)} images.")

    # 3. Iterate through all dataloaders and collect flattened latents
    all_latents_flat = []
    total_images = 0
    print("\nEncoding dataset to latent space...")
    with torch.no_grad():
        for loader in tqdm(data_loaders, desc="Processing Resolution Groups"):
            for batch in tqdm(loader, desc="Processing Batches", leave=False):
                if batch is None: continue
                
                images = batch.to(device, dtype=vae.dtype)
                latents = vae.encode(images).latent_dist.sample()
                
                # 修复 2: 将每个潜变量张量展平为一维向量再收集
                all_latents_flat.append(latents.cpu().flatten())
                total_images += images.shape[0]

    if not all_latents_flat:
        print("No latents were generated. Check dataset for valid images.")
        return

    # 4. Concatenate all 1D vectors and calculate statistics
    print("\nCalculating statistics from collected latents...")
    # 现在可以安全地拼接，因为它们都是一维向量
    latents_vector = torch.cat(all_latents_flat, dim=0)
    
    std_dev = torch.std(latents_vector)
    scaling_factor = 1.0 / std_dev

    print("\n--- VAE Scaling Factor Calculation Complete ---")
    print(f"Statistics based on {total_images} images from {len(filtered_groups)} resolution groups (with > 20 samples).")
    print(f"Total number of latent values analyzed: {latents_vector.numel()}")
    print(f"Calculated Standard Deviation: {std_dev.item():.6f}")
    print("-------------------------------------------------")
    print(f"✅ Recommended scaling_factor: {scaling_factor.item():.6f}")
    print("-------------------------------------------------")
    print("Use this value in your training and inference scripts for proper normalization.")


if __name__ == "__main__":
    calculate_scaling_factor()