import argparse
import os
from PIL import Image
import random
import math
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.transforms.functional as TF

# diffusers VAE
from diffusers import AutoencoderKL

# 尝试导入 accelerate（若不可用则回退为无加速器）
try:
    from accelerate import Accelerator
    has_accelerate = True
except Exception:
    Accelerator = None
    has_accelerate = False

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


class RecImageDataset(Dataset):
    """
    每组固定 target_size，但不同组可以有不同的 target_size（因此图片总体不统一缩放）。
    对于单通道灰度图，使用加权转换为3通道： [0.2989, 0.5870, 0.1140]
    每张图片会按该组 target_size 单独 resize（确保能被4整除）。
    augment_prob 控制是否对每张图片随机做 augment（在 __getitem__ 内随机决定）。
    输出 Tensor 已被 Normalize 到 [-1,1]（Normalize 0.5,0.5）。
    """
    def __init__(self, files, target_size, augment_prob=0.5):
        self.files = files
        self.target_size = target_size  # (W,H)
        self.augment_prob = augment_prob
        self.transform = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5],
                                                       [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.files)

    def _maybe_augment(self, img):
        if random.random() < self.augment_prob:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            k = random.choice([0, 1, 2, 3])
            if k:
                img = TF.rotate(img, angle=90 * k)
        return img

    def _convert_gray_to_rgb_weighted(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        w0, w1, w2 = 0.2989, 0.5870, 0.1140
        r = arr * w0
        g = arr * w1
        b = arr * w2
        rgb = np.stack([r, g, b], axis=2)
        rgb_img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
        return rgb_img

    def __getitem__(self, idx):
        path = self.files[idx]
        with Image.open(path) as img:
            mode = img.mode
            if mode in ('L', 'I', 'I;16', '1'):
                img = self._convert_gray_to_rgb_weighted(img)
            elif mode == 'RGB':
                pass
            else:
                img = img.convert('RGB')

            img = img.resize(self.target_size, resample=Image.BICUBIC)
            img = self._maybe_augment(img)
            tensor = self.transform(img)  # (C,H,W), in [-1,1]
            return tensor


def find_images_recursive(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if is_image_file(fn):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def group_images_by_adjusted_size(files):
    groups = {}
    for p in files:
        try:
            with Image.open(p) as im:
                mode = im.mode
                if mode not in ('RGB', 'L', 'I', 'I;16', '1'):
                    continue
                w, h = im.size
                adj = adjust_size_to_div4((w, h))
                groups.setdefault(adj, []).append(p)
        except Exception:
            continue
    return groups


# 预训练与保存路径（按你的要求）
PRETRAINED_VAE_DIR = "/home/zhangran/desktop/myProject/playground/sd-vae-ft-mse"
SAVE_FINETUNED_DIR = "/home/zhangran/desktop/myProject/playground/Image/checkpoints/vae/sd-vae-ft"


def build_vae(vae_latent_channels=4):
    # 使用预训练权重初始化，进行微调
    vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE_DIR)
    return vae


def try_encode_latent(vae, imgs):
    enc = vae.encode(imgs)
    if hasattr(enc, 'latent_dist'):
        try:
            lat = enc.latent_dist.sample() # This line works
        except Exception:
            lat = enc.latent_dist.mean
    elif isinstance(enc, dict):
        if 'latent_dist' in enc:
            try:
                lat = enc['latent_dist'].sample()
            except Exception:
                lat = enc['latent_dist'].mean
        elif 'sample' in enc:
            lat = enc['sample']
        else:
            lat = torch.as_tensor(enc)
    else:
        lat = enc
    return lat


def try_decode(vae, latents):
    out = vae.decode(latents)
    if isinstance(out, dict):
        if 'sample' in out:
            return out['sample'] # This line works
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
        raise RuntimeError("Unexpected vae.decode() return structure.")
    elif isinstance(out, torch.Tensor):
        return out
    else:
        return torch.as_tensor(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data directory (recursive)')
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs (default 1)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size (default 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--eval_frac', type=float, default=0.05, help='fraction for eval split (default 0.05)')
    args = parser.parse_args()

    files = find_images_recursive(args.data)
    if not files:
        print("No image files found under", args.data)
        return

    groups = group_images_by_adjusted_size(files)
    if not groups:
        print("No valid 1- or 3-channel images found.")
        return

    # 像素阈值: 1024*768 改为 300
    PIXEL_LIMIT = 300 * 300

    # 过滤太大的 size（直接忽略）
    filtered_groups = {}
    skipped = {}
    for size, flist in groups.items():
        w, h = size
        pixels = w * h
        if pixels > PIXEL_LIMIT:
            skipped[size] = len(flist)
            continue
        filtered_groups[size] = flist

    if skipped:
        print(f"Skipped {len(skipped)} size-groups due to exceeding pixel limit ({PIXEL_LIMIT}): {skipped}")

    groups = filtered_groups
    if not groups:
        print("No groups under pixel limit to train on. Exiting.")
        return

    # 新增：过滤小样本尺寸分组（样本数 <= 20 的直接丢弃，避免污染微调）
    small_groups = {size: len(flist) for size, flist in groups.items() if len(flist) <= 20}
    groups = {size: flist for size, flist in groups.items() if len(flist) > 20}
    if small_groups:
        print(f"Dropped size-groups with too few images (<=20): {small_groups}")
    if not groups:
        print("No size-groups with >20 images remain after filtering. Exiting.")
        return

    if has_accelerate:
        accelerator = Accelerator()
        is_main = accelerator.is_main_process
    else:
        accelerator = None
        is_main = True

    if is_main:
        print(f"Found image size groups: { {k: len(v) for k, v in groups.items()} }")
        print(f"Training VAE. Epochs={args.epoch}, batch_size={args.batch_size}")

    # 构建模型与优化器（先在 CPU 上构建）
    vae = build_vae(vae_latent_channels=4)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # 为每个 size group 构建 dataset / split / dataloader（一次性构建，便于 accelerate.prepare）
    train_loaders = []
    eval_loaders = []
    group_meta = {}  # 保存每组的元数据用于训练/日志: {size: {'train_loader':..., 'eval_loader':..., 'n':...}}
    rng = torch.Generator()
    rng.manual_seed(42)

    for size, flist in groups.items():
        n = len(flist)
        # skip empty
        if n == 0:
            continue
        n_eval = max(1, int(n * args.eval_frac)) if n > 1 else 0
        n_train = n - n_eval

        # 生成一致的拆分索引（一次性）
        perm = torch.randperm(n, generator=rng).tolist()
        train_idx = perm[:n_train]
        eval_idx = perm[n_train:] if n_eval > 0 else []

        ds_train_full = RecImageDataset(flist, size, augment_prob=0.5)
        ds_eval_full = RecImageDataset(flist, size, augment_prob=0.0)

        train_subset = Subset(ds_train_full, train_idx) if n_train > 0 else None
        eval_subset = Subset(ds_eval_full, eval_idx) if n_eval > 0 else None

        if train_subset is not None:
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            train_loaders.append(train_loader)
        else:
            train_loader = None

        if eval_subset is not None:
            eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            eval_loaders.append(eval_loader)
        else:
            eval_loader = None

        group_meta[size] = {'train_loader': train_loader, 'eval_loader': eval_loader, 'n': n}

    # 如果使用 accelerator，需要将模型、优化器和所有 dataloader 一次性 prepare
    if accelerator is not None:
        # 准备传入 accelerate.prepare 的对象顺序： model, optimizer, *train_loaders, *eval_loaders
        to_prepare = [vae, optimizer] + [dl for dl in train_loaders if dl is not None] + [dl for dl in eval_loaders if dl is not None]
        prepared = accelerator.prepare(*to_prepare)
        # 解析返回值
        vae = prepared[0]
        optimizer = prepared[1]
        n_train_dl = len([dl for dl in train_loaders if dl is not None])
        prepared_train = prepared[2:2 + n_train_dl]
        prepared_eval = prepared[2 + n_train_dl:]
        # 重新映射到 group_meta（按插入顺序）
        ti = 0
        ei = 0
        for size, meta in group_meta.items():
            if meta['train_loader'] is not None:
                meta['train_loader'] = prepared_train[ti]
                ti += 1
            if meta['eval_loader'] is not None:
                meta['eval_loader'] = prepared_eval[ei]
                ei += 1

    device = accelerator.device if accelerator is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if is_main:
        print(f"Using device: {device}")

    best_score = float('inf')
    best_path = os.path.join(os.getcwd(), "best.pth")
    latest_path = os.path.join(os.getcwd(), "latest.pth")

    for epoch in range(1, args.epoch + 1):
        vae.train()
        total_train_loss = 0.0
        total_train_batches = 0

        for size, meta in group_meta.items():
            train_loader = meta['train_loader']
            if train_loader is None:
                continue

            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train size={size}", disable=not is_main, leave=False)
            group_running_loss = 0.0
            group_batches = 0

            for batch in pbar:
                # 保持输入与模型参数 dtype 一致（修复 bfloat16/float32 mismatch）
                model_dtype = next(vae.parameters()).dtype
                imgs = batch.to(device=device, dtype=model_dtype)
                optimizer.zero_grad()
                try:
                    lat = try_encode_latent(vae, imgs)
                    recon = try_decode(vae, lat)
                except Exception as e:
                    if is_main:
                        print("Encode/Decode failed:", e)
                    raise

                if recon.shape != imgs.shape:
                    raise AssertionError(f"VAE output shape {tuple(recon.shape)} != input shape {tuple(imgs.shape)} for target_size={size}.")

                loss = F.mse_loss(recon, imgs)
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()

                val = loss.item() if not isinstance(loss, (list, tuple)) else float(loss[0])
                group_running_loss += val
                group_batches += 1
                total_train_loss += val
                total_train_batches += 1
                if is_main:
                    pbar.set_postfix({'group_train_loss': f"{(group_running_loss / group_batches):.6f}"})

        avg_train = total_train_loss / max(1, total_train_batches)

        # eval
        vae.eval()
        total_eval_loss = 0.0
        total_eval_batches = 0
        with torch.no_grad():
            for size, meta in group_meta.items():
                eval_loader = meta['eval_loader']
                if eval_loader is None:
                    continue
                pbar = tqdm(eval_loader, desc=f"Eval size={size}", disable=not is_main, leave=False)
                for batch in pbar:
                    model_dtype = next(vae.parameters()).dtype
                    imgs = batch.to(device=device, dtype=model_dtype)
                    lat = try_encode_latent(vae, imgs)
                    recon = try_decode(vae, lat)

                    if recon.shape != imgs.shape:
                        raise AssertionError(f"VAE output shape {tuple(recon.shape)} != input shape {tuple(imgs.shape)} during eval for target_size={size}.")

                    l = F.mse_loss(recon, imgs)
                    total_eval_loss += l.item()
                    total_eval_batches += 1

        avg_eval = total_eval_loss / max(1, total_eval_batches) if total_eval_batches > 0 else float('nan')

        # 保存 latest（仅主进程）
        if is_main:
            try:
                model_to_save = accelerator.unwrap_model(vae) if accelerator is not None else vae
                torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, latest_path)
            except Exception as e:
                print(f"Failed to save latest.pth: {e}")

        score = (avg_eval if not math.isnan(avg_eval) else 1e9) * 0.8 + avg_train * 0.2
        if score < best_score and is_main:
            best_score = score
            try:
                model_to_save = accelerator.unwrap_model(vae) if accelerator is not None else vae
                torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict(), 'score': score}, best_path)
                print(f"Saved new best model at epoch {epoch} with score {score:.6f} -> {best_path}")
            except Exception as e:
                print(f"Failed to save best.pth: {e}")

        if is_main:
            tqdm.write(f"Epoch {epoch}/{args.epoch}  train_loss={avg_train:.6f}  eval_loss={avg_eval if not math.isnan(avg_eval) else 'N/A'}  score={score:.6f}")

    if is_main:
        print("Training complete.")

    # 新增：保存微调后的 VAE 为 diffusers 兼容格式（save_pretrained）
    if is_main:
        try:
            os.makedirs(SAVE_FINETUNED_DIR, exist_ok=True)
            model_to_save = accelerator.unwrap_model(vae) if accelerator is not None else vae
            model_to_save.save_pretrained(SAVE_FINETUNED_DIR)
            print(f"Saved fine-tuned VAE to {SAVE_FINETUNED_DIR}")
        except Exception as e:
            print(f"Failed to save_pretrained to {SAVE_FINETUNED_DIR}: {e}")


if __name__ == "__main__":
    main()