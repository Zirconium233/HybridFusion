import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset import ImageFusionDataset
from model.traditional_fusion import LaplacianPyramidFusion
from metric.MetricGPU import (
    VIF_function_batch, Qabf_function_batch, SSIM_function_batch,
    PSNR_function_batch, MSE_function_batch, CC_function_batch, SCD_function_batch,
    Nabf_function_batch, MI_function_batch, EN_function_batch, SF_function_batch,
    SD_function_batch, AG_function_batch
)

# -------------------------------
# Config (test only, no training)
# -------------------------------
PROJECT_DIR = "./checkpoints/abl_traditional"
TEST_BATCH_SIZE = 2
NUM_WORKERS = 4
SAVE_IMAGES_TO_DIR = True
TEST_SET_NAMES = ["MSRS", "M3FD", "RS"]
DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "MSRS": {
        "train": {"dir_A": "./data/MSRS-main/MSRS-main/train/vi", "dir_B": "./data/MSRS-main/MSRS-main/train/ir"},
        "test":  {"dir_A": "./data/MSRS-main/MSRS-main/test/vi",  "dir_B": "./data/MSRS-main/MSRS-main/test/ir"},
    },
    "M3FD": {
        "train": {"dir_A": "./data/M3FD_Fusion/Vis", "dir_B": "./data/M3FD_Fusion/Ir"},
        "test":  {"dir_A": "./data/M3FD_Fusion/Vis", "dir_B": "./data/M3FD_Fusion/Ir"},
    },
    "RS": {
        "train": {"dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
                  "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared"},
        "test":  {"dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
                  "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared"},
    },
}

torch.backends.cudnn.benchmark = False


# -------------------------------
# Utils
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5

def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5

def make_3ch(x: torch.Tensor) -> torch.Tensor:
    return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

def save_image_grid(path: str, img: torch.Tensor, nrow: int = 4):
    x = img.detach().to(torch.float32).cpu().clamp(-1, 1)
    save_image(x, path, nrow=min(nrow, x.size(0)), normalize=True, value_range=(-1, 1), padding=2)

def build_test_loaders():
    loaders = {}
    for name in TEST_SET_NAMES:
        paths = DATASETS[name]["test"]
        ds = ImageFusionDataset(
            dir_A=paths["dir_A"], dir_B=paths["dir_B"], dir_C=None,
            is_train=False, is_getpatch=False, augment=False
        )
        loaders[name] = DataLoader(
            ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2), pin_memory=True
        )
    return loaders


# -------------------------------
# Heuristic weight map (W→IR)
# -------------------------------
def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: [-1,1], (B,C,H,W)
    if x.shape[1] == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

def sobel_grad_mag(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def gaussian_kernel(ks: int = 9, sigma: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    xs = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    g = torch.exp(-0.5 * (xs / sigma) ** 2)
    k1 = (g / g.sum()).view(1, 1, ks, 1)
    k2 = (g / g.sum()).view(1, 1, 1, ks)
    return k1, k2

def gaussian_blur(x: torch.Tensor, ks: int = 9, sigma: float = 3.0) -> torch.Tensor:
    k1, k2 = gaussian_kernel(ks, sigma, device=x.device, dtype=x.dtype)
    x = F.conv2d(x, k1, padding=(ks//2, 0), groups=x.shape[1])
    x = F.conv2d(x, k2, padding=(0, ks//2), groups=x.shape[1])
    return x

def build_weight_map(A: torch.Tensor, B: torch.Tensor, k: float = 3.0, beta: float = 0.5) -> torch.Tensor:
    """
    Return W in [0,1], where W->1 prefers B(IR), W->0 prefers A(VIS).
    A,B in [-1,1].
    """
    A_g = rgb_to_gray(A)
    B_g = rgb_to_gray(B)
    # luminance and gradient cues
    diff_lum = B_g - A_g
    gradA = sobel_grad_mag(A_g)
    gradB = sobel_grad_mag(B_g)
    diff_grad = gradB - gradA
    score = diff_lum + beta * diff_grad
    W = torch.sigmoid(k * score)
    W = gaussian_blur(W, ks=9, sigma=3.0)
    return W.clamp(0.0, 1.0)


# -------------------------------
# Two fusion methods
# -------------------------------
@torch.no_grad()
def fuse_laplacian(A: torch.Tensor, B: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Use provided LaplacianPyramidFusion (traditional)
    lp = LaplacianPyramidFusion(num_levels=4).to(device=A.device, dtype=A.dtype)
    A3 = make_3ch(A)
    return lp(A3, B, W)

@torch.no_grad()
def fuse_kernel(A: torch.Tensor, B: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Our "fusion kernel": smooth weight and directly per-pixel blend (no pyramid)
    Ws = gaussian_blur(W, ks=11, sigma=3.5).clamp(0.0, 1.0)
    A3 = make_3ch(A)
    B3 = make_3ch(B)
    F_hat = (1.0 - Ws) * A3 + Ws * B3
    return F_hat


# -------------------------------
# Evaluation
# -------------------------------
@torch.no_grad()
def evaluate_method(name: str, fuse_fn, loaders: Dict[str, DataLoader], device: torch.device):
    out_metrics = {}
    out_root = os.path.join(PROJECT_DIR, name)
    os.makedirs(out_root, exist_ok=True)

    for set_name, loader in loaders.items():
        if SAVE_IMAGES_TO_DIR:
            os.makedirs(os.path.join(out_root, "images", set_name), exist_ok=True)

        all_vif, all_qbf, all_ssim = [], [], []
        all_psnr, all_mse, all_cc, all_scd = [], [], [], []
        all_nabf, all_mi, all_ag, all_en, all_sf, all_sd = [], [], [], [], [], []

        pbar = tqdm(loader, desc=f"[{name}] {set_name}", leave=False)
        for i, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                A, B = batch[0].to(device), batch[1].to(device)
            else:
                raise RuntimeError("Expected (A,B[,C]) from dataset.")

            A = to_ch_last(A)
            B = to_ch_last(B)

            W = build_weight_map(A, B, k=3.0, beta=0.5)
            F_hat = fuse_fn(A, B, W)

            if SAVE_IMAGES_TO_DIR and i < 4:
                save_image_grid(os.path.join(out_root, "images", set_name, f"A_{i:04d}.png"), A)
                save_image_grid(os.path.join(out_root, "images", set_name, f"B_{i:04d}.png"), B)
                save_image_grid(os.path.join(out_root, "images", set_name, f"W_{i:04d}.png"), (W * 2 - 1))
                save_image_grid(os.path.join(out_root, "images", set_name, f"F_{i:04d}.png"), F_hat)

            A_255 = to_255(A).to(torch.float32)
            B_255 = to_255(B).to(torch.float32)
            F_255 = to_255(F_hat).to(torch.float32)

            try:
                all_vif.append(VIF_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_qbf.append(Qabf_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_ssim.append(SSIM_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_psnr.append(PSNR_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_mse.append(MSE_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_cc.append(CC_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_scd.append(SCD_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_nabf.append(Nabf_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_mi.append(MI_function_batch(A_255, B_255, F_255).reshape(-1).cpu())
                all_ag.append(AG_function_batch(F_255).reshape(-1).cpu())
                all_en.append(EN_function_batch(F_255).reshape(-1).cpu())
                all_sf.append(SF_function_batch(F_255).reshape(-1).cpu())
                all_sd.append(SD_function_batch(F_255).reshape(-1).cpu())
            except Exception as e:
                print(f"[{name}][{set_name}] metric error: {e}")
                continue

        # stack and mean
        def mean_cat(xs): 
            return torch.cat(xs).mean().item() if len(xs) else float("nan")

        metrics = {
            "VIF": mean_cat(all_vif), "Qabf": mean_cat(all_qbf), "SSIM": mean_cat(all_ssim),
            "PSNR": mean_cat(all_psnr), "MSE": mean_cat(all_mse), "CC": mean_cat(all_cc), "SCD": mean_cat(all_scd),
            "Nabf": mean_cat(all_nabf), "MI": mean_cat(all_mi), "AG": mean_cat(all_ag),
            "EN": mean_cat(all_en), "SF": mean_cat(all_sf), "SD": mean_cat(all_sd),
        }
        reward = (metrics["VIF"] + 1.5 * metrics["Qabf"] + metrics["SSIM"]) / 3.0
        metrics["Reward"] = reward
        out_metrics[set_name] = metrics

        print(f"[{name}] {set_name}: Reward={reward:.4f} | "
              f"VIF={metrics['VIF']:.4f} Qabf={metrics['Qabf']:.4f} SSIM={metrics['SSIM']:.4f} | "
              f"PSNR={metrics['PSNR']:.4f} MSE={metrics['MSE']:.4f} CC={metrics['CC']:.4f} | "
              f"SCD={metrics['SCD']:.4f} Nabf={metrics['Nabf']:.4f} MI={metrics['MI']:.4f} | "
              f"AG={metrics['AG']:.4f} EN={metrics['EN']:.4f} SF={metrics['SF']:.4f} SD={metrics['SD']:.4f}")

    # Save CSV
    import pandas as pd, json
    rows = []
    for ds, m in out_metrics.items():
        r = {"method": name, "dataset": ds}; r.update(m); rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    csv_path = os.path.join(PROJECT_DIR, f"metrics_{name}.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(PROJECT_DIR, f"metrics_{name}.json"), "w") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"[{name}] saved: {csv_path}")
    return out_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    loaders = build_test_loaders()

    print("Running ablations (no training)...")

    metrics_lap = evaluate_method("Laplacian", fuse_laplacian, loaders, device)
    metrics_kernel = evaluate_method("FusionKernel", fuse_kernel, loaders, device)

    print("\nDone.")


if __name__ == "__main__":
    main()