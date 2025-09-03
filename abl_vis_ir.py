import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset import ImageFusionDataset
from metric.MetricGPU import (
    VIF_function_batch, Qabf_function_batch, SSIM_function_batch,
    PSNR_function_batch, MSE_function_batch, CC_function_batch, SCD_function_batch,
    Nabf_function_batch, MI_function_batch, EN_function_batch, SF_function_batch,
    SD_function_batch, AG_function_batch
)

# -------------------------------
# 配置
# -------------------------------
PROJECT_DIR = "./checkpoints/abl_vis_ir"
TEST_BATCH_SIZE = 2
NUM_WORKERS = 4
SAVE_IMAGES_TO_DIR = True

# 六个数据集
TEST_SET_NAMES = ["MSRS", "M3FD", "RS", "PET", "CT", "SPECT"]

# 路径参考 train.py
DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "MSRS": {
        "train": {
            "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
        },
        "test": {
            "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
        },
    },
    "M3FD": {
        "train": {
            "dir_A": "",
            "dir_B": "",
        },
        "test": {
            "dir_A": "./data/M3FD_Fusion/Vis",
            "dir_B": "./data/M3FD_Fusion/Ir",
        },
    },
    "RS": {
        "train": {
            "dir_A": "",
            "dir_B": "",
        },
        "test": {
            "dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
            "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared",
        },
    },
    "PET": {
        "train": {
            "dir_A": "",
            "dir_B": "",
        },
        "test": {
            "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/PET",
            "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/MRI",
        },
    },
    "CT": {
        "train": {
            "dir_A": "",
            "dir_B": "",
        },
        "test": {
            "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/CT",
            "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/MRI",
        },
    },
    "SPECT": {
        "train": {
            "dir_A": ".",
            "dir_B": ".",
        },
        "test": {
            "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/SPECT",
            "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/MRI",
        },
    }
}

torch.backends.cudnn.benchmark = False


# -------------------------------
# 工具
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
# 评测：两条基线
# - VisAsFused: 直接用可见光当融合结果（3通道；CT 使用 repeat 的 CT 当 VIS）
# - IrAsFused:  直接用红外当融合结果（保持其原始通道数，通常为1）
# -------------------------------
@torch.no_grad()
def evaluate_baseline(name: str, loaders: Dict[str, DataLoader], device: torch.device):
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
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Expected (A,B[,C]) from dataset.")
            A, B = batch[0].to(device), batch[1].to(device)
            A = to_ch_last(A)
            B = to_ch_last(B)

            # CT: 将 CT repeat 当 VIS 源（以及 VIS 融合图）
            if set_name == "CT":
                A_vis = make_3ch(A)
            else:
                A_vis = A  # 其他数据集保持原样（若已是3通道则保持，若1通道则按原样）
            # 构造两条基线的 fused
            if name == "VisAsFused":
                F_hat = make_3ch(A_vis)  # 确保 3 通道
            elif name == "IrAsFused":
                F_hat = B  # 保持 IR 原通道（通常 1 通道）
            else:
                raise ValueError("Unknown baseline name")

            if SAVE_IMAGES_TO_DIR and i < 4:
                save_image_grid(os.path.join(out_root, "images", set_name, f"A_{i:04d}.png"), A_vis)
                save_image_grid(os.path.join(out_root, "images", set_name, f"B_{i:04d}.png"), B)
                save_image_grid(os.path.join(out_root, "images", set_name, f"F_{i:04d}.png"), F_hat)

            # 准备指标输入（范围 0..255，float32）
            A_255 = to_255(A_vis).to(torch.float32)  # 使用 A_vis（CT 为 repeat 后的3通道）
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

    # 保存 CSV/JSON
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

    print("Running VIS/IR baselines (no fusion)...")

    metrics_vis = evaluate_baseline("VisAsFused", loaders, device)
    metrics_ir = evaluate_baseline("IrAsFused", loaders, device)

    print("\nDone.")


if __name__ == "__main__":
    main()