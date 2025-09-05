import os
import cv2
import json
import math
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from model.policy_net import PolicyNet
from model.traditional_fusion import LaplacianPyramidFusion
from metric.MetricGPU import (
    VIF_function_batch, Qabf_function_batch, SSIM_function_batch,
    PSNR_function_batch, MSE_function_batch, CC_function_batch, SCD_function_batch,
    Nabf_function_batch, MI_function_batch, EN_function_batch, SF_function_batch,
    SD_function_batch, AG_function_batch
)

# -------------------------------
# 固定参数（预编码）
# -------------------------------
DATA_ROOT = "./data/M3SVD/Video/test"
VISIBLE_DIR = os.path.join(DATA_ROOT, "visible")
INFRA_DIR = os.path.join(DATA_ROOT, "infrared")
TARGET_SIZE = (480, 640)  # (H, W)
BATCH_SIZE = 48
NUM_WORKERS = 4
PIN_MEMORY = True

CKPT_ROOT = "./checkpoints/stochastic_policy_ycbcr"
EPOCHS_TO_EVAL = [2, 10]  # 需要评测的epoch
OUT_DIR = "./checkpoints/M3SVD_eval"
os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = False

# 新增：YCbCr 工具
def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5
def to_m11(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0
def rgb_to_ycbcr(x_rgb_m11: torch.Tensor):
    x = to_01(x_rgb_m11)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return to_m11(y), to_m11(cb), to_m11(cr)
def ycbcr_to_rgb(y_m11: torch.Tensor, cb_m11: torch.Tensor, cr_m11: torch.Tensor) -> torch.Tensor:
    y  = to_01(y_m11)
    cb = to_01(cb_m11) - 0.5
    cr = to_01(cr_m11) - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = torch.cat([r, g, b], dim=1).clamp(0, 1)
    return to_m11(rgb)


# -------------------------------
# 工具
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def to_255(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,255]
    return (x.clamp(-1, 1) + 1.0) * 127.5

def rgb_bgr_to_tensor_norm(img_bgr: np.ndarray, to_gray: bool, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    将 BGR uint8 图像处理为 tensor [-1,1]
    - to_gray=False: 输出 3xHxW (RGB)
    - to_gray=True:  输出 1xHxW (Gray)
    """
    H, W = size_hw
    if to_gray:
        if img_bgr.ndim == 3:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_bgr
        img_gray = cv2.resize(img_gray, (W, H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img_gray).float() / 127.5 - 1.0  # HxW
        return t.unsqueeze(0)  # 1xHxW
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img_rgb).float() / 127.5 - 1.0  # HxWxC
        t = t.permute(2, 0, 1).contiguous()  # CxHxW
        return t  # 3xHxW


# -------------------------------
# 数据集（内部实现，读取配对 mp4 并逐帧输出）
# -------------------------------
class M3SVDVideoPairDataset(Dataset):
    def __init__(self, vis_dir: str, ir_dir: str, target_size: Tuple[int, int] = (480, 640)):
        super().__init__()
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.target_size = target_size  # (H, W)

        vis_files = sorted([f for f in os.listdir(vis_dir) if f.lower().endswith(".mp4")])
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(".mp4")])

        # 按文件名配对（去扩展名）
        vis_map = {os.path.splitext(f)[0]: os.path.join(vis_dir, f) for f in vis_files}
        ir_map = {os.path.splitext(f)[0]: os.path.join(ir_dir, f) for f in ir_files}
        common_keys = sorted(set(vis_map.keys()).intersection(set(ir_map.keys())))
        if len(common_keys) == 0:
            raise RuntimeError("未在可见和红外目录下找到同名的 mp4 对。")

        self.pairs: List[Tuple[str, str]] = [(vis_map[k], ir_map[k]) for k in common_keys]

        # 预索引每对视频的帧数量，取两者最小值用于对齐
        self.frame_counts: List[int] = []
        for v_path, i_path in self.pairs:
            v_cap = cv2.VideoCapture(v_path)
            i_cap = cv2.VideoCapture(i_path)
            v_cnt = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            i_cnt = int(i_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            v_cap.release()
            i_cap.release()
            self.frame_counts.append(max(0, min(v_cnt, i_cnt)))

        # 建立全局索引：每个样本对应 (pair_idx, frame_idx)
        self.index_map: List[Tuple[int, int]] = []
        for pidx, n in enumerate(self.frame_counts):
            for fidx in range(n):
                self.index_map.append((pidx, fidx))

        total_frames = len(self.index_map)
        print(f"[M3SVD] 总视频对数: {len(self.pairs)}, 总帧数: {total_frames}")

        # 惰性打开的缓存（按 worker 进程独享）
        self._caps_vis: Dict[int, cv2.VideoCapture] = {}
        self._caps_ir: Dict[int, cv2.VideoCapture] = {}

    def __len__(self):
        return len(self.index_map)

    def _get_cap(self, path: str, cache: Dict[int, cv2.VideoCapture]) -> cv2.VideoCapture:
        key = hash(path)
        cap = cache.get(key, None)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(path)
            cache[key] = cap
        return cap

    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
        # 直接跳转到指定帧；部分编解码器不精确，但足够做评测
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # 尝试再次读取（容错），否则返回全零
            ok2, frame2 = cap.read()
            if not ok2 or frame2 is None:
                return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            frame = frame2
        return frame

    def __getitem__(self, idx: int):
        pair_idx, frame_idx = self.index_map[idx]
        v_path, i_path = self.pairs[pair_idx]

        cap_v = self._get_cap(v_path, self._caps_vis)
        cap_i = self._get_cap(i_path, self._caps_ir)

        v_bgr = self._read_frame(cap_v, frame_idx)  # HxWx3 BGR
        i_bgr = self._read_frame(cap_i, frame_idx)  # HxWx3 BGR (有些红外编码成3通道)

        A = rgb_bgr_to_tensor_norm(v_bgr, to_gray=False, size_hw=self.target_size)  # 3xHxW, [-1,1]
        B = rgb_bgr_to_tensor_norm(i_bgr, to_gray=True, size_hw=self.target_size)   # 1xHxW, [-1,1]
        return A, B

    def __del__(self):
        for cap in list(self._caps_vis.values()) + list(self._caps_ir.values()):
            try:
                cap.release()
            except Exception:
                pass


# -------------------------------
# 评测
# -------------------------------
@torch.no_grad()
def evaluate_one_checkpoint(
    device: torch.device,
    ckpt_path: str,
    loader: DataLoader,
    save_prefix: str,
    metric_mode: str = "mu",  # 'mu' or 'sample'
):
    if not os.path.isfile(ckpt_path):
        print(f"[Skip] 未找到模型: {ckpt_path}")
        return None

    # 模型与融合核
    # 仅在亮度通道与 IR 融合
    policy_net = PolicyNet(in_channels=2, out_channels=2).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    policy_net.load_state_dict(state, strict=True)
    policy_net.eval()

    fusion_kernel = LaplacianPyramidFusion(num_levels=4).to(device)

    # 聚合容器
    all_vif, all_qbf, all_ssim = [], [], []
    all_psnr, all_mse, all_cc, all_scd = [], [], [], []
    all_nabf, all_mi, all_ag, all_en, all_sf, all_sd = [], [], [], [], [], []

    pbar = tqdm(loader, desc=f"[Eval] {os.path.basename(os.path.dirname(ckpt_path))}")
    for A, B in pbar:
        A = to_ch_last(A.to(device=device, dtype=torch.float32))
        B = to_ch_last(B.to(device=device, dtype=torch.float32))
        # 只取 A 的亮度（Y），与 B(IR) 融合
        Y, Cb, Cr = rgb_to_ycbcr(A)
        mu, logvar = policy_net(Y, B)
        std = torch.exp(0.5 * logvar)

        F_Y_mu = fusion_kernel(Y, B, mu)
        sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)
        F_Y_sampled = fusion_kernel(Y, B, sampled_w)
        F_hat_mu = ycbcr_to_rgb(F_Y_mu, Cb, Cr)
        F_hat_sampled = ycbcr_to_rgb(F_Y_sampled, Cb, Cr)

        F_use = F_hat_mu if metric_mode == "mu" else F_hat_sampled

        # 指标用 0..255
        A_255 = to_255(A).to(torch.float32)
        B_255 = to_255(B).to(torch.float32)
        F_255 = to_255(F_use).to(torch.float32)

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
            print(f"[Metrics] 失败: {e}")
            continue

    def mean_cat(xs): 
        return torch.cat(xs).mean().item() if len(xs) else float("nan")

    metrics = {
        "VIF": mean_cat(all_vif), "Qabf": mean_cat(all_qbf), "SSIM": mean_cat(all_ssim),
        "PSNR": mean_cat(all_psnr), "MSE": mean_cat(all_mse), "CC": mean_cat(all_cc), "SCD": mean_cat(all_scd),
        "Nabf": mean_cat(all_nabf), "MI": mean_cat(all_mi),
        "AG": mean_cat(all_ag), "EN": mean_cat(all_en), "SF": mean_cat(all_sf), "SD": mean_cat(all_sd),
    }
    metrics["Reward"] = (metrics["VIF"] + 1.5 * metrics["Qabf"] + metrics["SSIM"]) / 3.0

    # 打印
    print(
        f"[Result] {os.path.basename(os.path.dirname(ckpt_path))} | "
        f"Reward={metrics['Reward']:.4f} | "
        f"VIF={metrics['VIF']:.4f} Qabf={metrics['Qabf']:.4f} SSIM={metrics['SSIM']:.4f} | "
        f"PSNR={metrics['PSNR']:.4f} MSE={metrics['MSE']:.4f} CC={metrics['CC']:.4f} SCD={metrics['SCD']:.4f} | "
        f"Nabf={metrics['Nabf']:.4f} MI={metrics['MI']:.4f} | "
        f"AG={metrics['AG']:.4f} EN={metrics['EN']:.4f} SF={metrics['SF']:.4f} SD={metrics['SD']:.4f}"
    )

    # 保存到 json（按模型一次一个，便于追溯）
    with open(f"{save_prefix}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集与 DataLoader
    ds = M3SVDVideoPairDataset(VISIBLE_DIR, INFRA_DIR, target_size=TARGET_SIZE)
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False
    )

    rows = []
    for ep in EPOCHS_TO_EVAL:
        ckpt_path = os.path.join(CKPT_ROOT, f"epoch_{ep}", "policy_net.pth")
        save_prefix = os.path.join(OUT_DIR, f"metrics_epoch_{ep}")
        metrics = evaluate_one_checkpoint(device, ckpt_path, dl, save_prefix, metric_mode="mu")
        if metrics is not None:
            row = {"epoch": ep}
            row.update(metrics)
            rows.append(row)

    if len(rows) == 0:
        print("[Done] 无可用结果（可能是模型权重不存在）。")
        return

    # 保存汇总 CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "metrics_M3SVD_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()