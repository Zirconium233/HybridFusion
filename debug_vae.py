import os
import math
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model.cvae import ConditionalVAE
from dataset import ImageFusionDataset
# 指标（GPU/CPU）
from metric.MetricGPU import (
    VIF_function_batch,
    Qabf_function_batch,
    SSIM_function_batch,
)

# -------------------------------
# 配置
# -------------------------------
CKPT_PATH = "./checkpoints/pretrain_vae/final/cvae.pth"
OUTPUT_DIR = "./save_images/debug_vae"
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda", 0) if USE_CUDA else torch.device("cpu")

# 与训练时一致的模型规模（如你在 pretrain_vae.py 中的设置）
ENC_BASE_CHS = (192, 256, 384)
Z_CH = 32
COND_CH = 64
DEC_CHS = (384, 256, 128, 96)

# 数据路径（测试集）
DATASETS: Dict[str, Dict[str, str]] = {
    "MSRS": {
        "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
        "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
    }
}
TEST_SET = "MSRS"
BATCH_SIZE = 4
NUM_WORKERS = 2

# 重复生成次数，用于观测随机性（同一批 A/B）
N_SAMPLES = 4
SEED_BASE = 20250829  # 可改为 None 以完全随机

# 随机性增强选项（仅调试脚本使用）
USE_PERTURBATION = True   # 是否对 z 引入额外扰动
NOISE_T = 1.5             # 采样温度，>1放大随机性
STD_FLOOR = 0.05          # std 的下限，避免退化为确定性
Z_JITTER_STD = 0.05       # 对采样后的 z 再加的高斯扰动强度

# [-1,1] -> [0,255] float32
def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5

# 将 3 通道或 1 通道张量转为灰度 0..255，用于 EN/SD
def to_gray255(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W] in [-1,1] or [0,255]? 这里假定输入是 [-1,1]，内部转 0..255
    x255 = to_255(x)
    if x255.shape[1] == 1:
        return x255
    # RGB -> Y (BT.601)
    r, g, b = x255[:, 0:1], x255[:, 1:2], x255[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

# 批量熵（每张图 256 桶直方图）
def entropy_batch(gray255: torch.Tensor) -> torch.Tensor:
    # gray255: [B,1,H,W], 0..255 float
    B, _, H, W = gray255.shape
    vals = gray255.clamp(0, 255).round().to(torch.int64).view(B, -1)  # [B,HW]
    ent = []
    for i in range(B):
        hist = torch.bincount(vals[i], minlength=256).to(torch.float32)
        p = hist / (H * W)
        p = p.clamp_min(1e-12)
        e = -(p * torch.log2(p)).sum()
        ent.append(e)
    return torch.stack(ent, dim=0)  # [B]

# 批量标准差（灰度）
def sd_batch(gray255: torch.Tensor) -> torch.Tensor:
    # gray255: [B,1,H,W]
    return gray255.view(gray255.size(0), -1).std(dim=1)

# 统计并打印指标（对每个样本在不同 F 之间的均值/方差/极差）
def compute_and_print_metrics(
    tag: str,
    A: torch.Tensor,
    B: torch.Tensor,
    F_list: List[torch.Tensor],
):
    """
    - 对每个样本，在多个 fused 上统计：VIF, SSIM, Qabf（相对 A/B），EN/SD（仅 F）
    - 打印每个样本的各指标的 mean/std/min/max/range
    - 同时打印跨样本的平均（mean of means / mean of stds）
    """
    device = A.device
    # 预先转 0..255 float32
    A255 = to_255(A).to(torch.float32)
    B255 = to_255(B).to(torch.float32)
    F255_list = [to_255(F).to(torch.float32) for F in F_list]

    # 累计每个 fused 的 batch 指标 -> 堆叠为 [S,B]
    vif_SB, qbf_SB, ssim_SB = [], [], []
    en_SB, sd_SB = [], []
    for F255, F in zip(F255_list, F_list):
        vif = VIF_function_batch(A255, B255, F255)  # [B]
        qbf = Qabf_function_batch(A255, B255, F255)  # [B]
        ssim = SSIM_function_batch(A255, B255, F255)  # [B]
        en = entropy_batch(to_gray255(F))  # [B]
        sd = sd_batch(to_gray255(F))       # [B]
        # 统一放到 CPU 便于打印
        vif_SB.append(vif.detach().flatten().cpu())
        qbf_SB.append(qbf.detach().flatten().cpu())
        ssim_SB.append(ssim.detach().flatten().cpu())
        en_SB.append(en.detach().flatten().cpu())
        sd_SB.append(sd.detach().flatten().cpu())

    def stack_stats(arr_list: List[torch.Tensor], name: str):
        # arr_list: S 个 [B] -> [S,B]
        SB = torch.stack(arr_list, dim=0)  # [S,B]
        mean_b = SB.mean(dim=0)            # [B]
        std_b = SB.std(dim=0, unbiased=False)
        min_b, _ = SB.min(dim=0)
        max_b, _ = SB.max(dim=0)
        rng_b = max_b - min_b
        # 打印每个样本
        for i in range(SB.shape[1]):
            print(f"[{tag}][sample {i:02d}] {name}: "
                  f"mean={mean_b[i]:.6f} std={std_b[i]:.6f} "
                  f"min={min_b[i]:.6f} max={max_b[i]:.6f} range={rng_b[i]:.6f}")
        # 跨样本汇总
        print(f"[{tag}][ALL] {name}: mean_of_means={mean_b.mean():.6f} "
              f"mean_std={std_b.mean():.6f} mean_range={rng_b.mean():.6f}")

    print(f"\n[Metrics Summary] {tag}")
    stack_stats(vif_SB, "VIF")
    stack_stats(qbf_SB, "Qabf")
    stack_stats(ssim_SB, "SSIM")
    stack_stats(en_SB, "EN")
    stack_stats(sd_SB, "SD")

# 每个样本单独保存对比图
def save_per_sample_grids(out_dir: str, A: torch.Tensor, B: torch.Tensor,
                          F_list: list[torch.Tensor], prefix: str, n_cols_cap: int = 8):
    """
    - A: [B,3,H,W] in [-1,1]
    - B: [B,1,H,W] or [B,3,H,W] in [-1,1]
    - F_list: List of [B,3,H,W] in [-1,1], len = N_SAMPLES
    为 batch 的每个样本保存一张图： [vis, ir(伪彩), fused1..k]
    """
    os.makedirs(out_dir, exist_ok=True)
    B3 = B.repeat(1, 3, 1, 1) if B.shape[1] == 1 else B
    Bsz = A.size(0)
    for b in range(Bsz):
        tiles = [A[b:b+1], B3[b:b+1]]
        for F in F_list:
            tiles.append(F[b:b+1])
        # 限制列数，避免过宽
        grid = torch.cat(tiles[:n_cols_cap], dim=0)  # [N,3,H,W]
        save_image(
            grid.detach().to(torch.float32).cpu().clamp(-1, 1),
            os.path.join(out_dir, f"{prefix}_sample{b:02d}.png"),
            nrow=min(grid.size(0), grid.size(0)),
            normalize=True, value_range=(-1, 1), padding=2
        )

def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = False

    # 构建模型并加载权重
    model = ConditionalVAE(
        in_ch=4,
        base_chs=ENC_BASE_CHS,
        z_ch=Z_CH,
        cond_ch=COND_CH,
        dec_chs=DEC_CHS,
    ).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] from {CKPT_PATH} | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(memory_format=torch.channels_last)

    model_dtype = next(model.parameters()).dtype
    print(f"[Model] dtype={model_dtype}, device={DEVICE}")

    # 数据
    ds_paths = DATASETS[TEST_SET]
    test_ds = ImageFusionDataset(
        dir_A=ds_paths["dir_A"],
        dir_B=ds_paths["dir_B"],
        dir_C=None,
        is_train=False,
        is_getpatch=False,
        augment=False,
    )
    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 仅取一个 batch
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        A, B = batch[0], batch[1]
    else:
        raise RuntimeError("Dataset should return a tuple/list (A,B[,C]).")

    A = to_ch_last(A.to(device=DEVICE, dtype=model_dtype))
    B = to_ch_last(B.to(device=DEVICE, dtype=model_dtype))

    # 编码一次，分析后验分布统计
    mu, logvar = model.encode(A, B)
    std = torch.exp(0.5 * logvar)
    kld = ConditionalVAE.kl_loss(mu, logvar, reduction="mean")

    mu_mean = mu.mean().item()
    mu_std = mu.std().item()
    std_mean = std.mean().item()
    std_min = std.min().item()
    std_max = std.max().item()
    tiny_std_ratio = (std < 1e-3).float().mean().item()

    print(f"[Posterior] mu_mean={mu_mean:.5f} mu_std={mu_std:.5f} "
          f"std_mean={std_mean:.5f} std_min={std_min:.5f} std_max={std_max:.5f} "
          f"tiny_std_ratio(<1e-3)={tiny_std_ratio:.6f}  KL(mean)={kld.item():.6f}")

    # 多次采样生成，观测随机性（分别生成 base 与 rand 两组）
    Fs_base, Zs_base = [], []
    Fs_rand, Zs_rand = [], []
    std = torch.exp(0.5 * logvar)
    std_eff_base = std  # 原始
    std_eff_rand = torch.clamp(std * NOISE_T, min=STD_FLOOR) if USE_PERTURBATION else std

    for s in range(N_SAMPLES):
        if SEED_BASE is not None:
            torch.manual_seed(SEED_BASE + s)
            if USE_CUDA:
                torch.cuda.manual_seed_all(SEED_BASE + s)
        # base 采样
        eps_b = torch.randn_like(std_eff_base)
        z_b = mu + eps_b * std_eff_base
        F_b = model.decode(z_b, B)
        Fs_base.append(F_b.detach())
        Zs_base.append(z_b.detach())
        # rand 采样（温度、下限与额外抖动）
        eps_r = torch.randn_like(std_eff_rand)
        z_r = mu + eps_r * std_eff_rand
        if USE_PERTURBATION and Z_JITTER_STD > 0:
            z_r = z_r + Z_JITTER_STD * torch.randn_like(z_r)
        F_r = model.decode(z_r, B)
        Fs_rand.append(F_r.detach())
        Zs_rand.append(z_r.detach())

    # 保存基线与扰动两组的“整个batch”的拼图（便于快速总览）
    save_image(
        torch.cat(Fs_base, dim=0).detach().to(torch.float32).cpu().clamp(-1, 1),
        os.path.join(OUTPUT_DIR, f"F_BASE_ALL_{N_SAMPLES}x.png"),
        nrow=min(4, N_SAMPLES * A.size(0)),
        normalize=True, value_range=(-1, 1), padding=2
    )
    save_image(
        torch.cat(Fs_rand, dim=0).detach().to(torch.float32).cpu().clamp(-1, 1),
        os.path.join(OUTPUT_DIR, f"F_RAND_ALL_{N_SAMPLES}x.png"),
        nrow=min(4, N_SAMPLES * A.size(0)),
        normalize=True, value_range=(-1, 1), padding=2
    )

    # 也保存输入 A/B 的全batch网格
    save_image(
        A.detach().to(torch.float32).cpu().clamp(-1, 1),
        os.path.join(OUTPUT_DIR, "A.png"),
        nrow=min(4, A.size(0)),
        normalize=True, value_range=(-1, 1), padding=2
    )
    save_image(
        B.detach().to(torch.float32).cpu().clamp(-1, 1),
        os.path.join(OUTPUT_DIR, "B.png"),
        nrow=min(4, B.size(0)),
        normalize=True, value_range=(-1, 1), padding=2
    )

    # 为每个样本保存对比图： [A, B(伪彩), F1..FN]
    save_per_sample_grids(os.path.join(OUTPUT_DIR, "grids_base"), A, B, Fs_base, prefix="base")
    save_per_sample_grids(os.path.join(OUTPUT_DIR, "grids_rand"), A, B, Fs_rand, prefix="rand")

    # 计算不同采样之间的差异（输出与潜变量）：分别统计 base 与 rand
    def pairwise_stats(tensors, name):
        # tensors: List[T], 形状一致
        n = len(tensors)
        l1s, mses, cos = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                x = tensors[i]
                y = tensors[j]
                l1 = (x - y).abs().mean()
                mse = (x - y).pow(2).mean()
                # 余弦相似度（展平后按 batch 计算）
                xb = x.flatten(1)
                yb = y.flatten(1)
                cs = torch.nn.functional.cosine_similarity(xb, yb, dim=1).mean()
                l1s.append(l1.item())
                mses.append(mse.item())
                cos.append(cs.item())
        if len(l1s) == 0:
            return
        print(f"[Diff][{name}] pairs={len(l1s)}  L1={sum(l1s)/len(l1s):.6f}  "
              f"MSE={sum(mses)/len(mses):.6f}  CosSim={sum(cos)/len(cos):.6f}")

    pairwise_stats(Fs_base, "F_base")
    pairwise_stats(Zs_base, "z_base")
    pairwise_stats(Fs_rand, "F_rand")
    pairwise_stats(Zs_rand, "z_rand")

    # 计算并打印指标（对同一 A/B 的多次 fused）
    compute_and_print_metrics("BASE", A, B, Fs_base)
    compute_and_print_metrics("RAND", A, B, Fs_rand)

    # 将“同一批次”的 base/rand 的第一个样本的多次结果并排保存，便于直观对比
    b0 = 0
    seq_base = torch.cat([F[b0:b0+1] for F in Fs_base], dim=0)
    seq_rand = torch.cat([F[b0:b0+1] for F in Fs_rand], dim=0)
    save_image(seq_base.detach().to(torch.float32).cpu().clamp(-1, 1),
               os.path.join(OUTPUT_DIR, f"sample{b0:02d}_seq_base.png"),
               nrow=seq_base.size(0), normalize=True, value_range=(-1, 1), padding=2)
    save_image(seq_rand.detach().to(torch.float32).cpu().clamp(-1, 1),
               os.path.join(OUTPUT_DIR, f"sample{b0:02d}_seq_rand.png"),
               nrow=seq_rand.size(0), normalize=True, value_range=(-1, 1), padding=2)

    print(f"[Done] results saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()