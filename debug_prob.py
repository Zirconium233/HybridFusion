import os
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

from dataset import ImageFusionDataset
from model.cvae import ConditionalVAE
from loss.loss import FusionLoss
from metric.MetricGPU import (
    VIF_function_batch,
    Qabf_function_batch,
    SSIM_function_batch,
)

# -------------------------------
# 配置（与 rl.py 对齐）
# -------------------------------
# 采样
DEBUG_BATCH_SIZE: int = 2        # 调试时用小 batch
NUM_WORKERS: int = 2
REPEAT_S: int = 4              # 每个样本的重复采样次数 S
MIXED_PRECISION: str = "bf16"

# PPO/优化参数（用于计算损失）
PPO_CLIP_EPS: float = 0.2
KL_WEIGHT_RL: float = 5e-4
ENTROPY_COEF: float = 1e-4
POLICY_LOSS_COEF: float = 10.0
AUX_DECODER_LOSS_W: float = 1e-4
UPDATE_DECODER: bool = True
ADV_NORMALIZE_BY_STD: bool = False

# 模型规模
ENC_BASE_CHS: Tuple[int, int, int] = (192, 256, 384)
Z_CH: int = 32
COND_CH: int = 64
DEC_CHS: Tuple[int, int, int, int] = (384, 256, 128, 96)

# 数据与模型路径
DATASET_PATH: Dict[str, str] = {
    "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
    "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
}
PRETRAIN_CKPT: str = "./checkpoints/pretrain_vae/final/cvae.pth"

# 调试配置
N_SAMPLES_TO_PRINT = 2  # 打印前 N 个样本的详细信息

torch.backends.cudnn.benchmark = False


def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5


def get_distribution(mu: torch.Tensor, logvar: torch.Tensor) -> Normal:
    return Normal(mu, torch.exp(0.5 * logvar))


def compute_reward(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    A255 = to_255(A).to(torch.float32)
    B255 = to_255(B).to(torch.float32)
    F255 = to_255(F).to(torch.float32)
    vif = VIF_function_batch(A255, B255, F255)
    qbf = Qabf_function_batch(A255, B255, F255)
    ssim = SSIM_function_batch(A255, B255, F255)
    return (vif + 1.5 * qbf + ssim) / 3.0


def main():
    # 使用最简单的 accelerator，仅用于设备和精度管理
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    
    print(f"[Config] Debugging with BS={DEBUG_BATCH_SIZE}, S={REPEAT_S}, MP={MIXED_PRECISION}")

    # --- 数据加载 ---
    ds = ImageFusionDataset(dir_A=DATASET_PATH["dir_A"], dir_B=DATASET_PATH["dir_B"], is_train=True)
    loader = DataLoader(ds, batch_size=DEBUG_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # --- 模型加载 ---
    model = ConditionalVAE(in_ch=4, base_chs=ENC_BASE_CHS, z_ch=Z_CH, cond_ch=COND_CH, dec_chs=DEC_CHS)
    if os.path.isfile(PRETRAIN_CKPT):
        model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location="cpu"), strict=False)
        print(f"[Init] Loaded pretrain from {PRETRAIN_CKPT}")
    else:
        print(f"[Warn] Pretrain ckpt not found: {PRETRAIN_CKPT}")

    model, loader = accelerator.prepare(model, loader)
    model.eval() # 设为 eval 模式，因为我们不训练
    
    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype
    fusion_loss_aux = FusionLoss().to(accelerator.device) if AUX_DECODER_LOSS_W > 0 else None

    # --- 取一个批次进行调试 ---
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Dataloader is empty.")
        return

    A, B = batch[0], batch[1]
    A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
    B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))
    Bsz = A.size(0)
    
    print("\n" + "="*20 + " 1. SAMPLING (OLD POLICY) " + "="*20)
    
    # --- 采样阶段 ---
    with torch.no_grad():
        mu, logvar = model.encode(A, B)
        dist = get_distribution(mu, logvar)
        mu_S = mu.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
        logvar_S = logvar.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
        dist_S = get_distribution(mu_S, logvar_S)
        z_BS = dist_S.rsample()
        logp_old_BS = dist_S.log_prob(z_BS).to(torch.float32).flatten(2).mean(dim=-1)
        z_flat = z_BS.reshape(Bsz * REPEAT_S, *z_BS.shape[2:])
        B_rep = B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1).reshape(Bsz * REPEAT_S, *B.shape[1:])
        A_rep = A.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1).reshape(Bsz * REPEAT_S, *A.shape[1:])
        F_rep = model.decode(z_flat, B_rep)
        R_flat = compute_reward(A_rep, B_rep, F_rep)
        R_BS = R_flat.view(Bsz, REPEAT_S)

        mean_R = R_BS.mean(dim=1, keepdim=True)
        if ADV_NORMALIZE_BY_STD:
            std_R = R_BS.std(dim=1, unbiased=False, keepdim=True) + 1e-6
            adv_BS = (R_BS - mean_R) / std_R
        else:
            adv_BS = (R_BS - mean_R)

    # --- 打印采样信息 ---
    for i in range(min(Bsz, N_SAMPLES_TO_PRINT)):
        print(f"\n--- Sample {i} ---")
        print(f"  Rewards   : {[f'{x:.4f}' for x in R_BS[i].tolist()]}")
        print(f"  Advantage : {[f'{x:.4f}' for x in adv_BS[i].tolist()]}")
        print(f"  LogProb(old): {[f'{x:.4f}' for x in logp_old_BS[i].tolist()]}")

    print("\n" + "="*20 + " 2. LOSS CALCULATION (NEW POLICY) " + "="*20)

    # --- PPO 损失计算 ---
    mu_new_B, logvar_new_B = model.encode(A, B)
    mu_new_BS = mu_new_B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
    logvar_new_BS = logvar_new_B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
    dist_new_BS = get_distribution(mu_new_BS, logvar_new_BS)
    new_logp_BS = dist_new_BS.log_prob(z_BS).to(torch.float32).flatten(2).mean(dim=-1)
    new_logp = new_logp_BS.reshape(-1)

    ratio = torch.exp((new_logp - logp_old_flat).clamp(-20, 20))
    surr1 = ratio * adv_flat
    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_flat
    policy_loss = -torch.min(surr1, surr2).mean() * POLICY_LOSS_COEF

    kld = ConditionalVAE.kl_loss(mu_new_B, logvar_new_B, reduction="mean")
    ent = get_distribution(mu_new_B, logvar_new_B).entropy().to(torch.float32).flatten(1).mean(dim=-1).mean()
    entropy_term = -ENTROPY_COEF * ent

    aux_loss = torch.tensor(0.0, device=accelerator.device)
    if UPDATE_DECODER and AUX_DECODER_LOSS_W > 0:
        z_det = z_flat.detach()
        B_det = B_rep.detach()
        A_det = A_rep.detach()
        F_current = model.decode(z_det, B_det)
        B_for_loss = B_det if B_det.shape[1] == 3 else B_det.repeat(1, 3, 1, 1)
        aux_loss = AUX_DECODER_LOSS_W * fusion_loss_aux(A_det, B_for_loss, F_current)

    total_loss = policy_loss + KL_WEIGHT_RL * kld + entropy_term + aux_loss

    # --- 打印损失信息 ---
    for i in range(min(Bsz, N_SAMPLES_TO_PRINT)):
        start, end = i * REPEAT_S, (i + 1) * REPEAT_S
        print(f"\n--- Sample {i} (Loss Calc) ---")
        print(f"  LogProb(new): {[f'{x:.4f}' for x in new_logp[start:end].tolist()]}")
        print(f"  Ratio       : {[f'{x:.4f}' for x in ratio[start:end].tolist()]}")
    # 额外打印整体 ratio 统计，辅助判断 policy 信号强弱
    print(f"\n[Ratio] mean={ratio.mean().item():.6f} std={ratio.std().item():.6f} "
          f"min={ratio.min().item():.6f} max={ratio.max().item():.6f}")

    print("\n" + "="*20 + " 3. FINAL LOSS SUMMARY " + "="*20)
    
    losses = {
        "Policy Loss": policy_loss.item(),
        "KL Loss (weighted)": (KL_WEIGHT_RL * kld).item(),
        "Entropy Bonus": entropy_term.item(),
        "Aux Decoder Loss": aux_loss.item(),
        "Total Loss": total_loss.item()
    }
    
    total_val = losses["Total Loss"]
    print(f"\n{'Component':<20} | {'Value':<15} | {'Percentage':<15}")
    print("-" * 55)
    for name, value in losses.items():
        percentage = f"{(value / total_val * 100):.2f}%" if total_val != 0 else "N/A"
        print(f"{name:<20} | {value:<15.6f} | {percentage:<15}")
        
    print("\nDebug script finished.")


if __name__ == "__main__":
    main()