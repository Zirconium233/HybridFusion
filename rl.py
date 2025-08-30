import os
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from accelerate import Accelerator, find_executable_batch_size
from tqdm.auto import tqdm
from torchvision.utils import save_image

from dataset import ImageFusionDataset
from model.cvae import ConditionalVAE
from loss.loss import FusionLoss
from metric.MetricGPU import (
    VIF_function_batch,
    Qabf_function_batch,
    SSIM_function_batch,
)

# -------------------------------
# 硬编码超参（按需修改）
# -------------------------------
# 训练与采样
EPOCHS: int = 200
TRAIN_BATCH_SIZE: int = 4
NUM_WORKERS: int = 4
REPEAT_S: int = 4
MIXED_PRECISION: str = "bf16"
PROJECT_DIR: str = "./checkpoints/rl_ppo_v2"
SAVE_IMAGES: bool = True
EVAL_FREQ: int = 1
SAVE_FREQ: int = 25

# PPO/优化
LR_ENCODER: float = 1e-5
LR_DECODER: float = 5e-6
UPDATE_DECODER: bool = True
PPO_CLIP_EPS: float = 0.2
PPO_EPOCHS: int = 2
KL_WEIGHT_RL: float = 5e-4
ENTROPY_COEF: float = 0.0  # 暂停熵奖励，先让reward抬升
MAX_GRAD_NORM: float = 1.0
POLICY_LOSS_COEF: float = 10.0
AUX_DECODER_LOSS_W: float = 5e-4  # 适度增大，防止decoder失配
# 优势标准化/策略
ADV_NORMALIZE_BY_STD: bool = True      # 启用标准化
ADV_MIN_STD: float = 1e-3              # 组内std下限，避免A几乎为0
ADV_RESCALE_TARGET_ABS: float = 1.0    # 将|A|的batch均值重标定到该目标
ADV_STRATEGY: str = "std"              # "std" | "center" | "wta" (winner-take-all)

# 验证多采样设置
EVAL_REPEAT_S: int = 4

# 模型规模（与预训练一致）
ENC_BASE_CHS: Tuple[int, int, int] = (192, 256, 384)
Z_CH: int = 32
COND_CH: int = 64
DEC_CHS: Tuple[int, int, int, int] = (384, 256, 128, 96)

# 数据集路径
DATASETS: Dict[str, Dict[str, str]] = {
    "MSRS_train": {
        "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
        "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
    },
    "MSRS_test": {
        "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
        "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
    }
}
PRETRAIN_DIR: str = "./checkpoints/pretrain_vae/final"  # 预训练模型路径

# 图像保存数量
SAVE_N_SAMPLES: int = 2
SAVE_N_COLS_CAP: int = 8

torch.backends.cudnn.benchmark = False


def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5


def get_distribution(mu: torch.Tensor, logvar: torch.Tensor) -> Normal:
    """从 mu 和 logvar 创建对角高斯分布"""
    return Normal(mu, torch.exp(0.5 * logvar))


def compute_reward(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """方式1：指标：(VIF + Qabf * 1.5 + SSIM) / 3"""
    A255 = to_255(A).to(torch.float32)
    B255 = to_255(B).to(torch.float32)
    F255 = to_255(F).to(torch.float32)
    vif = VIF_function_batch(A255, B255, F255)
    qbf = Qabf_function_batch(A255, B255, F255)
    ssim = SSIM_function_batch(A255, B255, F255)
    return (vif + 1.5 * qbf + ssim) / 3.0, (vif, qbf, ssim)


def build_dataloaders():
    train_paths = DATASETS["MSRS_train"]
    train_ds = ImageFusionDataset(dir_A=train_paths["dir_A"], dir_B=train_paths["dir_B"], is_train=True)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    test_paths = DATASETS["MSRS_test"]
    test_ds = ImageFusionDataset(dir_A=test_paths["dir_A"], dir_B=test_paths["dir_B"], is_train=False)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader


def save_compare_grids(out_dir: str, A: torch.Tensor, B: torch.Tensor, F_list: list[torch.Tensor], prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    B3 = B if B.shape[1] == 3 else B.repeat(1, 3, 1, 1)
    Bsz = min(A.size(0), SAVE_N_SAMPLES)
    for i in range(Bsz):
        tiles = [A[i:i+1], B3[i:i+1]]
        tiles.extend([F[i:i+1] for F in F_list])
        grid = torch.cat(tiles[:SAVE_N_COLS_CAP], dim=0)
        save_image(grid.detach().to(torch.float32).cpu(), os.path.join(out_dir, f"{prefix}_sample{i:02d}.png"),
                   nrow=grid.size(0), normalize=True, value_range=(-1, 1), padding=2)


@torch.no_grad()
def evaluate_and_log(accelerator: Accelerator, model: nn.Module, test_loader: DataLoader, epoch: int):
    model.eval()
    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype
    all_reward_oneshot, all_reward_meanS, all_reward_maxS = [], [], []
    all_vifs, all_qbfs, all_ssims = [], [], []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating", disable=not accelerator.is_main_process)):
        A, B = batch[0], batch[1]
        A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
        B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))

        # 评测：多次采样，统计 One-shot / Mean-of-S / Best-of-S
        mu, logvar = model.encode(A, B)
        mu_S = mu.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1)
        logvar_S = logvar.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1)
        dist_S = get_distribution(mu_S, logvar_S)
        z_BS = dist_S.rsample()
        Bsz = A.size(0)
        z_flat = z_BS.reshape(Bsz * EVAL_REPEAT_S, *z_BS.shape[2:])
        B_rep = B.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1).reshape(Bsz * EVAL_REPEAT_S, *B.shape[1:])
        A_rep = A.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1).reshape(Bsz * EVAL_REPEAT_S, *A.shape[1:])
        F_rep = model.decode(z_flat, B_rep)
        R_flat, _ = compute_reward(A_rep, B_rep, F_rep)
        R_BS = R_flat.view(Bsz, EVAL_REPEAT_S)
        reward_one = R_BS[:, 0]
        reward_meanS = R_BS.mean(dim=1)
        reward_maxS, _ = R_BS.max(dim=1)

        # 同时计算一次 One-shot 的指标（和之前保持一致）
        F_oneshot = F_rep.view(Bsz, EVAL_REPEAT_S, *F_rep.shape[1:])[:, 0]
        reward_for_metrics, (vif, qbf, ssim) = compute_reward(A, B, F_oneshot)

        # 聚合多卡结果
        reward_one_all, reward_meanS_all, reward_maxS_all, vif_all, qbf_all, ssim_all = accelerator.gather_for_metrics(
            (reward_one, reward_meanS, reward_maxS, vif, qbf, ssim)
        )
        all_reward_oneshot.append(reward_one_all.cpu())
        all_reward_meanS.append(reward_meanS_all.cpu())
        all_reward_maxS.append(reward_maxS_all.cpu())
        all_vifs.append(vif_all.cpu())
        all_qbfs.append(qbf_all.cpu())
        all_ssims.append(ssim_all.cpu())

        # 只在主进程保存首个batch的图片
        if i == 0 and accelerator.is_main_process and SAVE_IMAGES:
            out_dir = os.path.join(PROJECT_DIR, "eval_images", f"epoch_{epoch:04d}")
            # 保存该 batch 的多次采样前几张
            F_list = [F_oneshot.detach()]
            save_compare_grids(out_dir, A.detach(), B.detach(), F_list, prefix="eval")

    if accelerator.is_main_process:
        avg_reward_one = torch.cat(all_reward_oneshot).mean().item()
        avg_reward_meanS = torch.cat(all_reward_meanS).mean().item()
        avg_reward_maxS = torch.cat(all_reward_maxS).mean().item()
        avg_vif = torch.cat(all_vifs).mean().item()
        avg_qbf = torch.cat(all_qbfs).mean().item()
        avg_ssim = torch.cat(all_ssims).mean().item()
        
        print(f"\n[Eval Epoch {epoch}] "
              f"Reward(one)={avg_reward_one:.4f} | Reward(meanS)={avg_reward_meanS:.4f} | Reward(maxS)={avg_reward_maxS:.4f} "
              f"| VIF(one)={avg_vif:.4f} | Qabf(one)={avg_qbf:.4f} | SSIM(one)={avg_ssim:.4f}")
        log_dict = {
            "eval/reward_one": avg_reward_one,
            "eval/reward_meanS": avg_reward_meanS,
            "eval/reward_maxS": avg_reward_maxS,
            "eval/avg_vif": avg_vif,
            "eval/avg_qbf": avg_qbf,
            "eval/avg_ssim": avg_ssim,
        }
        accelerator.log(log_dict, step=epoch)


def main():
    accelerator = Accelerator(log_with="tensorboard", project_dir=PROJECT_DIR, mixed_precision=MIXED_PRECISION)
    if accelerator.is_main_process:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        print(f"[Config] epochs={EPOCHS}, bs={TRAIN_BATCH_SIZE}, S={REPEAT_S}, lr_enc={LR_ENCODER}, "
              f"clip={PPO_CLIP_EPS}, ppo_epochs={PPO_EPOCHS}, kl_w={KL_WEIGHT_RL}, mp={MIXED_PRECISION}")

    train_loader, test_loader = build_dataloaders()

    model = ConditionalVAE(in_ch=4, base_chs=ENC_BASE_CHS, z_ch=Z_CH, cond_ch=COND_CH, dec_chs=DEC_CHS)
    
    opt_params = [{"params": model.encoder.parameters(), "lr": LR_ENCODER}]
    if UPDATE_DECODER:
        opt_params.append({"params": model.decoder.parameters(), "lr": LR_DECODER})
    opt = torch.optim.AdamW(opt_params, weight_decay=1e-4, eps=1e-8)

    model, opt, train_loader, test_loader = accelerator.prepare(model, opt, train_loader, test_loader)
    
    # 尝试从检查点恢复
    try:
        accelerator.load_state(PROJECT_DIR)
        if accelerator.is_main_process: print(f"Resumed from checkpoint: {PROJECT_DIR}")
    except:
        # 如果恢复失败，则从预训练模型加载
        ckpt = os.path.join(PRETRAIN_DIR, "cvae.pth")
        if os.path.isfile(ckpt):
            accelerator.unwrap_model(model).load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
            if accelerator.is_main_process: print(f"[Init] Loaded pretrain from {ckpt}")
        else:
            if accelerator.is_main_process: print("[Init] No checkpoint or pretrain found. Starting from scratch.")

    accelerator.init_trackers("rl_ppo")
    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype
    fusion_loss_aux = FusionLoss().to(accelerator.device) if AUX_DECODER_LOSS_W > 0 else None

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            A, B = batch[0], batch[1]
            A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
            B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))
            Bsz = A.size(0)

            # ------------- 向量化采样阶段 (old) -------------
            with torch.no_grad():
                mu, logvar = model.encode(A, B)
                dist = get_distribution(mu, logvar)
                # 扩展 mu/logvar 以进行 S 次采样
                mu_S = mu.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
                logvar_S = logvar.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
                dist_S = get_distribution(mu_S, logvar_S)
                
                z_BS = dist_S.rsample()  # [B, S, C, h, w]
                # 关键修复：对 log_prob 做“均值归一化”，避免维度求和导致数值爆炸
                logp_old_BS = dist_S.log_prob(z_BS).to(torch.float32).flatten(2).mean(dim=-1)  # [B,S]

                # 展平以进行批处理解码
                z_flat = z_BS.reshape(Bsz * REPEAT_S, *z_BS.shape[2:])
                B_rep = B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1).reshape(Bsz * REPEAT_S, *B.shape[1:])
                A_rep = A.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1).reshape(Bsz * REPEAT_S, *A.shape[1:])
                
                F_rep = model.decode(z_flat, B_rep) # [B*S, 3, H, W]
                R_flat, _ = compute_reward(A_rep, B_rep, F_rep)
                R_BS = R_flat.view(Bsz, REPEAT_S)

                # 优势计算（支持三种策略）
            mean_R = R_BS.mean(dim=1, keepdim=True)
            if ADV_STRATEGY == "wta":
                adv_BS = torch.zeros_like(R_BS)
                idx = torch.argmax(R_BS, dim=1, keepdim=True)
                adv_BS.scatter_(1, idx, 1.0)  # 赢家得+1，其余0
            else:
                adv_BS = R_BS - mean_R
                if ADV_STRATEGY == "std" or ADV_NORMALIZE_BY_STD:
                    std_R = R_BS.std(dim=1, unbiased=False, keepdim=True)
                    std_R = torch.clamp(std_R, min=ADV_MIN_STD)
                    adv_BS = adv_BS / std_R

            # 重标定到固定平均幅度，保证policy梯度量级稳定
            adv_abs_mean = adv_BS.abs().mean().clamp_min(1e-8)
            adv_BS = adv_BS * (ADV_RESCALE_TARGET_ABS / adv_abs_mean)

            # 展平数据以进行 PPO 更新
            logp_old_flat = logp_old_BS.reshape(-1).detach()
            adv_flat = adv_BS.reshape(-1).detach()

            # ------------- PPO 更新阶段 (new) -------------
            for _ in range(PPO_EPOCHS):
                opt.zero_grad(set_to_none=True)
                
                # 关键优化：只对原始 B 个样本编码一次，再在 S 维上展开，避免 B*S 倍的显存/计算
                mu_new_B, logvar_new_B = model.encode(A, B)                                  # [B,zc,h,w]
                mu_new_BS = mu_new_B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)           # [B,S,zc,h,w]
                logvar_new_BS = logvar_new_B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)   # [B,S,zc,h,w]
                dist_new_BS = get_distribution(mu_new_BS, logvar_new_BS)
                new_logp_BS = dist_new_BS.log_prob(z_BS).to(torch.float32).flatten(2).mean(dim=-1)  # [B,S]
                new_logp = new_logp_BS.reshape(-1)

                ratio = torch.exp((new_logp - logp_old_flat).clamp(-20, 20))
                surr1 = ratio * adv_flat
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_flat
                policy_loss = -torch.min(surr1, surr2).mean() * POLICY_LOSS_COEF

                # KL/熵仅对 B 计算一次（与策略网络一致）
                kld = ConditionalVAE.kl_loss(mu_new_B, logvar_new_B, reduction="mean")
                # 熵同样对元素维取均值，避免量纲爆炸（仅 B）
                ent = get_distribution(mu_new_B, logvar_new_B).entropy().to(torch.float32).flatten(1).mean(dim=-1).mean()
                entropy_term = -ENTROPY_COEF * ent

                aux_loss = torch.tensor(0.0, device=accelerator.device)
                if UPDATE_DECODER and AUX_DECODER_LOSS_W > 0:
                     # 关键：detach 掉 encoder 相关的输入，避免 aux 反传进入 encoder
                     z_det = z_flat.detach()
                     B_det = B_rep.detach()
                     A_det = A_rep.detach()
                     F_current = model.decode(z_det, B_det)
                     B_for_loss = B_det if B_det.shape[1] == 3 else B_det.repeat(1, 3, 1, 1)
                     aux_loss = AUX_DECODER_LOSS_W * fusion_loss_aux(A_det, B_for_loss, F_current)

                total_loss = policy_loss + KL_WEIGHT_RL * kld + entropy_term + aux_loss
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

            # 日志与打印
            if accelerator.is_main_process:
                reward_mean, reward_std = R_BS.mean().item(), R_BS.std().item()
                adv_abs = adv_flat.abs().mean().item()
                ratio_mean, ratio_std = ratio.mean().item(), ratio.std().item()
                pbar.set_postfix({"reward_mean": f"{reward_mean:.4f}", "adv_abs": f"{adv_abs:.3f}", "ratio": f"{ratio_mean:.3f}"})
                log_dict = {
                    "rl/policy_loss": policy_loss.item(), "rl/kld": kld.item(),
                    "rl/entropy": ent.item() if ENTROPY_COEF > 0 else 0.0, "rl/aux_loss": aux_loss.item(),
                    "rl/reward_mean": reward_mean, "rl/reward_std": reward_std,
                    "rl/ratio_mean": ratio_mean, "rl/ratio_std": ratio_std,
                    "rl/adv_abs": adv_abs,
                 }
                accelerator.log(log_dict, step=global_step)
            global_step += 1

        # 评测与保存
        if epoch % EVAL_FREQ == 0:
            evaluate_and_log(accelerator, model, test_loader, epoch)
        
        if accelerator.is_main_process and (epoch % SAVE_FREQ == 0 or epoch == EPOCHS):
            accelerator.save_state(PROJECT_DIR)
            print(f"\n[Save] Epoch {epoch} state saved to {PROJECT_DIR}")

    if accelerator.is_main_process:
        accelerator.save_state(os.path.join(PROJECT_DIR, "final"))
        print(f"[Final] state saved to {os.path.join(PROJECT_DIR, 'final')}")

if __name__ == "__main__":
    main()