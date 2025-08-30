import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision.utils import save_image

from dataset import ImageFusionDataset
from model.policy_net import PolicyNet
from model.traditional_fusion import LaplacianPyramidFusion
from loss.loss import FusionLoss
from metric.MetricGPU import (
    VIF_function_batch,
    Qabf_function_batch,
    SSIM_function_batch,
)

# -------------------------------
# 超参
# -------------------------------
EPOCHS: int = 10
TRAIN_BATCH_SIZE: int = 4
NUM_WORKERS: int = 4
REPEAT_S: int = 4                 # 每张图采样次数
MIXED_PRECISION: str = "bf16"
PROJECT_DIR: str = "./checkpoints/method2_rl_ppo"
SAVE_IMAGES: bool = True
EVAL_FREQ: int = 1
SAVE_FREQ: int = 5
METRIC_MODE: str = "sample"  # "mu" | "sample" -> 评测时指标计算使用的权重类型

# PPO/优化
LR: float = 5e-5                  # 策略网络学习率
PPO_CLIP_EPS: float = 0.2
PPO_EPOCHS: int = 2
KL_WEIGHT_RL: float = 5e-3
ENTROPY_COEF: float = 0.0
MAX_GRAD_NORM: float = 1.0
POLICY_LOSS_COEF: float = 100.0
AUX_FUSION_LOSS_W: float = 1e-3   # 辅助融合损失，稳定画质

# 优势策略
ADV_NORMALIZE_BY_STD: bool = False
ADV_MIN_STD: float = 1e-3
ADV_RESCALE_TARGET_ABS: float = 1.0
ADV_STRATEGY: str = "std"         # "std" | "center" | "wta"

# 评测
EVAL_REPEAT_S: int = 4
SAVE_N_SAMPLES: int = 2
SAVE_N_COLS_CAP: int = 8

# 数据集路径（与其他脚本保持一致）
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
# 预训练策略初始化
PRETRAIN_POLICY: str = "./checkpoints/stochastic_policy_stage1/final/policy_net.pth"

torch.backends.cudnn.benchmark = False


# -------------------------------
# 工具
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5


def get_distribution(mu: torch.Tensor, logvar: torch.Tensor) -> Normal:
    return Normal(mu, torch.exp(0.5 * logvar))


def compute_reward(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor):
    A255 = to_255(A).to(torch.float32)
    B255 = to_255(B).to(torch.float32)
    F255 = to_255(F).to(torch.float32)
    vif = VIF_function_batch(A255, B255, F255)
    qbf = Qabf_function_batch(A255, B255, F255)
    ssim = SSIM_function_batch(A255, B255, F255)
    reward = (vif + 1.5 * qbf + ssim) / 3.0
    return reward, (vif, qbf, ssim)


def build_dataloaders():
    train_paths = DATASETS["MSRS_train"]
    train_ds = ImageFusionDataset(
        dir_A=train_paths["dir_A"], dir_B=train_paths["dir_B"], is_train=True, is_getpatch=False, augment=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    test_paths = DATASETS["MSRS_test"]
    test_ds = ImageFusionDataset(
        dir_A=test_paths["dir_A"], dir_B=test_paths["dir_B"], is_train=False, is_getpatch=False, augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=max(1, NUM_WORKERS // 2), pin_memory=True
    )
    return train_loader, test_loader


def save_compare_grids(out_dir: str, A: torch.Tensor, B: torch.Tensor, F_list: list[torch.Tensor], prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    B3 = B if B.shape[1] == 3 else B.repeat(1, 3, 1, 1)
    Bsz = min(A.size(0), SAVE_N_SAMPLES)
    for i in range(Bsz):
        tiles = [A[i:i+1], B3[i:i+1]]
        tiles.extend([F[i:i+1] for F in F_list])
        grid = torch.cat(tiles[:SAVE_N_COLS_CAP], dim=0)
        save_image(
            grid.detach().to(torch.float32).cpu(),
            os.path.join(out_dir, f"{prefix}_sample{i:02d}.png"),
            nrow=grid.size(0),
            normalize=True,
            value_range=(-1, 1),
            padding=2,
        )


# -------------------------------
# 评测
# -------------------------------
@torch.no_grad()
def evaluate_and_log(
    accelerator: Accelerator,
    policy: PolicyNet,
    fusion_kernel: LaplacianPyramidFusion,
    test_loader: DataLoader,
    epoch: int,
    metric_mode: str = 'sample',  # 'sample' or 'mu'
):
    policy.eval()
    model_dtype = next(accelerator.unwrap_model(policy).parameters()).dtype
    all_reward_oneshot, all_reward_meanS, all_reward_maxS = [], [], []
    all_vifs, all_qbfs, all_ssims = [], [], []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating", disable=not accelerator.is_main_process)):
        A, B = batch[0], batch[1]
        A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
        B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))
        Bsz = A.size(0)

        # 分布参数
        mu, logvar = policy(A, B)  # mu in (0,1), logvar free

        # 多次采样
        mu_S = mu.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1)          # [B,S,1,H,W]
        logvar_S = logvar.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1)  # [B,S,1,H,W]
        dist_S = get_distribution(mu_S, logvar_S)
        w_raw_BS = dist_S.rsample()
        w_BS = torch.clamp(w_raw_BS, 0.0, 1.0)

        # 展平以批量融合
        w_flat = w_BS.reshape(Bsz * EVAL_REPEAT_S, *w_BS.shape[2:])
        A_rep = A.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1).reshape(Bsz * EVAL_REPEAT_S, *A.shape[1:])
        B_rep = B.unsqueeze(1).expand(-1, EVAL_REPEAT_S, -1, -1, -1).reshape(Bsz * EVAL_REPEAT_S, *B.shape[1:])
        F_rep = fusion_kernel(A_rep, B_rep, w_flat)
        # 使用均值权重进行一次确定性融合（用于 mu 模式）
        F_mu = fusion_kernel(A, B, mu.clamp(0.0, 1.0))

        # 回到 [B,S]
        R_flat, _ = compute_reward(A_rep, B_rep, F_rep)
        R_BS = R_flat.view(Bsz, EVAL_REPEAT_S)
        reward_one = R_BS[:, 0]
        reward_meanS = R_BS.mean(dim=1)
        reward_maxS, _ = R_BS.max(dim=1)

        # 根据 metric_mode 选择用于指标计算的融合结果
        F_oneshot = F_rep.view(Bsz, EVAL_REPEAT_S, *F_rep.shape[1:])[:, 0]
        F_metric = F_mu if metric_mode == "mu" else F_oneshot
        _, (vif, qbf, ssim) = compute_reward(A, B, F_metric)

        # 聚合
        reward_one_all, reward_meanS_all, reward_maxS_all, vif_all, qbf_all, ssim_all = accelerator.gather_for_metrics(
            (reward_one, reward_meanS, reward_maxS, vif, qbf, ssim)
        )
        all_reward_oneshot.append(reward_one_all.cpu())
        all_reward_meanS.append(reward_meanS_all.cpu())
        all_reward_maxS.append(reward_maxS_all.cpu())
        all_vifs.append(vif_all.cpu())
        all_qbfs.append(qbf_all.cpu())
        all_ssims.append(ssim_all.cpu())

        # 保存图片
        if i == 0 and accelerator.is_main_process and SAVE_IMAGES:
            out_dir = os.path.join(PROJECT_DIR, "eval_images", f"epoch_{epoch:04d}")
            # 同时保存 mu 与 sample 结果，便于肉眼对比
            save_compare_grids(out_dir, A.detach(), B.detach(),
                               [F_mu.detach(), F_oneshot.detach()],
                               prefix=f"eval_{metric_mode}")

    if accelerator.is_main_process:
        avg_reward_one = torch.cat(all_reward_oneshot).mean().item()
        avg_reward_meanS = torch.cat(all_reward_meanS).mean().item()
        avg_reward_maxS = torch.cat(all_reward_maxS).mean().item()
        avg_vif = torch.cat(all_vifs).mean().item()
        avg_qbf = torch.cat(all_qbfs).mean().item()
        avg_ssim = torch.cat(all_ssims).mean().item()

        print(
            f"\n[Eval Epoch {epoch}] "
            f"Reward(one)={avg_reward_one:.4f} | Reward(meanS)={avg_reward_meanS:.4f} | Reward(maxS)={avg_reward_maxS:.4f} "
            f"| VIF({metric_mode})={avg_vif:.4f} | Qabf({metric_mode})={avg_qbf:.4f} | SSIM({metric_mode})={avg_ssim:.4f}"
        )
        accelerator.log(
            {
                "eval/reward_one": avg_reward_one,
                "eval/reward_meanS": avg_reward_meanS,
                "eval/reward_maxS": avg_reward_maxS,
                f"eval/{metric_mode}/vif": avg_vif,
                f"eval/{metric_mode}/qabf": avg_qbf,
                f"eval/{metric_mode}/ssim": avg_ssim,
            },
            step=epoch,
        )


# -------------------------------
# 训练（PPO微调）
# -------------------------------
def main():
    accelerator = Accelerator(log_with="tensorboard", project_dir=PROJECT_DIR, mixed_precision=MIXED_PRECISION)
    if accelerator.is_main_process:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        print(
            f"[Config] epochs={EPOCHS}, bs={TRAIN_BATCH_SIZE}, S={REPEAT_S}, lr={LR}, "
            f"clip={PPO_CLIP_EPS}, ppo_epochs={PPO_EPOCHS}, kl_w={KL_WEIGHT_RL}, mp={MIXED_PRECISION}"
        )

    # 数据
    train_loader, test_loader = build_dataloaders()

    # 模型与环境
    policy = PolicyNet(in_channels=4, out_channels=2)
    if os.path.isfile(PRETRAIN_POLICY):
        try:
            policy.load_state_dict(torch.load(PRETRAIN_POLICY, map_location="cpu"), strict=False)
            if accelerator.is_main_process:
                print(f"[Init] Loaded pretrained policy from {PRETRAIN_POLICY}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Init] Failed to load pretrained policy: {e}")
    fusion_kernel = LaplacianPyramidFusion(num_levels=4)
    fusion_kernel.eval()  # 固定环境

    # 优化器与辅助损失
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4, eps=1e-8)
    fusion_loss_aux = FusionLoss().to(accelerator.device) if AUX_FUSION_LOSS_W > 0 else None

    # 准备
    policy, opt, train_loader, test_loader = accelerator.prepare(policy, opt, train_loader, test_loader)
    model_dtype = next(accelerator.unwrap_model(policy).parameters()).dtype
    fusion_kernel = fusion_kernel.to(device=accelerator.device, dtype=model_dtype)

    accelerator.init_trackers("method2_rl_ppo")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        policy.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            A, B = batch[0], batch[1]
            A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
            B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))
            Bsz = A.size(0)

            # --------- 采样阶段（old）---------
            with torch.no_grad():
                mu, logvar = policy(A, B)  # [B,1,H,W]
                # 扩展到 S 次采样
                mu_S = mu.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)           # [B,S,1,H,W]
                logvar_S = logvar.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)   # [B,S,1,H,W]
                dist_S = get_distribution(mu_S, logvar_S)
                w_raw_BS = dist_S.rsample()                                       # [B,S,1,H,W]
                # log_prob 对元素维做均值，避免维度求和数值爆炸
                logp_old_BS = dist_S.log_prob(w_raw_BS).to(torch.float32).flatten(2).mean(dim=-1)  # [B,S]

                # 用裁剪后的 w 进行融合与奖励
                w_BS = torch.clamp(w_raw_BS, 0.0, 1.0)
                w_flat = w_BS.reshape(Bsz * REPEAT_S, *w_BS.shape[2:])
                A_rep = A.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1).reshape(Bsz * REPEAT_S, *A.shape[1:])
                B_rep = B.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1).reshape(Bsz * REPEAT_S, *B.shape[1:])
                F_rep = fusion_kernel(A_rep, B_rep, w_flat)
                # 使用均值权重进行一次确定性融合（用于 mu 模式）
                F_mu = fusion_kernel(A, B, mu.clamp(0.0, 1.0))

                # 回到 [B,S]
                R_flat, _ = compute_reward(A_rep, B_rep, F_rep)
                R_BS = R_flat.view(Bsz, REPEAT_S)

            # 优势
            mean_R = R_BS.mean(dim=1, keepdim=True)
            if ADV_STRATEGY == "wta":
                adv_BS = torch.zeros_like(R_BS)
                idx = torch.argmax(R_BS, dim=1, keepdim=True)
                adv_BS.scatter_(1, idx, 1.0)
            else:
                adv_BS = R_BS - mean_R
                if ADV_STRATEGY == "std" or ADV_NORMALIZE_BY_STD:
                    std_R = R_BS.std(dim=1, unbiased=False, keepdim=True).clamp_min(ADV_MIN_STD)
                    adv_BS = adv_BS / std_R

            # 重标定到固定平均幅度
            adv_abs_mean = adv_BS.abs().mean().clamp_min(1e-8)
            adv_BS = adv_BS * (ADV_RESCALE_TARGET_ABS / adv_abs_mean)

            # 展平
            logp_old_flat = logp_old_BS.reshape(-1).detach()
            adv_flat = adv_BS.reshape(-1).detach()
            w_raw_flat = w_raw_BS.reshape(Bsz * REPEAT_S, *w_raw_BS.shape[2:]).detach()  # 作为旧样本

            # --------- PPO 更新（new）---------
            for _ in range(PPO_EPOCHS):
                opt.zero_grad(set_to_none=True)

                mu_new, logvar_new = policy(A, B)  # [B,1,H,W]
                mu_new_BS = mu_new.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
                logvar_new_BS = logvar_new.unsqueeze(1).expand(-1, REPEAT_S, -1, -1, -1)
                dist_new_BS = get_distribution(mu_new_BS, logvar_new_BS)
                new_logp_BS = dist_new_BS.log_prob(
                    w_raw_BS  # 注意：用未裁剪样本评估对数概率，保持一致性
                ).to(torch.float32).flatten(2).mean(dim=-1)  # [B,S]
                new_logp = new_logp_BS.reshape(-1)

                ratio = torch.exp((new_logp - logp_old_flat).clamp(-20, 20))
                surr1 = ratio * adv_flat
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_flat
                policy_loss = -torch.min(surr1, surr2).mean() * POLICY_LOSS_COEF

                # KL 正则（以 B 级统计）
                kld = (-0.5 * (1 + logvar_new - mu_new.pow(2) - logvar_new.exp())).mean()

                # 熵（可选）
                ent = dist_new_BS.entropy().to(torch.float32).flatten(2).mean(dim=-1).mean()
                entropy_term = -ENTROPY_COEF * ent

                # 可选辅助融合损失（防止退化）
                aux_loss = torch.tensor(0.0, device=accelerator.device)
                if AUX_FUSION_LOSS_W > 0 and fusion_loss_aux is not None:
                    # 使用当前策略的均值权重进行一次确定性融合，避免反传到 A/B
                    with torch.no_grad():
                        w_mu = mu_new.clamp(0.0, 1.0)
                        F_mu = fusion_kernel(A, B, w_mu)
                    B_for_loss = B if B.shape[1] == 3 else B.repeat(1, 3, 1, 1)
                    aux_loss = AUX_FUSION_LOSS_W * fusion_loss_aux(A, B_for_loss, F_mu)

                total_loss = policy_loss + KL_WEIGHT_RL * kld + entropy_term + aux_loss
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()

            # 日志
            if accelerator.is_main_process:
                reward_mean, reward_std = R_BS.mean().item(), R_BS.std().item()
                adv_abs = adv_flat.abs().mean().item()
                ratio_mean, ratio_std = ratio.mean().item(), ratio.std().item()
                pbar.set_postfix({"reward_mean": f"{reward_mean:.4f}", 
                                  "adv_abs": f"{adv_abs:.3f}", 
                                  "ratio": f"{ratio_mean:.3f}", 
                                  "loss": f"{total_loss.item():.3f}", 
                                  "policy_loss": f"{policy_loss.item():.3f}", 
                                  "kld": f"{kld.item():.3f}",
                                  "aux_loss": f"{aux_loss.item():.3f}"})
                accelerator.log(
                    {
                        "rl/policy_loss": policy_loss.item(),
                        "rl/kld": kld.item(),
                        "rl/entropy": ent.item() if ENTROPY_COEF > 0 else 0.0,
                        "rl/aux_loss": aux_loss.item(),
                        "rl/reward_mean": reward_mean,
                        "rl/reward_std": reward_std,
                        "rl/ratio_mean": ratio_mean,
                        "rl/ratio_std": ratio_std,
                        "rl/adv_abs": adv_abs,
                    },
                    step=global_step,
                )
            global_step += 1

        # 评测与保存
        if epoch % EVAL_FREQ == 0:
            evaluate_and_log(accelerator, policy, fusion_kernel, test_loader, epoch, metric_mode=METRIC_MODE)

        if accelerator.is_main_process and (epoch % SAVE_FREQ == 0 or epoch == EPOCHS):
            # 仅保存策略网络权重
            save_dir = os.path.join(PROJECT_DIR, f"epoch_{epoch:04d}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(policy).state_dict(), os.path.join(save_dir, "policy_net.pth"))
            print(f"\n[Save] Epoch {epoch} model -> {save_dir}")

    if accelerator.is_main_process:
        final_dir = os.path.join(PROJECT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(accelerator.unwrap_model(policy).state_dict(), os.path.join(final_dir, "policy_net.pth"))
        print(f"[Final] model -> {final_dir}")


if __name__ == "__main__":
    main()