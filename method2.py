import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision.utils import save_image

from dataset import ImageFusionDataset
from model.policy_net import PolicyNet
from loss.loss import FusionLoss
from model.traditional_fusion import LaplacianPyramidFusion
from metric.MetricGPU import VIF_function_batch, Qabf_function_batch, SSIM_function_batch

# -------------------------------
# 超参数配置
# -------------------------------
EPOCHS: int = 10
LR: float = 1e-4
# KL 散度权重：鼓励策略网络在自监督阶段保留一定随机性
KL_WEIGHT: float = 1e-5
# 融合自监督损失缩放（便于将KLD控制在合适量级）
LOSS_SCALE_FACTOR: float = 0.1
# 混合精度
MIXED_PRECISION: str = "bf16"  # "no" | "fp16" | "bf16"
PROJECT_DIR: str = "./checkpoints/stochastic_policy_stage1"
SAVE_IMAGES_TO_DIR: bool = True
TRAIN_BATCH_SIZE: int = 16
TEST_BATCH_SIZE: int = 2
NUM_WORKERS: int = 4
GRAD_ACCUM_STEPS: int = 2
MAX_GRAD_NORM: float = 1.0
TEST_FREQ: int = 1
SAVE_FREQ: int = 5

# -------------------------------
# 数据集路径（默认 MSRS，与 pretrain_vae.py 对齐）
# -------------------------------
DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "MSRS": {
        "train": {
            "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
            # C 可选，这里不需要监督标签
        },
        "test": {
            "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
        },
    }
}
TEST_SET_NAMES = ["MSRS"]

torch.backends.cudnn.benchmark = False


# -------------------------------
# 工具函数（与 pretrain_vae.py 一致风格）
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def to_01(x: torch.Tensor) -> torch.Tensor:
    # 输入范围 [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) * 0.5


def to_255(x: torch.Tensor) -> torch.Tensor:
    # 输入范围 [-1,1] -> [0,255]
    return (x.clamp(-1, 1) + 1.0) * 127.5


def save_image_grid(path: str, img: torch.Tensor, nrow: int = 4):
    """
    保存 B,C,H,W 张量到 path，范围[-1,1]；使用 torchvision.utils.save_image 生成网格。
    """
    x = img.detach().to(torch.float32).cpu().clamp(-1, 1)
    save_image(x, path, nrow=min(nrow, x.size(0)), normalize=True, value_range=(-1, 1), padding=2)


def log_images_tb(accelerator: Accelerator, tag: str, images: torch.Tensor, step: int):
    """
    将 [-1,1] 的 B,C,H,W 图像写入 TensorBoard（0..1）。
    """
    if not accelerator.is_main_process or len(accelerator.trackers) == 0:
        return
    writer = None
    for t in accelerator.trackers:
        if getattr(t, "name", "") == "tensorboard":
            writer = getattr(t, "writer", None)
            break
    if writer is None:
        return
    imgs = to_01(images.detach().to(torch.float32).cpu())
    writer.add_images(tag, imgs, global_step=step)


def build_dataloaders(accelerator: Accelerator):
    train_paths = DATASETS["MSRS"]["train"]
    train_ds = ImageFusionDataset(
        dir_A=train_paths["dir_A"],
        dir_B=train_paths["dir_B"],
        dir_C=None,
        is_train=True,
        is_getpatch=False,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loaders = {}
    for name in TEST_SET_NAMES:
        paths = DATASETS[name]["test"]
        test_ds = ImageFusionDataset(
            dir_A=paths["dir_A"],
            dir_B=paths["dir_B"],
            dir_C=None,
            is_train=False,
            is_getpatch=False,
            augment=False,
        )
        test_loaders[name] = DataLoader(
            test_ds,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=True,
        )
    return train_loader, test_loaders


# -------------------------------
# 训练主流程
# -------------------------------
def main():
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=PROJECT_DIR,
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    )
    accelerator.init_trackers("stochastic_policy_stage1")

    if accelerator.is_main_process:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        print(f"[Config] epochs={EPOCHS}, lr={LR}, kl_w={KL_WEIGHT}, loss_scale={LOSS_SCALE_FACTOR}, mp={MIXED_PRECISION}")
        print(f"[Dirs] project_dir={PROJECT_DIR}")

    # 数据
    train_loader, test_loaders = build_dataloaders(accelerator)

    # 1) 策略网络（可训练）
    policy_net = PolicyNet(in_channels=4, out_channels=2)
    # 2) 固定的拉普拉斯金字塔融合核（不可训练）
    fusion_kernel = LaplacianPyramidFusion(num_levels=4)
    # 3) 自监督融合损失
    fusion_loss_fn = FusionLoss()

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=1e-4)

    # 加速准备：仅将可训练模块和训练 dataloader 交给 accelerate
    policy_net, optimizer, train_loader = accelerator.prepare(policy_net, optimizer, train_loader)

    # 模型参数 dtype（bf16/fp16/fp32），使输入与之对齐
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype

    # 无参数模块放到正确 device/dtype
    fusion_kernel = fusion_kernel.to(device=accelerator.device, dtype=model_dtype)
    fusion_loss_fn = fusion_loss_fn.to(device=accelerator.device, dtype=model_dtype)

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        policy_net.train()
        # 统计（epoch 级别）
        epoch_total_loss_sum = 0.0
        epoch_fusion_loss_sum = 0.0
        epoch_kld_sum = 0.0
        epoch_sample_count = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            # 兼容数据集返回 (A,B) 或 (A,B,C)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    A, B = batch[0], batch[1]
                else:
                    raise RuntimeError("Train batch should provide (A,B[,C]).")
            else:
                raise RuntimeError("Dataset should return a tuple/list (A,B[,C]).")

            # 入模前对齐 device/dtype，并保持 channels_last
            A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
            B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))

            with accelerator.accumulate(policy_net):
                # 1) 策略网络输出分布参数（均值/对数方差）
                mu, logvar = policy_net(A, B)  # mu in (0,1), logvar unrestricted

                # 2) 采样权重图（重参数化），并裁剪到 [0,1]
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                sampled_w = torch.clamp(mu + eps * std, 0.0, 1.0)

                # 3) 固定融合核执行融合
                #    注意：融合核期望 B 是单通道（若 B 为单通道，这里直接传）
                F_hat = fusion_kernel(A, B, sampled_w)

                # 4) 融合自监督损失（需要三者通道一致；将 IR 扩 3 通道）
                B_for_loss = B if B.shape[1] == 3 else B.repeat(1, 3, 1, 1)
                fusion_loss = fusion_loss_fn(A, B_for_loss, F_hat)

                # 5) KL 正则（权重图分布 ~ N(0, I)）
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss = fusion_loss * LOSS_SCALE_FACTOR + KL_WEIGHT * kld_loss

                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            bsz = A.shape[0]
            epoch_total_loss_sum += float(total_loss.item()) * bsz
            epoch_fusion_loss_sum += float(fusion_loss.item()) * bsz
            epoch_kld_sum += float(kld_loss.item()) * bsz
            epoch_sample_count += bsz
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "fusion": f"{fusion_loss.item():.2f}",
                    "kld": f"{kld_loss.item():.4f}",
                })
                accelerator.log(
                    {
                        "train/total_loss": float(total_loss.item()),
                        "train/fusion_loss": float(fusion_loss.item()),
                        "train/kld_loss": float(kld_loss.item()),
                    },
                    step=global_step,
                )

        # 跨进程聚合 epoch 级统计
        stats_local = torch.tensor(
            [epoch_total_loss_sum, epoch_fusion_loss_sum, epoch_kld_sum, epoch_sample_count],
            device=accelerator.device,
            dtype=torch.float64,
        )
        stats_all = accelerator.gather_for_metrics(stats_local)  # [world, 4]
        if stats_all.ndim == 1:
            stats_all = stats_all.unsqueeze(0)
        totals = stats_all.sum(dim=0)
        total_samples = max(1.0, float(totals[3].item()))
        avg_total = float(totals[0].item()) / total_samples
        avg_fusion = float(totals[1].item()) / total_samples
        avg_kld = float(totals[2].item()) / total_samples

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] avg_total={avg_total:.4f}  avg_fusion={avg_fusion:.4f}  avg_kld={avg_kld:.6f}")
            accelerator.log(
                {"epoch/avg_total": avg_total, "epoch/avg_fusion": avg_fusion, "epoch/avg_kld": avg_kld},
                step=epoch,
            )

        # 测试与可视化
        if epoch % TEST_FREQ == 0 or epoch == EPOCHS:
            evaluate_and_log(accelerator, policy_net, fusion_kernel, test_loaders, epoch)

        # 保存
        if epoch % SAVE_FREQ == 0 and accelerator.is_main_process:
            save_dir = os.path.join(PROJECT_DIR, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(policy_net)
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "policy_net.pth"))
            print(f"[Save] model -> {save_dir}")

    # 最终保存
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(PROJECT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy_net)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "policy_net.pth"))
        print(f"[Final] model -> {final_dir}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


# -------------------------------
# 评估与日志
# -------------------------------
@torch.no_grad()
def evaluate_and_log(
    accelerator: Accelerator,
    policy_net: PolicyNet,
    fusion_kernel: LaplacianPyramidFusion,
    test_loaders: Dict[str, DataLoader],
    epoch: int,
):
    policy_net.eval()
    device = accelerator.device
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype

    for set_name, loader in test_loaders.items():
        if accelerator.is_main_process:
            print(f"\n[Eval] Epoch {epoch} - {set_name}")

        all_vif_cpu = []
        all_qbf_cpu = []
        all_ssim_cpu = []

        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    A, B = batch[0], batch[1]
                else:
                    raise RuntimeError("Test batch should provide (A,B[,C]).")
            else:
                raise RuntimeError("Test dataset should return a tuple/list (A,B[,C]).")

            A = to_ch_last(A.to(device=device, dtype=model_dtype))
            B = to_ch_last(B.to(device=device, dtype=model_dtype))

            # 策略均值图（确定性融合）与一次随机采样（随机融合）
            mu, logvar = policy_net(A, B)
            std = torch.exp(0.5 * logvar)

            F_hat_mu = fusion_kernel(A, B, mu)
            sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)
            F_hat_sampled = fusion_kernel(A, B, sampled_w)

            # 仅第一个 batch 记录图像
            if batch_idx == 0:
                tag = f"images/{set_name}"
                log_images_tb(accelerator, f"{tag}/A_vis", A, step=epoch)
                log_images_tb(accelerator, f"{tag}/B_ir", B, step=epoch)
                log_images_tb(accelerator, f"{tag}/Control_mu", mu, step=epoch)
                log_images_tb(accelerator, f"{tag}/Control_std", std, step=epoch)
                log_images_tb(accelerator, f"{tag}/Fused_deterministic", F_hat_mu, step=epoch)
                log_images_tb(accelerator, f"{tag}/Fused_stochastic", F_hat_sampled, step=epoch)

                if accelerator.is_main_process and SAVE_IMAGES_TO_DIR:
                    out_dir = os.path.join(PROJECT_DIR, "images", f"epoch_{epoch}", set_name)
                    os.makedirs(out_dir, exist_ok=True)
                    save_image_grid(os.path.join(out_dir, f"A_e{epoch:04d}.png"), A)
                    save_image_grid(os.path.join(out_dir, f"B_e{epoch:04d}.png"), B)
                    save_image_grid(os.path.join(out_dir, f"mu_e{epoch:04d}.png"), mu)
                    save_image_grid(os.path.join(out_dir, f"F_mu_e{epoch:04d}.png"), F_hat_mu)
                    save_image_grid(os.path.join(out_dir, f"F_sample_e{epoch:04d}.png"), F_hat_sampled)

            # 指标（在确定性融合上评估）
            A_255 = to_255(A).to(torch.float32)
            B_255 = to_255(B).to(torch.float32)
            F_255 = to_255(F_hat_mu).to(torch.float32)

            try:
                vif = VIF_function_batch(A_255, B_255, F_255)
                qbf = Qabf_function_batch(A_255, B_255, F_255)
                ssim = SSIM_function_batch(A_255, B_255, F_255)
                # 聚合各进程
                vif_all = accelerator.gather_for_metrics(vif.reshape(-1)).float().cpu()
                qbf_all = accelerator.gather_for_metrics(qbf.reshape(-1)).float().cpu()
                ssim_all = accelerator.gather_for_metrics(ssim.reshape(-1)).float().cpu()
                all_vif_cpu.append(vif_all)
                all_qbf_cpu.append(qbf_all)
                all_ssim_cpu.append(ssim_all)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[Metrics][{set_name}] Failed: {e}")
                continue

        if len(all_vif_cpu) > 0:
            vif_mean = torch.cat(all_vif_cpu).mean().item()
            qbf_mean = torch.cat(all_qbf_cpu).mean().item()
            ssim_mean = torch.cat(all_ssim_cpu).mean().item()
            reward_mean = (vif_mean + 1.5 * qbf_mean + ssim_mean) / 3.0
            if accelerator.is_main_process:
                print(f"[Metrics][{set_name}] VIF={vif_mean:.4f}  Qabf={qbf_mean:.4f}  SSIM={ssim_mean:.4f}  Reward={reward_mean:.4f}")
                accelerator.log(
                    {
                        f"test/{set_name}/VIF": vif_mean,
                        f"test/{set_name}/Qabf": qbf_mean,
                        f"test/{set_name}/SSIM": ssim_mean,
                        f"test/{set_name}/Reward": reward_mean,
                    },
                    step=epoch,
                )


if __name__ == "__main__":
    main()