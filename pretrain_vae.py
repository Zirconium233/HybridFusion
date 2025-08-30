import os
import math
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image
from torchvision.utils import save_image

# 本项目内模块
from dataset import ImageFusionDataset  # 期望返回 (A,B,optional C)
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
EPOCHS: int = 3000
LR: float = 1e-4
KL_WEIGHT: float = 1e-3
USE_FUSION_LOSS: bool = True  # True: 无监督FusionLoss；False: label监督MSE
MIXED_PRECISION: str = "bf16"  # "no" | "fp16" | "bf16"
PROJECT_DIR: str = "./checkpoints/pretrain_vae_3000"  # 也作为 accelerator.project_dir
SAVE_IMAGES_TO_DIR: bool = True  # 保存到tensorboard的同时，落地保存一份
TRAIN_BATCH_SIZE: int = 16        # 全分辨率建议小batch
TEST_BATCH_SIZE: int = 4
NUM_WORKERS: int = 4
GRAD_ACCUM_STEPS: int = 1
MAX_GRAD_NORM: float = 1.0
TEST_FREQ: int = 50               # 每多少epoch测试一次
SAVE_FREQ: int = 50              # 每多少epoch保存一次

# 数据集路径（默认 MSRS，与 pretrain.yml 对齐，可扩展多个测试集）
DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "MSRS": {
        "train": {
            "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
            "dir_C": "./data/MSRS-main/MSRS-main/train/label",
        },
        "test": {
            "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
            # MSRS 测试集通常无 label，可不填
        },
    }
}
TEST_SET_NAMES = ["MSRS"]  # 默认测试集列表

# 模型默认规模（约1.5M参数，见你当前设置）
ENC_BASE_CHS: Tuple[int, int, int] = (192, 256, 384)
Z_CH: int = 32
COND_CH: int = 64
DEC_CHS: Tuple[int, int, int, int] = (384, 256, 128, 96)

# 其他
torch.backends.cudnn.benchmark = False


def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def to_01(x: torch.Tensor) -> torch.Tensor:
    # 输入 [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) * 0.5


def to_255(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,255]
    return (x.clamp(-1, 1) + 1.0) * 127.5


def save_image_grid(path: str, img: torch.Tensor, nrow: int = 4):
    """
    保存 B,C,H,W 张量到 path，范围[-1,1]；使用 torchvision.utils.save_image 生成网格。
    兼容单通道/三通道，自动归一化到 [-1,1]。
    """
    x = img.detach().to(torch.float32).cpu().clamp(-1, 1)
    # save_image 支持 1/3 通道张量；nrow 控制每行图片数
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
    # add_images 期望 [B, C, H, W] 且范围 [0,1]
    writer.add_images(tag, imgs, global_step=step)


def build_dataloaders(accelerator: Accelerator):
    train_paths = DATASETS["MSRS"]["train"]
    train_ds = ImageFusionDataset(
        dir_A=train_paths["dir_A"],
        dir_B=train_paths["dir_B"],
        dir_C=train_paths.get("dir_C", None),
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
            dir_C=paths.get("dir_C", None),
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


def main():
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=PROJECT_DIR,
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    )
    accelerator.init_trackers("pretrain_vae")
    if accelerator.is_main_process:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        print(f"[Config] epochs={EPOCHS}, lr={LR}, kl_w={KL_WEIGHT}, fusion_loss={USE_FUSION_LOSS}, mp={MIXED_PRECISION}")
        print(f"[Dirs] project_dir={PROJECT_DIR}")

    # 数据
    train_loader, test_loaders = build_dataloaders(accelerator)

    # 模型与损失
    model = ConditionalVAE(
        in_ch=4,
        base_chs=ENC_BASE_CHS,
        z_ch=Z_CH,
        cond_ch=COND_CH,
        dec_chs=DEC_CHS,
    )
    fusion_loss_fn = FusionLoss() if USE_FUSION_LOSS else None

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Accelerate 准备
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    # 在 prepare 之后初始化 tracker，确保只在主进程上进行
    accelerator.init_trackers("pretrain_vae")
    # 记录模型参数 dtype，训练时将输入对齐到相同 dtype，避免 bf16/float32 混用
    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype
    # 将无参数的 loss 模块放到正确的 device/dtype（不需要 prepare）
    if fusion_loss_fn is not None:
        fusion_loss_fn = fusion_loss_fn.to(
            device=accelerator.device,
            dtype=model_dtype
        )

    # 训练循环
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        # 统计本 epoch 的加权和（按样本数），便于多卡聚合
        epoch_loss_sum = 0.0
        epoch_recon_sum = 0.0
        epoch_kld_sum = 0.0
        epoch_sample_count = 0
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            # 兼容数据集返回 (A,B) 或 (A,B,C)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    A, B, C = batch
                elif len(batch) == 2:
                    A, B = batch
                    C = None
                else:
                    raise RuntimeError(f"Unexpected batch tuple length: {len(batch)}")
            else:
                raise RuntimeError("Dataset should return a tuple/list (A,B[,C]).")

            # 对齐到模型 dtype 并保持 channels_last，避免 bf16 权重 + fp32 输入导致的报错
            A = to_ch_last(A.to(dtype=model_dtype))
            B = to_ch_last(B.to(dtype=model_dtype))
            C = to_ch_last(C.to(dtype=model_dtype)) if C is not None else None

            with accelerator.accumulate(model):
                F_hat, mu, logvar, z = model(A, B)
                # KL
                kld = ConditionalVAE.kl_loss(mu, logvar, reduction="mean")

                if USE_FUSION_LOSS:
                    # FusionLoss 里的强度项需要三者通道一致；将 1 通道 IR 扩展为 3 通道
                    B_for_loss = B if B.shape[1] == 3 else B.repeat(1, 3, 1, 1)
                    B_for_loss = to_ch_last(B_for_loss)
                    loss_rec = fusion_loss_fn(A, B_for_loss, F_hat)
                else:
                    if C is None:
                        # 若无 label，退化为 A 的重建（兼容）
                        loss_rec = F.mse_loss(F_hat, A)
                    else:
                        loss_rec = F.mse_loss(F_hat, C)

                loss = loss_rec + KL_WEIGHT * kld

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            running += loss.item()
            bsz = A.shape[0]
            epoch_loss_sum += float(loss.item()) * bsz
            epoch_recon_sum += float(loss_rec.item()) * bsz
            epoch_kld_sum += float(kld.item()) * bsz
            epoch_sample_count += bsz
            global_step += 1
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                accelerator.log(
                    {"train/loss": float(loss.item()), "train/recon": float(loss_rec.item()), "train/kld": float(kld.item())},
                    step=global_step,
                )
        # 跨进程聚合 epoch 级统计并打印
        stats_local = torch.tensor(
            [epoch_loss_sum, epoch_recon_sum, epoch_kld_sum, epoch_sample_count],
            device=accelerator.device, dtype=torch.float64
        )
        stats_all = accelerator.gather_for_metrics(stats_local)  # [world, 4]
        if stats_all.ndim == 1:
            stats_all = stats_all.unsqueeze(0)
        totals = stats_all.sum(dim=0)  # 4,
        total_samples = max(1.0, float(totals[3].item()))
        avg_loss = float(totals[0].item()) / total_samples
        avg_recon = float(totals[1].item()) / total_samples
        avg_kld = float(totals[2].item()) / total_samples
        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}  avg_recon={avg_recon:.4f}  avg_kld={avg_kld:.6f}")
            accelerator.log(
                {"epoch/avg_loss": avg_loss, "epoch/avg_recon": avg_recon, "epoch/avg_kld": avg_kld},
                step=epoch,  # epoch 级别的日志使用 epoch 作为 step
            )

        # 测试与可视化
        if epoch % TEST_FREQ == 0:
            evaluate_and_log(accelerator, model, test_loaders, epoch)

        # 保存
        if epoch % SAVE_FREQ == 0 and accelerator.is_main_process:
            save_dir = os.path.join(PROJECT_DIR, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "cvae.pth"))
            print(f"[Save] model -> {save_dir}")

    # 最终保存
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(PROJECT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "cvae.pth"))
        print(f"[Final] model -> {final_dir}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


@torch.no_grad()
def evaluate_and_log(accelerator: Accelerator, model: nn.Module, test_loaders: Dict[str, DataLoader], epoch: int):
    model.eval()
    device = accelerator.device
    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype

    for set_name, loader in test_loaders.items():
        # 只在主进程记录
        if accelerator.is_main_process:
            print(f"\n[Eval] Epoch {epoch} - {set_name}")

        # 指标累计（全测试集批处理）
        all_vif_cpu = []
        all_qbf_cpu = []
        all_ssim_cpu = []

        for batch_idx, batch in enumerate(loader):
             # 兼容数据集返回 (A,B) 或 (A,B,C)
             if isinstance(batch, (list, tuple)):
                 if len(batch) >= 2:
                     A, B = batch[0], batch[1]
                 else:
                     raise RuntimeError("Test batch should provide (A,B[,C]).")
             else:
                 raise RuntimeError("Test dataset should return a tuple/list (A,B[,C]).")

             A = to_ch_last(A.to(device=device, dtype=model_dtype))
             B = to_ch_last(B.to(device=device, dtype=model_dtype))

             F_hat, _, _, _ = model(A, B)

             # 仅第一个 batch 保存/记录图片，避免过多输出
             if batch_idx == 0:
                tag = f"images/{set_name}"
                log_images_tb(accelerator, f"{tag}/A", A, step=epoch)
                log_images_tb(accelerator, f"{tag}/B", B, step=epoch)
                log_images_tb(accelerator, f"{tag}/F_hat", F_hat, step=epoch)
                if accelerator.is_main_process and SAVE_IMAGES_TO_DIR:
                    out_dir = os.path.join(PROJECT_DIR, "images", f"epoch_{epoch}", set_name)
                    os.makedirs(out_dir, exist_ok=True)
                    save_image_grid(os.path.join(out_dir, f"A_e{epoch:04d}.png"), A)
                    save_image_grid(os.path.join(out_dir, f"B_e{epoch:04d}.png"), B)
                    save_image_grid(os.path.join(out_dir, f"F_hat_e{epoch:04d}.png"), F_hat)

             # 指标（GPU batch 版本）：输入应为 0..255
             # 使用 float32 计算，避免 bf16/fp16 带来的数值损失
             A_255 = to_255(A).to(torch.float32)
             B_255 = to_255(B).to(torch.float32)
             F_255 = to_255(F_hat).to(torch.float32)

             try:
                vif = VIF_function_batch(A_255, B_255, F_255)
                qbf = Qabf_function_batch(A_255, B_255, F_255)
                ssim = SSIM_function_batch(A_255, B_255, F_255)
                # 聚合到所有进程，再转 CPU 累加
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

        # 全测试集平均
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