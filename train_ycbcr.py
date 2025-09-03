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
# 超参数配置（与 train.py 基本一致）
# -------------------------------
EPOCHS: int = 500
LR: float = 1e-4
KL_WEIGHT: float = 1e-5
LOSS_SCALE_FACTOR: float = 0.1
MIXED_PRECISION: str = "bf16"  # "no" | "fp16" | "bf16"
PROJECT_DIR: str = "./checkpoints/stochastic_policy_ycbcr_final"
SAVE_IMAGES_TO_DIR: bool = True
TRAIN_BATCH_SIZE: int = 16
TEST_BATCH_SIZE: int = 2
NUM_WORKERS: int = 4
GRAD_ACCUM_STEPS: int = 2
MAX_GRAD_NORM: float = 1.0
TEST_FREQ: int = 1
SAVE_FREQ: int = 50
METRIC_MODE: str = 'mu' # mu or sample
SAVE_MODELS: bool = True
EVAL_CALLBACK = None

# FusionLoss 权重
LOSS_MAX_RATIO: float = 10.0
LOSS_CONSIST_RATIO: float = 2.0
LOSS_GRAD_RATIO: float = 40.0
LOSS_SSIM_IR_RATIO: float = 1.0
LOSS_SSIM_RATIO: float = 1.0
LOSS_IR_COMPOSE: float = 2.0
LOSS_COLOR_RATIO: float = 2.0
LOSS_MAX_MODE: str = "l1"
LOSS_CONSIST_MODE: str = "l1"
LOSS_SSIM_WINDOW: int = 48

# -------------------------------
# 数据集路径（与 train.py 对齐）
# -------------------------------
DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "MSRS": {
        "train": {"dir_A": "./data/MSRS-main/MSRS-main/train/vi",
                  "dir_B": "./data/MSRS-main/MSRS-main/train/ir"},
        "test":  {"dir_A": "./data/MSRS-main/MSRS-main/test/vi",
                  "dir_B": "./data/MSRS-main/MSRS-main/test/ir"},
    },
    "M3FD": {
        "train": {"dir_A": "", "dir_B": ""},
        "test":  {"dir_A": "./data/M3FD_Fusion/Vis",
                  "dir_B": "./data/M3FD_Fusion/Ir"},
    },
    "RS": {
        "train": {"dir_A": "", "dir_B": ""},
        "test":  {"dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
                  "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared"},
    },
    "PET": {
        "train": {"dir_A": "", "dir_B": ""},
        "test":  {"dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/PET",
                  "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/MRI"},
    },
    "CT": {
        "train": {"dir_A": "", "dir_B": ""},
        "test":  {"dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/CT",
                  "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/MRI"},
    },
    "SPECT": {
        "train": {"dir_A": ".", "dir_B": "."},
        "test":  {"dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/SPECT",
                  "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/MRI"},
    }
}
TEST_SET_NAMES = ["MSRS","M3FD", "RS", "PET", "SPECT", "CT"]

torch.backends.cudnn.benchmark = False


# -------------------------------
# 工具函数
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5

def to_m11(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0

def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5

def save_image_grid(path: str, img: torch.Tensor, nrow: int = 4):
    x = img.detach().to(torch.float32).cpu().clamp(-1, 1)
    save_image(x, path, nrow=min(nrow, x.size(0)), normalize=True, value_range=(-1, 1), padding=2)

def rgb_to_ycbcr(x_rgb_m11: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x_rgb_m11: [-1,1] RGB -> 返回 (Y, Cb, Cr), 皆为 [-1,1]
    使用 BT.601 全范围，先映射到 [0,1] 再计算
    """
    x = to_01(x_rgb_m11)
    r, g, b = x[:, 0:1], x[:, 1:1+1], x[:, 2:2+1]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return to_m11(y), to_m11(cb), to_m11(cr)

def ycbcr_to_rgb(y_m11: torch.Tensor, cb_m11: torch.Tensor, cr_m11: torch.Tensor) -> torch.Tensor:
    """
    输入 Y/Cb/Cr 为 [-1,1]，返回 RGB[-1,1]
    """
    y  = to_01(y_m11)
    cb = to_01(cb_m11) - 0.5
    cr = to_01(cr_m11) - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = torch.cat([r, g, b], dim=1).clamp(0, 1)
    return to_m11(rgb)

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
# 训练主流程（融合仅作用在 Y 通道）
# -------------------------------
def main():
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=PROJECT_DIR,
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    )
    accelerator.init_trackers("stochastic_policy_ycbcr")

    if accelerator.is_main_process:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        print(f"[Config] epochs={EPOCHS}, lr={LR}, kl_w={KL_WEIGHT}, loss_scale={LOSS_SCALE_FACTOR}, mp={MIXED_PRECISION}")
        print(f"[Dirs] project_dir={PROJECT_DIR}")

    # 数据
    train_loader, test_loaders = build_dataloaders(accelerator)

    # 策略网络：仅接收 2 通道输入（A_Y 和 B_IR）
    policy_net = PolicyNet(in_channels=2, out_channels=2)
    fusion_kernel = LaplacianPyramidFusion(num_levels=4)
    fusion_loss_fn = FusionLoss(
        max_ratio=LOSS_MAX_RATIO,
        consist_ratio=LOSS_CONSIST_RATIO,
        grad_ratio=LOSS_GRAD_RATIO,
        ssim_ir_ratio=LOSS_SSIM_IR_RATIO,
        ssim_ratio=LOSS_SSIM_RATIO,
        ir_compose=LOSS_IR_COMPOSE,
        color_ratio=LOSS_COLOR_RATIO,
        max_mode=LOSS_MAX_MODE,
        consist_mode=LOSS_CONSIST_MODE,
        ssim_window_size=LOSS_SSIM_WINDOW,
    )

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=1e-4)

    policy_net, optimizer, train_loader = accelerator.prepare(policy_net, optimizer, train_loader)
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype

    fusion_kernel = fusion_kernel.to(device=accelerator.device, dtype=model_dtype)
    fusion_loss_fn = fusion_loss_fn.to(device=accelerator.device, dtype=model_dtype)

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        policy_net.train()
        epoch_total_loss_sum = 0.0
        epoch_fusion_loss_sum = 0.0
        epoch_kld_sum = 0.0
        epoch_sample_count = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Train batch should provide (A,B[,C]).")
            A_rgb, B_ir = batch[0], batch[1]

            # 对齐 device/dtype 并保持 channels_last
            A_rgb = to_ch_last(A_rgb.to(device=accelerator.device, dtype=model_dtype))
            B_ir = to_ch_last(B_ir.to(device=accelerator.device, dtype=model_dtype))  # 1 通道

            # 处理 A 的通道：如果是 3 通道则转换到 YCbCr（训练时 PET/SPECT），
            # 如果是单通道（CT），直接当作 Y（不做 Cb/Cr）
            if A_rgb.shape[1] == 3:
                Y, Cb, Cr = rgb_to_ycbcr(A_rgb)
            else:
                # 保证为 1 通道
                Y = A_rgb if A_rgb.shape[1] == 1 else A_rgb.mean(dim=1, keepdim=True)
                Cb = Cr = None

            B1 = B_ir if B_ir.shape[1] == 1 else B_ir.mean(dim=1, keepdim=True)

            with accelerator.accumulate(policy_net):
                # 策略网络（2通道：Y 与 B）
                mu, logvar = policy_net(Y, B1)
                std = torch.exp(0.5 * logvar)

                # 融合仅在 Y 通道
                F_Y = fusion_kernel(Y, B1, mu)
                # 颜色由 A 直接提供：复原到 RGB
                if Cb is not None:
                    F_rgb = ycbcr_to_rgb(F_Y, Cb, Cr)
                else:
                    # CT / 单通道：把 fused Y 扩为 3 通道用于损失计算（直接灰度复用）
                    F_rgb = F_Y.repeat(1, 3, 1, 1)

                # 损失：A_rgb vs (IR 扩 3 通道) vs F_rgb
                B_for_loss = B1.repeat(1, 3, 1, 1)
                # A_rgb 可能是单通道或三通道；统一到三通道用于 loss
                if A_rgb.shape[1] == 3:
                    A_for_loss = A_rgb
                else:
                    A_for_loss = Y.repeat(1, 3, 1, 1)
                fusion_loss = fusion_loss_fn(A_for_loss, B_for_loss, F_rgb)

                # KL 正则
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = fusion_loss * LOSS_SCALE_FACTOR + KL_WEIGHT * kld_loss

                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            bsz = A_rgb.shape[0]
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

        # 聚合并打印
        stats_local = torch.tensor(
            [epoch_total_loss_sum, epoch_fusion_loss_sum, epoch_kld_sum, epoch_sample_count],
            device=accelerator.device,
            dtype=torch.float64,
        )
        stats_all = accelerator.gather_for_metrics(stats_local)
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
            evaluate_and_log(accelerator, policy_net, fusion_kernel, test_loaders, epoch, METRIC_MODE)

        # 保存
        if SAVE_MODELS and SAVE_FREQ > 0 and (epoch % SAVE_FREQ == 0) and accelerator.is_main_process:
            save_dir = os.path.join(PROJECT_DIR, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(policy_net)
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "policy_net.pth"))
            print(f"[Save] model -> {save_dir}")

    # 最终保存
    accelerator.wait_for_everyone()
    if SAVE_MODELS and accelerator.is_main_process:
        final_dir = os.path.join(PROJECT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy_net)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "policy_net.pth"))
        print(f"[Final] model -> {final_dir}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


# -------------------------------
# 评估与日志（融合仅作用在 Y，输出 F_rgb）
# -------------------------------
@torch.no_grad()
def evaluate_and_log(
    accelerator: Accelerator,
    policy_net: PolicyNet,
    fusion_kernel: LaplacianPyramidFusion,
    test_loaders: Dict[str, DataLoader],
    epoch: int,
    metric_mode: str = 'mu',
):
    policy_net.eval()
    device = accelerator.device
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype

    from metric.MetricGPU import (
        PSNR_function_batch, MSE_function_batch, CC_function_batch, SCD_function_batch,
        Nabf_function_batch, MI_function_batch, EN_function_batch, SF_function_batch,
        SD_function_batch, AG_function_batch
    )

    results_all_sets = {}
    for set_name, loader in test_loaders.items():
        if accelerator.is_main_process:
            print(f"\n[Eval] Epoch {epoch} - {set_name}")

        all_vif_cpu = []; all_qbf_cpu = []; all_ssim_cpu = []
        all_psnr_cpu, all_mse_cpu, all_cc_cpu, all_scd_cpu = [], [], [], []
        all_nabf_cpu, all_mi_cpu = [], []
        all_ag_cpu, all_en_cpu, all_sf_cpu, all_sd_cpu = [], [], [], []

        for batch_idx, batch in enumerate(loader):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Test batch should provide (A,B[,C]).")
            A_rgb, B_ir = batch[0], batch[1]

            A_rgb = to_ch_last(A_rgb.to(device=device, dtype=model_dtype))
            B_ir = to_ch_last(B_ir.to(device=device, dtype=model_dtype))
            # 兼容 CT（单通道）与 RGB（3通道）
            if A_rgb.shape[1] == 3:
                Y, Cb, Cr = rgb_to_ycbcr(A_rgb)
                A_for_metric = A_rgb
            else:
                Y = A_rgb if A_rgb.shape[1] == 1 else A_rgb.mean(dim=1, keepdim=True)
                Cb = Cr = None
                # 评测时将 A 扩为 3 通道以计算与 fused 的指标（灰度复用）
                A_for_metric = Y.repeat(1, 3, 1, 1)
            B1 = B_ir if B_ir.shape[1] == 1 else B_ir.mean(dim=1, keepdim=True)

            mu, logvar = policy_net(Y, B1)
            std = torch.exp(0.5 * logvar)
            F_Y_mu = fusion_kernel(Y, B1, mu)
            sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)
            F_Y_sampled = fusion_kernel(Y, B1, sampled_w)

            if Cb is not None:
                F_hat_mu = ycbcr_to_rgb(F_Y_mu, Cb, Cr)
                F_hat_sampled = ycbcr_to_rgb(F_Y_sampled, Cb, Cr)
            else:
                F_hat_mu = F_Y_mu.repeat(1, 3, 1, 1)
                F_hat_sampled = F_Y_sampled.repeat(1, 3, 1, 1)

            # 仅第一个 batch 记录图像
            if batch_idx == 0:
                def to01(t): return to_01(t.detach().to(torch.float32).cpu())
                if accelerator.is_main_process and SAVE_IMAGES_TO_DIR:
                    out_dir = os.path.join(PROJECT_DIR, "images", f"epoch_{epoch}", set_name)
                    os.makedirs(out_dir, exist_ok=True)
                    # 保存用于可视化的 A（若为 CT，会是灰度3通道副本）
                    save_image(to01(A_for_metric), os.path.join(out_dir, f"A_rgb_e{epoch:04d}.png"))
                    save_image(to01(B1.repeat(1,3,1,1)), os.path.join(out_dir, f"B_ir_e{epoch:04d}.png"))
                    save_image(to01(mu), os.path.join(out_dir, f"mu_e{epoch:04d}.png"))
                    save_image(to01(F_hat_mu), os.path.join(out_dir, f"F_mu_e{epoch:04d}.png"))
                    save_image(to01(F_hat_sampled), os.path.join(out_dir, f"F_sample_e{epoch:04d}.png"))

            # 指标
            A_255 = to_255(A_for_metric).to(torch.float32)
            B_255 = to_255(B1).to(torch.float32)
            F_use = F_hat_mu if metric_mode == 'mu' else F_hat_sampled
            F_255 = to_255(F_use).to(torch.float32)

            try:
                vif = VIF_function_batch(A_255, B_255, F_255)
                qbf = Qabf_function_batch(A_255, B_255, F_255)
                ssim = SSIM_function_batch(A_255, B_255, F_255)
                psnr = PSNR_function_batch(A_255, B_255, F_255)
                mse = MSE_function_batch(A_255, B_255, F_255)
                cc = CC_function_batch(A_255, B_255, F_255)
                scd = SCD_function_batch(A_255, B_255, F_255)
                nabf = Nabf_function_batch(A_255, B_255, F_255)
                mi = MI_function_batch(A_255, B_255, F_255)
                ag = AG_function_batch(F_255)
                en = EN_function_batch(F_255)
                sf = SF_function_batch(F_255)
                sd = SD_function_batch(F_255)

                # 聚合各进程
                vif_all = accelerator.gather_for_metrics(vif.reshape(-1)).float().cpu()
                qbf_all = accelerator.gather_for_metrics(qbf.reshape(-1)).float().cpu()
                ssim_all = accelerator.gather_for_metrics(ssim.reshape(-1)).float().cpu()
                psnr_all = accelerator.gather_for_metrics(psnr.reshape(-1)).float().cpu()
                mse_all = accelerator.gather_for_metrics(mse.reshape(-1)).float().cpu()
                cc_all = accelerator.gather_for_metrics(cc.reshape(-1)).float().cpu()
                scd_all = accelerator.gather_for_metrics(scd.reshape(-1)).float().cpu()
                nabf_all = accelerator.gather_for_metrics(nabf.reshape(-1)).float().cpu()
                mi_all = accelerator.gather_for_metrics(mi.reshape(-1)).float().cpu()
                ag_all = accelerator.gather_for_metrics(ag.reshape(-1)).float().cpu()
                en_all = accelerator.gather_for_metrics(en.reshape(-1)).float().cpu()
                sf_all = accelerator.gather_for_metrics(sf.reshape(-1)).float().cpu()
                sd_all = accelerator.gather_for_metrics(sd.reshape(-1)).float().cpu()

                all_vif_cpu.append(vif_all); all_qbf_cpu.append(qbf_all); all_ssim_cpu.append(ssim_all)
                all_psnr_cpu.append(psnr_all); all_mse_cpu.append(mse_all); all_cc_cpu.append(cc_all); all_scd_cpu.append(scd_all)
                all_nabf_cpu.append(nabf_all); all_mi_cpu.append(mi_all)
                all_ag_cpu.append(ag_all); all_en_cpu.append(en_all); all_sf_cpu.append(sf_all); all_sd_cpu.append(sd_all)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[Metrics][{set_name}] Failed: {e}")
                continue

        if len(all_vif_cpu) > 0:
            vif_mean = torch.cat(all_vif_cpu).mean().item()
            qbf_mean = torch.cat(all_qbf_cpu).mean().item()
            ssim_mean = torch.cat(all_ssim_cpu).mean().item()
            psnr_mean = torch.cat(all_psnr_cpu).mean().item()
            mse_mean = torch.cat(all_mse_cpu).mean().item()
            cc_mean = torch.cat(all_cc_cpu).mean().item()
            scd_mean = torch.cat(all_scd_cpu).mean().item()
            nabf_mean = torch.cat(all_nabf_cpu).mean().item()
            mi_mean = torch.cat(all_mi_cpu).mean().item()
            ag_mean = torch.cat(all_ag_cpu).mean().item()
            en_mean = torch.cat(all_en_cpu).mean().item()
            sf_mean = torch.cat(all_sf_cpu).mean().item()
            sd_mean = torch.cat(all_sd_cpu).mean().item()
            reward_mean = (vif_mean + 1.5 * qbf_mean + ssim_mean) / 3.0
            if accelerator.is_main_process:
                print(
                    f"[Metrics][{set_name}] "
                    f"VIF={vif_mean:.4f}  Qabf={qbf_mean:.4f}  SSIM={ssim_mean:.4f}  Reward={reward_mean:.4f}  "
                    f"PSNR={psnr_mean:.4f}  MSE={mse_mean:.4f}  CC={cc_mean:.4f}  SCD={scd_mean:.4f}  "
                    f"Nabf={nabf_mean:.4f}  MI={mi_mean:.4f}  AG={ag_mean:.4f}  EN={en_mean:.4f}  "
                    f"SF={sf_mean:.4f}  SD={sd_mean:.4f}"
                )
                accelerator.log(
                    {
                        f"test/{set_name}/VIF": vif_mean,
                        f"test/{set_name}/Qabf": qbf_mean,
                        f"test/{set_name}/SSIM": ssim_mean,
                        f"test/{set_name}/Reward": reward_mean,
                        f"test/{set_name}/PSNR": psnr_mean,
                        f"test/{set_name}/MSE": mse_mean,
                        f"test/{set_name}/CC": cc_mean,
                        f"test/{set_name}/SCD": scd_mean,
                        f"test/{set_name}/Nabf": nabf_mean,
                        f"test/{set_name}/MI": mi_mean,
                        f"test/{set_name}/AG": ag_mean,
                        f"test/{set_name}/EN": en_mean,
                        f"test/{set_name}/SF": sf_mean,
                        f"test/{set_name}/SD": sd_mean,
                    },
                    step=epoch,
                )
            results_all_sets[set_name] = {
                "Reward": reward_mean, "VIF": vif_mean, "Qabf": qbf_mean, "SSIM": ssim_mean,
                "PSNR": psnr_mean, "MSE": mse_mean, "CC": cc_mean, "SCD": scd_mean,
                "Nabf": nabf_mean, "MI": mi_mean, "AG": ag_mean, "EN": en_mean, "SF": sf_mean, "SD": sd_mean,
            }

    if callable(EVAL_CALLBACK):
        try:
            EVAL_CALLBACK(epoch, results_all_sets)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Eval Callback] Error: {e}")
    return results_all_sets


if __name__ == "__main__":
    main()