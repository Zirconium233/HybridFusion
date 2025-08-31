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
from metric.MetricGPU import VIF_function_batch, Qabf_function_batch, SSIM_function_batch

# -------------------------------
# Hyper-params (keep same as train.py unless specified)
# -------------------------------
EPOCHS: int = 10          # ablation: run 10 epochs
LR: float = 1e-4
KL_WEIGHT: float = 1e-5
LOSS_SCALE_FACTOR: float = 0.1
MIXED_PRECISION: str = "bf16"  # "no" | "fp16" | "bf16"
PROJECT_DIR: str = "./checkpoints/abl_no_kernel_weighted_avg"
SAVE_IMAGES_TO_DIR: bool = True
TRAIN_BATCH_SIZE: int = 16
TEST_BATCH_SIZE: int = 2
NUM_WORKERS: int = 4
GRAD_ACCUM_STEPS: int = 2
MAX_GRAD_NORM: float = 1.0
TEST_FREQ: int = 1        # ablation: eval every epoch
SAVE_FREQ: int = 0        # ablation: no intermediate save
METRIC_MODE: str = 'mu'   # 'mu' or 'sample'
SAVE_MODELS: bool = True
EVAL_CALLBACK = None

# FusionLoss weights (can be overridden externally)
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
# Datasets
# -------------------------------
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
            "dir_A": "./data/M3FD_Fusion/Vis",
            "dir_B": "./data/M3FD_Fusion/Ir",
        },
        "test": {
            "dir_A": "./data/M3FD_Fusion/Vis",
            "dir_B": "./data/M3FD_Fusion/Ir",
        },
    },
    "RS": {
        "train": {
            "dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
            "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared",
        },
        "test": {
            "dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
            "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared",
        },
    },
}
TEST_SET_NAMES = ["MSRS", "M3FD", "RS"]

torch.backends.cudnn.benchmark = False


# -------------------------------
# Utils
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) * 0.5


def to_255(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,255]
    return (x.clamp(-1, 1) + 1.0) * 127.5


def make_3ch(x: torch.Tensor) -> torch.Tensor:
    return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)


def save_image_grid(path: str, img: torch.Tensor, nrow: int = 4):
    x = img.detach().to(torch.float32).cpu().clamp(-1, 1)
    save_image(x, path, nrow=min(nrow, x.size(0)), normalize=True, value_range=(-1, 1), padding=2)


def log_images_tb(accelerator: Accelerator, tag: str, images: torch.Tensor, step: int):
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
# Training
# -------------------------------
def main():
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=PROJECT_DIR,
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    )
    accelerator.init_trackers("stochastic_policy_stage1_ablation")

    if accelerator.is_main_process:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        print(f"[Config] epochs={EPOCHS}, lr={LR}, kl_w={KL_WEIGHT}, loss_scale={LOSS_SCALE_FACTOR}, mp={MIXED_PRECISION}")
        print(f"[Dirs] project_dir={PROJECT_DIR}")

    # Data
    train_loader, test_loaders = build_dataloaders(accelerator)

    # Model and loss
    policy_net = PolicyNet(in_channels=4, out_channels=2)
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
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                A, B = batch[0], batch[1]
            else:
                raise RuntimeError("Train batch should provide (A,B[,C]).")

            A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
            B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))

            with accelerator.accumulate(policy_net):
                # Predict weight distribution
                mu, logvar = policy_net(A, B)  # mu in (0,1)
                # Deterministic fusion by weighted average (no traditional fusion kernel)
                A3 = make_3ch(A)
                B3 = make_3ch(B)
                F_hat = mu * A3 + (1.0 - mu) * B3

                # Fusion self-supervised loss
                B_for_loss = B3
                fusion_loss = fusion_loss_fn(A3, B_for_loss, F_hat)

                # KL regularization
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

        # epoch stats across processes
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

        # Eval
        if epoch % TEST_FREQ == 0 or epoch == EPOCHS:
            evaluate_and_log(accelerator, policy_net, test_loaders, epoch, METRIC_MODE)

        # Save: only final (SAVE_FREQ=0)
        if SAVE_MODELS and SAVE_FREQ > 0 and (epoch % SAVE_FREQ == 0) and accelerator.is_main_process:
            pass

    # Final save
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
# Evaluation
# -------------------------------
@torch.no_grad()
def evaluate_and_log(
    accelerator: Accelerator,
    policy_net: PolicyNet,
    test_loaders: Dict[str, DataLoader],
    epoch: int,
    metric_mode: str = 'mu',  # 'mu' or 'sample'
):
    policy_net.eval()
    device = accelerator.device
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype

    # more metrics
    from metric.MetricGPU import (
        PSNR_function_batch, MSE_function_batch, CC_function_batch, SCD_function_batch,
        Nabf_function_batch, MI_function_batch, EN_function_batch, SF_function_batch,
        SD_function_batch, AG_function_batch
    )

    def blend(A: torch.Tensor, B: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        A3 = make_3ch(A)
        B3 = make_3ch(B)
        return torch.clamp(W, 0.0, 1.0) * A3 + (1.0 - torch.clamp(W, 0.0, 1.0)) * B3

    results_all_sets = {}
    for set_name, loader in test_loaders.items():
        if accelerator.is_main_process:
            print(f"\n[Eval] Epoch {epoch} - {set_name}")

        all_vif_cpu = []
        all_qbf_cpu = []
        all_ssim_cpu = []
        all_psnr_cpu, all_mse_cpu, all_cc_cpu, all_scd_cpu = [], [], [], []
        all_nabf_cpu, all_mi_cpu = [], []
        all_ag_cpu, all_en_cpu, all_sf_cpu, all_sd_cpu = [], [], [], []

        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                A, B = batch[0], batch[1]
            else:
                raise RuntimeError("Test batch should provide (A,B[,C]).")

            A = to_ch_last(A.to(device=device, dtype=model_dtype))
            B = to_ch_last(B.to(device=device, dtype=model_dtype))

            mu, logvar = policy_net(A, B)
            std = torch.exp(0.5 * logvar)

            F_hat_mu = blend(A, B, mu)
            sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)
            F_hat_sampled = blend(A, B, sampled_w)

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

            # Metrics (choose fused by metric_mode)
            A_255 = to_255(A).to(torch.float32)
            B_255 = to_255(B).to(torch.float32)
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