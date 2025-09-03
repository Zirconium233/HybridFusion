import os
import argparse
from typing import Dict, List, Tuple

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
# 配置
# -------------------------------
EPOCHS: int = 500
LR: float = 1e-4
KL_WEIGHT: float = 1e-5
LOSS_SCALE_FACTOR: float = 0.1
MIXED_PRECISION: str = "bf16"  # "no" | "fp16" | "bf16"
BASE_PROJECT_DIR: str = "./checkpoints/Med_ycbcr_500"
SAVE_IMAGES_TO_DIR: bool = True
TRAIN_BATCH_SIZE: int = 8
TEST_BATCH_SIZE: int = 2
NUM_WORKERS: int = 4
GRAD_ACCUM_STEPS: int = 1
MAX_GRAD_NORM: float = 1.0
TEST_FREQ: int = 1
SAVE_FREQ: int = 50
SAVE_MODELS: bool = True
METRIC_MODE: str = "mu"  # mu or sample

# FusionLoss 权重（与 train.py 对齐）
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
# 数据集路径（参考 Image/train.py）
# A 为 PET/CT/SPECT，B 为 MRI
# -------------------------------
MED_DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
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
    },
}

torch.backends.cudnn.benchmark = False


# -------------------------------
# 工具函数
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5

def to_255(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 127.5

def save_image_grid(path: str, img: torch.Tensor, nrow: int = 4):
    x = img.detach().to(torch.float32).cpu().clamp(-1, 1)
    save_image(x, path, nrow=min(nrow, x.size(0)), normalize=True, value_range=(-1, 1), padding=2)

# 新增：YCbCr 工具（仅在 PET/SPECT 走 Y 通道融合；CT 直接灰度融合）
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

def make_3ch(x: torch.Tensor) -> torch.Tensor:
    return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

def make_1ch(x: torch.Tensor) -> torch.Tensor:
    # 若为3通道，转为灰度；若为1通道，直接返回
    if x.shape[1] == 1:
        return x
    # 简单平均到灰度（避免引入额外依赖）
    return x.mean(dim=1, keepdim=True)

def build_train_loader_for_task(task: str) -> DataLoader:
    # 仅使用 test 集做训练（公平对比）
    paths = MED_DATASETS[task]["test"]
    ds = ImageFusionDataset(
        dir_A=paths["dir_A"], dir_B=paths["dir_B"], dir_C=None,
        is_train=True, is_getpatch=False, augment=False
    )
    return DataLoader(
        ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

def build_test_loaders_for_targets(targets: List[str]) -> Dict[str, DataLoader]:
    loaders = {}
    for name in targets:
        paths = MED_DATASETS[name]["test"]
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
# 评估
# -------------------------------
@torch.no_grad()
def evaluate_and_log_med(
    accelerator: Accelerator,
    policy_net: PolicyNet,
    fusion_kernel: LaplacianPyramidFusion,
    test_loaders: Dict[str, DataLoader],
    project_dir: str,
    epoch: int,
    metric_mode: str = "mu",
):
    # 更多指标
    from metric.MetricGPU import (
        PSNR_function_batch, MSE_function_batch, CC_function_batch, SCD_function_batch,
        Nabf_function_batch, MI_function_batch, EN_function_batch, SF_function_batch,
        SD_function_batch, AG_function_batch
    )

    policy_net.eval()
    device = accelerator.device
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype

    results_all_sets = {}

    for set_name, loader in test_loaders.items():
        if accelerator.is_main_process:
            print(f"\n[Eval] Epoch {epoch} - {set_name}")

        all_vif, all_qbf, all_ssim = [], [], []
        all_psnr, all_mse, all_cc, all_scd = [], [], [], []
        all_nabf, all_mi, all_ag, all_en, all_sf, all_sd = [], [], [], [], [], []

        for batch_idx, batch in enumerate(loader):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Test batch should provide (A,B[,C]).")
            A, B = batch[0], batch[1]

            # 对齐 device/dtype，并处理通道
            A = to_ch_last(A.to(device=device, dtype=model_dtype))
            B = to_ch_last(B.to(device=device, dtype=model_dtype))
            B1 = make_1ch(B)          # MRI -> 1通道红外侧
            # PET/SPECT: 仅在亮度 Y 与 IR 融合；CT: 直接灰度融合
            if A.shape[1] == 3:
                A3 = make_3ch(A)
                Y, Cb, Cr = rgb_to_ycbcr(A3)
            else:
                Y = make_1ch(A)
                Cb = Cr = None

            # Policy 输出
            mu, logvar = policy_net(Y, B1)
            std = torch.exp(0.5 * logvar)

            # 融合
            F_Y_mu = fusion_kernel(Y, B1, mu)
            sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)
            F_Y_sampled = fusion_kernel(Y, B1, sampled_w)
            if Cb is not None:
                F_hat_mu = ycbcr_to_rgb(F_Y_mu, Cb, Cr)
                F_hat_sampled = ycbcr_to_rgb(F_Y_sampled, Cb, Cr)
                A_for_metric = A3
            else:
                F_hat_mu = F_Y_mu.repeat(1, 3, 1, 1)
                F_hat_sampled = F_Y_sampled.repeat(1, 3, 1, 1)
                A_for_metric = Y.repeat(1, 3, 1, 1)

            # 仅第一个 batch 保存图像
            if batch_idx == 0 and SAVE_IMAGES_TO_DIR and accelerator.is_main_process:
                out_dir = os.path.join(project_dir, "images", f"epoch_{epoch}", set_name)
                os.makedirs(out_dir, exist_ok=True)
                save_image_grid(os.path.join(out_dir, f"A_e{epoch:04d}.png"), A_for_metric)
                save_image_grid(os.path.join(out_dir, f"B_e{epoch:04d}.png"), B1)
                save_image_grid(os.path.join(out_dir, f"mu_e{epoch:04d}.png"), mu)
                save_image_grid(os.path.join(out_dir, f"F_mu_e{epoch:04d}.png"), F_hat_mu)
                save_image_grid(os.path.join(out_dir, f"F_sample_e{epoch:04d}.png"), F_hat_sampled)

            # 指标
            A_255 = to_255(A_for_metric).to(torch.float32)
            B_255 = to_255(B1).to(torch.float32)
            F_use = F_hat_mu if metric_mode == "mu" else F_hat_sampled
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

                # 多进程聚合
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

                all_vif.append(vif_all); all_qbf.append(qbf_all); all_ssim.append(ssim_all)
                all_psnr.append(psnr_all); all_mse.append(mse_all); all_cc.append(cc_all); all_scd.append(scd_all)
                all_nabf.append(nabf_all); all_mi.append(mi_all)
                all_ag.append(ag_all); all_en.append(en_all); all_sf.append(sf_all); all_sd.append(sd_all)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[Metrics][{set_name}] Failed: {e}")
                continue

        def mean_cat(xs): 
            return torch.cat(xs).mean().item() if len(xs) else float("nan")

        vif_mean = mean_cat(all_vif); qbf_mean = mean_cat(all_qbf); ssim_mean = mean_cat(all_ssim)
        psnr_mean = mean_cat(all_psnr); mse_mean = mean_cat(all_mse); cc_mean = mean_cat(all_cc); scd_mean = mean_cat(all_scd)
        nabf_mean = mean_cat(all_nabf); mi_mean = mean_cat(all_mi)
        ag_mean = mean_cat(all_ag); en_mean = mean_cat(all_en); sf_mean = mean_cat(all_sf); sd_mean = mean_cat(all_sd)
        reward_mean = (vif_mean + 1.5 * qbf_mean + ssim_mean) / 3.0

        if accelerator.is_main_process:
            print(
                f"[Metrics][{set_name}] "
                f"VIF={vif_mean:.4f} Qabf={qbf_mean:.4f} SSIM={ssim_mean:.4f} Reward={reward_mean:.4f} | "
                f"PSNR={psnr_mean:.4f} MSE={mse_mean:.4f} CC={cc_mean:.4f} SCD={scd_mean:.4f} | "
                f"Nabf={nabf_mean:.4f} MI={mi_mean:.4f} AG={ag_mean:.4f} EN={en_mean:.4f} SF={sf_mean:.4f} SD={sd_mean:.4f}"
            )

        results_all_sets[set_name] = {
            "Reward": reward_mean, "VIF": vif_mean, "Qabf": qbf_mean, "SSIM": ssim_mean,
            "PSNR": psnr_mean, "MSE": mse_mean, "CC": cc_mean, "SCD": scd_mean,
            "Nabf": nabf_mean, "MI": mi_mean, "AG": ag_mean, "EN": en_mean, "SF": sf_mean, "SD": sd_mean,
        }

    # 保存 CSV/JSON
    if accelerator.is_main_process:
        import pandas as pd, json
        rows = []
        for ds, m in results_all_sets.items():
            r = {"dataset": ds}; r.update(m); rows.append(r)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(project_dir, f"metrics_epoch{epoch:02d}.csv")
        json_path = os.path.join(project_dir, f"metrics_epoch{epoch:02d}.json")
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(results_all_sets, f, indent=2)
        print(f"[Eval] Saved {csv_path}")

    return results_all_sets


# -------------------------------
# 单任务训练与评测
# -------------------------------
def train_one_task(task: str, eval_targets: List[str]):
    task_dir = os.path.join(BASE_PROJECT_DIR, task)
    os.makedirs(task_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=task_dir,
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    )
    accelerator.init_trackers(f"med_{task}")

    if accelerator.is_main_process:
        print(f"\n[Task] {task} | project_dir={task_dir} | epochs={EPOCHS}")
        for t in eval_targets:
            print(f"[Eval Target] {t}")

    # 数据：使用 test 集训练
    train_loader = build_train_loader_for_task(task)
    test_loaders = build_test_loaders_for_targets(eval_targets)

    # 模型与损失
    # 统一使用 Y+IR 两通道输入；CT 也为 1+1 通道
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

    # accelerate
    policy_net, optimizer, train_loader = accelerator.prepare(policy_net, optimizer, train_loader)

    # 放置到 device/dtype
    model_dtype = next(accelerator.unwrap_model(policy_net).parameters()).dtype
    fusion_kernel = fusion_kernel.to(device=accelerator.device, dtype=model_dtype)
    fusion_loss_fn = fusion_loss_fn.to(device=accelerator.device, dtype=model_dtype)

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        policy_net.train()
        epoch_total, epoch_fusion, epoch_kld, epoch_n = 0.0, 0.0, 0.0, 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"[{task}] Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Train batch should provide (A,B[,C]).")
            A, B = batch[0], batch[1]

            # 对齐 device/dtype 并通道处理
            A = to_ch_last(A.to(device=accelerator.device, dtype=model_dtype))
            B = to_ch_last(B.to(device=accelerator.device, dtype=model_dtype))
            B1 = make_1ch(B)    # MRI -> 1ch
            # PET/SPECT: 仅在亮度 Y 与 IR 融合；CT: 直接灰度融合
            if A.shape[1] == 3:
                A3 = make_3ch(A)
                Y, Cb, Cr = rgb_to_ycbcr(A3)
            else:
                Y = make_1ch(A)
                Cb = Cr = None

            with accelerator.accumulate(policy_net):
                # 策略网络（输入两通道：Y 与 B1）
                mu, logvar = policy_net(Y, B1)
                std = torch.exp(0.5 * logvar)

                # 融合（确定性：mu）在 Y 上进行；CT 时直接灰度，PET/SPECT 复原 RGB
                F_Y = fusion_kernel(Y, B1, mu)
                if Cb is not None:
                    F_hat = ycbcr_to_rgb(F_Y, Cb, Cr)
                    A_for_loss = A3
                else:
                    F_hat = F_Y.repeat(1, 3, 1, 1)
                    A_for_loss = Y.repeat(1, 3, 1, 1)

                # 自监督损失（B_for_loss 扩到3通道）
                B_for_loss = B1.repeat(1, 3, 1, 1)
                fusion_loss = fusion_loss_fn(A_for_loss, B_for_loss, F_hat)

                # KL 正则
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = fusion_loss * LOSS_SCALE_FACTOR + KL_WEIGHT * kld_loss

                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            bsz = A.shape[0]
            epoch_total += float(total_loss.item()) * bsz
            epoch_fusion += float(fusion_loss.item()) * bsz
            epoch_kld += float(kld_loss.item()) * bsz
            epoch_n += bsz
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "fusion": f"{fusion_loss.item():.2f}",
                    "kld": f"{kld_loss.item():.4f}",
                })
                accelerator.log(
                    {"train/total_loss": float(total_loss.item()),
                     "train/fusion_loss": float(fusion_loss.item()),
                     "train/kld_loss": float(kld_loss.item())},
                    step=global_step,
                )

        # 统计
        stats_local = torch.tensor([epoch_total, epoch_fusion, epoch_kld, epoch_n], device=accelerator.device, dtype=torch.float64)
        stats_all = accelerator.gather_for_metrics(stats_local)
        if stats_all.ndim == 1: stats_all = stats_all.unsqueeze(0)
        totals = stats_all.sum(dim=0)
        n = max(1.0, float(totals[3].item()))
        avg_total = float(totals[0].item()) / n
        avg_fusion = float(totals[1].item()) / n
        avg_kld = float(totals[2].item()) / n
        if accelerator.is_main_process:
            print(f"[{task}][Epoch {epoch}] avg_total={avg_total:.4f}  avg_fusion={avg_fusion:.4f}  avg_kld={avg_kld:.6f}")
            accelerator.log({"epoch/avg_total": avg_total, "epoch/avg_fusion": avg_fusion, "epoch/avg_kld": avg_kld}, step=epoch)

        # 评测
        if epoch % TEST_FREQ == 0 or epoch == EPOCHS:
            _ = evaluate_and_log_med(
                accelerator=accelerator,
                policy_net=policy_net,
                fusion_kernel=fusion_kernel,
                test_loaders=test_loaders,
                project_dir=task_dir,
                epoch=epoch,
                metric_mode=METRIC_MODE,
            )

        # 保存
        if SAVE_MODELS and SAVE_FREQ > 0 and (epoch % SAVE_FREQ == 0) and accelerator.is_main_process:
            save_dir = os.path.join(task_dir, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(policy_net)
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "policy_net.pth"))
            print(f"[{task}][Save] -> {save_dir}")

    accelerator.wait_for_everyone()
    if SAVE_MODELS and accelerator.is_main_process:
        final_dir = os.path.join(task_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy_net)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "policy_net.pth"))
        print(f"[{task}][Final] -> {final_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    torch.cuda.empty_cache()


# -------------------------------
# 主流程：顺序跑 PET、SPECT、CT
# PET/SPECT 互相 cross-eval；CT 仅自测
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ALL", choices=["ALL", "PET", "SPECT", "CT"],
                        help="选择要运行的任务；默认 ALL 顺序运行 PET、SPECT、CT")
    args = parser.parse_args()

    os.makedirs(BASE_PROJECT_DIR, exist_ok=True)

    runs: List[Tuple[str, List[str]]] = []
    if args.task == "ALL":
        runs = [
            ("PET",   ["PET", "SPECT"]),  # 自测 PET + 跨任务 SPECT
            ("SPECT", ["SPECT", "PET"]),  # 自测 SPECT + 跨任务 PET
            ("CT",    ["CT"]),            # 仅自测
        ]
    elif args.task == "PET":
        runs = [("PET", ["PET", "SPECT"])]
    elif args.task == "SPECT":
        runs = [("SPECT", ["SPECT", "PET"])]
    elif args.task == "CT":
        runs = [("CT", ["CT"])]

    for task, targets in runs:
        train_one_task(task, targets)


if __name__ == "__main__":
    main()