# 直接在 VAE 潜空间用 XRestormer 训练一个模型，不使用 diffusion，快速验证效果
# 仅参数：--epochs，其余从本文件 CONFIG 读取（硬编码）
import os
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL

# =========================
# 全局快速配置（可直接修改）
# =========================
CONFIG = {
    "run_name": "fusion_XRestormer_v1",
    "output_dir": "./checkpoints/pretrain/",
    "model_config": {
        "vae": {
            "pretrain": "/home/zhangran/desktop/myProject/playground/sd-vae-ft-mse"
        }
    },
    "training": {
        "epoch": 100,
        "train_batch_size": 16,
        "learning_rate": 1.0e-4,
        "num_workers": 4,
        "test_freq": 10,
        "save_freq": 10
    },
    "datasets": {
        "MSRS": {
            "train": {
                "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
                "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
                "dir_C": "./data/MSRS-main/MSRS-main/train/label"
            },
            "test": {
                "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
                "dir_B": "./data/MSRS-main/MSRS-main/test/ir"
            }
        }
    },
    "train_dataset": {"name": "MSRS"},
    "pipeline": {
        "use_shortcut": False
    }
}

# 数据集
from dataset import ImageFusionDataset
# 使用“无下采样”的潜空间 XRestormer
from model.latent_xrestormer import LatentXRestormerNoDown


def build_vae(pretrain_path: str, mode: str = "official") -> AutoencoderKL:
    if mode == "official":
        vae = AutoencoderKL.from_pretrained(pretrain_path)
    else:
        vae = AutoencoderKL(
            sample_size=128,
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[64, 128, 256],
            latent_channels=4,
            # scale_factor=4,
            scaling_factor=0.057867,
        )
        vae.load_state_dict(torch.load(pretrain_path)['model_state_dict'])
    return vae


@torch.no_grad()
def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    """
    images: [-1,1], [B,C,H,W]，C=1/3
    返回未缩放 latent（不乘 scaling_factor）
    """
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    enc_out = vae.encode(images)
    lat = enc_out.latent_dist.sample() if hasattr(enc_out, "latent_dist") else enc_out[0]
    # 确保输出与输入/vae dtype 一致（训练/推理统一 bf16）
    lat = lat.to(images.dtype)
    return lat


@torch.no_grad()
def precompute_train_latents_to_memory(
    vae: AutoencoderKL,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    预编码训练集到内存：
    - 推理统一 bf16，减少显存压力
    - vis_lat, ir_lat, target: 未缩放 latent
    - 存储到内存时转为 CPU float32
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    vae.eval().to(device=device, dtype=torch.bfloat16)

    vis_list: List[torch.Tensor] = []
    ir_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []

    pbar = tqdm(dl, desc="预编码训练集到内存")
    for vis, ir, label in pbar:
        vis = vis.to(device, dtype=torch.bfloat16, non_blocking=True)
        ir = ir.to(device, dtype=torch.bfloat16, non_blocking=True)
        label = label.to(device, dtype=torch.bfloat16, non_blocking=True)

        vis_lat = encode_images_to_latents(vae, vis)     # 未缩放
        ir_lat = encode_images_to_latents(vae, ir)       # 未缩放
        label_lat = encode_images_to_latents(vae, label) # 未缩放

        vis_list.append(vis_lat.detach().to("cpu", dtype=torch.float32))
        ir_list.append(ir_lat.detach().to("cpu", dtype=torch.float32))
        tgt_list.append(label_lat.detach().to("cpu", dtype=torch.float32))

    all_vis = torch.cat(vis_list, dim=0)
    all_ir = torch.cat(ir_list, dim=0)
    all_tgt = torch.cat(tgt_list, dim=0)
    return all_vis, all_ir, all_tgt


@torch.no_grad()
def save_tensor_image(x: torch.Tensor, save_path: str):
    """
    x: [-1,1] float tensor, [C,H,W] or [1,C,H,W]
    保存为 PNG
    """
    from PIL import Image
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)  # [0,255]
    if x.shape[0] == 1:
        img = Image.fromarray(x[0].numpy(), mode="L")
    else:
        img = Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def build_latent_xrestormer(latent_ch: int) -> nn.Module:
    """
    构建用于 latent 空间的“无下采样”XRestormer：
    - 输入: vis latent（latent_ch）与 ir latent（latent_ch）
    - 输出: 融合 latent（latent_ch），不做额外缩放/反缩放
    - 内部按窗口动态 padding，支持任意分辨率
    """
    model = LatentXRestormerNoDown(
        latent_channels=latent_ch,
        dim=32,                 # 可调容量：32/48/64
        num_blocks=6,           # 堆叠块数
        window_size=8,
        num_channel_heads=2,
        num_spatial_heads=4,
        spatial_dim_head=16,
        ffn_expansion_factor=2.66,
        overlap_ratio=0.5,
        use_residual_to_A=False
    )
    return model


def train():
    torch.backends.cudnn.benchmark = True
    cfg = CONFIG
    epochs = CONFIG['training']['epoch']

    # 路径和训练超参
    run_name = cfg.get("run_name", "latent_train_v2")
    out_root = cfg.get("output_dir", "./checkpoints/pretrain/")
    out_dir = os.path.join(out_root, f"{run_name}-v2-latent-xrestormer")
    os.makedirs(out_dir, exist_ok=True)

    train_bs = int(cfg.get("training", {}).get("train_batch_size", 8))
    lr = float(cfg.get("training", {}).get("learning_rate", 2e-4))
    num_workers = int(cfg.get("training", {}).get("num_workers", 4))
    test_freq = int(cfg.get("training", {}).get("test_freq", 10))

    # pipeline 配置（日志用）
    pipeline_cfg = cfg.get("pipeline", {}) or {}
    use_shortcut = bool(pipeline_cfg.get("use_shortcut", False))

    # 构建 VAE（推理全 bf16）
    vae_cfg = cfg.get("model_config", {}).get("vae", {})
    pretrain_path = vae_cfg.get("pretrain", None)
    if pretrain_path is None:
        raise RuntimeError("CONFIG.model_config.vae.pretrain 未配置（需指向 sd-vae-ft-mse 或等效目录）")
    mode = 'official' if 'pretrain' in vae_cfg else 'unofficial'
    vae = build_vae(pretrain_path, mode=mode)
    latent_ch = int(getattr(vae.config, "latent_channels", 4))

    # 数据集
    ds_name = cfg.get("train_dataset", {}).get("name", "MSRS")
    tr_paths = cfg.get("datasets", {}).get(ds_name, {}).get("train", {})
    te_paths = cfg.get("datasets", {}).get(ds_name, {}).get("test", {})

    train_set = ImageFusionDataset(
        dir_A=tr_paths["dir_A"], dir_B=tr_paths["dir_B"], dir_C=tr_paths.get("dir_C", None),
        is_train=True, is_getpatch=False, augment=False
    )
    test_set = ImageFusionDataset(
        dir_A=te_paths["dir_A"], dir_B=te_paths["dir_B"], dir_C=None,
        is_train=False, is_getpatch=False
    )

    # 预编码或在线编码
    precompute_to_mem = os.environ.get("PRECOMPUTE_TO_MEMORY", "1") != "0"
    if precompute_to_mem:
        # 预编码推理 bf16；存 CPU float32
        vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_vis_cpu, train_ir_cpu, train_tgt_cpu = precompute_train_latents_to_memory(
            vae=vae, dataset=train_set, batch_size=train_bs, num_workers=num_workers, device=vae_device
        )
        train_dl = DataLoader(
            TensorDataset(train_vis_cpu, train_ir_cpu, train_tgt_cpu),
            batch_size=train_bs, shuffle=True, num_workers=0, pin_memory=True
        )
        encode_on_the_fly = False
    else:
        # 在线编码（更省内存，慢）
        train_dl = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
        encode_on_the_fly = True

    # 构建模型与优化器
    model = build_latent_xrestormer(latent_ch=latent_ch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 使用 Accelerate（默认 bf16）
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # VAE 在训练/导出阶段使用 bf16 以节省显存
    vae = vae.to(device=device, dtype=torch.bfloat16).eval()

    # 将组件交给 accelerator
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    if accelerator.is_main_process:
        print(f"use_shortcut={use_shortcut}, latent_ch={latent_ch}, encode_to_mem={precompute_to_mem}")
        print(f"输出目录: {out_dir}")

    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_dl, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            if encode_on_the_fly:
                # 在线编码（VAE/张量统一 bf16）
                vis, ir, label = batch
                vis = vis.to(device, dtype=torch.bfloat16, non_blocking=True)
                ir = ir.to(device, dtype=torch.bfloat16, non_blocking=True)
                label = label.to(device, dtype=torch.bfloat16, non_blocking=True)
                with torch.no_grad():
                    vis_lat = encode_images_to_latents(vae, vis)
                    ir_lat = encode_images_to_latents(vae, ir)
                    target = encode_images_to_latents(vae, label)  # 未缩放 latent
            else:
                # 预编码：CPU float32 -> 设备 bf16
                vis_lat, ir_lat, target = batch
                vis_lat = vis_lat.to(device, dtype=torch.bfloat16, non_blocking=True)
                ir_lat = ir_lat.to(device, dtype=torch.bfloat16, non_blocking=True)
                target = target.to(device, dtype=torch.bfloat16, non_blocking=True)

            with accelerator.autocast():
                pred = model(vis_lat, ir_lat)   # 直接输出融合后 latent（未缩放）
                loss = mse(pred, target)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            steps += 1
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{running_loss / steps:.4f}"})

        avg_loss = running_loss / max(1, steps)
        if accelerator.is_main_process:
            print(f"[Train] Epoch {epoch} - loss: {avg_loss:.6f}")

        # 简单验证/导出若干测试样例（在线编码 + bf16 解码）
        if (epoch % test_freq == 0) or (epoch == epochs):
            if accelerator.is_main_process:
                export_samples(vae, accelerator.unwrap_model(model), test_set, device, out_dir, epoch)

        # 保存权重
        if (epoch % max(cfg["training"]["save_freq"], test_freq) == 0) or (epoch == epochs):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(out_dir, f"epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(ckpt_dir, "latent_xrestormer.pth"))
                print(f"已保存模型到: {ckpt_dir}")
            accelerator.wait_for_everyone()


@torch.no_grad()
def export_samples(
    vae: AutoencoderKL,
    model: nn.Module,
    test_set: Dataset,
    device: torch.device,
    out_dir: str,
    epoch: int,
    max_batches: int = 3,
    batch_size: int = 4,
):
    model.eval()
    vae.eval()  # 已在训练中放到 bf16/device

    dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    save_dir = os.path.join(out_dir, f"samples_epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        vis, ir = batch[0].to(device, dtype=torch.bfloat16), batch[1].to(device, dtype=torch.bfloat16)

        vis_lat = encode_images_to_latents(vae, vis)     # 未缩放
        ir_lat = encode_images_to_latents(vae, ir)       # 未缩放

        fused_lat = model(vis_lat, ir_lat)               # 未缩放 latent
        # 确保解码输入与 VAE 权重 dtype 一致，避免 float vs bfloat16 冲突
        fused = vae.decode(fused_lat.to(dtype=vae.dtype)).sample  # [-1,1]

        # 保存
        for b in range(fused.shape[0]):
            save_tensor_image(fused[b], os.path.join(save_dir, f"fused_{cnt:05d}.png"))
            save_tensor_image(vis[b], os.path.join(save_dir, f"vis_{cnt:05d}.png"))
            if ir.shape[1] == 1:
                save_tensor_image(ir[b], os.path.join(save_dir, f"ir_{cnt:05d}.png"))
            cnt += 1

    print(f"[Eval] 已导出测试样例到: {save_dir}")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, required=True, help="训练轮数")
    # args = parser.parse_args()
    train()


if __name__ == "__main__":
    main()