import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel

from dataset import ImageFusionDataset
from model.pipeline import ConditioningEncoder

# ============ 全局配置（尽量与 pretrain_v2 保持一致，仅模型相关不同） ============
CONFIG = {
    "run_name": "fusion_UNet_one_step_v3",
    "output_dir": "./checkpoints/pretrain/",
    "model_config": {
        "vae": {
            "pretrain": "/home/zhangran/desktop/myProject/playground/sd-vae-ft-mse"
        },
        # 来自 config/pretrain.yml 的 UNet 定义（保持一致）
        "unet": {
            "block_out_channels": [64, 128, 256, 256],
            "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
            "cross_attention_dim": 512,
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 160
        },
        "encoder": {
            "in_channels": 8,
            "out_channels": 512,
            "base_channels": 128,
            "layer_blocks": [2, 2, 2]
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

# ============ 通用工具：VAE/编码/保存，与 v2 保持一致 ============
def build_vae(pretrain_path: str) -> AutoencoderKL:
    return AutoencoderKL.from_pretrained(pretrain_path)

@torch.no_grad()
def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    enc_out = vae.encode(images)
    lat = enc_out.latent_dist.sample() if hasattr(enc_out, "latent_dist") else enc_out[0]
    return lat.to(images.dtype)

@torch.no_grad()
def precompute_train_latents_to_memory(
    vae: AutoencoderKL,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    vae.eval().to(device=device, dtype=torch.bfloat16)

    vis_list, ir_list, tgt_list = [], [], []
    pbar = tqdm(dl, desc="预编码训练集到内存")
    for vis, ir, label in pbar:
        vis = vis.to(device, dtype=torch.bfloat16, non_blocking=True)
        ir = ir.to(device, dtype=torch.bfloat16, non_blocking=True)
        label = label.to(device, dtype=torch.bfloat16, non_blocking=True)

        vis_lat = encode_images_to_latents(vae, vis)
        ir_lat = encode_images_to_latents(vae, ir)
        label_lat = encode_images_to_latents(vae, label)

        vis_list.append(vis_lat.detach().to("cpu", dtype=torch.float32))
        ir_list.append(ir_lat.detach().to("cpu", dtype=torch.float32))
        tgt_list.append(label_lat.detach().to("cpu", dtype=torch.float32))

    return torch.cat(vis_list), torch.cat(ir_list), torch.cat(tgt_list)

@torch.no_grad()
def save_tensor_image(x: torch.Tensor, save_path: str):
    from PIL import Image
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    if x.shape[0] == 1:
        img = Image.fromarray(x[0].numpy(), mode="L")
    else:
        img = Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

# ============ UNet one-step: 动态对齐尺寸到 2^(L-1) 的倍数 ============
def _unet_downscale_factor(down_block_types: List[str]) -> int:
    # diffusers 的 UNet 会在每个 down block（除了最后一个）下采样一次
    return 2 ** (max(0, len(down_block_types) - 1))

def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad = (0, pad_w, 0, pad_h)
    if pad_h or pad_w:
        x = F.pad(x, pad, mode="replicate")
    return x, pad

def _unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    _, _, h, w = x.shape
    l, r, t, b = pad
    return x[:, :, : h - (t + b), : w - (l + r)]

def build_unet_and_encoder(cfg_model: dict) -> Tuple[UNet2DConditionModel, ConditioningEncoder, int]:
    unet = UNet2DConditionModel(**cfg_model["unet"])
    encoder = ConditioningEncoder(**cfg_model["encoder"])
    factor = _unet_downscale_factor(cfg_model["unet"]["down_block_types"])
    return unet, encoder, factor

# ============ 训练（与 v2 尽量一致，仅替换模型与前向） ============
def train():
    torch.backends.cudnn.benchmark = True
    cfg = CONFIG
    epochs = int(cfg["training"]["epoch"])

    run_name = cfg.get("run_name", "latent_train_v3_unet")
    out_root = cfg.get("output_dir", "./checkpoints/pretrain/")
    out_dir = os.path.join(out_root, f"{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    train_bs = int(cfg["training"].get("train_batch_size", 8))
    lr = float(cfg["training"].get("learning_rate", 2e-4))
    num_workers = int(cfg["training"].get("num_workers", 4))
    test_freq = int(cfg["training"].get("test_freq", 10))

    use_shortcut = bool(cfg.get("pipeline", {}).get("use_shortcut", False))

    # VAE
    pretrain_path = cfg["model_config"]["vae"]["pretrain"]
    vae = build_vae(pretrain_path)
    latent_ch = int(getattr(vae.config, "latent_channels", 4))

    # 数据集
    ds_name = cfg["train_dataset"]["name"]
    tr_paths = cfg["datasets"][ds_name]["train"]
    te_paths = cfg["datasets"][ds_name]["test"]
    train_set = ImageFusionDataset(tr_paths["dir_A"], tr_paths["dir_B"], tr_paths.get("dir_C"), is_train=True, is_getpatch=False, augment=False)
    test_set = ImageFusionDataset(te_paths["dir_A"], te_paths["dir_B"], None, is_train=False, is_getpatch=False)

    # 预编码
    precompute_to_mem = os.environ.get("PRECOMPUTE_TO_MEMORY", "1") != "0"
    if precompute_to_mem:
        vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_vis_cpu, train_ir_cpu, train_tgt_cpu = precompute_train_latents_to_memory(
            vae=vae, dataset=train_set, batch_size=train_bs, num_workers=num_workers, device=vae_device
        )
        train_dl = DataLoader(TensorDataset(train_vis_cpu, train_ir_cpu, train_tgt_cpu),
                              batch_size=train_bs, shuffle=True, num_workers=0, pin_memory=True)
        encode_on_the_fly = False
    else:
        train_dl = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
        encode_on_the_fly = True

    # 模型与优化器
    unet, cond_encoder, unet_factor = build_unet_and_encoder(cfg["model_config"])
    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(cond_encoder.parameters()), lr=lr, weight_decay=1e-4)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    vae = vae.to(device=device, dtype=torch.bfloat16).eval()

    # prepare
    unet, cond_encoder, optimizer, train_dl = accelerator.prepare(unet, cond_encoder, optimizer, train_dl)

    if accelerator.is_main_process:
        print(f"use_shortcut={use_shortcut}, latent_ch={latent_ch}, encode_to_mem={precompute_to_mem}, unet_factor={unet_factor}")
        print(f"输出目录: {out_dir}")

    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        unet.train(); cond_encoder.train()
        running_loss, steps = 0.0, 0

        pbar = tqdm(train_dl, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            if encode_on_the_fly:
                vis, ir, label = batch
                vis = vis.to(device, dtype=torch.bfloat16, non_blocking=True)
                ir = ir.to(device, dtype=torch.bfloat16, non_blocking=True)
                label = label.to(device, dtype=torch.bfloat16, non_blocking=True)
                with torch.no_grad():
                    vis_lat = encode_images_to_latents(vae, vis)
                    ir_lat = encode_images_to_latents(vae, ir)
                    target = encode_images_to_latents(vae, label)
            else:
                vis_lat, ir_lat, target = batch
                vis_lat = vis_lat.to(device, dtype=torch.bfloat16, non_blocking=True)
                ir_lat = ir_lat.to(device, dtype=torch.bfloat16, non_blocking=True)
                target = target.to(device, dtype=torch.bfloat16, non_blocking=True)

            # 条件与输入
            cond_in = torch.cat([vis_lat, ir_lat], dim=1)  # [B,8,H,W]
            # 动态对齐到 UNet 下采样倍数
            x_in, pad = _pad_to_multiple(vis_lat, unet_factor)
            cond_in, _ = _pad_to_multiple(cond_in, unet_factor)

            with accelerator.autocast():
                cond_embeds = cond_encoder(cond_in)  # encoder_hidden_states
                timesteps = torch.zeros(x_in.shape[0], dtype=torch.long, device=device)
                pred_pad = unet(x_in, timesteps, encoder_hidden_states=cond_embeds).sample
                pred = _unpad(pred_pad, pad)
                loss = mse(pred, target)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(list(unet.parameters()) + list(cond_encoder.parameters()), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.detach().cpu()); steps += 1
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{running_loss/steps:.4f}"})

        if accelerator.is_main_process:
            print(f"[Train] Epoch {epoch} - loss: {running_loss/max(1,steps):.6f}")

        if (epoch % test_freq == 0) or (epoch == epochs):
            if accelerator.is_main_process:
                export_samples(vae, accelerator.unwrap_model(unet), accelerator.unwrap_model(cond_encoder), unet_factor, test_set, device, out_dir, epoch)

        if (epoch % max(cfg["training"]["save_freq"], test_freq) == 0) or (epoch == epochs):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(out_dir, f"epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(accelerator.unwrap_model(unet).state_dict(), os.path.join(ckpt_dir, "unet.pth"))
                torch.save(accelerator.unwrap_model(cond_encoder).state_dict(), os.path.join(ckpt_dir, "cond_encoder.pth"))
                print(f"已保存模型到: {ckpt_dir}")
            accelerator.wait_for_everyone()

@torch.no_grad()
def export_samples(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    cond_encoder: nn.Module,
    unet_factor: int,
    test_set: Dataset,
    device: torch.device,
    out_dir: str,
    epoch: int,
    max_batches: int = 3,
    batch_size: int = 4,
):
    unet.eval(); cond_encoder.eval(); vae.eval()

    dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    save_dir = os.path.join(out_dir, f"samples_epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        vis, ir = batch[0].to(device, dtype=torch.bfloat16), batch[1].to(device, dtype=torch.bfloat16)

        vis_lat = encode_images_to_latents(vae, vis)
        ir_lat = encode_images_to_latents(vae, ir)

        cond_in = torch.cat([vis_lat, ir_lat], dim=1)
        x_in, pad = _pad_to_multiple(vis_lat, unet_factor)
        cond_in, _ = _pad_to_multiple(cond_in, unet_factor)

        cond_embeds = cond_encoder(cond_in)
        t = torch.zeros(x_in.shape[0], dtype=torch.long, device=device)
        fused_pad = unet(x_in, t, encoder_hidden_states=cond_embeds).sample
        fused_lat = _unpad(fused_pad, pad)

        fused = vae.decode(fused_lat.to(dtype=vae.dtype)).sample

        for b in range(fused.shape[0]):
            save_tensor_image(fused[b], os.path.join(save_dir, f"fused_{cnt:05d}.png"))
            save_tensor_image(vis[b], os.path.join(save_dir, f"vis_{cnt:05d}.png"))
            if ir.shape[1] == 1:
                save_tensor_image(ir[b], os.path.join(save_dir, f"ir_{cnt:05d}.png"))
            cnt += 1

    print(f"[Eval] 已导出测试样例到: {save_dir}")

def main():
    train()

if __name__ == "__main__":
    main()