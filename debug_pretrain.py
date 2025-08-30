# 调试版 pretrain：内部使用 dict 作为 config，仅作最小改动，
# 训练时 encoder 接收 VAE latent（concatenate vis+ir 的 latent），epoch 固定 50，并保存前4张测试图。
import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from model.pipeline import ConditioningEncoder
from dataset import ImageFusionDataset
import torchvision.utils as vutils
import torch.nn.functional as F

# cuDNN 加速
torch.backends.cudnn.benchmark = True

# --------------------------
# 内置 config（可直接修改）
# --------------------------
cfg = {
    "run_name": "fusion_diffusion_pretrain_v3",
    "output_dir": "./checkpoints/pretrain/debug",
    "model_config": {
        "unet": {
            "sample_size": 160,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 2,
            "block_out_channels": (128, 128, 256, 256, 512, 512),
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            "cross_attention_dim": 512,
        },
        "encoder": {
            "in_channels": None,    # 将在下方根据 VAE latent 计算并赋值
            "out_channels": 512,
            "base_channels": 128,
            "layer_blocks": (2, 2, 2),
        },
        "vae": {
            "pretrain": "/home/zhangran/desktop/myProject/playground/sd-vae-ft-mse"
        },
    },
    "diffusion": {
        "num_train_timesteps": 1000,
        "num_inference_steps": 20,
        "beta_schedule": "squaredcos_cap_v2",
    },
    "training": {
        "num_epochs": 50,                 # 固定为 50（按要求）
        "train_batch_size": 16,
        "learning_rate": 2e-5,
        "optimizer": {
            "type": "AdamW",
            "args": {"weight_decay": 0.0001, "betas": (0.9, 0.999)},
        },
        "lr_warmup_steps": 500,
        "test_freq": 5,
        "save_freq": 50,
        "num_workers": 4,
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
    },
    "dataset": {
        "train": {
            "dir_A": './data/MSRS-main/MSRS-main/train/vi',
            "dir_B": './data/MSRS-main/MSRS-main/train/ir',
            "dir_C": './data/MSRS-main/MSRS-main/train/label'
        },
        "test":{
            "dir_A": './data/MSRS-main/MSRS-main/test/vi',
            "dir_B": './data/MSRS-main/MSRS-main/test/ir'
        }
    },
    "pipeline": {
        "use_shortcut": False,
        "use_ddim_logprob": False,
    },
}

# 依据 VAE latent channels 重新计算 encoder in_channels（vis_lat + ir_lat concat）
cfg["model_config"]["encoder"]["in_channels"] = cfg["model_config"]["vae"].get("latent_channels", 4) * 2

# 根据是否使用预训练VAE调整UNet sample_size
if 'pretrain' in cfg["model_config"]["vae"]:
    # 预训练VAE使用8倍下采样
    vae_scale_factor = 8
    cfg["model_config"]["unet"]["sample_size"] = 640 // vae_scale_factor  # 80
else:
    # 自定义VAE使用4倍下采样
    vae_scale_factor = 4
    cfg["model_config"]["unet"]["sample_size"] = 640 // vae_scale_factor  # 160

# --------------------------
# 简单工具
# --------------------------
def find_vae_checkpoint(path):
    return path if os.path.exists(path) else None

# --------------------------
# 主流程（尽量与 pretrain.py 对齐，最小改动）
# --------------------------
def main():
    os.makedirs(cfg["output_dir"], exist_ok=True)
    project_dir = cfg["output_dir"]
    accelerator = Accelerator(log_with="tensorboard", project_dir=project_dir)
    if accelerator.is_main_process:
        print("Debug pretrain start. Writing to", project_dir)

    # VAE 初始化
    vae_cfg = cfg["model_config"]["vae"]
    
    # Check if using pretrained VAE
    if 'pretrain' in vae_cfg:
        pretrain_path = vae_cfg['pretrain']
        if accelerator.is_main_process:
            print(f"Loading pretrained VAE from: {pretrain_path}")
        vae = AutoencoderKL.from_pretrained(pretrain_path)
        if accelerator.is_main_process:
            print("Pretrained VAE loaded successfully.")
    else:
        vae = AutoencoderKL(
            sample_size=vae_cfg["sample_size"],
            in_channels=vae_cfg["in_channels"],
            out_channels=vae_cfg["out_channels"],
            down_block_types=vae_cfg["down_block_types"],
            up_block_types=vae_cfg["up_block_types"],
            block_out_channels=vae_cfg["block_out_channels"],
            latent_channels=vae_cfg["latent_channels"],
            scaling_factor=vae_cfg["scaling_factor"],
        )
        vae_ckpt = find_vae_checkpoint(vae_cfg["checkpoint_dir"])
        if vae_ckpt is None:
            if accelerator.is_main_process:
                print("VAE checkpoint not found at", vae_cfg["checkpoint_dir"], " -> abort for reliable test")
            return
        sd = torch.load(vae_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        vae.load_state_dict(sd, strict=True)
        if accelerator.is_main_process:
            print("Loaded VAE:", vae_ckpt)

    # Dataset / Dataloader
    train_dataset = ImageFusionDataset(
        dir_A=cfg["dataset"]["train"]["dir_A"],
        dir_B=cfg["dataset"]["train"]["dir_B"],
        dir_C=cfg["dataset"]["train"]["dir_C"],
        is_train=True,
    )
    # 与 pretrain 对齐：num_workers 用配置，不强制 >=1；评测用单独 loader，防止与 Accelerator 包装冲突
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["train_batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],   # 不再 max(1, ...)
        pin_memory=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["train_batch_size"],
        shuffle=False,
        num_workers=0,         # 避免评测时 DataLoader worker/信号死锁
        pin_memory=False
    )

    # --------------------------
    # 训练前：预先 VAE 编码训练集到内存
    # --------------------------
    # 依据设置判断是否使用 latent 作为条件（encoder 输入为 latent*2 即启用）
    latent_ch = vae_cfg.get("latent_channels", 4)
    use_latent_condition = (cfg["model_config"]["encoder"]["in_channels"] == latent_ch * 2)
    use_shortcut = bool(cfg.get("pipeline", {}).get("use_shortcut", False))
    scaling_factor = vae.config.scaling_factor

    class InMemoryLatentDataset(torch.utils.data.Dataset):
        def __init__(self, conditions: torch.Tensor, targets: torch.Tensor):
            self.conditions = conditions
            self.targets = targets
        def __len__(self):
            return self.conditions.shape[0]
        def __getitem__(self, idx):
            return self.conditions[idx], self.targets[idx]

    pre_bs = cfg["training"]["train_batch_size"]
    pre_dl = DataLoader(
        train_dataset,
        batch_size=pre_bs,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    vae_device = accelerator.device
    vae = vae.to(device=vae_device)  # 预计算阶段使用 fp32 更稳
    vae.eval()
    from tqdm.auto import tqdm as _tqdm
    precomputed_conditions: list[torch.Tensor] = []
    precomputed_targets: list[torch.Tensor] = []
    pbar_pre = _tqdm(pre_dl, disable=not accelerator.is_main_process, desc="Precomputing VAE latents")
    with torch.no_grad():
        for batch in pbar_pre:
            if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                raise RuntimeError(f"Train dataset must return (vis, ir, label), got {type(batch)} with len={len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            vis_images, ir_images, label_images = batch
            vis_images = vis_images.to(device=vae_device, dtype=torch.float32, non_blocking=True)
            ir_images = ir_images.to(device=vae_device, dtype=torch.float32, non_blocking=True)
            label_images = label_images.to(device=vae_device, dtype=torch.float32, non_blocking=True)

            # 计算目标 latent
            if use_shortcut:
                vis_lat_t = vae.encode(vis_images).latent_dist.sample()
                label_lat_t = vae.encode(label_images).latent_dist.sample()
                lat_target = (label_lat_t - vis_lat_t) * scaling_factor
            else:
                label_lat_t = vae.encode(label_images).latent_dist.sample()
                lat_target = label_lat_t * scaling_factor

            # 计算条件输入
            if use_latent_condition:
                ir_in = ir_images.repeat(1, 3, 1, 1) if ir_images.shape[1] == 1 else ir_images
                enc_out = vae.encode(torch.cat([vis_images, ir_in], dim=0))
                lat_all = enc_out.latent_dist.sample() if hasattr(enc_out, "latent_dist") else (
                    enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
                )
                B = vis_images.shape[0]
                vis_lat_c, ir_lat_c = lat_all[:B], lat_all[B:2*B]
                condition_input = torch.cat([vis_lat_c, ir_lat_c], dim=1)
            else:
                condition_input = torch.cat([vis_images, ir_images], dim=1)

            # 存为 CPU float16，节省内存
            precomputed_conditions.append(condition_input.detach().to(device="cpu", dtype=torch.float16))
            precomputed_targets.append(lat_target.detach().to(device="cpu", dtype=torch.float16))

    if len(precomputed_conditions) == 0:
        raise RuntimeError("No training samples found during VAE precompute.")
    all_conditions = torch.cat(precomputed_conditions, dim=0)
    all_targets = torch.cat(precomputed_targets, dim=0)
    if all_conditions.shape[0] != all_targets.shape[0]:
        raise RuntimeError(f"Precomputed sizes mismatch: conditions={all_conditions.shape[0]} vs targets={all_targets.shape[0]}")

    # 替换训练 DataLoader 为内存数据集
    inmem_ds = InMemoryLatentDataset(all_conditions, all_targets)
    train_loader = DataLoader(
        inmem_ds,
        batch_size=cfg["training"]["train_batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # model 初始化：UNet + ConditioningEncoder（encoder in_channels 改为 latent*2）
    latent_ch = vae_cfg.get("latent_channels", 4)
    cfg["model_config"]["encoder"]["in_channels"] = latent_ch * 2

    unet = UNet2DConditionModel(**cfg["model_config"]["unet"])
    encoder = ConditioningEncoder(in_channels=cfg["model_config"]["encoder"]["in_channels"],
                                  out_channels=cfg["model_config"]["encoder"]["out_channels"])

    # wrapper 保持与 pretrain 类似
    class ModelWrapper(torch.nn.Module):
        def __init__(self, unet, encoder):
            super().__init__()
            self.unet = unet
            self.encoder = encoder

        def forward(self, noisy_latent, timesteps, condition_latent):
            # condition_latent: concatenated vis_lat + ir_lat
            cond_emb = self.encoder(condition_latent)
            pred = self.unet(noisy_latent, timesteps, encoder_hidden_states=cond_emb).sample
            return pred

    model_wrapper = ModelWrapper(unet, encoder)

    # optimizer & lr_scheduler
    OptimCls = getattr(torch.optim, cfg["training"]["optimizer"]["type"])
    optimizer = OptimCls(model_wrapper.parameters(), lr=cfg["training"]["learning_rate"], **cfg["training"]["optimizer"].get("args", {}))
    total_steps = len(train_loader) * cfg["training"]["num_epochs"]
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg["training"]["lr_warmup_steps"], num_training_steps=total_steps)

    # diffusion scheduler
    diffusion = DDIMScheduler(num_train_timesteps=cfg["diffusion"]["num_train_timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"])

    # accelerate prepare
    model_wrapper, optimizer, train_loader, lr_scheduler = accelerator.prepare(model_wrapper, optimizer, train_loader, lr_scheduler)
    device = accelerator.device
    model_dtype = next(model_wrapper.parameters()).dtype
    vae = vae.to(device=device, dtype=model_dtype)
    vae.eval()

    # training loop （与 pretrain 保持相同逻辑，encoder 输入为 VAE latents）
    num_epochs = cfg["training"]["num_epochs"]
    test_freq = cfg["training"]["test_freq"]  # 只在第 50 轮评测，避免每轮评测拖慢/卡死
    for epoch in range(num_epochs):
        model_wrapper.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = tuple(t.to(device=device, dtype=model_dtype) for t in batch)
            with accelerator.accumulate(model_wrapper):
                # 支持两种 batch：
                # 1) 预计算路径：(condition_input, lat_target)
                # 2) 旧路径：(vis_images, ir_images, label_images) —— 为兼容保留
                if len(batch) == 2:
                    condition_input, lat_target = batch
                elif len(batch) == 3:
                    vis_images, ir_images, label_images = batch
                    # 在线路径（不推荐）：仍旧执行最少一次 VAE，保持兼容
                    with torch.no_grad():
                        B = vis_images.shape[0]
                        ir_in = ir_images.repeat(1, 3, 1, 1) if ir_images.shape[1] == 1 else ir_images
                        enc_all = vae.encode(torch.cat([vis_images, ir_in, label_images], dim=0))
                        lat_all = enc_all.latent_dist.sample() if hasattr(enc_all, "latent_dist") else (
                            enc_all[0] if isinstance(enc_all, (tuple, list)) else enc_all
                        )
                        vis_lat, ir_lat, label_lat = lat_all[:B], lat_all[B:2*B], lat_all[2*B:3*B]
                        lat_target = (label_lat - vis_lat) * scaling_factor if use_shortcut else (label_lat * scaling_factor)
                        condition_input = torch.cat([vis_lat, ir_lat], dim=1) if use_latent_condition else torch.cat([vis_images, ir_images], dim=1)
                else:
                    raise RuntimeError(f"Unexpected batch size: expected 2 or 3 elements, got {len(batch)}")

                bs = lat_target.shape[0]
                timesteps = torch.randint(0, diffusion.config.num_train_timesteps, (bs,), device=device).long()
                noise = torch.randn_like(lat_target)
                noisy_latent = diffusion.add_noise(lat_target, noise, timesteps)

                pred_noise = model_wrapper(noisy_latent, timesteps, condition_input)
                loss = F.mse_loss(pred_noise, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), cfg["training"]["max_grad_norm"])
                optimizer.step()
                try:
                    lr_scheduler.step()
                except Exception:
                    pass
                optimizer.zero_grad()
            pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

        # 仅在第 5 轮评测（与 test_freq 对齐，避免每轮评测导致速度减半与潜在死锁）
        if accelerator.is_main_process and ((epoch + 1) % test_freq == 0):
            print(f"Epoch {epoch+1} finished. Running quick sample test...")
            model_wrapper.eval()
            unwrapped = accelerator.unwrap_model(model_wrapper)
            unet_eval = unwrapped.unet
            encoder_eval = unwrapped.encoder

            # 使用独立 eval_loader，避免复用 train_loader（Accelerator 包装后复用可能阻塞）
            eval_batch = next(iter(eval_loader))
            eval_batch = tuple(t.to(device=device, dtype=model_dtype) for t in eval_batch)
            vis_test, ir_test, _ = eval_batch

            vis_t = vis_test
            ir_t = ir_test.repeat(1, 3, 1, 1) if ir_test.shape[1] == 1 else ir_test

            with torch.inference_mode():
                B_t = vis_t.shape[0]
                enc_cond = vae.encode(torch.cat([vis_t, ir_t], dim=0))
                lat_cond_all = enc_cond.latent_dist.sample() if hasattr(enc_cond, "latent_dist") else (
                    enc_cond[0] if isinstance(enc_cond, (tuple, list)) else enc_cond
                )
                vis_lat_t, ir_lat_t = lat_cond_all[:B_t], lat_cond_all[B_t:2*B_t]
                cond_lat_t = torch.cat([vis_lat_t, ir_lat_t], dim=1)

                lat_channels = vae.config.latent_channels
                lat_h, lat_w = vis_lat_t.shape[2], vis_lat_t.shape[3]
                latents = torch.randn((B_t, lat_channels, lat_h, lat_w), device=device, dtype=model_dtype)
                diffusion.set_timesteps(cfg["diffusion"]["num_inference_steps"])
                for t in diffusion.timesteps:
                    noise_pred = unet_eval(latents, t, encoder_hidden_states=encoder_eval(cond_lat_t)).sample
                    latents = diffusion.step(noise_pred, t, latents).prev_sample

                if hasattr(vae.config, "scaling_factor"):
                    latents = latents / vae.config.scaling_factor

                fused = vae.decode(latents).sample  # [-1,1] -> [0,1]
                fused_norm = ((fused.clamp(-1, 1) + 1.0) / 2.0).detach().cpu()
                save_dir = os.path.join(cfg["output_dir"], "debug_samples")
                os.makedirs(save_dir, exist_ok=True)
                for i in range(min(4, fused_norm.shape[0])):
                    vutils.save_image(fused_norm[i], os.path.join(save_dir, f"epoch{epoch+1:03d}_sample_{i}.png"))
                print("Saved sample images to", save_dir)

    # 最终保存模型（解包）
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model_wrapper)
        torch.save(unwrapped.unet.state_dict(), os.path.join(project_dir, "unet_debug.pth"))
        torch.save(unwrapped.encoder.state_dict(), os.path.join(project_dir, "encoder_debug.pth"))
        print("Saved debug model weights to", project_dir)


if __name__ == "__main__":
    main()