import os
import yaml
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel

# 项目内模块
from model.pipeline import ConditioningEncoder, ImageFusionPipeline
from dataset import ImageFusionDataset

# 配置区（根据需要调整）
PRETRAIN_BASE = "./checkpoints/pretrain/fusion_diffusion_pretrain_v4"
VAE_FALLBACK = "./checkpoints/vae/best.pth"
NUM_BATCHES = 2        # 每个 loader 检查多少批次
NUM_INFERENCE_STEPS = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1        # 每多少个 batch 打印一次详细表格
USE_LATENT_COND = True

def find_latest_run(base):
    cand = sorted(Path(base).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No runs found under {base}")
    return cand[0]

def find_weights(base):
    final = Path(base) / "final"
    if final.exists():
        u = final / "unet.pth"
        e = final / "encoder.pth"
        if u.exists() and e.exists():
            return u, e
    epochs = sorted(Path(base).glob("epoch_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ep in epochs:
        u = ep / "unet.pth"
        e = ep / "encoder.pth"
        if u.exists() and e.exists():
            return u, e
    u = Path(base) / "unet.pth"
    e = Path(base) / "encoder.pth"
    if u.exists() and e.exists():
        return u, e
    return None, None

def tensor_stats(t: torch.Tensor, name="tensor"):
    # returns dict of stats (over full tensor)
    a = t.detach().cpu().ravel().numpy()
    return {
        "name": name,
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "median": float(np.nanmedian(a)),
        "p10": float(np.nanpercentile(a, 10)),
        "p90": float(np.nanpercentile(a, 90)),
        "abs_mean": float(np.nanmean(np.abs(a))),
        "var": float(np.nanvar(a)),
    }

def per_sample_stats(tensor: torch.Tensor):
    # tensor shape (B, C, H, W) or (B, ...). Return list of dicts per sample.
    B = tensor.shape[0]
    out = []
    for i in range(B):
        out.append(tensor_stats(tensor[i], name=f"sample{i}"))
    return out

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    # flatten per-sample cosine similarity
    a_f = a.reshape(a.shape[0], -1)
    b_f = b.reshape(b.shape[0], -1)
    num = (a_f * b_f).sum(dim=1)
    denom = (a_f.norm(dim=1) * b_f.norm(dim=1)).clamp(min=eps)
    return (num / denom).detach().cpu().numpy()

def run_debug_on_loader(loader, name, unet, encoder, vae, scheduler, scaling_factor, use_shortcut=False, max_batches=NUM_BATCHES):
    print(f"\n=== Debug dataset: {name} (device={DEVICE}) ===")
    unet.eval(); encoder.eval(); vae.eval()
    batch_idx = 0
    for batch in loader:
        if batch_idx >= max_batches:
            break
        batch = tuple(t.to(device=DEVICE, dtype=torch.float32) for t in batch)
        vis = batch[0]
        ir = batch[1]
        label = batch[2] if len(batch) > 2 else None

        B = vis.shape[0]
        with torch.no_grad():
            # encode latents (robust)
            def safe_encode(x):
                enc = vae.encode(x)
                if hasattr(enc, "latent_dist"):
                    try:
                        s = enc.latent_dist.sample()
                    except Exception:
                        s = enc.latent_dist.mean
                elif isinstance(enc, dict):
                    if "latent_dist" in enc:
                        s = enc["latent_dist"].sample()
                    elif "sample" in enc:
                        s = enc["sample"]
                    else:
                        s = torch.as_tensor(enc)
                else:
                    s = enc
                return s

            vis_lat = safe_encode(vis).to(DEVICE)
            if label is not None:
                label_lat = safe_encode(label).to(DEVICE)
            else:
                label_lat = None

            # 两种训练目标：shortcut=True -> residual (label - vis)，否则 -> label_lat
            if label_lat is not None:
                lat_res_raw = label_lat - vis_lat       # 未缩放的残差 (latent space)
                lat_label_raw = label_lat
            else:
                lat_res_raw = None
                lat_label_raw = None

            if use_shortcut:
                if lat_res_raw is not None:
                    lat_target_raw = lat_res_raw
                else:
                    lat_target_raw = None
            else:
                lat_target_raw = lat_label_raw

            # scaled target (训练时使用)
            if lat_target_raw is not None:
                lat_target_scaled = lat_target_raw * scaling_factor
            else:
                lat_target_scaled = None

            # 随机 timesteps & noise (模拟训练中的 add_noise)
            ts = None
            noise = None
            noisy = None
            if lat_target_scaled is not None:
                ts = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
                noise = torch.randn_like(lat_target_scaled)
                try:
                    noisy = scheduler.add_noise(lat_target_scaled, noise, ts)
                except Exception:
                    noisy = lat_target_scaled + noise
                # helper: build condition input for encoder (一次性 encode vis+ir when using latent cond)
            def build_condition_input(vis_tensor: torch.Tensor, ir_tensor: torch.Tensor) -> torch.Tensor:
                if USE_LATENT_COND:
                    ir_in = ir_tensor.repeat(1, 3, 1, 1) if ir_tensor.shape[1] == 1 else ir_tensor
                    with torch.no_grad():
                        enc = vae.encode(torch.cat([vis_tensor, ir_in], dim=0))
                        lat = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else (enc[0] if isinstance(enc, (tuple, list)) else enc)
                    B = vis_tensor.shape[0]
                    vis_lat, ir_lat = lat[:B], lat[B:2*B]
                    return torch.cat([vis_lat, ir_lat], dim=1)
                else:
                    return torch.cat([vis_tensor, ir_tensor], dim=1)
            # get condition embeds: 使用 build_condition_input 自动选择 pixel/latent；IR 单通道时 repeat 到 3
            cond_in = build_condition_input(vis, ir).to(device=DEVICE, dtype=unet.dtype)
            cond_emb = encoder(cond_in)

            # predicted noise (如果有 noisy)
            pred_noise = None
            if noisy is not None:
                pred_noise = unet(noisy, ts, encoder_hidden_states=cond_emb).sample

            # ---- statistics ----
            print(f"\nBatch {batch_idx} --- B={B} (use_shortcut={use_shortcut})")
            if lat_target_raw is not None:
                print("Scaling factor:", scaling_factor)
                print("Latent channels/shape:", lat_target_raw.shape)

            # items to report
            items = [("vis_lat", vis_lat)]
            if label_lat is not None:
                items += [("label_lat", label_lat), ("lat_res_raw", lat_res_raw)]
                items += [("lat_target_raw", lat_target_raw), ("lat_target_scaled", lat_target_scaled)]
            if noise is not None:
                items += [("noise", noise)]
            if pred_noise is not None:
                items += [("pred_noise", pred_noise)]

            # 打印均值/方差等
            for k, t in items:
                if t is None:
                    continue
                s = tensor_stats(t)
                print(f"{k}: mean={s['mean']:.6f}, std={s['std']:.6f}, var={s['var']:.6f}, abs_mean={s['abs_mean']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}")

            # 额外统计：noise 与 target 的总体均值/方差以及 SNR（按方差比）
            if lat_target_scaled is not None and noise is not None:
                # 全局方差（每样本再平均）
                target_var_per_sample = lat_target_scaled.reshape(B, -1).var(dim=1).detach().cpu().numpy()
                target_mean_per_sample = lat_target_scaled.reshape(B, -1).mean(dim=1).detach().cpu().numpy()
                noise_var_per_sample = noise.reshape(B, -1).var(dim=1).detach().cpu().numpy() + 1e-12
                noise_mean_per_sample = noise.reshape(B, -1).mean(dim=1).detach().cpu().numpy()
                snr_per_sample = target_var_per_sample / noise_var_per_sample
                print("Per-sample target_mean, target_var, noise_mean, noise_var, SNR(var_target/var_noise):")
                for i in range(B):
                    print(f"  sample{i}: t_mean={target_mean_per_sample[i]:.6e}, t_var={target_var_per_sample[i]:.6e}, n_mean={noise_mean_per_sample[i]:.6e}, n_var={noise_var_per_sample[i]:.6e}, SNR={snr_per_sample[i]:.6f}")
                print("SNR stats: mean {:.4f}, median {:.4f}, min {:.4f}, max {:.4f}".format(np.mean(snr_per_sample), np.median(snr_per_sample), np.min(snr_per_sample), np.max(snr_per_sample)))

                # 如果 pred_noise 存在，打印 pred 的均值/方差并与 noise 比较
                if pred_noise is not None:
                    pred_mean_per_sample = pred_noise.reshape(B, -1).mean(dim=1).detach().cpu().numpy()
                    pred_var_per_sample = pred_noise.reshape(B, -1).var(dim=1).detach().cpu().numpy()
                    print("Predicted noise mean/var per-sample and ratio to true noise var:")
                    for i in range(B):
                        ratio = pred_var_per_sample[i] / (noise_var_per_sample[i] + 1e-12)
                        print(f"  sample{i}: pred_mean={pred_mean_per_sample[i]:.6e}, pred_var={pred_var_per_sample[i]:.6e}, var_ratio={ratio:.6f}")

                    # 预测噪声与真实噪声的cosine与MSE
                    cos_sim = cosine_sim(pred_noise, noise)
                    mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=tuple(range(1, pred_noise.dim()))).detach().cpu().numpy()
                    print("Cosine sim(pred,noise) per-sample:", cos_sim.tolist())
                    print("MSE(pred,noise) per-sample:", mse_per_sample.tolist())

            # 分析 scaling_factor 对 raw target 的影响（显示 raw std 和 scaled std）
            if lat_target_raw is not None:
                raw_std = lat_target_raw.reshape(B, -1).std(dim=1).detach().cpu().numpy()
                scaled_std = lat_target_scaled.reshape(B, -1).std(dim=1).detach().cpu().numpy()
                print("Target std before/after scaling_factor per-sample:")
                for i in range(B):
                    print(f"  sample{i}: raw_std={raw_std[i]:.6e}, scaled_std={scaled_std[i]:.6e}, scaling_factor={scaling_factor:.6e}")

            # compare vis_lat vs label_lat norms (how big the residual is relative to vis)
            if label_lat is not None:
                vis_norm = vis_lat.reshape(B, -1).norm(dim=1).detach().cpu().numpy()
                lab_norm = label_lat.reshape(B, -1).norm(dim=1).detach().cpu().numpy()
                res_norm = (label_lat - vis_lat).reshape(B, -1).norm(dim=1).detach().cpu().numpy()
                for i in range(B):
                    print(f"Latent norms sample{i}: vis_norm={vis_norm[i]:.6e}, label_norm={lab_norm[i]:.6e}, res_norm={res_norm[i]:.6e}, res/vis_ratio={(res_norm[i]/(vis_norm[i]+1e-12)):.6f}")

        batch_idx += 1

def main():
    # find run and config
    run_dir = find_latest_run(PRETRAIN_BASE)
    config_path = Path(PRETRAIN_BASE) / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 读取 pipeline 配置（short cut）
    pipeline_cfg = config.get("pipeline", {})
    use_shortcut = bool(pipeline_cfg.get("use_shortcut", False))

    print("Using run_dir:", run_dir)
    # load VAE
    vae_cfg = config["model_config"]["vae"]
    
    # Check if using pretrained VAE
    if 'pretrain' in vae_cfg:
        pretrain_path = vae_cfg['pretrain']
        print(f"Loading pretrained VAE from: {pretrain_path}")
        vae = AutoencoderKL.from_pretrained(pretrain_path)
        print("Pretrained VAE loaded successfully.")
    else:
        vae_init_kwargs = {k: v for k, v in vae_cfg.items() if k != "checkpoint_dir"}
        vae = AutoencoderKL(**vae_init_kwargs)
        vae_ckpt_path = vae_cfg.get("checkpoint_dir", VAE_FALLBACK)
        if not os.path.exists(vae_ckpt_path):
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_ckpt_path}")
        sd = torch.load(vae_ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        vae.load_state_dict(sd, strict=True)
    
    vae.to(DEVICE).eval()
    scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is None:
        raise RuntimeError("vae.config.scaling_factor missing")
    print("Loaded VAE, scaling_factor =", scaling_factor)

    # load UNet & Encoder
    unet_cfg = config["model_config"]["unet"]
    unet = UNet2DConditionModel(**unet_cfg)
    encoder = ConditioningEncoder(**config["model_config"]["encoder"])
    unet_path, encoder_path = find_weights(run_dir)
    if unet_path is None or encoder_path is None:
        raise FileNotFoundError(f"Could not find unet/encoder weights under {run_dir}")
    unet.load_state_dict(torch.load(unet_path, map_location="cpu"))
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    unet.to(DEVICE).eval(); encoder.to(DEVICE).eval()
    print("Loaded UNet and Encoder weights")

    # 自动判断是否使用 latent 条件（encoder 输入通道 == vae.latent_channels*2）
    vae_latent_ch = int(getattr(vae.config, "latent_channels", 4))
    enc_in_ch = int(getattr(getattr(encoder, "conv1", None), "in_channels", 0) or 0)
    USE_LATENT_COND = (enc_in_ch == vae_latent_ch * 2)
    print(f"USE_LATENT_COND={USE_LATENT_COND} (encoder_in={enc_in_ch}, vae_latent_ch={vae_latent_ch})")

    # scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # data loaders (use same as pretrain)
    train_ds_config = config["train_dataset"]
    train_paths = config["datasets"][train_ds_config["name"]]["train"]
    train_dataset = ImageFusionDataset(dir_A=train_paths["dir_A"], dir_B=train_paths["dir_B"], dir_C=train_paths.get("dir_C"), is_train=True, is_getpatch=False, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=min(4, len(train_dataset) or 1), shuffle=False, num_workers=2)

    test_loaders = {}
    for test_set in config.get("test_sets", []):
        name = test_set["name"]
        test_paths = config["datasets"][name]["test"]
        ds = ImageFusionDataset(dir_A=test_paths["dir_A"], dir_B=test_paths["dir_B"], dir_C=test_paths.get("dir_C"), is_train=False, is_getpatch=False)
        test_loaders[name] = DataLoader(ds, batch_size=test_set.get("test_batch_size", 4), shuffle=False)

    # run debug on train and test first sets
    run_debug_on_loader(train_loader, "train", unet, encoder, vae, scheduler, scaling_factor, use_shortcut=use_shortcut, max_batches=NUM_BATCHES)
    for name, loader in test_loaders.items():
        run_debug_on_loader(loader, name, unet, encoder, vae, scheduler, scaling_factor, use_shortcut=use_shortcut, max_batches=NUM_BATCHES)

if __name__ == "__main__":
    main()