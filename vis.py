# 新增 notebook cell：加载已保存权重，用 pipeline 进行 train/test 推理，打印图与指标，并对训练集计算一次 proxy loss
# 说明：请根据实际 run_name/保存路径调整 RUN_NAME / CHECKPOINT_DIR 变量
import os, yaml, glob, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from IPython.display import display
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel

# 项目内模块
from model.pipeline import ImageFusionPipeline, ConditioningEncoder
from dataset import ImageFusionDataset
import metric

# --- 配置区域：根据实际调整 ---
RUN_NAME = None  # 若已知设置为 "your_run_name"，否则留 None 自动选择 checkpoints/pretrain 下最新的 run
PRETRAIN_BASE = "./checkpoints/pretrain/fusion_diffusion_pretrain_v3"
VAE_FALLBACK = "./checkpoints/vae/best.pth"
NUM_SHOW_BATCHES = 1   # 每个数据集展示多少个批次
NUM_INFERENCE_STEPS = 20
SAVE_DIR = 'save_images/v3'

# --- 自动查找 run_dir 和 config.yml ---
if RUN_NAME is None:
    cand = sorted(Path(PRETRAIN_BASE).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No runs found under {PRETRAIN_BASE}. Set RUN_NAME manually.")
    run_dir = cand[0]
else:
    run_dir = Path(PRETRAIN_BASE) / RUN_NAME
config_path = Path(PRETRAIN_BASE) / "config.yml"
if not config_path.exists():
    raise FileNotFoundError(f"Config not found at {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print(f"Using run_dir = {run_dir}")
print(f"Loaded config from {config_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 构建模型结构并加载权重 ---
# 1) VAE
vae_cfg = config["model_config"]["vae"]
vae_init_kwargs = {k: v for k, v in vae_cfg.items() if k != "checkpoint_dir"}
vae = AutoencoderKL(**vae_init_kwargs)
vae_ckpt_path = vae_cfg.get("checkpoint_dir", VAE_FALLBACK)
if not os.path.exists(vae_ckpt_path):
    raise FileNotFoundError(f"VAE checkpoint not found at {vae_ckpt_path}")
sd = torch.load(vae_ckpt_path, map_location="cpu")
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
vae.load_state_dict(sd, strict=True)
vae.to(device).eval()
scaling_factor = getattr(vae.config, "scaling_factor", None)
if scaling_factor is None:
    raise RuntimeError("vae.config.scaling_factor missing")

# 2) UNet & Encoder
unet_cfg = config["model_config"]["unet"]
unet = UNet2DConditionModel(**unet_cfg)
encoder = ConditioningEncoder(**config["model_config"]["encoder"])

# 寻找保存的 unet/encoder 权重（优先 final，再找 epoch_*）
def find_weights(base):
    final = Path(base) / "final"
    if final.exists():
        u = final / "unet.pth"
        e = final / "encoder.pth"
        if u.exists() and e.exists():
            return u, e
    # 寻找最新 epoch_x
    epochs = sorted(Path(base).glob("epoch_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ep in epochs:
        u = ep / "unet.pth"
        e = ep / "encoder.pth"
        if u.exists() and e.exists():
            return u, e
    # fallback to files directly under run_dir
    u = Path(base) / "unet.pth"
    e = Path(base) / "encoder.pth"
    if u.exists() and e.exists():
        return u, e
    return None, None

unet_path, encoder_path = find_weights(run_dir)
if unet_path is None or encoder_path is None:
    raise FileNotFoundError(f"Could not find unet/encoder weights under {run_dir}")
unet.load_state_dict(torch.load(unet_path, map_location="cpu"))
encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
unet.to(device).eval()
encoder.to(device).eval()

# scheduler for inference / for computing add_noise during proxy loss
diffusion_cfg = config.get("diffusion", {})
num_train_timesteps = 1000
scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2")

# 构造 pipeline
pipeline = ImageFusionPipeline(unet=unet, scheduler=scheduler, encoder=encoder, vae=vae, vae_scale_factor=4).to(device)

# --- 数据集 loaders（参考 pretrain.py） ---
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

# --- metric 准备（优先使用 *_function_batch） ---
metric_batch_funcs = {}
metric_single_funcs = {}
for name in dir(metric):
    if name.endswith("_function_batch") and callable(getattr(metric, name)):
        metric_batch_funcs[name[:-15]] = getattr(metric, name)
    if name.endswith("_function") and callable(getattr(metric, name)):
        metric_single_funcs[name[:-9]] = getattr(metric, name)
all_metric_names = sorted(set(metric_batch_funcs.keys()) | set(metric_single_funcs.keys()))
print("Metrics detected:", all_metric_names)

# 辅助：tensor -> uint8 image for display
def tensor_to_uint8(img_tensor):
    # expect [-1,1] float tensor (B, C, H, W)
    img = img_tensor.clamp(-1,1).add(1).mul(127.5).cpu().numpy().astype(np.uint8)
    return img

# os.makedirs("./play_outputs", exist_ok=True)

# --- 函数：对一个 loader 做推理/度量/显示 ---
def run_and_report(loader, name, max_batches=NUM_SHOW_BATCHES):
    print(f"\n===== Dataset: {name} =====")
    results = []
    batch_idx = 0
    for batch in loader:
        if batch_idx >= max_batches:
            break
        # batch may be tuple (vis, ir, label) or (vis, ir)
        batch = tuple(t.to(device=device, dtype=torch.float32) for t in batch)
        vis = batch[0]
        ir = batch[1]
        label = batch[2] if len(batch) > 2 else None

        # inference
        with torch.no_grad():
            start = time.time()
            fused = pipeline(vis, ir, num_inference_steps=NUM_INFERENCE_STEPS)
            elapsed = time.time() - start

        # metrics: prefer batch implementations
        metric_scores = {}
        B = vis.shape[0]
        # prepare metric args in "pixel 0..255" like pretrain
        vis_m = ((vis.to(dtype=torch.float32) + 1.0) * 127.5).clamp(0,255)
        ir_m = ((ir.to(dtype=torch.float32) + 1.0) * 127.5).clamp(0,255)
        fused_m = ((fused.to(dtype=torch.float32) + 1.0) * 127.5).clamp(0,255)

        for m in all_metric_names:
            if m in metric_batch_funcs:
                try:
                    func = metric_batch_funcs[m]
                    # try 3-arg, 2-arg, 1-arg
                    import inspect
                    sig = inspect.signature(func)
                    params = len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
                    if params >= 3:
                        vals = func(vis_m, ir_m, fused_m)
                    elif params == 2:
                        vals = func(vis_m, fused_m)
                    else:
                        vals = func(fused_m)
                    # normalize to numpy per-sample
                    if isinstance(vals, torch.Tensor):
                        arr = vals.detach().cpu().numpy()
                    else:
                        arr = np.asarray(vals)
                except Exception as e:
                    print(f"Metric {m} batch impl failed: {e}")
                    arr = np.full((B,), np.nan)
            elif m in metric_single_funcs and label is not None:
                # fallback to per-sample CPU functions (slower)
                arr = []
                func = metric_single_funcs[m]
                for i in range(B):
                    try:
                        # per-sample expects HWC maybe; pass flattened tensors similar to batch form
                        v = func(((vis[i:i+1]+1.0)*127.5).clamp(0,255), ((ir[i:i+1]+1.0)*127.5).clamp(0,255), ((fused[i:i+1]+1.0)*127.5).clamp(0,255))
                        arr.append(v)
                    except Exception:
                        arr.append(np.nan)
                arr = np.asarray(arr)
            else:
                arr = np.full((B,), np.nan)
            metric_scores[m] = arr

        # 训练 proxy loss (仅当 label 存在且 data 来自训练集)
        losses = None
        if label is not None:
            with torch.no_grad():
                # encode target latent
                enc = vae.encode(label)
                lat_target = getattr(enc, "latent_dist", None)
                if lat_target is not None:
                    try:
                        lat_target = enc.latent_dist.sample()
                    except Exception:
                        lat_target = enc.latent_dist.mean
                elif isinstance(enc, dict):
                    lat_target = enc.get("latent_dist", enc.get("sample", torch.as_tensor(enc))).sample \
                                 if "latent_dist" in enc or "sample" in enc else torch.as_tensor(enc)
                else:
                    lat_target = enc
                lat_target = lat_target.to(device)
                lat_target = lat_target - vae.encode(vis.to(device)).latent_dist.sample() # residual
                lat_target = lat_target * scaling_factor

                # random timesteps & noise (proxy estimate of training loss)
                B = lat_target.shape[0]
                ts = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
                noise = torch.randn_like(lat_target)
                try:
                    noisy = scheduler.add_noise(lat_target, noise, ts)
                except Exception:
                    noisy = lat_target + noise
                # predicted noise by saved models
                enc_cond = encoder(torch.cat([vis, ir], dim=1))
                pred_noise = unet(noisy, ts, encoder_hidden_states=enc_cond).sample
                # per-sample L1
                per_sample = F.l1_loss(pred_noise, noise, reduction="none")
                # mean over non-batch dims
                mean_dims = tuple(range(1, per_sample.dim()))
                losses = per_sample.mean(dim=mean_dims).detach().cpu().numpy()
        else:
            losses = np.full((vis.shape[0],), np.nan)

        # 打印 summary
        print(f"Batch {batch_idx} - elapsed {elapsed:.3f}s - metric samples count {B}")
        # 打印 metric averages for the batch
        for m, arr in metric_scores.items():
            mean_val = float(np.nanmean(arr)) if arr.size else float("nan")
            print(f"  {m}: {mean_val:.4f}")

        # 打印 proxy losses
        for i, l in enumerate(losses):
            print(f"  sample[{i}] proxy L1 loss: {l:.6f}")

        # 显示图片：将 fused / label / vis (只展示第一样本)
        vis_uint8 = tensor_to_uint8(vis)
        fused_uint8 = tensor_to_uint8(fused)
        if label is not None:
            label_uint8 = tensor_to_uint8(label)
        B = vis_uint8.shape[0]
        # ensure save dir
        os.makedirs(f"./{SAVE_DIR}", exist_ok=True)
        for i in range(B):
            fig, axs = plt.subplots(1, 3 if label is not None else 2, figsize=(12,4))
            axs[0].imshow(vis_uint8[i].transpose(1,2,0)); axs[0].set_title("VIS"); axs[0].axis("off")
            axs[1].imshow(fused_uint8[i].transpose(1,2,0)); axs[1].set_title("Fused"); axs[1].axis("off")
            if label is not None:
                axs[2].imshow(label_uint8[i].transpose(1,2,0)); axs[2].set_title("Label"); axs[2].axis("off")
            plt.suptitle(f"{name} batch{batch_idx} sample{i}")
            # plt.show()
            plt.savefig(f"./{SAVE_DIR}/{name}_b{batch_idx}_s{i}_compare.png")
            out_path = f"./{SAVE_DIR}/{name}_b{batch_idx}_s{i}.png"
            plt.imsave(out_path, fused_uint8[i].transpose(1,2,0))
        batch_idx += 1
        results.append({"metrics": metric_scores, "losses": losses})
    return results

# --- 运行 train & test ---
train_results = run_and_report(train_loader, "train", max_batches=NUM_SHOW_BATCHES)
test_results = {}
for tname, loader in test_loaders.items():
    test_results[tname] = run_and_report(loader, tname, max_batches=NUM_SHOW_BATCHES)

print(f"\nAll done. Generated images saved to ./{SAVE_DIR}")