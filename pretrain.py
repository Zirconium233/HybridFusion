import os
import yaml
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL

from model.pipeline import ConditioningEncoder
from dataset import ImageFusionDataset
import metric

class ModelWrapper(nn.Module):
    """
    A wrapper to hold the UNet and ConditioningEncoder together.
    This is necessary for compatibility with DeepSpeed, which expects a single model.
    """
    def __init__(self, unet, encoder):
        super().__init__()
        self.unet = unet
        self.encoder = encoder

    def forward(self, noisy_latent, timesteps, condition_img):
        condition_embeds = self.encoder(condition_img)
        predicted_noise = self.unet(noisy_latent, timesteps, encoder_hidden_states=condition_embeds).sample
        return predicted_noise

def find_vae_checkpoint(ckpt_dir="./checkpoints/vae/best.pth"):
    """Returns the checkpoint path if it exists."""
    return ckpt_dir if os.path.exists(ckpt_dir) else None

def main(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    args, _ = parser.parse_known_args([])

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    project_dir = os.path.join(config.get('output_dir', "./checkpoints/pretrain/"), config['run_name'])
    os.makedirs(project_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=project_dir,
    )

    if accelerator.is_main_process:
        config_copy_path = os.path.join(project_dir, "config.yml")
        shutil.copy2(args.config, config_copy_path)
        print(f"Configuration saved to: {config_copy_path}")
        print("Configuration:")
        print(yaml.dump(config, indent=2))

    # --- VAE Initialization ---
    vae_cfg = config['model_config'].get('vae')
    if vae_cfg is None:
        raise RuntimeError("VAE configuration missing. Pretraining requires a VAE.")

    vae_init_kwargs = {k: v for k, v in vae_cfg.items() if k != 'checkpoint_dir'}
    vae = AutoencoderKL(**vae_init_kwargs)
    
    if 'scaling_factor' not in vae.config or vae.config.scaling_factor is None:
        raise ValueError("VAE config in yaml MUST contain 'scaling_factor' with the calculated value.")
    
    scaling_factor = vae.config.scaling_factor
    if accelerator.is_main_process:
        print(f"Successfully loaded VAE scaling_factor: {scaling_factor}")

    vae_scale_factor = 4  # Hardcoded downsampling factor
    if accelerator.is_main_process:
        print(f"Using HARDCODED VAE downsampling scale factor: {vae_scale_factor}")

    vae_ckpt = find_vae_checkpoint(ckpt_dir=vae_cfg.get('checkpoint_dir', "./checkpoints/vae/best.pth"))
    if vae_ckpt is None:
        raise RuntimeError(f"No VAE checkpoint found at the specified path: {vae_cfg.get('checkpoint_dir')}.")
    
    if accelerator.is_main_process:
        print(f"Loading VAE checkpoint from {vae_ckpt}")
    
    sd = torch.load(vae_ckpt, map_location="cpu")
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    
    vae.load_state_dict(sd, strict=True)
    print("VAE checkpoint loaded successfully in strict mode.")

    # --- Model and Optimizer Initialization ---
    unet = UNet2DConditionModel(**config['model_config']['unet'])
    encoder = ConditioningEncoder(**config['model_config']['encoder'])
    model_wrapper = ModelWrapper(unet, encoder)

    optimizer_config = config['training']['optimizer']
    OptimCls = getattr(torch.optim, optimizer_config['type'])
    optimizer = OptimCls(model_wrapper.parameters(),
                         lr=config['training']['learning_rate'], **optimizer_config.get('args', {}))

    scheduler_lr_config = config['training']['scheduler']
    LRSchedulerCls = getattr(torch.optim.lr_scheduler, scheduler_lr_config['type'])
    lr_scheduler = LRSchedulerCls(optimizer, **scheduler_lr_config.get('args', {}))
    
    diffusion_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )

    # --- DataLoaders ---
    train_ds_config = config['train_dataset']
    train_paths = config['datasets'][train_ds_config['name']]['train']
    train_dataset = ImageFusionDataset(
        dir_A=train_paths['dir_A'], dir_B=train_paths['dir_B'], dir_C=train_paths.get('dir_C'),
        is_train=True, is_getpatch=False, augment=False
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['train_batch_size'],
                              shuffle=True,
                              num_workers=config['training'].get('num_workers', 4),
                              pin_memory=True)

    test_loaders_list = []
    test_set_names = []
    for test_set_config in config.get('test_sets', []):
        set_name = test_set_config['name']
        test_paths = config['datasets'][set_name]['test']
        test_dataset = ImageFusionDataset(
            dir_A=test_paths['dir_A'], dir_B=test_paths['dir_B'], dir_C=test_paths.get('dir_C'),
            is_train=False, is_getpatch=False
        )
        test_batch_size = test_set_config.get('test_batch_size', 4)
        test_loaders_list.append(DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False))
        test_set_names.append(set_name)

    # --- Accelerate Preparation ---
    # 修复：移除 vae，只准备需要训练的模型和数据加载器
    components_to_prepare = [model_wrapper, optimizer, lr_scheduler, train_loader] + test_loaders_list
    prepared_components = accelerator.prepare(*components_to_prepare)
    
    # 修复：重新解包，注意索引的变化
    idx = 0
    model_wrapper, optimizer, lr_scheduler, train_loader = prepared_components[idx:idx+4]; idx += 4
    prepared_test_loaders_list = prepared_components[idx:]
    prepared_test_loaders = dict(zip(test_set_names, prepared_test_loaders_list))

    # 修复：手动将 VAE 移至设备，因为它没有被 accelerator.prepare 管理
    # 给vae换成accelerator的dtype
    device = accelerator.device
    model_dtype = next(model_wrapper.parameters()).dtype
    vae = vae.to(device, dtype=model_dtype)

    loss_fn = F.l1_loss
    num_train_epochs = config['training']['num_epochs']
    
    # --- Training Loop ---
    for epoch in range(num_train_epochs):
        model_wrapper.train()
        
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}/{num_train_epochs}")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device, dtype=model_dtype) for t in batch)
            with accelerator.accumulate(model_wrapper):
                if len(batch) != 3:
                    raise RuntimeError(f"Training requires labels (dir_C). Batch provided has {len(batch)} items.")
                
                vis_images, ir_images, label_images = batch

                vae.eval()
                with torch.no_grad():
                    lat_target = vae.encode(label_images).latent_dist.sample()
                    lat_target *= scaling_factor

                batch_size = lat_target.shape[0]
                timesteps = torch.randint(0, diffusion_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                noise = torch.randn_like(lat_target)
                noisy_latent = diffusion_scheduler.add_noise(lat_target, noise, timesteps)
                
                condition_img = torch.cat([vis_images, ir_images], dim=1)
                
                predicted_noise = model_wrapper(noisy_latent, timesteps, condition_img)
                
                loss = loss_fn(predicted_noise, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                 pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

        lr_scheduler.step()

        # --- Testing Loop ---
        if (epoch + 1) % config['training']['test_freq'] == 0 and accelerator.is_main_process:
            print(f"\n--- Running Test at Epoch {epoch+1} ---")
            model_wrapper.eval()
            vae.eval()

            unwrapped_model = accelerator.unwrap_model(model_wrapper)
            unet_eval, encoder_eval = unwrapped_model.unet, unwrapped_model.encoder

            results = {}
            import inspect
            # 自动收集 metric 模块中以 "_function" 结尾的函数（CPU 版，保留仅用于列名）
            metric_funcs = {}
            for name in dir(metric):
                if name.endswith("_function"):
                    func = getattr(metric, name)
                    if callable(func):
                        short_name = name[:-9]
                        metric_funcs[short_name] = func

            # 仅收集 GPU batch 版本（_function_batch）
            metric_batch_funcs = {}
            for name in dir(metric):
                if name.endswith("_function_batch"):
                    func = getattr(metric, name)
                    if callable(func):
                        short_name = name[:-15]
                        metric_batch_funcs[short_name] = func

            if len(metric_batch_funcs) == 0:
                print("[Warning] No GPU batch metric implementations found. All metrics will be NaN.")

            for set_name, test_loader in prepared_test_loaders.items():
                # 只针对在 module 中存在的指标名做收集（包含没有 batch 实现的也会被标为 NaN）
                all_metric_names = sorted(set(metric_funcs.keys()) | set(metric_batch_funcs.keys()))
                metric_scores = {m: [] for m in all_metric_names}

                missing_batch_metrics = sorted(list(set(metric_funcs.keys()) - set(metric_batch_funcs.keys())))
                if missing_batch_metrics:
                    print(f"[Info] The following metrics have no GPU batch impl and will be NaN: {missing_batch_metrics}")

                batch_idx = 0
                for batch in tqdm(test_loader, desc=f"Testing {set_name}", leave=False):
                    batch = tuple(t.to(device, dtype=model_dtype) for t in batch)
                    vis_test, ir_test = batch[0], batch[1]

                    with torch.no_grad():
                        condition_img_test = torch.cat([vis_test, ir_test], dim=1)
                        condition_embeds_test = encoder_eval(condition_img_test)

                        B, _, H, W = vis_test.shape
                        lat_h, lat_w = H // vae_scale_factor, W // vae_scale_factor
                        lat_channels = vae.config.latent_channels
                        latents_shape = (B, lat_channels, lat_h, lat_w)

                        latents = torch.randn(latents_shape, device=device, dtype=unet_eval.dtype)
                        diffusion_scheduler.set_timesteps(config['diffusion'].get('num_inference_steps', 10))
                        for t in diffusion_scheduler.timesteps:
                            noise_pred = unet_eval(latents, t, encoder_hidden_states=condition_embeds_test).sample
                            latents = diffusion_scheduler.step(noise_pred, t, latents).prev_sample
                        latents /= scaling_factor
                        fused = vae.decode(latents).sample

                    # 检测模型输出 NaN（模型问题）
                    if torch.isnan(fused).any():
                        print(f"[Error] NaN detected in model fused outputs at epoch {epoch+1}, set {set_name}, batch {batch_idx}.")
                        # 打印统计用于定位
                        try:
                            print(f" fused stats: min={torch.nanmin(fused).item()}, max={torch.nanmax(fused).item()}, mean={torch.nanmean(fused).item()}")
                        except Exception:
                            print(" unable to compute fused stats due to NaNs.")
                        # 替换 NaN 以便继续评估（nan_to_num 行为可调整）
                        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)

                    # ---- GPU batch metrics 路径（移除 CPU 回退，所有指标仅用 GPU batch 实现） ----
                    # 确保为 float32（可能原为 bf16）
                    vis_f = vis_test.to(dtype=torch.float32)
                    ir_f = ir_test.to(dtype=torch.float32)
                    fused_f = fused.to(dtype=torch.float32)

                    # 把 -1..1 转到 0..255，截断到 [0,255]
                    vis_metric = ((vis_f + 1.0) * 127.5).clamp(0.0, 255.0)
                    ir_metric = ((ir_f + 1.0) * 127.5).clamp(0.0, 255.0)
                    fused_metric = ((fused_f + 1.0) * 127.5).clamp(0.0, 255.0)

                    # 对每个 gpu-batch 指标进行计算
                    for metric_name in all_metric_names:
                        if metric_name not in metric_batch_funcs:
                            # 没有 GPU batch 实现，直接填充 NaN
                            for _ in range(B):
                                metric_scores[metric_name].append(np.nan)
                            continue

                        batch_func = metric_batch_funcs[metric_name]

                        # 根据函数签名决定传入参数数量：
                        # - 单参数 (params==1)：传 fused_metric（与 CPU 单输入行为一致）
                        # - 三参数及以上 (params>=3)：传 (vis_metric, ir_metric, fused_metric)
                        # - 两参数 (params==2)：尝试 (vis_metric, fused_metric)
                        # 如果签名无法解析则尝试三参调用，回退到一参调用
                        try:
                            sig = inspect.signature(batch_func)
                            params = len([p for p in sig.parameters.values()
                                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
                        except Exception:
                            params = None

                        vals = None
                        # 优先依据 params 分支安全调用
                        try:
                            if params == 1:
                                vals_t = batch_func(fused_metric)
                            elif params == 2:
                                vals_t = batch_func(vis_metric, fused_metric)
                            elif params is None or params >= 3:
                                # 尝试三参调用
                                vals_t = batch_func(vis_metric, ir_metric, fused_metric)
                            else:
                                # 兜底：尝试三参再一参
                                try:
                                    vals_t = batch_func(vis_metric, ir_metric, fused_metric)
                                except TypeError:
                                    vals_t = batch_func(fused_metric)
                        except Exception as e:
                            print(f"[Error] GPU batch metric '{metric_name}' raised exception at epoch {epoch+1}, set {set_name}, batch {batch_idx}: {e}")
                            vals = np.full((B,), np.nan)

                        # 如果成功调用并得到返回值，统一转换为 numpy array
                        if vals is None:
                            try:
                                if isinstance(vals_t, torch.Tensor):
                                    vals = vals_t.detach().cpu().numpy()
                                else:
                                    vals = np.asarray(vals_t)
                            except Exception as e:
                                print(f"[Error] Failed to convert metric '{metric_name}' output to numpy at epoch {epoch+1}, set {set_name}, batch {batch_idx}: {e}")
                                vals = np.full((B,), np.nan)

                        # 若返回包含 NaN，标记并打印指标名（指标实现问题）
                        if np.isnan(vals).any():
                            nan_idx = np.where(np.isnan(vals))[0].tolist()
                            print(f"[Warning] GPU batch metric '{metric_name}' produced NaN for indices {nan_idx} at epoch {epoch+1}, set {set_name}, batch {batch_idx}.")

                        # 追加到 metric_scores
                        for i in range(B):
                            v = vals[i] if i < len(vals) else np.nan
                            metric_scores[metric_name].append(float(v) if not np.isnan(v) else np.nan)

                    batch_idx += 1

                # 汇总每个 metric 的平均值（保持 NaN 行为）
                results[set_name] = {m: (np.nan if len(s) == 0 else float(np.nanmean(s))) for m, s in metric_scores.items()}

            if results:
                df = pd.DataFrame(results).T
                print(df.to_string(float_format="%.4f"))
            print("--- Test Finished ---\n")

        # --- Saving Models ---
        if (epoch + 1) % config['training']['save_freq'] == 0:
            # 首先确保所有 rank 都到达保存点，避免分布式 save 的死锁
            accelerator.wait_for_everyone()

            # 只在主进程保存解包后的模型权重，避免 DeepSpeed/ZeRO 的分片导致 save_state 卡死
            if accelerator.is_main_process:
                save_dir = os.path.join(project_dir, f"epoch_{epoch+1}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model_wrapper)
                torch.save(unwrapped.unet.state_dict(), os.path.join(save_dir, "unet.pth"))
                torch.save(unwrapped.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
                print(f"Model weights saved to {save_dir}")
            # 再次同步，确保主进程保存完成后其他进程继续
            accelerator.wait_for_everyone()

    # --- Final Save ---
    if accelerator.is_main_process:
        final_dir = os.path.join(project_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model_wrapper)
        torch.save(unwrapped_model.unet.state_dict(), os.path.join(final_dir, "unet.pth"))
        torch.save(unwrapped_model.encoder.state_dict(), os.path.join(final_dir, "encoder.pth"))
        print(f"Final unwrapped models saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    main(args.config)