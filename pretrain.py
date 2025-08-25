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
from diffusers.optimization import get_cosine_schedule_with_warmup

from model.pipeline import ConditioningEncoder, ddim_step_with_logprob
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

    # 读取 pipeline 配置（不要在此处使用 accelerator）
    pipeline_cfg = config.get("pipeline", {})
    use_shortcut = bool(pipeline_cfg.get("use_shortcut", False))
    use_ddim_logprob = bool(pipeline_cfg.get("use_ddim_logprob", False))

    project_dir = os.path.join(config.get('output_dir', "./checkpoints/pretrain/"), config['run_name'])
    os.makedirs(project_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=project_dir,
    )

    # 现在 accelerator 已创建，可以安全打印
    if accelerator.is_main_process:
        print(f"Pipeline config: use_shortcut={use_shortcut}, use_ddim_logprob={use_ddim_logprob}")

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

    # 使用 diffusers 提供的 cosine schedule with warmup
    # 需要知道总训练步数：len(train_loader) * num_epochs（train_loader 已创建下面）
    # 因此先创建 train_loader（见下），然后再构建 lr_scheduler。
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

    # 计算总训练步数并构建 lr_scheduler（支持配置 lr_warmup_steps 或 warmup.warmup_steps）
    num_epochs = int(config['training']['num_epochs'])
    steps_per_epoch = max(1, len(train_loader))
    total_training_steps = num_epochs * steps_per_epoch

    # 优先读取 top-level lr_warmup_steps，兼容旧 warmup 配置
    warmup_cfg = config['training'].get('warmup', {}) if 'training' in config else {}
    lr_warmup_steps = config['training'].get('lr_warmup_steps', None) or warmup_cfg.get('warmup_steps', None) or 0
    lr_warmup_steps = int(lr_warmup_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=total_training_steps,
    )
    if accelerator.is_main_process:
        print(f"[LR] total_steps={total_training_steps}, warmup_steps={lr_warmup_steps}")

    # --- Noise / Diffusion scheduler（必须初始化，供训练中的 add_noise 与测试采样使用） ---
    diffusion_cfg = config.get("diffusion", {}) or {}
    diffusion_num_train_timesteps = int(diffusion_cfg.get("num_train_timesteps", diffusion_cfg.get("num_inference_steps", 1000)))
    diffusion_beta_schedule = diffusion_cfg.get("beta_schedule", "squaredcos_cap_v2")
    diffusion_scheduler = DDIMScheduler(num_train_timesteps=diffusion_num_train_timesteps, beta_schedule=diffusion_beta_schedule)
    if accelerator.is_main_process:
        print(f"[Diffusion] DDIMScheduler initialized: num_train_timesteps={diffusion_num_train_timesteps}, beta_schedule={diffusion_beta_schedule}")

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

    # --- LR Warmup wrapper（可选） ---
    # 配置入口: config['training']['warmup'] -> {'warmup_steps': int} 或 {'warmup_ratio': float}
    warmup_cfg = config['training'].get('warmup', {}) if 'training' in config else {}
    warmup_steps = warmup_cfg.get('warmup_steps', None)
    warmup_ratio = warmup_cfg.get('warmup_ratio', None)

    # 如果用户只提供 warmup_ratio，则估算 warmup_steps = warmup_ratio * total_training_steps
    if warmup_steps is None and warmup_ratio:
        # 估计总 step = epochs * steps_per_epoch
        steps_per_epoch = max(1, len(train_loader))
        total_steps = int(config['training']['num_epochs'] * steps_per_epoch)
        warmup_steps = int(max(0, warmup_ratio) * total_steps)

    # 如果启用了预热且 warmup_steps>0，则用 SequentialLR 把 LinearLR (warmup) 和原 lr_scheduler 串联
    if warmup_steps and warmup_steps > 0:
        try:
            # Linear warmup from 1e-6 * lr -> 1.0 * lr over warmup_steps
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
            # SequentialLR 将在 warmup 完成后切换到原有 lr_scheduler
            if hasattr(torch.optim.lr_scheduler, 'SequentialLR'):
                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
            else:
                # 旧版本回退：使用 LambdaLR 做线性 warmup，然后继续由 lr_scheduler 控制（注意：此路径可能需要手动 step 两个 scheduler）
                def _warmup_lambda(step):
                    if step >= warmup_steps:
                        return 1.0
                    return float(step) / float(max(1, warmup_steps))
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
            if accelerator.is_main_process:
                print(f"[LR] Enabled warmup: warmup_steps={warmup_steps}")
        except Exception as e:
            # 若构造失败则回退到原 lr_scheduler
            if accelerator.is_main_process:
                print(f"[LR] Warning: failed to enable warmup scheduler ({e}), continuing without warmup.")
    else:
        if accelerator.is_main_process:
            print("[LR] Warmup disabled (no warmup_steps/warmup_ratio configured)")

    # --- Accelerate Preparation ---
    # 将 model, optimizer, dataloader, lr_scheduler 一起交给 accelerator.prepare（顺序需一致）
    components_to_prepare = [model_wrapper, optimizer, train_loader, lr_scheduler] + test_loaders_list
    prepared = accelerator.prepare(*components_to_prepare)

    # 解包
    idx = 0
    model_wrapper, optimizer, train_loader, lr_scheduler = prepared[idx:idx+4]; idx += 4
    prepared_test_loaders_list = prepared[idx:]
    prepared_test_loaders = dict(zip(test_set_names, prepared_test_loaders_list))

    # 将 VAE 移到 accelerator.device 并与模型 dtype 对齐（vae 未包含在 wrapper 中）
    device = accelerator.device
    model_dtype = next(model_wrapper.parameters()).dtype
    vae = vae.to(device, dtype=model_dtype)

    # 选择损失函数（保持之前逻辑）
    loss_name = config.get('training', {}).get('loss_function', 'l2')
    loss_name = (loss_name or 'l2').lower()
    if loss_name in ('l1', 'mae'):
        loss_fn = F.l1_loss
    elif loss_name in ('mse', 'l2'):
        # 均方差（L2）
        loss_fn = F.mse_loss
    else:
        # 未知配置则回退为 MSE，并打印警告（仅主进程可见）
        if accelerator.is_main_process:
            print(f"[Warning] Unknown loss_function '{loss_name}' in config, defaulting to 'mse'")
        loss_fn = F.mse_loss

    num_train_epochs = config['training']['num_epochs']
    
    # 全局 step 计数，用于 step 级别日志
    global_step = 0

    # --- Training Loop: 在每次 optimizer.step() 后调用 lr_scheduler.step()（step-level 调度） ---
    global_step = 0
    for epoch in range(num_train_epochs):
        model_wrapper.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}/{num_train_epochs}")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device, dtype=model_dtype) for t in batch)
            with accelerator.accumulate(model_wrapper):
                if len(batch) != 3:
                    raise RuntimeError(f"Training requires labels (dir_C). Batch provided has {len(batch)} items.")
                
                vis_images, ir_images, label_images = batch

                vae.eval()
                with torch.no_grad():
                    if use_shortcut:
                        # 教模型预测残差：label_lat - vis_lat（与 pipeline 加回 vis_lat 的行为一致）
                        vis_lat = vae.encode(vis_images).latent_dist.sample()
                        label_lat = vae.encode(label_images).latent_dist.sample()
                        lat_target = (label_lat - vis_lat) * scaling_factor
                    else:
                        # 教模型直接预测 label latent（不依赖 shortcut），与推理时不加回 vis_lat 一致
                        label_lat = vae.encode(label_images).latent_dist.sample()
                        lat_target = label_lat * scaling_factor

                batch_size = lat_target.shape[0]
                timesteps = torch.randint(0, diffusion_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                noise = torch.randn_like(lat_target)
                noisy_latent = diffusion_scheduler.add_noise(lat_target, noise, timesteps)
                
                condition_img = torch.cat([vis_images, ir_images], dim=1)
                
                # 计算 predicted_noise / loss
                predicted_noise = model_wrapper(noisy_latent, timesteps, condition_img)
                loss = loss_fn(predicted_noise, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), config['training'].get('max_grad_norm', 1.0))
                optimizer.step()
                # 在 optimizer.step() 之后立即更新 lr（step-level 调度）
                try:
                    lr_scheduler.step()
                except Exception:
                    # 某些 scheduler 需要 step 时传入 epoch/step，兜底忽略
                    pass
                optimizer.zero_grad()

                # logging
                epoch_loss_sum += loss.item()
                epoch_loss_count += 1
                global_step += 1
                try:
                    current_lr = lr_scheduler.get_last_lr()[0]
                except Exception:
                    current_lr = float(next(iter(optimizer.param_groups))['lr'])
                accelerator.log({"train/step/loss": float(loss.item()), "train/step/lr": float(current_lr)}, step=global_step)

            if accelerator.sync_gradients:
                pbar.set_postfix({"loss": loss.item(), "lr": current_lr})

        # epoch 结束：记录 epoch 级别指标（lr 用 lr_scheduler.get_last_lr）
        epoch_avg_loss = (epoch_loss_sum / epoch_loss_count) if epoch_loss_count > 0 else float('nan')
        try:
            epoch_lr = lr_scheduler.get_last_lr()[0]
        except Exception:
            epoch_lr = float(next(iter(optimizer.param_groups))['lr'])
        accelerator.log({"train/epoch/loss": float(epoch_avg_loss), "train/epoch/lr": float(epoch_lr)}, step=epoch+1)

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

                        # 初始 latent（与 pipeline 保持一致）
                        latents = torch.randn(latents_shape, device=device, dtype=unet_eval.dtype)
                        diffusion_scheduler.set_timesteps(config['diffusion'].get('num_inference_steps', 20))
                        for t in diffusion_scheduler.timesteps:
                            # 保存原始 x_t 用于可选 logprob 计算
                            x_t = latents
                            noise_pred = unet_eval(x_t, t, encoder_hidden_states=condition_embeds_test).sample
                            step_out = diffusion_scheduler.step(noise_pred, t, x_t)
                            prev_sample = step_out.prev_sample
                            # 如果配置要求，尝试计算 ddin logprob（函数返回 prev_sample, log_prob）
                            if use_ddim_logprob:
                                try:
                                    _, _ = ddim_step_with_logprob(diffusion_scheduler, noise_pred, t, x_t, prev_sample=prev_sample, eta=1.0)
                                except Exception:
                                    # 忽略 logprob 失败，不影响原本 sampling
                                    pass
                            latents = prev_sample

                        # 反缩放
                        if hasattr(vae.config, 'scaling_factor'):
                            latents = latents / vae.config.scaling_factor

                        # 根据 config 决定是否加回 vis latent（shortcut）
                        if use_shortcut:
                            latents = latents + vae.encode(vis_test.to(device=device, dtype=model_dtype)).latent_dist.sample()

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
                # 使用 accelerator.log 上报所有 test 指标（以 epoch 编号为 step）
                # 组织为键值对： test/{set_name}/{metric_name} -> value
                metrics_to_log = {}
                for set_name_k, metrics_dict in results.items():
                    for metric_k, val in metrics_dict.items():
                        metrics_to_log[f"test/{set_name_k}/{metric_k}"] = float(val) if not np.isnan(val) else float('nan')
                accelerator.log(metrics_to_log, step=epoch+1)
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