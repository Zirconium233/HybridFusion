import os
import yaml
import argparse
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL

from model.pipeline import ImageFusionPipeline, ConditioningEncoder, ddim_step_with_logprob
# optional value network
try:
    from value_network import ValueNetwork
    VALUE_NET_IMPORT_OK = True
except Exception:
    ValueNetwork = None
    VALUE_NET_IMPORT_OK = False

from dataset import ImageFusionDataset

# ------------------ 新增：ModelWrapper，避免在使用 DeepSpeed/Accelerate 时将多个独立模型传入 prepare ------------------
import torch.nn as nn
class ModelWrapper(nn.Module):
    """
    Wrap UNet 和 ConditioningEncoder 为单个 module，避免向 accelerator.prepare 传入多个模型导致 DeepSpeed 报错。
    行为与 pretrain.py 中的 Wrapper 保持一致：forward 返回 unet 的 predicted noise。
    """
    def __init__(self, unet: nn.Module, encoder: nn.Module):
        super().__init__()
        self.unet = unet
        self.encoder = encoder

    def forward(self, noisy_latent, timesteps, condition_img):
        condition_embeds = self.encoder(condition_img)
        out = self.unet(noisy_latent, timesteps, encoder_hidden_states=condition_embeds)
        # diffusers UNet 返回的是 ModelOutput-like or tuple; 保证返回 .sample 以兼容已有代码
        pred = out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, (list, tuple)) else out)
        return pred


def find_vae_checkpoint(ckpt_dir="./checkpoints/vae/best"):
    # 兼容传入文件路径或目录路径
    ckpt_dir = os.path.expanduser(ckpt_dir)
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    if os.path.isdir(ckpt_dir):
        cand = [
            os.path.join(ckpt_dir, "best.pth"),
            os.path.join(ckpt_dir, "vae.pth"),
            os.path.join(ckpt_dir, "best.ckpt"),
            os.path.join(ckpt_dir, "checkpoint.pth"),
            os.path.join(ckpt_dir, "model.pth"),
            os.path.join(ckpt_dir, "latest.pth"),
        ]
        for p in cand:
            if os.path.exists(p):
                return p
    # 兜底：直接返回给定路径（如果存在），否则 None
    return ckpt_dir if os.path.exists(ckpt_dir) else None


def main(config_path: str, test_run: bool = False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    project_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(project_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=project_dir,
    )

    if accelerator.is_main_process:
        config_copy = os.path.join(project_dir, "config.yml")
        shutil.copy2(config_path, config_copy)
        print(f"Saved config -> {config_copy}")

    # --- build models ---
    unet_cfg = config['model_config']['unet']
    enc_cfg = config['model_config']['encoder']
    vae_cfg = config['model_config'].get('vae', None)

    unet = UNet2DConditionModel(**unet_cfg)
    encoder = ConditioningEncoder(**enc_cfg)

    # 用 wrapper 将 unet 与 encoder 包装为单个模块，避免 accelerate.prepare 时传入多个模型
    model_wrapper = ModelWrapper(unet, encoder)

    # VAE: optional, try to load checkpoint if provided
    vae = None
    if vae_cfg is not None:
        vae_init_kwargs = {k: v for k, v in vae_cfg.items() if k in ["sample_size", "in_channels", "out_channels", "down_block_types", "up_block_types", "block_out_channels", "latent_channels", "scaling_factor"]}
        vae = AutoencoderKL(**vae_init_kwargs)
        vae_ckpt = find_vae_checkpoint(vae_cfg.get('checkpoint_dir', "./checkpoints/vae/best"))
        if vae_ckpt:
            if accelerator.is_main_process:
                print(f"Loading VAE checkpoint from {vae_ckpt}")
            sd = torch.load(vae_ckpt, map_location="cpu")
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            try:
                vae.load_state_dict(sd)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Partial VAE load: {e}")
                own = vae.state_dict()
                filtered = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
                own.update(filtered)
                vae.load_state_dict(own)
        else:
            if accelerator.is_main_process:
                print("Warning: no VAE checkpoint found; proceeding with randomly init VAE (not recommended).")

    # scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    # pipeline: pass vae_scale_factor if provided in config (spatial downsample factor), otherwise let pipeline infer
    vae_scale_factor = vae_cfg.get('vae_scale_factor') if vae_cfg is not None else None
    # pipeline = ImageFusionPipeline(unet=unet, scheduler=scheduler, encoder=encoder, vae=vae, vae_scale_factor=vae_scale_factor)
    # pipeline 后面 prepare 完成后定义

    # optimizers
    policy_optimizer = torch.optim.AdamW(list(unet.parameters()) + list(encoder.parameters()), lr=config['rl_training']['policy_learning_rate'])

    # optional value net
    use_value_net = bool(config.get('rl_training', {}).get('use_value_net', False)) and VALUE_NET_IMPORT_OK
    if bool(config.get('rl_training', {}).get('use_value_net', False)) and not VALUE_NET_IMPORT_OK and accelerator.is_main_process:
        print("ValueNetwork import failed; falling back to batch-based advantages.")
    value_net = None
    value_optimizer = None
    if use_value_net:
        latent_dim = config['rl_training'].get('value_latent_dim', 4*32*32)
        condition_dim = unet_cfg['cross_attention_dim']
        value_net = ValueNetwork(latent_dim=latent_dim, condition_dim=condition_dim)
        value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=config['rl_training']['value_learning_rate'])

    # dataloader or test dummy
    if test_run:
        bs = 4
        # 为 accelerate.prepare / DeepSpeed 提供一个合法的 DataLoader（必须有 batch_size 属性）
        # 使用单个小批次的 TensorDataset 作为最小封装，避免 prepare 报错
        vis = torch.randn(bs, 3, 480, 640)
        ir = torch.randn(bs, 1, 480, 640)
        dataset = torch.utils.data.TensorDataset(vis, ir)
        train_loader = DataLoader(dataset, batch_size=bs)
    else:
        train_ds_config = config['data']
        train_dataset = ImageFusionDataset(
            dir_A=train_ds_config['train']['dir_A'],
            dir_B=train_ds_config['train']['dir_B'],
            dir_C=train_ds_config['train'].get('dir_C', None),
            is_train=True,    # 与 pretrain 保持一致：使用全分辨率训练 / 不裁patch
            is_getpatch=False, 
            augment=train_ds_config['augment']
        )
        train_loader = DataLoader(train_dataset, batch_size=train_ds_config['train_batch_size'], shuffle=True, num_workers=train_ds_config.get('num_workers', 4))

    # Prepare with accelerator: DO NOT include VAE (keep consistent with pretrain)
    # 传入 model_wrapper 而不是单独的 unet/encoder，避免 DeepSpeed 报错
    prepare_list = [model_wrapper, policy_optimizer, train_loader]
    if use_value_net:
        prepare_list += [value_net, value_optimizer]
    prepared = accelerator.prepare(*prepare_list)

    # unpack
    idx = 0
    model_wrapper = prepared[idx]; idx += 1
    policy_optimizer = prepared[idx]; idx += 1
    if use_value_net:
        value_net = prepared[idx]; idx += 1
        value_optimizer = prepared[idx]; idx += 1
    # train_loader 也会被 prepare（test_run 时是 Dummy DataLoader）
    train_loader = prepared[idx]; idx += 1

    # move VAE to device manually if present
    device = accelerator.device
    model_dtype = next(unet.parameters()).dtype
    if vae is not None:
        vae = vae.to(device=device, dtype=model_dtype)

    # --- CRITICAL FIX: ensure pipeline references the prepared/moved modules ---
    # pipeline was created之前，self.unet/self.encoder 可能还指向旧实例。
    # 这里把 pipeline 中的模块替换为 prepare 后、已在 device 上的实例。
    # try:
    #     # 使用 DiffusionPipeline.register_modules 更安全地注册（覆盖旧引用）
    #     pipeline.register_modules(unet=unet, encoder=encoder, vae=vae, scheduler=scheduler)
    # except Exception:
    #     # 兜底：直接赋值
    #     pipeline.unet = unet
    #     pipeline.encoder = encoder
    #     pipeline.vae = vae
    #     pipeline.scheduler = scheduler
    # 直接在这里定义pipeline
    # 从 wrapper 中取出已 prepare 的子模块用于 pipeline（注意 model_wrapper 可能是 accelerate 包装后的对象）
    # 使用属性访问，以确保 pipeline 使用同一套参数/设备
    unet = model_wrapper.unet
    encoder = model_wrapper.encoder
    pipeline = ImageFusionPipeline(unet=unet, scheduler=scheduler, encoder=encoder, vae=vae, vae_scale_factor=vae_scale_factor)

    # ensure scheduler timesteps set
    num_inf_steps = config['data']['num_inference_steps']
    scheduler.set_timesteps(num_inf_steps)

    # simple reward placeholder
    def reward_fn(images, **kwargs):
        return torch.randn(images.shape[0], device=accelerator.device)

    # training loop (simplified DDPO-like)
    for epoch in range(config['rl_training']['num_epochs']):
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"RL Epoch {epoch+1}")
        for batch in pbar:
            batch = tuple(t.to(device, dtype=model_dtype) for t in batch)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                vis_images, ir_images, _ = batch
            else:
                vis_images, ir_images = batch

            vis_images = vis_images.to(accelerator.device)
            ir_images = ir_images.to(accelerator.device)
            # 原始条件图（供 ModelWrapper 内部调用 encoder）
            condition_img = torch.cat([vis_images, ir_images], dim=1)
            # 如果需要 condition_embeds（value_net 分支仍可使用），可以提前计算
            condition_embeds = None
            # rollout
            with torch.no_grad():
                decoded, latents_history, old_log_probs = pipeline.forward_with_logprob(
                    vis_image=vis_images, ir_image=ir_images, num_inference_steps=num_inf_steps
                )
            # latents_history: list len = steps+1, each (B, C, H, W)
            # old_log_probs: (B, steps)

            rewards = reward_fn(decoded)
            # advantages
            if use_value_net and value_net is not None:
                # train value net quickly
                condition_embeds = encoder(torch.cat([vis_images, ir_images], dim=1))
                # expand rewards per timestep
                target_values = rewards.view(-1,1).expand(-1, num_inf_steps).reshape(-1,1)
                # states: use latents_history[:-1] as states before action
                states = torch.stack(latents_history[:-1], dim=1).view(-1, *latents_history[0].shape[1:])  # (B*steps, C,H,W)
                timesteps_batch = scheduler.timesteps.unsqueeze(0).repeat(vis_images.shape[0],1).view(-1)
                cond_batch = condition_embeds.repeat_interleave(num_inf_steps, dim=0)

                # simple value update steps
                for _ in range(config['rl_training'].get('value_update_steps', 1)):
                    preds = value_net(states, timesteps_batch, cond_batch)
                    vloss = F.mse_loss(preds, target_values)
                    accelerator.backward(vloss)
                    value_optimizer.step(); value_optimizer.zero_grad()

                with torch.no_grad():
                    preds = value_net(states, timesteps_batch, cond_batch).view(vis_images.shape[0], num_inf_steps)
                    advantages = (target_values.view(vis_images.shape[0], num_inf_steps).mean(dim=1) - preds.mean(dim=1))
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages_expanded = advantages.view(-1,1).expand(-1, num_inf_steps).reshape(-1)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                advantages_expanded = advantages.view(-1,1).expand(-1, num_inf_steps).reshape(-1)
                condition_embeds = encoder(torch.cat([vis_images, ir_images], dim=1))

            # 使用 DDPO 风格的时间步采样与按步训练，避免一次性把 (B * steps) 展开到显存中
            # 1) 形成张量 (B, steps+1, C, H, W) 与 old_log_probs (B, steps)
            latents_tensor = torch.stack(latents_history, dim=1)  # (B, steps+1, C, H, W)
            old_log_probs = old_log_probs  # (B, steps)
            B = vis_images.shape[0]
            num_steps = latents_tensor.shape[1] - 1  # 通常 == num_inf_steps

            # --- NEW: 构造 advantages 的 (B, num_steps) 版本，便于按时间步索引 ---
            # advantages 在上面已经计算为形状 (B,)（或 value_net 分支得到的 (B,)）
            # 保证它的长度等于 batch size
            if advantages.numel() != B:
                # 兜底：若 advantages 被展平为 (B*num_steps,) 的形式，恢复到 (B,)
                advantages = advantages.view(B, -1).mean(dim=1)
            adv_matrix = advantages.view(B, 1).expand(B, num_steps)  # (B, num_steps)

            # 2) 选择用于训练的时间步数量（timestep_fraction 避免训练全部时间步）
            tf_frac = float(config['rl_training'].get('timestep_fraction', 0.4))
            num_train_timesteps = max(1, int(num_steps * tf_frac))

            # 3) 为训练构建小批次：按时间步循环，每步只移动当前 (B, C, H, W) 到 GPU
            # 使用 accelerator.accumulate + 自动混合精度（若可用）减少显存与提高稳定性
            autocast = accelerator.autocast
            for _step_iter in range(config['rl_training'].get('policy_update_steps', 1)):
                # 随机打乱时间维度以获得多样性（与 ddpo 一致）
                perm = torch.randperm(num_steps, device=latents_tensor.device)[:num_train_timesteps]
                for j in perm:
                    # 使用已 prepare 的 model_wrapper 进行 accumulate / forward，避免 graph/模块不一致
                    with accelerator.accumulate(model_wrapper):
                        with autocast():
                            # 当前与下一个状态
                            state = latents_tensor[:, j].to(accelerator.device)
                            next_state = latents_tensor[:, j + 1].to(accelerator.device)
                            # timesteps 标量扩展为向量
                            t_batch = scheduler.timesteps[j].to(accelerator.device).repeat(B)
                            # 直接把原始条件图传入 wrapper，由 wrapper 内部调用 encoder
                            noise_pred = model_wrapper(state, t_batch, condition_img)
                            # model_wrapper 已保证返回 predicted noise tensor
                            _, new_log_prob = ddim_step_with_logprob(scheduler, noise_pred, t_batch, state, prev_sample=next_state, eta=1.0)

                        old_step = old_log_probs[:, j].to(new_log_prob.device)
                        # ratio shape: (B,)
                        ratio = torch.exp(new_log_prob - old_step)
                        # 从 adv_matrix 取当前时间步的 advantage
                        adv_step = adv_matrix[:, j].to(ratio.device)

                        unclipped = -adv_step * ratio
                        clipped = -adv_step * torch.clamp(ratio, 1.0 - config['rl_training']['ppo_clip_range'], 1.0 + config['rl_training']['ppo_clip_range'])
                        policy_loss = torch.mean(torch.maximum(unclipped, clipped))

                        total_loss = config['rl_training']['reward_weight'] * policy_loss
                        if config['rl_training'].get('use_kl', False):
                            kl_weight = float(config['rl_training'].get('kl_weight', 0.0))
                            approx_kl = torch.mean(new_log_prob - old_step)
                            total_loss = total_loss + kl_weight * approx_kl

                        accelerator.backward(total_loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config['rl_training'].get('max_grad_norm', 1.0))
                        policy_optimizer.step()
                        policy_optimizer.zero_grad()

            # 显存整理

            if accelerator.is_main_process and test_run:
                print("Test step complete.")

        # optional checkpoint save
        if accelerator.is_main_process and (epoch+1) % config['logging']['save_freq'] == 0:
            save_dir = os.path.join(config['output_dir'], config['run_name'], f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            # unwrap model_wrapper，然后保存其子模块
            unwrapped_wrapper = accelerator.unwrap_model(model_wrapper)
            torch.save(unwrapped_wrapper.unet.state_dict(), os.path.join(save_dir, "unet.pth"))
            torch.save(unwrapped_wrapper.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
            print(f"Saved checkpoint to {save_dir}")

        if test_run:
            if accelerator.is_main_process:
                print("Test run finished.")
            break

        if accelerator.is_main_process:
            final_dir = os.path.join(config['output_dir'], config['run_name'], "final")
            os.makedirs(final_dir, exist_ok=True)
            unwrapped_wrapper = accelerator.unwrap_model(model_wrapper)
            torch.save(unwrapped_wrapper.unet.state_dict(), os.path.join(final_dir, "unet.pth"))
            torch.save(unwrapped_wrapper.encoder.state_dict(), os.path.join(final_dir, "encoder.pth"))
            print(f"Saved final models to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()
    main(args.config, args.test_run)