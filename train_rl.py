# train_rl.py

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
# 尝试可选导入 ValueNetwork；若失败则降级到 batch-based advantages（GRPO 风格）
try:
    from value_network import ValueNetwork
    VALUE_NET_IMPORT_OK = True
except Exception:
    ValueNetwork = None
    VALUE_NET_IMPORT_OK = False

from dataset import ImageFusionDataset


def find_vae_checkpoint(ckpt_dir="./checkpoints/vae/best"):
    """查找VAE checkpoint文件"""
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
    return None


def main(config, test_run):
    # --- 1. 初始化和配置加载 ---
    project_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(project_dir, exist_ok=True)
    
    accelerator = Accelerator(
        log_with="tensorboard", 
        project_dir=project_dir
    )
    
    if accelerator.is_main_process:
        # 保存配置文件副本
        config_copy_path = os.path.join(project_dir, "config.yml")
        with open(config_copy_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_copy_path}")
        print(f"Starting Run: {config['run_name']}")
        print("Configuration:")
        print(yaml.dump(config, indent=2))

    # --- 2. 模型、优化器和数据加载器设置 ---
    
    # 加载Stage 1预训练好的模型
    stage1_dir = config['stage1_checkpoint_dir']
    # 使用配置字典实例化模型
    unet = UNet2DConditionModel(**config['model_config']['unet'])
    encoder = ConditioningEncoder(**config['model_config']['encoder'])
    
    # 仅在真实训练时加载 Stage1 的权重（test_run 保持随机初始用于快速验证）
    if not test_run:
        if accelerator.is_main_process:
            print(f"Loading Stage 1 models from {stage1_dir}")
        try:
            unet_sd = torch.load(os.path.join(stage1_dir, "unet.pth"), map_location="cpu")
            encoder_sd = torch.load(os.path.join(stage1_dir, "encoder.pth"), map_location="cpu")
            # 如果保存的是 dict 包装（state_dict），尝试提取
            if isinstance(unet_sd, dict) and 'model_state_dict' in unet_sd:
                unet_sd = unet_sd['model_state_dict']
            if isinstance(encoder_sd, dict) and 'model_state_dict' in encoder_sd:
                encoder_sd = encoder_sd['model_state_dict']
            unet.load_state_dict(unet_sd)
            encoder.load_state_dict(encoder_sd)
            if accelerator.is_main_process:
                print("Stage 1 models loaded successfully")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: Failed to load Stage 1 models: {e}")
    
    # 创建一个冻结的参考网络，用于计算KL散度 (DPOK的核心思想)
    unet_ref = None
    if config.get('rl_training', {}).get('use_kl', False):
        unet_ref = UNet2DConditionModel(**config['model_config']['unet'])
        unet_ref.load_state_dict(unet.state_dict())
        unet_ref.requires_grad_(False)
    
    
    # 是否使用 value network，由 config 控制且导入成功才开启
    use_value_net = bool(config.get('rl_training', {}).get('use_value_net', False)) and VALUE_NET_IMPORT_OK
    if bool(config.get('rl_training', {}).get('use_value_net', False)) and not VALUE_NET_IMPORT_OK:
        if accelerator.is_main_process:
            print("Warning: ValueNetwork import failed; falling back to batch advantages (GRPO-style).")
    if use_value_net:
        # 初始化价值网络（假定 latent 维度/条件维度由 config 提供或使用默认）
        latent_dim = config.get('rl_training', {}).get('value_latent_dim', 4*32*32)
        condition_dim = config['model_config']['unet']['cross_attention_dim']
        value_net = ValueNetwork(latent_dim=latent_dim, condition_dim=condition_dim)
    else:
        value_net = None

    # 实例化VAE并创建Pipeline（使用config中的 vae 配置），并尝试加载外部训练好的 VAE
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    vae_cfg = config.get('model_config', {}).get('vae', None)
    if vae_cfg is not None:
        vae = AutoencoderKL(**{k: v for k, v in vae_cfg.items() if k in ["sample_size", "in_channels", "out_channels", "down_block_types", "up_block_types", "block_out_channels", "latent_channels"]})
        # 查找外部 vae checkpoint（默认路径可在 rl.yml 中配置）
        vae_ckpt_dir = vae_cfg.get('checkpoint_dir', "./checkpoints/vae/best")
        vae_ckpt = find_vae_checkpoint(vae_ckpt_dir)
        if vae_ckpt is not None:
            if accelerator.is_main_process:
                print(f"Loading VAE from {vae_ckpt}")
            sd = torch.load(vae_ckpt, map_location="cpu")
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            try:
                vae.load_state_dict(sd)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Partial load of VAE state_dict, error: {e}")
                own_state = vae.state_dict()
                filtered = {k: v for k, v in sd.items() if k in own_state and own_state[k].shape == v.shape}
                own_state.update(filtered)
                vae.load_state_dict(own_state)
        else:
            if accelerator.is_main_process:
                print("No VAE checkpoint found for RL; continuing with randomly initialized VAE (not recommended)")

        vae_scale = vae_cfg.get('scale_factor', None)
    else:
        vae = None
        vae_scale = None

    pipeline = ImageFusionPipeline(unet=unet, scheduler=scheduler, encoder=encoder, vae=vae, vae_scale_factor=vae_scale)

    # 优化器
    policy_optimizer = torch.optim.AdamW(list(unet.parameters()) + list(encoder.parameters()), lr=config['rl_training']['policy_learning_rate'])
    value_optimizer = None
    if use_value_net:
        value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=config['rl_training']['value_learning_rate'])

    # 数据加载器：加载 label (dir_C) 但暂时不使用 label 信号
    if not test_run:
        train_ds_config = config['data']
        train_dataset = ImageFusionDataset(
            dir_A=train_ds_config['train']['dir_A'],
            dir_B=train_ds_config['train']['dir_B'],
            dir_C=train_ds_config['train'].get('dir_C', None),  # load labels if provided, not used currently
            is_train=True, is_getpatch=True, patch_size=train_ds_config['patch_size'], augment=train_ds_config['augment']
        )
        train_loader = DataLoader(train_dataset, batch_size=train_ds_config['train_batch_size'], shuffle=True)
    
    # 准备所有组件（有选择地包含value_net / value_optimizer）
    prepare_list = [unet, encoder, policy_optimizer]
    if use_value_net:
        prepare_list.extend([value_net, value_optimizer])
    if not test_run:
        prepare_list.append(train_loader)
    # 如果 VAE 存在，把它也放进 prepare（仅用于推理 decode / encode）
    if vae is not None:
        prepare_list.insert(0, vae)
    prepared = accelerator.prepare(*prepare_list)
    # unpack prepared items
    idx = 0
    if vae is not None:
        vae = prepared[idx]; idx += 1
    unet = prepared[idx]; idx += 1
    encoder = prepared[idx]; idx += 1
    policy_optimizer = prepared[idx]; idx += 1
    if use_value_net:
        value_net = prepared[idx]; idx += 1
        value_optimizer = prepared[idx]; idx += 1
    if not test_run:
        train_loader = prepared[idx]; idx += 1

    if unet_ref is not None:
        unet_ref.to(accelerator.device)
    if use_value_net and value_net is not None:
        value_net.to(accelerator.device)

    # 保证 scheduler 在外部也设置 timesteps（供 timesteps_batch 使用）
    num_inf_steps = config['data']['num_inference_steps']
    scheduler.set_timesteps(num_inf_steps)

    # --- 3. Reward函数 (Placeholder) ---
    def reward_fn(images, **kwargs):
        """Placeholder for the reward function. Returns a random score."""
        return torch.randn(images.shape[0], device=accelerator.device)

    # --- 4. 测试运行逻辑 ---
    if test_run:
        if accelerator.is_main_process:
            print("\n--- Starting Test Run ---")
        # 创建虚拟数据
        bs = 2
        ps = config['data']['patch_size']
        vis_images = torch.randn(bs, 3, ps, ps, device=accelerator.device)
        ir_images = torch.randn(bs, 1, ps, ps, device=accelerator.device)
        # 将虚拟数据包装成一个批次
        test_batch = (vis_images, ir_images)
        # 只跑一个批次
        train_loader = [test_batch]
    
    num_inf_steps = config['data']['num_inference_steps']

    # --- 5. 训练循环 ---
    for epoch in range(config['rl_training']['num_epochs']):
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"RL Epoch {epoch+1}")
        
        for batch in pbar:
            # 支持 (vis, ir, label) 或 (vis, ir)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                vis_images, ir_images, _label_images = batch
            else:
                vis_images, ir_images = batch

            # --- 5.1 采样/收集轨迹 (Rollout) ---
            with torch.no_grad():
                final_images, latents_history, old_log_probs = pipeline.forward_with_logprob(
                    vis_image=vis_images, ir_image=ir_images, num_inference_steps=num_inf_steps
                )
            
            # --- 5.2 计算奖励 ---
            rewards = reward_fn(final_images)  # shape: (batch,)
            if accelerator.is_main_process and test_run: print("Step 2: Reward calculation successful.")
                
            # --- 5.3 价值网络更新 或 使用 batch-based advantages ---
            if use_value_net and value_net is not None:
                # 使用价值网络：构造 per-timestep target values 并训练 value_net
                condition_embeds = encoder(torch.cat([vis_images, ir_images], dim=1))
                # target_values: expand final reward 到每个 timestep
                target_values = rewards.view(-1, 1).expand(-1, num_inf_steps).reshape(-1, 1)

                # 将历史latents展平 (batch * num_steps, C, H, W)
                states_batch = torch.stack(latents_history[:-1], dim=1).view(-1, *latents_history[0].shape[1:])
                # timesteps_batch：重复 timesteps for batch
                timesteps_batch = scheduler.timesteps.repeat(vis_images.shape[0])
                condition_embeds_batch = condition_embeds.repeat_interleave(num_inf_steps, dim=0)

                for _ in range(config['rl_training']['value_update_steps']):
                    predicted_values = value_net(states_batch, timesteps_batch, condition_embeds_batch)
                    value_loss = F.mse_loss(predicted_values, target_values)
                    
                    accelerator.backward(value_loss)
                    value_optimizer.step()
                    value_optimizer.zero_grad()
                if accelerator.is_main_process and test_run: print("Step 3: Value network update successful.")

                # 计算 advantages = target - predicted (并标准化)
                with torch.no_grad():
                    predicted_values = value_net(states_batch, timesteps_batch, condition_embeds_batch)
                    advantages = (target_values - predicted_values).view(vis_images.shape[0], num_inf_steps).mean(dim=1)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    # 为 policy 更新扩展到每个时刻的向量（flatten）
                    advantages_expanded = advantages.view(-1,1).expand(-1, num_inf_steps).reshape(-1)
            else:
                # 不使用 value_net：直接使用 batch-based advantages（GRPO风格）
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # shape (batch,)
                # 扩展到每 timestep 的 advantage 向量（flatten）
                advantages_expanded = advantages.view(-1,1).expand(-1, num_inf_steps).reshape(-1)
                # 仍需构造 states_batch / timesteps_batch / condition_embeds_batch 用于后续计算
                condition_embeds = encoder(torch.cat([vis_images, ir_images], dim=1))
                states_batch = torch.stack(latents_history[:-1], dim=1).view(-1, *latents_history[0].shape[1:])
                timesteps_batch = scheduler.timesteps.repeat(vis_images.shape[0])
                condition_embeds_batch = condition_embeds.repeat_interleave(num_inf_steps, dim=0)
                if accelerator.is_main_process and test_run: print("Using batch advantages (no value net).")
            
            # --- 5.4 策略网络更新 ---
            for _ in range(config['rl_training']['policy_update_steps']):
                with torch.no_grad():
                    # 如果使用 value_net，上面已生成 predicted_values 用于计算 advantages
                    # 如果未使用，则 advantages_expanded 已准备好
                    # 标准化已经在上面完成
                    pass
                
                # 重新计算当前策略下的动作概率
                noise_pred_train = unet(states_batch, timesteps_batch, condition_embeds_batch).sample
                if config.get('rl_training', {}).get('use_kl', False):
                    with torch.no_grad():
                        noise_pred_ref = unet_ref(states_batch, timesteps_batch, condition_embeds_batch).sample
                
                # 计算 new_log_prob：ddim_step_with_logprob 支持向量化的 timestep 参数（timesteps_batch）
                prev_samples_flat = torch.stack(latents_history[1:], dim=1).view(-1, *latents_history[0].shape[1:])
                _, new_log_prob = ddim_step_with_logprob(
                    scheduler, noise_pred_train, timesteps_batch, states_batch, 
                    prev_sample=prev_samples_flat, eta=1.0
                )
                
                # KL Loss (DPOK) 简化为预测噪声的MSE
                if config.get('rl_training', {}).get('use_kl', False):
                    if accelerator.is_main_process:
                        print("rl_training.use_kl=True but KL loss is not implemented. Raising NotImplementedError.")
                    raise NotImplementedError("KL loss computation not implemented. Set rl_training.use_kl=False to skip.")
                else:
                    kl_loss = torch.tensor(0.0, device=accelerator.device)

                # PPO Clipped Loss (DDPO)
                # new_log_prob 与 old_log_probs_flat 顺序需一致（均为 batch-major flatten）
                old_log_probs_flat = old_log_probs.view(-1)
                ratio = torch.exp(new_log_prob - old_log_probs_flat.to(new_log_prob.device))
                unclipped_loss = -advantages_expanded.to(ratio.device) * ratio
                clipped_loss = -advantages_expanded.to(ratio.device) * torch.clamp(ratio, 1.0 - config['rl_training']['ppo_clip_range'], 1.0 + config['rl_training']['ppo_clip_range'])
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                
                # 总损失
                total_loss = config['rl_training']['reward_weight'] * policy_loss + config['rl_training']['kl_weight'] * kl_loss
                
                accelerator.backward(total_loss)
                policy_optimizer.step()
                policy_optimizer.zero_grad()
            if accelerator.is_main_process and test_run: print("Step 4: Policy network update successful.")

        if test_run:
            if accelerator.is_main_process: print("\n--- Test Run Successful! ---")
            break # 测试运行只跑一个批次就退出

        # --- 6. 保存模型 ---
        if (epoch + 1) % config['logging']['save_freq'] == 0 and accelerator.is_main_process:
            save_dir = os.path.join(config['output_dir'], config['run_name'], f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 使用accelerator保存unwrapped模型
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_encoder = accelerator.unwrap_model(encoder)
            
            torch.save(unwrapped_unet.state_dict(), os.path.join(save_dir, "unet.pth"))
            torch.save(unwrapped_encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
            print(f"RL model saved to {save_dir}")

    if accelerator.is_main_process:
        # 保存最终模型
        save_dir = os.path.join(config['output_dir'], config['run_name'], "final")
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用accelerator保存unwrapped模型
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_encoder = accelerator.unwrap_model(encoder)
        
        torch.save(unwrapped_unet.state_dict(), os.path.join(save_dir, "unet.pth"))
        torch.save(unwrapped_encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        print(f"Final RL model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the RL config file.")
    parser.add_argument("--test_run", action="store_true", help="Run a test with dummy data.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config, args.test_run)