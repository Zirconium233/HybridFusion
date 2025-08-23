# pretrain.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd # 用于美化输出
import os
from diffusers import UNet2DConditionModel, DDIMScheduler
import numpy as np

# 假设您的模型、数据集和指标代码都在这些文件中
from model.pipeline import ImageFusionPipeline, ConditioningEncoder
from dataset import ImageFusionDataset
import metric # 导入您提供的 metric.py

def main(config_path):
    # --- 1. 初始化和配置加载 ---
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=os.path.join(config['output_dir'], config['run_name'])
    )
    
    # 打印配置信息
    if accelerator.is_main_process:
        print("Configuration:")
        print(yaml.dump(config, indent=2))
        
    # --- 2. 模型、优化器和数据加载器设置 ---
    
    # 初始化模型 (使用分层配置：unet / encoder / vae)
    unet = UNet2DConditionModel(**config['model_config']['unet'])
    encoder = ConditioningEncoder(**config['model_config']['encoder'])
    # Pipeline 不需要，因为我们直接训练U-Net和Encoder
    
    # 优化器
    optimizer_config = config['training']['optimizer']
    optimizer = getattr(torch.optim, optimizer_config['type'])(
        list(unet.parameters()) + list(encoder.parameters()),
        lr=config['training']['learning_rate'],
        **optimizer_config['args']
    )

    # 学习率调度器
    scheduler_config = config['training']['scheduler']
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_config['type'])(
        optimizer, **scheduler_config['args']
    )
    
    # 强制使用 L1 loss（按要求）
    loss_fn = F.l1_loss
    
    # 训练数据集：不使用 patch（全分辨率），使用 dir_C 作为 label（若配置提供）
    train_ds_config = config['train_dataset']
    train_dataset_paths = config['datasets'][train_ds_config['name']]['train']
    train_dataset = ImageFusionDataset(
        dir_A=train_dataset_paths['dir_A'],
        dir_B=train_dataset_paths['dir_B'],
        dir_C=train_dataset_paths.get('dir_C', None),
        is_train=True,
        is_getpatch=False,               # 全分辨率，不切 patch
        patch_size=train_ds_config.get('patch_size', 128),
        augment=train_ds_config.get('augment', False),
        is_ycbcrA=False,
        is_ycbcrB=False,
        is_ycbcrC=False,
        scale=1
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['train_batch_size'], 
        shuffle=True
    )
    
    # 准备所有组件
    unet, encoder, optimizer, scheduler, train_loader = accelerator.prepare(
        unet, encoder, optimizer, scheduler, train_loader
    )

    # --- 3. 训练循环 ---

    for epoch in range(config['training']['num_epochs']):
        unet.train()
        encoder.train()
        
        # 训练进度条
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            # 支持 (vis, ir, label) 或 (vis, ir)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                vis_images, ir_images, label_images = batch
            else:
                vis_images, ir_images = batch
                label_images = vis_images  # 若未提供 C，则回退到 vis 作为 target（兼容）
            
            with accelerator.accumulate(unet, encoder):
                # 拼接条件图像并通过编码器
                condition_img = torch.cat([vis_images, ir_images], dim=1).to(accelerator.device)
                condition_embeds = encoder(condition_img)
                
                # 目标为 label_images（已由 dataset 读取 dir_C）
                target_image = label_images.to(accelerator.device)
                noise = torch.randn_like(target_image)
                timesteps = torch.randint(0, 1000, (target_image.shape[0],), device=accelerator.device).long()
                
                # 简化目标：让U-Net直接预测目标图像
                predicted_image = unet(noise, timesteps, encoder_hidden_states=condition_embeds).sample
                
                # 计算 L1 损失（使用 label C）
                loss = loss_fn(predicted_image, target_image)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({"loss": loss.item()})
        
        scheduler.step()
        
        # --- 4. 测试流程 ---
        if (epoch + 1) % config['training']['test_freq'] == 0:
            if accelerator.is_main_process:
                print(f"\n--- Running Test at Epoch {epoch+1} ---")
                
                unet.eval()
                encoder.eval()
                
                results = {}
                # 遍历config中定义的所有测试集
                for test_set_config in config['test_sets']:
                    set_name = test_set_config['name']
                    test_dataset_paths = config['datasets'][set_name]['test']
                    
                    # 创建测试数据集（不使用patch和增强），如果有 dir_C 一并加载
                    test_dataset = ImageFusionDataset(
                        dir_A=test_dataset_paths['dir_A'],
                        dir_B=test_dataset_paths['dir_B'],
                        dir_C=test_dataset_paths.get('dir_C', None),
                        is_train=False,
                        is_getpatch=False
                    )
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                    
                    # 存储每个指标的结果
                    metric_scores = {m: [] for m in metric.__all__} # 假设metric.py有__all__
                    
                    for batch in tqdm(test_loader, desc=f"Testing on {set_name}"):
                        # 支持 (vis, ir, label) 或 (vis, ir)
                        if isinstance(batch, (list, tuple)) and len(batch) == 3:
                            vis_test, ir_test, label_test = batch
                        else:
                            vis_test, ir_test = batch
                            label_test = None
                        with torch.no_grad():
                            condition_img_test = torch.cat([vis_test, ir_test], dim=1).to(accelerator.device)
                            condition_embeds_test = encoder(condition_img_test)
                            noise_test = torch.randn_like(vis_test, device=accelerator.device)
                            
                            fused_image = unet(noise_test, torch.tensor(0, device=accelerator.device), condition_embeds_test).sample
                        
                        # 将Tensor转为Numpy array用于计算指标
                        vis_np = vis_test.squeeze().cpu().numpy()
                        ir_np = ir_test.squeeze().cpu().numpy()
                        fused_np = fused_image.squeeze().cpu().numpy()
                        
                        # 计算所有指标
                        for metric_name in metric_scores.keys():
                            metric_func = getattr(metric, f"{metric_name}_function")
                            score = metric_func(vis_np, ir_np, fused_np)
                            metric_scores[metric_name].append(score)
                    
                    # 计算平均分并存入总结果
                    results[set_name] = {m: np.mean(s) for m, s in metric_scores.items()}
                
                # 使用Pandas美化并打印结果表格
                df = pd.DataFrame(results).T
                print(df.to_string(float_format="%.4f"))
                print("--- Test Finished ---\n")
                
        # --- 5. 保存模型 ---
        if (epoch + 1) % config['training']['save_freq'] == 0:
            if accelerator.is_main_process:
                save_dir = os.path.join(config['output_dir'], config['run_name'], f"epoch_{epoch+1}")
                os.makedirs(save_dir, exist_ok=True)
                
                # 使用accelerator保存unwrapped模型
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_encoder = accelerator.unwrap_model(encoder)
                
                torch.save(unwrapped_unet.state_dict(), os.path.join(save_dir, "unet.pth"))
                torch.save(unwrapped_encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
                print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    # 使用时，传入配置文件的路径
    # 例如: python pretrain.py --config config/pretrain_config.yml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    main(args.config)