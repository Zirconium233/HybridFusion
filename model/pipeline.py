# model.py (Optimized Version)
import torch
import torch.nn as nn
import math
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from typing import List, Optional, Tuple
from tqdm.auto import tqdm
import time

# 尝试导入FLOPs计算库
try:
    from fvcore.nn import FlopCountAnalysis
    has_fvcore = True
except ImportError:
    has_fvcore = False


def _left_broadcast(t, shape):
    """广播张量到指定形状"""
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(scheduler, timestep, prev_timestep):
    """获取方差"""
    alpha_prod_t = torch.gather(scheduler.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        scheduler.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def _infer_vae_scale(vae) -> int:
    return 4 # 不知道为什么，3次下采样实际缩放是4，可能和diffuser实现有关系

# 新增：按通道自动选择可整除组数的 GroupNorm 创建器
def _make_groupnorm(num_channels: int, max_groups: int = 32) -> nn.Module:
    groups = min(max_groups, num_channels)
    # 降低组数直到能整除通道数
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    # groups==1 时，相当于 LayerNorm 风格（在 GroupNorm 中）
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    prev_sample: torch.FloatTensor,
    eta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM步进函数，基于DDPO实现，同时计算对数概率
    """
    # 1. 获取前一个时间步
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = torch.clamp(torch.as_tensor(prev_timestep), 0, scheduler.config.num_train_timesteps - 1)
    timestep_tensor = torch.as_tensor(timestep).to(sample.device)
    
    # 2. 计算alphas和betas
    alpha_prod_t = scheduler.alphas_cumprod.gather(0, timestep_tensor.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        scheduler.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    
    beta_prod_t = 1 - alpha_prod_t
    
    # 3. 计算预测的原始样本
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    
    # 4. 计算方差
    variance = _get_variance(scheduler, timestep_tensor, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)
    
    # 5. 计算指向x_t的方向
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output
    
    # 6. 计算x_{t-1}的均值
    prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    
    # 7. 计算对数概率（基于DDPO实现）
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # 对除batch维度外的所有维度求平均
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob


class ConditioningEncoder(nn.Module):
    """
    小型ResNet风格的条件编码器，将拼接的VIS+IR图像(4通道)编码为特征序列。
    维持在10-30M参数范围内，处理速度快。
    """
    def __init__(self, in_channels=4, out_channels=512, base_channels=128, layer_blocks=(2,2,2)):
        super().__init__()
        
        # ResNet-style blocks
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # 使用 GroupNorm 替代 BatchNorm，避免 diffusion 训练中 BatchNorm 的不稳定性
        self.bn1 = _make_groupnorm(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # allow configurable block counts (increase capacity)
        self.layer1 = self._make_layer(base_channels, base_channels, layer_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels*2, layer_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, layer_blocks[2], stride=2)
        
        # 输出投影，保持输出维度为 out_channels（通常与 UNet cross_attention_dim 对齐）
        self.final_conv = nn.Conv2d(base_channels*4, out_channels, kernel_size=1)
        
    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    @property
    def dtype(self):
        """返回模型的数据类型，用于兼容DiffusionPipeline"""
        return next(self.parameters()).dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (B, 4, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.final_conv(x)
        # 输出特征图: (B, out_channels, H', W')
        
        # 转换为序列: (B, H'*W', out_channels) -> (B, SeqLen, Dim)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        return x


class BasicBlock(nn.Module):
    """基础ResNet块"""
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _make_groupnorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _make_groupnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _make_groupnorm(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ImageFusionPipeline(DiffusionPipeline):
    """
    使用小号VAE的图像融合Pipeline。
    新增参数:
      - use_shortcut: 是否在最终 latent 上加回 vis 的 latent (残差 shortcut)
      - use_ddim_logprob: forward_with_logprob 时是否使用 ddim_step_with_logprob 计算 logprob
    """
    def __init__(self, unet: UNet2DConditionModel, scheduler: DDIMScheduler, encoder: ConditioningEncoder, vae: AutoencoderKL, vae_scale_factor: Optional[int] = None, use_shortcut: bool = False, use_ddim_logprob: bool = False):
        super().__init__()
        # 注册模块
        self.register_modules(unet=unet, scheduler=scheduler, encoder=encoder, vae=vae)
        self.vae_scale_factor = vae_scale_factor if vae_scale_factor is not None else _infer_vae_scale(vae)
        if vae_scale_factor is None:
            print(f"[Info] inferred vae_scale_factor={self.vae_scale_factor} from vae.config")
        # 新增配置
        self.use_shortcut = bool(use_shortcut)
        self.use_ddim_logprob = bool(use_ddim_logprob)

    @property
    def device(self):
        """返回模型的设备"""
        return next(self.unet.parameters()).device

    @torch.no_grad()
    def __call__(
        self,
        vis_image: torch.FloatTensor,
        ir_image: torch.FloatTensor,
        num_inference_steps: int = 20,
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        device = self.device
        dtype = self.unet.dtype

        # 原像素大小检查保留
        _, _, H, W = vis_image.shape
        if H % self.vae_scale_factor != 0 or W % self.vae_scale_factor != 0:
            raise ValueError(f"输入 H,W 必须能被 vae_scale_factor({self.vae_scale_factor}) 整除，当前 H={H}, W={W}")
        
        # 根据 encoder 输入通道判断是否使用 latent 条件（效率：一次 encode vis+ir）
        latent_channels = getattr(self.vae.config, "latent_channels", 4)
        cond_in_ch = getattr(self.encoder.conv1, "in_channels", None)
        use_latent_condition = (cond_in_ch == latent_channels * 2)

        if use_latent_condition:
            ir_for_encode = ir_image.repeat(1, 3, 1, 1) if ir_image.shape[1] == 1 else ir_image
            enc = self.vae.encode(torch.cat([vis_image, ir_for_encode], dim=0))
            lat_all = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else (enc[0] if isinstance(enc, (tuple, list)) else enc)
            B = vis_image.shape[0]
            vis_lat_c, ir_lat_c = lat_all[:B], lat_all[B:2*B]
            condition_input = torch.cat([vis_lat_c, ir_lat_c], dim=1).to(device=device, dtype=dtype)
        else:
            condition_input = torch.cat([vis_image, ir_image], dim=1).to(device=device, dtype=dtype)

        condition_embeds = self.encoder(condition_input)
        
        # 潜空间采样保持不变
        b, _, H, W = vis_image.shape
        latent_h = H // self.vae_scale_factor
        latent_w = W // self.vae_scale_factor
        latent_channels = getattr(self.vae.config, "latent_channels", 4)

        latents_shape = (b, latent_channels, latent_h, latent_w)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        for t in timesteps:
            noise_pred = self.unet(latents, t, encoder_hidden_states=condition_embeds).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        if hasattr(self.vae.config, 'scaling_factor'):
            latents = latents / self.vae.config.scaling_factor

        if self.use_shortcut:
            vis_for_encode = vis_image.to(device=device, dtype=dtype)
            with torch.no_grad():
                vis_lat = self.vae.encode(vis_for_encode).latent_dist.sample()
            latents = latents + vis_lat

        try:
            decoded = self.vae.decode(latents).sample
        except Exception:
            out = self.vae.decode(latents)
            decoded = out["sample"] if isinstance(out, dict) and "sample" in out else out

        return decoded

    @torch.no_grad()
    def forward_with_logprob(
        self,
        vis_image: torch.FloatTensor,
        ir_image: torch.FloatTensor,
        num_inference_steps: int = 10,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        device = self.device
        dtype = self.unet.dtype

        # 条件：像素 or latent（一次 encode vis+ir）
        latent_channels = getattr(self.vae.config, "latent_channels", 4)
        cond_in_ch = getattr(self.encoder.conv1, "in_channels", None)
        use_latent_condition = (cond_in_ch == latent_channels * 2)
        if use_latent_condition:
            ir_for_encode = ir_image.repeat(1, 3, 1, 1) if ir_image.shape[1] == 1 else ir_image
            enc = self.vae.encode(torch.cat([vis_image, ir_for_encode], dim=0))
            lat_all = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else (enc[0] if isinstance(enc, (tuple, list)) else enc)
            B = vis_image.shape[0]
            vis_lat_c, ir_lat_c = lat_all[:B], lat_all[B:2*B]
            condition_input = torch.cat([vis_lat_c, ir_lat_c], dim=1).to(device=device, dtype=dtype)
        else:
            condition_input = torch.cat([vis_image, ir_image], dim=1).to(device=device, dtype=dtype)

        condition_embeds = self.encoder(condition_input)

        # 其余保持不变
        b, _, H, W = vis_image.shape
        latent_h = H // self.vae_scale_factor
        latent_w = W // self.vae_scale_factor
        latent_channels = getattr(self.vae.config, "latent_channels", 4)

        latents_shape = (b, latent_channels, latent_h, latent_w)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        all_latents = [latents]
        all_log_probs = []

        for t in timesteps:
            noise_pred = self.unet(latents, t, encoder_hidden_states=condition_embeds).sample
            next_step = self.scheduler.step(noise_pred, t, latents)
            prev_sample = next_step.prev_sample
            if self.use_ddim_logprob:
                _, log_prob = ddim_step_with_logprob(self.scheduler, noise_pred, t, latents, prev_sample=prev_sample, eta=1.0)
                all_log_probs.append(log_prob)
            else:
                all_log_probs.append(torch.zeros(b, device=device, dtype=dtype))
            latents = prev_sample
            all_latents.append(latents)

        final_latents = latents

        if hasattr(self.vae.config, 'scaling_factor'):
            final_latents = final_latents / self.vae.config.scaling_factor

        if self.use_shortcut:
            vis_for_encode = vis_image.to(device=device, dtype=dtype)
            with torch.no_grad():
                vis_latent = self.vae.encode(vis_for_encode).latent_dist.sample()
            final_latents = final_latents + vis_latent

        try:
            decoded = self.vae.decode(final_latents).sample
        except Exception:
            out = self.vae.decode(final_latents)
            decoded = out["sample"] if isinstance(out, dict) and "sample" in out else out

        log_probs_history = torch.stack(all_log_probs, dim=1)
        
        return decoded, all_latents, log_probs_history

if __name__ == "__main__":
    
    print("--- 1. Model Initialization ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use pretrained VAE instead of custom VAE
    vae = AutoencoderKL.from_pretrained("/home/zhangran/desktop/myProject/playground/sd-vae-ft-mse")
    vae_latent_channels = vae.config.latent_channels
    vae_scale_factor = 8  # Pretrained VAE uses 8x downsampling
    scaling_factor = vae.config.scaling_factor
    print(f"[Info] using pretrained VAE: latent_channels={vae_latent_channels}, scale_factor={vae_scale_factor}, scaling_factor={scaling_factor}")
 
     # U-Net 配置 - 输入通道应匹配 VAE 的潜空间通道数
    unet_config = {
         'block_out_channels': (64, 128, 256, 256),  # 减小参数量
         'down_block_types': ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
         'up_block_types': ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
         'cross_attention_dim': 512,  # 匹配encoder输出
         'in_channels': vae_latent_channels,    # 改为 VAE 潜空间通道
         'out_channels': vae_latent_channels,   # 输出也在潜空间
         'sample_size': 640 // vae_scale_factor,  # UNet 在潜空间的空间尺寸（640//8=80）
    }
    unet = UNet2DConditionModel(**unet_config)
    
    # 使用本地scheduler，默认10步
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    
    # 初始化我们的条件编码器 - 匹配unet的cross_attention_dim
    encoder = ConditioningEncoder(in_channels=vae_latent_channels*2, out_channels=unet_config['cross_attention_dim'])
    
    # 改为开启 logprob 计算，避免 forward_with_logprob 返回全 0
    pipeline = ImageFusionPipeline(
        unet=unet, scheduler=scheduler, encoder=encoder, vae=vae,
        vae_scale_factor=vae_scale_factor, use_shortcut=False, use_ddim_logprob=True
    ).to(device)

    # 计算参数量
    params_unet = sum(p.numel() for p in pipeline.unet.parameters() if p.requires_grad)
    params_encoder = sum(p.numel() for p in pipeline.encoder.parameters() if p.requires_grad)
    params_vae = sum(p.numel() for p in pipeline.vae.parameters() if p.requires_grad)
    total_params = params_unet + params_encoder + params_vae
    print(f"U-Net initialized with {params_unet / 1e6:.2f}M parameters.")
    print(f"ConditioningEncoder initialized with {params_encoder / 1e6:.2f}M parameters.")
    print(f"VAE initialized with {params_vae / 1e6:.2f}M parameters.")
    print(f"Total model parameters: {total_params / 1e6:.2f}M")
    
    print("\n--- 2. Full Resolution Inference Test (with VAE) ---")
    
    # 测试 640x480 分辨率 (VGA)
    print("Testing 640x480 resolution...")
    vis_image_vga = torch.randn(1, 3, 480, 640, device=device)  # (H, W) = (480, 640)
    ir_image_vga = torch.randn(1, 1, 480, 640, device=device)
    
    start_time = time.time()
    fused_image_vga = pipeline(vis_image_vga, ir_image_vga, num_inference_steps=20)
    end_time = time.time()
    time_vga = end_time - start_time
    
    print(f"Inference on 640x480 finished.")
    print(f"Input VIS shape: {vis_image_vga.shape}, IR shape: {ir_image_vga.shape}")
    print(f"Output fused image shape: {fused_image_vga.shape}")
    print(f"Inference time: {time_vga:.2f} seconds.")

    # 测试 1024x768 分辨率 (XGA)
    print("\nTesting 1024x768 resolution...")
    vis_image_xga = torch.randn(1, 3, 768, 1024, device=device)  # (H, W) = (768, 1024)
    ir_image_xga = torch.randn(1, 1, 768, 1024, device=device)
    
    start_time = time.time()
    fused_image_xga = pipeline(vis_image_xga, ir_image_xga, num_inference_steps=20)
    end_time = time.time()
    time_xga = end_time - start_time

    print(f"Inference on 1024x768 finished.")
    print(f"Output fused image shape: {fused_image_xga.shape}")
    print(f"Inference time: {time_xga:.2f} seconds.")

    print("\n--- 3. Loss & Log Probability Test ---")
    
    # 使用640x480的图像进行概率测试
    with torch.no_grad():
        _, latents_history, old_log_probs = pipeline.forward_with_logprob(
            vis_image_vga, ir_image_vga, num_inference_steps=20
        )
    
    print(f"Generated {len(latents_history)} latent states.")
    print(f"Log probabilities shape: {old_log_probs.shape}")
    print(f"Sample log prob values: {old_log_probs[0, :5]}")  # 前5步的log prob
    
    # 测试loss计算（使用 LATENT 条件，而不是像素 3+1）
    dummy_rewards = torch.randn(1, device=device)
    advantages = (dummy_rewards - dummy_rewards.mean())
    
    t_idx = 5  # 中间的一步
    current_latents = latents_history[t_idx]
    next_latents_action = latents_history[t_idx+1]
    old_log_prob_t = old_log_probs[:, t_idx]
    
    # 构造与 pipeline 一致的 latent 条件：一次性编码 vis+ir
    with torch.no_grad():
        ir_vga_for_encode = ir_image_vga.repeat(1, 3, 1, 1) if ir_image_vga.shape[1] == 1 else ir_image_vga
        enc_pair = pipeline.vae.encode(torch.cat([vis_image_vga, ir_vga_for_encode], dim=0))
        lat_pair = enc_pair.latent_dist.sample() if hasattr(enc_pair, "latent_dist") else (
            enc_pair[0] if isinstance(enc_pair, (tuple, list)) else enc_pair
        )
        Bv = vis_image_vga.shape[0]
        vis_lat_vga, ir_lat_vga = lat_pair[:Bv], lat_pair[Bv:2*Bv]
        cond_input_vga = torch.cat([vis_lat_vga, ir_lat_vga], dim=1)

    # 模拟一次带梯度的前向传播（确保 dtype/device 一致）
    pipeline.unet.train()
    condition_embeds = pipeline.encoder(cond_input_vga.to(device=pipeline.device, dtype=pipeline.unet.dtype))
    noise_pred_new = pipeline.unet(current_latents, scheduler.timesteps[t_idx], condition_embeds).sample
    # 重新计算该步的 log_prob（使用和 forward_with_logprob 一致的 DDIM 公式）
    _, new_log_prob_t = ddim_step_with_logprob(
        pipeline.scheduler, noise_pred_new, scheduler.timesteps[t_idx], current_latents, 
        eta=1.0, prev_sample=next_latents_action
    )

    ratio = torch.exp(new_log_prob_t - old_log_prob_t)
    policy_loss = -advantages[0] * ratio.mean()
    
    print(f"Policy loss calculation successful: {policy_loss.item():.4f}")
    print(f"Probability ratio: {ratio.mean().item():.4f}")
    
    print("\n--- 4. Model Size & Performance Verification ---")
    if total_params / 1e6 < 100:
        print(f"✓ Model size ({total_params / 1e6:.1f}M params) is within 10-100M range")
    else:
        print(f"✗ Model size ({total_params / 1e6:.1f}M params) exceeds 100M")
        
    # 检查两个分辨率的推理时间
    if time_vga < 5:
        print(f"✓ 640x480 inference time ({time_vga:.2f}s) is under 5 seconds")
    else:
        print(f"✗ 640x480 inference time ({time_vga:.2f}s) exceeds 5 seconds")
        
    if time_xga < 5:
        print(f"✓ 1024x768 inference time ({time_xga:.2f}s) is under 5 seconds")
    else:
        print(f"✗ 1024x768 inference time ({time_xga:.2f}s) exceeds 5 seconds")
        
    print(f"\nPerformance Summary:")
    print(f"  - 640x480: {time_vga:.2f}s")
    print(f"  - 1024x768: {time_xga:.2f}s")
        
    print("\n--- Pipeline Test Complete ---")