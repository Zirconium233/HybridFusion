import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LaplacianPyramidFusion(nn.Module):
    """
    一个固定的、非神经网络的拉普拉斯金字塔融合模块。
    它作为一个可被策略图控制的 "环境" 或 "渲染器"。
    """
    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        # 创建一个固定的高斯核用于金字塔构建
        kernel = self.create_gaussian_kernel(kernel_size=5, sigma=1.0)
        self.register_buffer('gaussian_kernel', kernel)

    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """创建一个2D高斯核"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= (kernel_size - 1) / 2.0
        g = coords**2
        g = -(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)
        kernel = g.exp().unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """使用高斯核进行下采样"""
        # 扩展核以匹配输入通道
        C = x.shape[1]
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1).to(device=x.device, dtype=x.dtype)
        # 卷积 + 步长为2的采样
        return F.conv2d(x, kernel, stride=2, padding=2, groups=C)

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        """使用转置卷积进行上采样"""
        C = x.shape[1]
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1).to(device=x.device, dtype=x.dtype) * 4
        return F.conv_transpose2d(x, kernel, stride=2, padding=2, output_padding=1, groups=C)

    def build_laplacian_pyramid(self, img: torch.Tensor):
        pyramid = []
        current_img = img
        for _ in range(self.num_levels):
            down = self.downsample(current_img)
            up = self.upsample(down)
            # 确保尺寸一致
            if up.shape[-2:] != current_img.shape[-2:]:
                up = F.interpolate(up, size=current_img.shape[-2:], mode='bilinear', align_corners=False)
            laplacian = current_img - up
            pyramid.append(laplacian)
            current_img = down
        pyramid.append(current_img) # 最后一层是高斯基层
        return pyramid

    def reconstruct_from_pyramid(self, pyramid):
        current_img = pyramid[-1] # 从高斯基层开始
        for level in range(self.num_levels - 1, -1, -1):
            up = self.upsample(current_img)
            laplacian = pyramid[level]
            # 确保尺寸一致
            if up.shape[-2:] != laplacian.shape[-2:]:
                 up = F.interpolate(up, size=laplacian.shape[-2:], mode='bilinear', align_corners=False)
            current_img = up + laplacian
        return current_img

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor, weight_map: torch.Tensor):
        """
        Args:
            img_a (torch.Tensor): 可见光图像 (B, C, H, W)
            img_b (torch.Tensor): 红外图像 (B, 1, H, W)
            weight_map (torch.Tensor): 采样得到的控制图 (B, 1, H, W)，值在[0,1]
        Returns:
            torch.Tensor: 融合后的图像
        """
        # 将单通道的红外图像B扩展以匹配A的通道数，便于融合
        if img_a.shape[1] != img_b.shape[1]:
            img_b_fused = img_b.repeat(1, img_a.shape[1], 1, 1)
        else:
            img_b_fused = img_b

        pyr_a = self.build_laplacian_pyramid(img_a)
        pyr_b = self.build_laplacian_pyramid(img_b_fused)
        pyr_fused = []

        for level in range(self.num_levels + 1):
            # 将 weight_map 缩放到当前金字塔层的尺寸
            h, w = pyr_a[level].shape[-2:]
            w_level = F.interpolate(weight_map, size=(h, w), mode='bilinear', align_corners=False)

            # 使用权重图进行加权融合
            # w_level 接近 1 -> 偏向 B (IR)
            # w_level 接近 0 -> 偏向 A (VIS)
            fused_level = (1 - w_level) * pyr_a[level] + w_level * pyr_b[level]
            pyr_fused.append(fused_level)

        return self.reconstruct_from_pyramid(pyr_fused)