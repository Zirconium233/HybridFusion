import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LaplacianPyramidFusion(nn.Module):
    """
    A fixed, non-neural network Laplacian Pyramid fusion module.
    It acts as an "environment" or "renderer" controlled by a policy map.
    """
    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        # Create a fixed Gaussian kernel for pyramid construction
        kernel = self.create_gaussian_kernel(kernel_size=5, sigma=1.0)
        self.register_buffer('gaussian_kernel', kernel)

    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Creates a 2D Gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= (kernel_size - 1) / 2.0
        g = coords**2
        g = -(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)
        kernel = g.exp().unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples using the Gaussian kernel"""
        # Expand kernel to match input channels
        C = x.shape[1]
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1).to(device=x.device, dtype=x.dtype)
        # Convolution + stride-2 sampling
        return F.conv2d(x, kernel, stride=2, padding=2, groups=C)

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples using transposed convolution"""
        C = x.shape[1]
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1).to(device=x.device, dtype=x.dtype) * 4
        return F.conv_transpose2d(x, kernel, stride=2, padding=2, output_padding=1, groups=C)

    def build_laplacian_pyramid(self, img: torch.Tensor):
        pyramid = []
        current_img = img
        for _ in range(self.num_levels):
            down = self.downsample(current_img)
            up = self.upsample(down)
            # Ensure size consistency
            if up.shape[-2:] != current_img.shape[-2:]:
                up = F.interpolate(up, size=current_img.shape[-2:], mode='bilinear', align_corners=False)
            laplacian = current_img - up
            pyramid.append(laplacian)
            current_img = down
        pyramid.append(current_img) # The last level is the Gaussian base
        return pyramid

    def reconstruct_from_pyramid(self, pyramid):
        current_img = pyramid[-1] # Start from the Gaussian base
        for level in range(self.num_levels - 1, -1, -1):
            up = self.upsample(current_img)
            laplacian = pyramid[level]
            # Ensure size consistency
            if up.shape[-2:] != laplacian.shape[-2:]:
                up = F.interpolate(up, size=laplacian.shape[-2:], mode='bilinear', align_corners=False)
            current_img = up + laplacian
        return current_img

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor, weight_map: torch.Tensor):
        """
        Args:
            img_a (torch.Tensor): Visible image (B, C, H, W)
            img_b (torch.Tensor): Infrared image (B, 1, H, W)
            weight_map (torch.Tensor): Sampled control map (B, 1, H, W), values in [0,1]
        Returns:
            torch.Tensor: The fused image
        """
        # Expand single-channel infrared image B to match the channels of A for fusion
        if img_a.shape[1] != img_b.shape[1]:
            img_b_fused = img_b.repeat(1, img_a.shape[1], 1, 1)
        else:
            img_b_fused = img_b

        pyr_a = self.build_laplacian_pyramid(img_a)
        pyr_b = self.build_laplacian_pyramid(img_b_fused)
        pyr_fused = []

        for level in range(self.num_levels + 1):
            # Scale the weight_map to the size of the current pyramid level
            h, w = pyr_a[level].shape[-2:]
            w_level = F.interpolate(weight_map, size=(h, w), mode='bilinear', align_corners=False)

            # Perform weighted fusion using the weight map
            # w_level close to 1 -> favors B (IR)
            # w_level close to 0 -> favors A (VIS)
            fused_level = (1 - w_level) * pyr_a[level] + w_level * pyr_b[level]
            pyr_fused.append(fused_level)

        return self.reconstruct_from_pyramid(pyr_fused)