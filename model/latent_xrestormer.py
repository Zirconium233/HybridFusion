import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 仅复用最底层的 TransformerBlock，避免金字塔下采样引入整除约束
try:
    from .xrestormer import TransformerBlock  # package 方式运行: python -m Image.model.latent_xrestormer
except ImportError:
    from model.xrestormer import TransformerBlock  # 脚本方式运行: python Image/model/latent_xrestormer.py


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    x: [B, C, H, W]
    返回: x_pad, pad (left, right, top, bottom)
    仅右/下 padding，便于后续裁剪恢复。
    """
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad = (0, pad_w, 0, pad_h)
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, pad, mode="replicate")
    return x, pad


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    _, _, h, w = x.shape
    left, right, top, bottom = pad
    h_out = h - (top + bottom)
    w_out = w - (left + right)
    return x[:, :, :h_out, :w_out]


class WindowSafeBlock(nn.Module):
    """
    对单个 TransformerBlock 做窗口对齐 padding，前后保持输入输出分辨率不变。
    """
    def __init__(self, block: TransformerBlock, window_size: int):
        super().__init__()
        self.block = block
        self.window_size = int(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, pad = pad_to_multiple(x, self.window_size)
        y = self.block(x_pad)
        y = unpad(y, pad)
        return y


class LatentXRestormerNoDown(nn.Module):
    """
    无下采样的潜空间 XRestormer：
    - 输入: A/B latent（C=latent_channels）
    - 将 A/B 各自编码到 dim，拼接后 1x1 融合到 dim，再堆叠 N 个 TransformerBlock（窗口注意力）
    - 每个 block 内部做 window 对齐 padding，避免整除限制
    - 输出: 融合 latent（C=latent_channels）
    """
    def __init__(
        self,
        latent_channels: int = 4,
        dim: int = 32,
        num_blocks: int = 6,
        window_size: int = 8,
        num_channel_heads: int = 2,
        num_spatial_heads: int = 4,
        spatial_dim_head: int = 16,
        ffn_expansion_factor: float = 2.66,
        overlap_ratio: float = 0.5,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        use_residual_to_A: bool = False,
    ):
        super().__init__()
        self.use_residual_to_A = use_residual_to_A
        self.window_size = int(window_size)

        # A/B 各自 stem
        self.stem_A = nn.Sequential(
            nn.Conv2d(latent_channels, dim, 3, padding=1, bias=bias),
            nn.GELU(),
        )
        self.stem_B = nn.Sequential(
            nn.Conv2d(latent_channels, dim, 3, padding=1, bias=bias),
            nn.GELU(),
        )

        # 融合降维
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=bias),
            nn.GELU(),
        )

        # N 个 TransformerBlock（无下采样），逐层窗口安全
        blocks: List[nn.Module] = []
        for _ in range(num_blocks):
            tb = TransformerBlock(
                dim=dim,
                window_size=self.window_size,
                overlap_ratio=overlap_ratio,
                num_channel_heads=num_channel_heads,
                num_spatial_heads=num_spatial_heads,
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            )
            blocks.append(WindowSafeBlock(tb, window_size=self.window_size))
        self.blocks = nn.Sequential(*blocks)

        # 输出头
        self.head = nn.Conv2d(dim, latent_channels, 3, padding=1, bias=True)

        # 可选 residual to A（与 use_shortcut 类似）
        if self.use_residual_to_A:
            self.resproj = nn.Identity() if latent_channels == latent_channels else nn.Conv2d(latent_channels, latent_channels, 1, bias=False)


    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: [B,C,H,W]（可任意分辨率）
        fa = self.stem_A(a)
        fb = self.stem_B(b)
        x = torch.cat([fa, fb], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        y = self.head(x)
        if self.use_residual_to_A:
            y = y + a  # 直接在 latent 空间做残差
        return y


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _test_once(model: nn.Module, device: torch.device, shape: Tuple[int, int], dtype=torch.bfloat16):
    h, w = shape
    c = 4
    a = torch.randn(1, c, h, w, device=device, dtype=dtype)
    b = torch.randn(1, c, h, w, device=device, dtype=dtype)
    t0 = time.time()
    y = model(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0
    print(f"in: ({h},{w}) -> out: {tuple(y.shape[-2:])}, time: {dt*1000:.2f} ms")


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LatentXRestormerNoDown(
        latent_channels=4,
        dim=32,                 # 调大到 48/64 可提升容量
        num_blocks=6,           # 叠加更多 block 增强能力
        window_size=8,
        num_channel_heads=2,
        num_spatial_heads=4,
        spatial_dim_head=16,
        ffn_expansion_factor=2.66,
        overlap_ratio=0.5,
        use_residual_to_A=False,
    ).to(device=device, dtype=torch.bfloat16).eval()

    params = count_params(model)
    print(f"LatentXRestormerNoDown params: {params/1e6:.3f} M")

    # 覆盖你提到的尺寸（latent 分辨率）
    sizes: List[Tuple[int, int]] = [
        (80, 60),   # W=80, H=64
        (60, 80),   # H/W 互换
        (160, 120),
        (120, 160),
        (128, 96),
        (96, 128),
        (128, 128),
        (256, 256),
        (32, 32),
    ]
    for s in sizes:
        _test_once(model, device, s, dtype=torch.bfloat16)


if __name__ == "__main__":
    main()