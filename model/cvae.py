import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
# 新增：一个标准卷积块用于首层 stem，避免 depthwise 在大分辨率上的索引限制
class StandardConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.SiLU(inplace=True)

        # 轻量残差（仅在尺寸与通道一致时使用）
        self.use_skip = (stride == 1 and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        if self.use_skip:
            x = x + identity
        return x


class DepthwiseSeparableConv(nn.Module):
    """DW + PW，减少参数量和算量."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, act: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """两次DW+PW叠加."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid = out_ch
        self.conv1 = DepthwiseSeparableConv(in_ch, mid, kernel_size=3, stride=stride, act=True)
        self.conv2 = DepthwiseSeparableConv(mid, out_ch, kernel_size=3, stride=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class Encoder(nn.Module):
    """
    条件编码器：输入 concat([A(3c), B(1c)]) -> 下采样到 1/8 -> 输出均值与对数方差 (mu, logvar).
    """
    def __init__(self, in_ch: int = 4, base_chs: Tuple[int, int, int] = (32, 64, 96), z_ch: int = 8):
        super().__init__()
        c1, c2, c3 = base_chs
        # 首层使用标准卷积块，避免大分辨率 + depthwise 在 bf16 下触发 32-bit 索引限制
        self.stem = StandardConvBlock(in_ch, c1, stride=1)       # H, W
        self.down1 = ConvBlock(c1, c2, stride=2)         # H/2, W/2
        self.down2 = ConvBlock(c2, c3, stride=2)         # H/4, W/4
        self.down3 = ConvBlock(c3, c3, stride=2)         # H/8, W/8
        self.bottleneck = ConvBlock(c3, c3, stride=1)    # H/8, W/8
        # 头部用 DW+PW 产生 mu/logvar
        self.to_mu = DepthwiseSeparableConv(c3, z_ch, kernel_size=3, stride=1, act=False)
        self.to_logvar = DepthwiseSeparableConv(c3, z_ch, kernel_size=3, stride=1, act=False)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([A, B], dim=1)  # shape: [B, 4, H, W]
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.bottleneck(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        # Clamp logvar 以避免数值不稳定
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        return mu, logvar


class CondPyramid(nn.Module):
    """
    条件特征金字塔（来自 B 的条件，极简实现）:
      - 先以 1x1 将 B:1c 提到 cond_ch，再构造四个尺度: 1/8, 1/4, 1/2, 1/1
    """
    def __init__(self, cond_ch: int = 8):
        super().__init__()
        self.embed = nn.Conv2d(1, cond_ch, kernel_size=1, bias=False)

    def forward(self, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        # B: [B,1,H,W]
        b0 = self.embed(B)                          # H, W
        b1 = F.avg_pool2d(b0, kernel_size=2, stride=2)   # H/2, W/2
        b2 = F.avg_pool2d(b1, kernel_size=2, stride=2)   # H/4, W/4
        b3 = F.avg_pool2d(b2, kernel_size=2, stride=2)   # H/8, W/8
        return {"r1": b0, "r2": b1, "r4": b2, "r8": b3}


class Decoder(nn.Module):
    """
    条件解码器：z 与 B 的条件特征逐级拼接并上采样，恢复到 3 通道图像.
    解码通道少，降低显存占用： [64, 48, 24, 8]
    """
    def __init__(self, z_ch: int = 8, cond_ch: int = 8, dec_chs: Tuple[int, int, int, int] = (64, 48, 24, 8)):
        super().__init__()
        d8, d4, d2, d1 = dec_chs
        # r8: [z(8) + cond(8)] -> 64
        self.r8 = ConvBlock(z_ch + cond_ch, d8, stride=1)
        # r4: up + cond -> 48
        self.r4_conv = ConvBlock(d8 + cond_ch, d4, stride=1)
        # r2:
        self.r2_conv = ConvBlock(d4 + cond_ch, d2, stride=1)
        # r1:
        self.r1_conv = ConvBlock(d2 + cond_ch, d1, stride=1)
        # out
        self.out = nn.Conv2d(d1, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, z: torch.Tensor, cond_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # r8
        x = torch.cat([z, cond_feats["r8"]], dim=1)
        x = self.r8(x)
        # r4
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, cond_feats["r4"]], dim=1)
        x = self.r4_conv(x)
        # r2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, cond_feats["r2"]], dim=1)
        x = self.r2_conv(x)
        # r1
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, cond_feats["r1"]], dim=1)
        x = self.r1_conv(x)
        # 输出 -1..1
        y = torch.tanh(self.out(x))
        return y


class ConditionalVAE(nn.Module):
    """
    轻量级图像融合条件VAE：
      - encode(A, cond=B) -> mu, logvar
      - reparameterize(mu, logvar) -> z
      - decode(z, cond=B) -> F_hat
      - forward(A, B) -> F_hat, mu, logvar, z
    设计目标：
      - 参数量 << 10M
      - 大分辨率批次前向可承受（尽量早下采样、解码通道少）
    """
    def __init__(
        self,
        in_ch: int = 4,               # A(3) + B(1)
        base_chs: Tuple[int, int, int] = (32, 64, 96),
        z_ch: int = 8,
        cond_ch: int = 8,
        dec_chs: Tuple[int, int, int, int] = (64, 48, 24, 8),
    ):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, base_chs=base_chs, z_ch=z_ch)
        self.cond_pyr = CondPyramid(cond_ch=cond_ch)
        self.decoder = Decoder(z_ch=z_ch, cond_ch=cond_ch, dec_chs=dec_chs)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        # 使用与 mu 相同的 dtype/device
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        # KL(N(mu, sigma) || N(0,1))
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == "mean":
            return kld.mean()
        elif reduction == "sum":
            return kld.sum()
        else:
            return kld

    def encode(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(A, B)

    def decode(self, z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        cond_feats = self.cond_pyr(B)
        return self.decoder(z, cond_feats)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(A, B)
        z = self.reparameterize(mu, logvar)
        # 直接在 z 的 1/8 分辨率上解码（条件金字塔内部自带 1/8,1/4,1/2,1/1）
        F_hat = self.decode(z, B)
        return F_hat, mu, logvar, z


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 前向/显存测试：batch=16, res=1024x768
    # 关闭 benchmark 避免选到不稳定的算法；使用 channels_last 更友好
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    # 若可用CUDA，默认使用 bf16 以减轻显存
    dtype = torch.bfloat16 if (device.type == "cuda") else torch.float32

    # 实例化模型（保持极轻量）
    model = ConditionalVAE(
        in_ch=4,
        # 提升模型容量至 ~1.5M 参数（仍 < 10M）
        base_chs=(192, 256, 384),
        z_ch=32,
        cond_ch=64,
        dec_chs=(384, 256, 128, 96),
    ).to(device=device, dtype=dtype).to(memory_format=torch.channels_last)

    n_params = count_parameters(model)
    print(f"[Model] ConditionalVAE parameters: {n_params/1e6:.3f} M ({n_params} params)")
    assert n_params < 10_000_000, "参数量超过 10M，请调低通道数。"

    BATCH, H, W = 16, 768, 1024  # 1024x768（W x H）
    # 构造随机输入（范围 [-1,1]）
    A = (torch.rand(BATCH, 3, H, W, device=device, dtype=dtype) * 2.0 - 1.0).contiguous(memory_format=torch.channels_last)
    B = (torch.rand(BATCH, 1, H, W, device=device, dtype=dtype) * 2.0 - 1.0).contiguous(memory_format=torch.channels_last)

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        # 仅编码测试（更轻）：检查能否跑通且不过度占用显存
        mu, logvar = model.encode(A, B)
        print(f"[Encode] mu shape: {tuple(mu.shape)}, logvar shape: {tuple(logvar.shape)}")

        # 完整前向（encode->reparam->decode）
        F_hat, mu2, logvar2, z = model(A, B)
        print(f"[Forward] F_hat shape: {tuple(F_hat.shape)}, z shape: {tuple(z.shape)}")

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[CUDA] Peak memory allocated: {peak_mem:.2f} MB (dtype={str(dtype).split('.')[-1]})")