import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PolicyNet(nn.Module):
    """
    一个U-Net结构的策略网络。
    输入 (VIS, IR) 图像，输出控制图的均值和对数方差。
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 2, bilinear: bool = True):
        super(PolicyNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, vis: torch.Tensor, ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([vis, ir], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        
        
        
        mu = torch.sigmoid(logits[:, 0:1, :, :])
        logvar = logits[:, 1:2, :, :]
        
        return mu, logvar

if __name__ == "__main__":
    import time

    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    def to_human(n: float) -> str:
        if n >= 1e9:
            return f"{n/1e9:.3f}G"
        if n >= 1e6:
            return f"{n/1e6:.3f}M"
        if n >= 1e3:
            return f"{n/1e3:.3f}K"
        return str(int(n))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyNet().to(device).eval()

    
    N, H, W = 16, 480, 640
    vis = torch.randn(N, 3, H, W, device=device)
    ir = torch.randn(N, 1, H, W, device=device)

    
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        mu, logvar = model(vis, ir)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000.0

    
    params = count_params(model)

    
    flops_total = None
    backend = None
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_total = FlopCountAnalysis(model, (vis, ir)).total()
        backend = "fvcore(FLOPs)"
    except Exception:
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(vis, ir), verbose=False)
            flops_total = macs  
            backend = "thop(MACs)"
        except Exception:
            backend = "none"

    print(f"Device: {device}")
    print(f"Input: VIS {tuple(vis.shape)}, IR {tuple(ir.shape)}")
    print(f"Output: mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}")
    print(f"Params: {params} ({to_human(params)} params)")
    if flops_total is not None:
        print(f"FLOPs[{backend}]: {flops_total:.0f} ({to_human(flops_total)} ops)")
    else:
        print("FLOPs: fvcore/thop not installed, pip install fvcore thop")
    print(f"Forward time: {dt_ms:.2f} ms")