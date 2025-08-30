import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def cc(img1, img2):
    # 使 eps 与输入 dtype 对齐，避免混合精度时类型提升
    eps = torch.finfo(img1.dtype).eps
    """Correlation coefficient for (N, C, H, W) tensors in [0.,1.] or arbitrary range."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc_val = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc_val = torch.clamp(cc_val, -1., 1.)
    return cc_val.mean()


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        # 若通道不足3，直接返回0损失，避免YCBCR定义不明导致错误
        if image_visible.shape[1] < 3 or image_fused.shape[1] < 3:
            return torch.tensor(0.0, device=image_visible.device, dtype=image_visible.dtype)

        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image


class L_Intensity_Max_RGB(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, max_mode="l1"):
        # 支持任意通道数，但要求三者通道一致
        assert image_visible.shape[1] == image_infrared.shape[1] == image_fused.shape[1], \
            "image_visible, image_infrared, image_fused 的通道数需一致"
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        # mask 与输入 dtype 对齐，避免混合精度下类型提升
        mask = (gray_infrared > gray_visible).to(dtype=image_visible.dtype)

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        if max_mode == "l1":
            loss_intensity = F.l1_loss(fused_image, image_fused)
        else:
            loss_intensity = F.mse_loss(fused_image, image_fused)
        return loss_intensity


class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            loss_intensity = (F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused)) / 2
        else:
            loss_intensity = (F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused)) / 2
        return loss_intensity


class GradientMaxLoss(nn.Module):
    """边缘/梯度最大保留损失（对x、y方向分别对齐到两者梯度最大值）"""
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        # 使用register_buffer保证自动跟随到正确设备
        sobel_x = torch.FloatTensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        grad_A_x, grad_A_y = self.gradient(image_A)
        grad_B_x, grad_B_y = self.gradient(image_B)
        grad_F_x, grad_F_y = self.gradient(image_fuse)
        loss = F.l1_loss(grad_F_x, torch.max(grad_A_x, grad_B_x)) + \
               F.l1_loss(grad_F_y, torch.max(grad_A_y, grad_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        # 卷积核按输入 dtype 对齐
        gradient_x = F.conv2d(image, self.sobel_x.to(dtype=image.dtype, device=image.device), padding=0)
        gradient_y = F.conv2d(image, self.sobel_y.to(dtype=image.dtype, device=image.device), padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)


class L_Grad(nn.Module):
    """可选：合并x/y梯度幅值的梯度损失（未在FusionLoss中默认使用）"""
    def __init__(self):
        super(L_Grad, self).__init__()
        sobel_x = torch.FloatTensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)
        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        # 卷积核按输入 dtype 对齐
        gradient_x = F.conv2d(image, self.sobel_x.to(dtype=image.dtype, device=image.device), padding=0)
        gradient_y = F.conv2d(image, self.sobel_y.to(dtype=image.dtype, device=image.device), padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)

    def tensor_RGB2GRAY(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


def gaussian(window_size, sigma, *, dtype=torch.float32, device=None):
    # 新增 dtype/device 以便与输入对齐
    vals = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.tensor(vals, dtype=dtype, device=device)
    return gauss / gauss.sum()


def create_window(window_size, channel=1, *, dtype=torch.float32, device=None):
    # 新增 dtype/device 以便与输入对齐
    _1D_window = gaussian(window_size, 1.5, dtype=dtype, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # 支持数据范围自适应
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        device = img1.device
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel, dtype=img1.dtype, device=device)
    else:
        # 使传入窗口与输入 dtype/device 对齐
        window = window.to(device=img1.device, dtype=img1.dtype)

    # C1/C2 与输入 dtype/device 对齐
    C1 = torch.tensor((0.01 * L) ** 2, dtype=img1.dtype, device=img1.device)
    C2 = torch.tensor((0.03 * L) ** 2, dtype=img1.dtype, device=img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    # 返回 1 - SSIM 作为loss
    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        # 初始以 float32 存储，前向时会cast到输入的 dtype/device
        window = create_window(window_size, dtype=torch.float32, device=None)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            # 若img1为单通道而img2为多通道，将img1复制到3通道以对齐
            img1 = torch.cat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.window.size(0):
            window = self.window.to(device=img1.device, dtype=img1.dtype)
        else:
            window = create_window(self.window_size, channel, dtype=img1.dtype, device=img1.device)
            # 更新buffer以适配新的通道数（以当前dtype存储）
            self.register_buffer('window', window)

        return ssim(img1, img2, window=window, window_size=self.window_size)


def structure_loss(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # 可选结构一致性项（当前未在FusionLoss中使用）
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1 = F.conv2d(img1, window, padding=padd, groups=channel) - mu1
    sigma2 = F.conv2d(img2, window, padding=padd, groups=channel) - mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C2 = (0.03 * L) ** 2
    loss = (2 * sigma12 + C2) / (2 * sigma1 * sigma2 + C2)

    if size_average:
        ret = loss.mean()
    else:
        ret = loss.mean(1).mean(1).mean(1)
    if full:
        return 1 - ret
    return ret


def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 1e-4))
    return grad_norm


class FusionLoss(nn.Module):
    """
    融合损失（无文本/任务依赖，完全批量并行）
    组成：
      - SSIM 可见->融合 + 赤外灰度->融合灰度
      - 强度最大（逐像素从A/B中选择更亮者）
      - 强度一致性（与可见/赤外灰度一致）
      - 颜色一致性（Cb/Cr一致）
      - 梯度最大（基于Sobel的边缘对齐）
    默认权重来自你提供配置的 default 项：
      max_ratio=24, consist_ratio=40, grad_ratio(原text_ratio)=48, ssim_ratio=2, color_ratio=12, ir_compose=1
    """
    def __init__(
        self,
        max_ratio: float = 10.0,
        consist_ratio: float = 2.0,
        grad_ratio: float = 40.0,       # 原实现的 text_ratio（实为texture/gradient项）
        ssim_ir_ratio: float = 1.0,
        ssim_ratio: float = 1.0,
        ir_compose: float = 2.0,
        color_ratio: float = 2.0,
        max_mode: str = "l1",
        consist_mode: str = "l1",
        ssim_window_size: int = 48
    ):
        super().__init__()
        # 组件
        self.loss_ssim = L_SSIM(window_size=ssim_window_size)
        self.loss_gradmax = GradientMaxLoss()
        self.loss_max = L_Intensity_Max_RGB()
        self.loss_consist = L_Intensity_Consist()
        self.loss_color = L_color()

        # 权重与配置
        self.max_ratio = float(max_ratio)
        self.consist_ratio = float(consist_ratio)
        self.grad_ratio = float(grad_ratio)
        self.ssim_ir_ratio = float(ssim_ir_ratio)
        self.ssim_ratio = float(ssim_ratio)
        self.ir_compose = float(ir_compose)
        self.color_ratio = float(color_ratio)
        self.max_mode = max_mode
        self.consist_mode = consist_mode

    @staticmethod
    def rgb2gray(image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

    def forward(self, image_A: torch.Tensor, image_B: torch.Tensor, image_fused: torch.Tensor) -> torch.Tensor:
        # 并行批次处理，无显式循环
        image_A_gray = self.rgb2gray(image_A)
        image_B_gray = self.rgb2gray(image_B)
        image_F_gray = self.rgb2gray(image_fused)

        # SSIM (可见RGB vs 融合RGB) + (赤外Gray vs 融合Gray)
        loss_ssim = self.ssim_ratio * (
            self.loss_ssim(image_A, image_fused) +
            self.ssim_ir_ratio * self.loss_ssim(image_B_gray, image_F_gray)
        )

        # 强度最大项（像素级从A/B选最大）与一致性项（灰度）
        loss_max = self.max_ratio * self.loss_max(image_A, image_B, image_fused, self.max_mode)
        loss_consist = self.consist_ratio * self.loss_consist(image_A_gray, image_B_gray, image_F_gray, self.ir_compose, self.consist_mode)

        # 颜色项（仅当可见与融合为3通道时有效，否则为0）
        loss_color = self.color_ratio * self.loss_color(image_A, image_fused)

        # 梯度最大项（基于灰度）
        loss_grad = self.grad_ratio * self.loss_gradmax(image_A_gray, image_B_gray, image_F_gray)

        total = loss_ssim + loss_max + loss_consist + loss_color + loss_grad
        return total


if __name__ == "__main__":
    # 简单批量测试：128x128, 640x480, 1024x768
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    loss_fn = FusionLoss().to(device=device, dtype=dtype)

    def run_case(n, c, h, w):
        image_A = torch.rand(n, c, h, w, device=device, dtype=dtype)
        image_B = torch.rand(n, c, h, w, device=device, dtype=dtype)
        image_F = torch.rand(n, c, h, w, device=device, dtype=dtype)
        loss = loss_fn(image_A, image_B, image_F)
        print(f"Batch={n}, C={c}, H={h}, W={w} -> loss={loss.item():.6f}")

    # 设定批量大小和通道数（3通道）
    run_case(n=4, c=3, h=128, w=128)
    run_case(n=2, c=3, h=480, w=640)     # 640x480
    run_case(n=2, c=3, h=768, w=1024)    # 1024x768