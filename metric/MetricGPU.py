import time
import math
import torch
import torch.nn.functional as F
import numpy as np


from .Metric import (
    EN_function as EN_cpu,
    SF_function as SF_cpu,
    SD_function as SD_cpu,
    PSNR_function as PSNR_cpu,
    MSE_function as MSE_cpu,
    VIF_function as VIF_cpu,
    CC_function as CC_cpu,
    SCD_function as SCD_cpu,
    Qabf_function as Qabf_cpu,
    Nabf_function as Nabf_cpu,
    MI_function as MI_cpu,
    AG_function as AG_cpu,
    SSIM_function as SSIM_cpu,
    MS_SSIM_function as MS_SSIM_cpu,
)


from . import ssim as ssim_module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def _to_tensor_batch(x, device=DEVICE, dtype=torch.float32):
    """
    Accepts x as:
      - numpy array: (H,W,C), (C,H,W), (B,H,W,C), (B,C,H,W)
      - torch tensor: same shapes
    Returns (B, C, H, W) float32 tensor on device
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x

    t = t.to(dtype=torch.float32)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)  
    elif t.dim() == 3:
        
        if t.shape[0] == 3:  
            t = t.unsqueeze(0)  
        else:
            
            t = t.permute(2, 0, 1).unsqueeze(0)  
    elif t.dim() == 4:
        
        if t.shape[-1] == 3 and t.shape[1] != 3:
            
            t = t.permute(0, 3, 1, 2)
    else:
        raise ValueError("Unsupported tensor shape for _to_tensor_batch: %s" % (t.shape,))

    return t.to(device=device, dtype=dtype)


def preprocess_to_gray_batch(x):
    t = _to_tensor_batch(x)  
    if t.shape[1] == 1:
        gray = t
    else:
        
        r = t[:, 0:1, :, :]
        g = t[:, 1:2, :, :]
        b = t[:, 2:3, :, :]
        
        gray_f = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        gray = torch.clamp(gray_f, 0.0, 255.0)
        gray = gray.to(torch.long).to(torch.float32)  
    return gray  

def EN_function_batch(x):
    """
    Consistent with CPU: Perform histogram statistics on pixels of all channels of the original image (without grayscale conversion), bins=256, range=[0,255]
    Returns: (B,)
    """
    t = _to_tensor_batch(x)  
    B = t.shape[0]
    vals = torch.clamp(t, 0.0, 255.0).to(torch.long)  
    vals = vals.reshape(B, -1)
    outs = []
    for i in range(B):
        cnt = torch.bincount(vals[i], minlength=256).to(torch.float32)
        prob = cnt / cnt.sum().clamp(min=1.0)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-7))
        outs.append(entropy)
    return torch.stack(outs)


def SF_function_batch(x):
    """
    Behavior is fully aligned with CPU(Bug level aligned. SF lower than MATLAB version, not used in paper):
    - Simulate uint8 differential wrapping in integer domain (mod 256)
    - Also simulate mod 256 overflow for uint8 squares in integer domain (i.e., (v*v) % 256)
    - Finally convert to high-precision float for mean/sqrt to reproduce numpy(uint8) path
    """
    t = _to_tensor_batch(x)  
    
    t_int = torch.clamp(t, 0.0, 255.0).to(torch.int32)

    
    rf_mod = torch.remainder(t_int[:, :, 1:, :] - t_int[:, :, :-1, :], 256)  
    cf_mod = torch.remainder(t_int[:, :, :, 1:] - t_int[:, :, :, :-1], 256)  

    
    rf_sq_mod = torch.remainder(rf_mod * rf_mod, 256)  
    cf_sq_mod = torch.remainder(cf_mod * cf_mod, 256)

    
    B = rf_sq_mod.shape[0]
    rf_sq_f = rf_sq_mod.to(torch.float64).reshape(B, -1)
    cf_sq_f = cf_sq_mod.to(torch.float64).reshape(B, -1)

    rf1 = torch.sqrt(rf_sq_f.mean(dim=1))
    cf1 = torch.sqrt(cf_sq_f.mean(dim=1))

    SF = torch.sqrt(rf1.pow(2) + cf1.pow(2)).to(torch.float32)
    return SF


def SD_function_batch(x):
    gray = preprocess_to_gray_batch(x)  
    B = gray.shape[0]
    
    vals = gray.view(B, -1)
    mu = vals.mean(dim=1, keepdim=True)
    SD = torch.sqrt(((vals - mu) ** 2).sum(dim=1) / vals.shape[1])
    return SD


def MSE_function_batch(A, B, F):
    A_t = preprocess_to_gray_batch(A) / 255.0  
    B_t = preprocess_to_gray_batch(B) / 255.0
    F_t = preprocess_to_gray_batch(F) / 255.0
    Bsize = F_t.shape[0]
    m = F_t.shape[2]
    n = F_t.shape[3]
    mse_af = ((F_t - A_t) ** 2).view(Bsize, -1).mean(dim=1)
    mse_bf = ((F_t - B_t) ** 2).view(Bsize, -1).mean(dim=1)
    MSE = 0.5 * mse_af + 0.5 * mse_bf
    return MSE


def PSNR_function_batch(A, B, F):
    MSE = MSE_function_batch(A, B, F)
    
    psnr = 20.0 * torch.log10(torch.tensor(255.0, device=MSE.device) / torch.sqrt(MSE.clamp(min=1e-12)))
    return psnr


def AG_function_batch(x):
    
    gray = preprocess_to_gray_batch(x)  
    B, C, H, W = gray.shape
    
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    
    if H >= 3:
        gy[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]) / 2.0
    else:
        gy[:, :, 1:-1, :] = 0.0
    if W >= 3:
        gx[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]) / 2.0
    else:
        gx[:, :, :, 1:-1] = 0.0
    
    gy[:, :, 0, :] = gray[:, :, 1, :] - gray[:, :, 0, :]
    gy[:, :, -1, :] = gray[:, :, -1, :] - gray[:, :, -2, :]
    gx[:, :, :, 0] = gray[:, :, :, 1] - gray[:, :, :, 0]
    gx[:, :, :, -1] = gray[:, :, :, -1] - gray[:, :, :, -2]
    s = torch.sqrt((gx ** 2 + gy ** 2) / 2.0)
    
    denom = max(1, (W - 1) * (H - 1))
    AG = s.view(B, -1).sum(dim=1) / float(denom)
    return AG


def corr2_batch(a, b):
    
    B = a.shape[0]
    a_v = a.view(B, -1)
    b_v = b.view(B, -1)
    a_m = a_v.mean(dim=1, keepdim=True)
    b_m = b_v.mean(dim=1, keepdim=True)
    num = ((a_v - a_m) * (b_v - b_m)).sum(dim=1)
    den = torch.sqrt(((a_v - a_m) ** 2).sum(dim=1) * ((b_v - b_m) ** 2).sum(dim=1)).clamp(min=1e-12)
    return num / den

def CC_function_batch(A, B, F):
    A_t = preprocess_to_gray_batch(A)
    B_t = preprocess_to_gray_batch(B)
    F_t = preprocess_to_gray_batch(F)
    rAF = corr2_batch(A_t, F_t)
    rBF = corr2_batch(B_t, F_t)
    return 0.5 * (rAF + rBF)


def SCD_function_batch(A, B, F):
    A_t = preprocess_to_gray_batch(A)
    B_t = preprocess_to_gray_batch(B)
    F_t = preprocess_to_gray_batch(F)
    r = corr2_batch(F_t - B_t, A_t) + corr2_batch(F_t - A_t, B_t)
    return r


def fspecial_gaussian_torch(shape, sigma, device=DEVICE):
    m = (shape[0] - 1.0) / 2.0
    n = (shape[1] - 1.0) / 2.0
    y = torch.arange(-m, m + 1, device=device, dtype=torch.float32).view(-1, 1)
    x = torch.arange(-n, n + 1, device=device, dtype=torch.float32).view(1, -1)
    h = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h = h * (h >= (torch.finfo(h.dtype).eps * h.max()))
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def vifp_mscale_batch(ref, dist):
    sigma_nsq = 2.0
    num = ref.new_zeros(ref.shape[0])
    den = ref.new_zeros(ref.shape[0])
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian_torch((N, N), N / 5.0, device=ref.device)  
        win_t = win.view(1, 1, N, N)
        if scale > 1:
            
            ref = F.conv2d(ref, win_t, padding=0)
            dist = F.conv2d(dist, win_t, padding=0)
            ref = ref[:, :, ::2, ::2]
            dist = dist[:, :, ::2, ::2]
        mu1 = F.conv2d(ref, win_t, padding=0)
        mu2 = F.conv2d(dist, win_t, padding=0)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(ref * ref, win_t, padding=0) - mu1_sq
        sigma2_sq = F.conv2d(dist * dist, win_t, padding=0) - mu2_sq
        sigma12 = F.conv2d(ref * dist, win_t, padding=0) - mu1_mu2
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        mask1 = sigma1_sq < 1e-10
        g = torch.where(mask1, torch.zeros_like(g), g)
        sv_sq = torch.where(mask1, sigma2_sq, sv_sq)
        sigma1_sq = torch.where(mask1, torch.zeros_like(sigma1_sq), sigma1_sq)

        mask2 = sigma2_sq < 1e-10
        g = torch.where(mask2, torch.zeros_like(g), g)
        sv_sq = torch.where(mask2, torch.zeros_like(sv_sq), sv_sq)

        sv_sq = torch.where(g < 0, sigma2_sq, sv_sq)
        g = torch.where(g < 0, torch.zeros_like(g), g)
        sv_sq = torch.clamp(sv_sq, min=1e-10)

        num = num + torch.sum(torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)), dim=[1, 2, 3])
        den = den + torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=[1, 2, 3])
    vifp = num / den
    return vifp


def VIF_function_batch(A, B, F):
    A_t = preprocess_to_gray_batch(A)
    B_t = preprocess_to_gray_batch(B)
    F_t = preprocess_to_gray_batch(F)
    vA = vifp_mscale_batch(A_t, F_t)
    vB = vifp_mscale_batch(B_t, F_t)
    return vA + vB



def get_Qabf_batch(pA, pB, pF):
    
    B = pA.shape[0]
    device = pA.device
    
    h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def getArray(img):
        
        SAx = F.conv2d(img, h3, padding=1)
        SAy = F.conv2d(img, h1, padding=1)
        gA = torch.sqrt(SAx * SAx + SAy * SAy)
        
        zero_mask = (SAx == 0)
        aA = torch.atan(torch.where(zero_mask, torch.ones_like(SAy) * 1.0, SAy / (SAx + 1e-12)))
        aA = torch.where(zero_mask, torch.ones_like(aA) * (math.pi / 2), aA)
        return gA, aA

    gA, aA = getArray(pA)
    gB, aB = getArray(pB)
    gF, aF = getArray(pF)

    
    Tg = 0.9994; kg = -15; Dg = 0.5
    Ta = 0.9879; ka = -22; Da = 0.8

    def getQabf(aA_, gA_, aF_, gF_):
        mask = (gA_ > gF_)
        
        
        equal_mask = (gA_ == gF_)
        GAF = torch.where(mask, gF_ / (gA_ + 1e-12), torch.where(equal_mask, gF_, gA_ / (gF_ + 1e-12)))
        AAF = 1 - torch.abs(aA_ - aF_) / (math.pi / 2)
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)
    deno = (gA + gB).sum(dim=[1, 2, 3])
    nume = (QAF * gA + QBF * gB).sum(dim=[1, 2, 3])
    output = nume / deno
    return output


def _per_extn_im_fn_torch(x, wsize):
    
    hwsize = (wsize - 1) // 2
    B, C, p, q = x.shape
    pad = wsize - 1
    out = x.new_zeros((B, C, p + pad, q + pad))
    out[:, :, hwsize: p + hwsize, hwsize: q + hwsize] = x
    
    
    
    if wsize - 1 == hwsize + 1:
        out[:, :, 0:hwsize, :] = out[:, :, 2:3, :].expand(-1, -1, hwsize, -1)
        out[:, :, p + hwsize: p + wsize - 1, :] = out[:, :, -3:-2, :].expand(-1, -1, hwsize, -1)
    out[:, :, :, 0:hwsize] = out[:, :, :, 2:3].expand(-1, -1, -1, hwsize)
    out[:, :, :, q + hwsize: q + wsize - 1] = out[:, :, :, -3:-2].expand(-1, -1, -1, hwsize)
    return out


def sobel_fn_torch(x):
    
    vtemp = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3) / 8.0
    htemp = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3) / 8.0
    a, b = htemp.shape[-2], htemp.shape[-1]
    x_ext = _per_extn_im_fn_torch(x, a)
    gv = F.conv2d(x_ext, vtemp, padding=0)
    gh = F.conv2d(x_ext, htemp, padding=0)
    return gv, gh


def get_Nabf_batch(I1, I2, f):
    
    Td = 2.0
    wt_min = 0.001
    P = 1
    Lg = 1.5
    Nrg = 0.9999
    kg = 19.0
    sigmag = 0.5
    Nra = 0.9995
    ka = 22.0
    sigmaa = 0.5

    xrcw = f
    x1 = I1
    x2 = I2

    gvA, ghA = sobel_fn_torch(x1)
    gA = torch.sqrt(ghA ** 2 + gvA ** 2)
    gvB, ghB = sobel_fn_torch(x2)
    gB = torch.sqrt(ghB ** 2 + gvB ** 2)
    gvF, ghF = sobel_fn_torch(xrcw)
    gF = torch.sqrt(ghF ** 2 + gvF ** 2)

    gAF = torch.zeros_like(gA)
    gBF = torch.zeros_like(gB)
    aA = torch.where((gvA == 0) & (ghA == 0), torch.zeros_like(gvA), torch.atan(gvA / (ghA + 1e-12)))
    aB = torch.where((gvB == 0) & (ghB == 0), torch.zeros_like(gvB), torch.atan(gvB / (ghB + 1e-12)))
    aF = torch.where((gvF == 0) & (ghF == 0), torch.zeros_like(gvF), torch.atan(gvF / (ghF + 1e-12)))

    maskAF1 = (gA == 0) | (gF == 0)
    maskAF2 = (gA > gF)
    gAF = torch.where(~maskAF1, torch.where(maskAF2, gF / (gA + 1e-12), gA / (gF + 1e-12)), gAF)
    maskBF1 = (gB == 0) | (gF == 0)
    maskBF2 = (gB > gF)
    gBF = torch.where(~maskBF1, torch.where(maskBF2, gF / (gB + 1e-12), gB / (gF + 1e-12)), gBF)

    aAF = torch.abs(torch.abs(aA - aF) - math.pi / 2) * 2 / math.pi
    aBF = torch.abs(torch.abs(aB - aF) - math.pi / 2) * 2 / math.pi

    QgAF = Nrg / (1 + torch.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + torch.exp(-ka * (aAF - sigmaa)))
    QAF = torch.sqrt(QgAF * QaAF)
    QgBF = Nrg / (1 + torch.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + torch.exp(-ka * (aBF - sigmaa)))
    QBF = torch.sqrt(QgBF * QaBF)

    wtA = torch.where(gA >= Td, (wt_min * torch.ones_like(gA)) * (gA ** Lg), torch.zeros_like(gA))
    wtB = torch.where(gB >= Td, (wt_min * torch.ones_like(gB)) * (gB ** Lg), torch.zeros_like(gB))

    wt_sum = (wtA + wtB).sum(dim=[1, 2, 3])

    QAF_wtsum = (QAF * wtA).sum(dim=[1, 2, 3]) / wt_sum
    QBF_wtsum = (QBF * wtB).sum(dim=[1, 2, 3]) / wt_sum
    QABF = QAF_wtsum + QBF_wtsum

    rr = torch.where(gF <= torch.min(gA, gB), torch.ones_like(gF), torch.zeros_like(gF))
    LABF = (rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)).sum(dim=[1, 2, 3]) / wt_sum

    na = torch.where((gF > gA) & (gF > gB), torch.ones_like(gF), torch.zeros_like(gF))
    NABF = (na * ((1 - QAF) * wtA + (1 - QBF) * wtB)).sum(dim=[1, 2, 3]) / wt_sum
    return NABF


def Qabf_function_batch(A, B, F):
    A_t = preprocess_to_gray_batch(A)
    B_t = preprocess_to_gray_batch(B)
    F_t = preprocess_to_gray_batch(F)
    return get_Qabf_batch(A_t, B_t, F_t)


def Nabf_function_batch(A, B, F):
    A_t = preprocess_to_gray_batch(A)
    B_t = preprocess_to_gray_batch(B)
    F_t = preprocess_to_gray_batch(F)
    return get_Nabf_batch(A_t, B_t, F_t)





def Hab_batch(im1, im2, gray_level=256):
    B = im1.shape[0]
    im1_inds = torch.clamp(im1, 0.0, 255.0).to(torch.long).view(B, -1)
    im2_inds = torch.clamp(im2, 0.0, 255.0).to(torch.long).view(B, -1)
    out = []
    for i in range(B):
        idx1 = im1_inds[i]
        idx2 = im2_inds[i]
        idx_flat = idx1 * gray_level + idx2
        cnt = torch.bincount(idx_flat, minlength=gray_level * gray_level).to(torch.float32)
        h = cnt.view(gray_level, gray_level)
        h = h / h.sum().clamp(min=1.0)
        im1_marg = h.sum(dim=0)
        im2_marg = h.sum(dim=1)
        
        H_x = torch.sum(im1_marg[im1_marg > 0] * torch.log2(im1_marg[im1_marg > 0]))
        H_y = torch.sum(im2_marg[im2_marg > 0] * torch.log2(im2_marg[im2_marg > 0]))
        H_xy = torch.sum(h[h > 0] * torch.log2(h[h > 0]))
        MI = H_xy - H_x - H_y
        out.append(MI)
    return torch.stack(out)


def MI_function_batch(A, B, F, gray_level=256):
    A_t = preprocess_to_gray_batch(A)
    B_t = preprocess_to_gray_batch(B)
    F_t = preprocess_to_gray_batch(F)
    MIA = Hab_batch(A_t, F_t, gray_level)
    MIB = Hab_batch(B_t, F_t, gray_level)
    return MIA + MIB


def _fspecial_gauss_1d_torch(size, sigma, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * (torch.tensor(sigma, dtype=dtype, device=device) ** 2)))
    g = g / g.sum()
    return g  

def _gaussian_filter_separable(X, g1d):
    """
    Perform 1D Gaussian convolution on (B,C,H,W) using 'valid' convolution (no padding)
    """
    B, C, H, W = X.shape
    k = g1d.numel()
    
    wv = g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)
    Yv = F.conv2d(X, wv, padding=0, groups=C)
    
    wh = g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)
    Y = F.conv2d(Yv, wh, padding=0, groups=C)
    return Y

def _ssim_torch(X, Y, data_range=255.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)

    X = X * 255.0
    Y = Y * 255.0

    K1, K2 = K
    device = X.device
    C = X.shape[1]
    g1d = _fspecial_gauss_1d_torch(win_size, win_sigma, device, dtype=torch.float64)
    
    
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter_separable(X, g1d)
    mu2 = _gaussian_filter_separable(Y, g1d)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter_separable(X * X, g1d) - mu1_sq
    sigma2_sq = _gaussian_filter_separable(Y * Y, g1d) - mu2_sq
    sigma12 = _gaussian_filter_separable(X * Y, g1d) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    B = X.shape[0]
    ssim_per_channel = ssim_map.view(B, C, -1).mean(-1)
    cs = cs_map.view(B, C, -1).mean(-1)
    return ssim_per_channel, cs

def ssim_batch(X, Y, data_range=255.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03), size_average=True):
    ssim_per_channel, _ = _ssim_torch(X, Y, data_range=data_range, win_size=win_size, win_sigma=win_sigma, K=K)
    if size_average:
        return ssim_per_channel.mean(dim=1).to(torch.float32)  
    else:
        return ssim_per_channel.to(torch.float32)  

def ms_ssim_batch(X, Y, data_range=255.0, win_size=11, win_sigma=1.5, weights=None, K=(0.01,0.03), size_average=True):
    
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64, device=X.device)
    levels = weights.shape[0]
    mcs = []
    X_i = X
    Y_i = Y
    for i in range(levels):
        ssim_per_channel, cs = _ssim_torch(X_i, Y_i, data_range=data_range, win_size=win_size, win_sigma=win_sigma, K=K)
        if i < levels - 1:
            mcs.append(torch.clamp(cs, min=0.0))
            
            X_i = F.avg_pool2d(X_i, kernel_size=2, stride=2)
            Y_i = F.avg_pool2d(Y_i, kernel_size=2, stride=2)
    ssim_per_channel = torch.clamp(ssim_per_channel, min=0.0)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  
    w = weights.view(-1, 1, 1)
    ms = torch.prod(mcs_and_ssim ** w, dim=0)  
    if size_average:
        return ms.mean(dim=1).to(torch.float32)  
    else:
        return ms.to(torch.float32)  

def SSIM_function_batch(A, B, F):
    
    A_t = preprocess_to_gray_batch(A)  
    F_t = preprocess_to_gray_batch(F)
    B_t = preprocess_to_gray_batch(B)
    ssim_AF = ssim_batch(A_t, F_t, size_average=True)  
    ssim_BF = ssim_batch(B_t, F_t, size_average=True)
    return ssim_AF + ssim_BF


def _run_test(): # for CPU vs GPU consistency and speedup
    import time
    torch.manual_seed(2)
    np.random.seed(2)
    B = 2
    H = 480
    W = 640
    C = 3
    
    vis = (np.random.rand(B, H, W, C) * 255).astype(np.uint8)
    ir = (np.random.rand(B, H, W, C) * 255).astype(np.uint8)
    fused = (np.random.rand(B, H, W, C) * 255).astype(np.uint8)

    
    device = DEVICE
    print("Using device:", device)

    
    start = time.time()
    cpu_results = {}
    cpu_results['EN_vis'] = [EN_cpu(vis[i]) for i in range(B)]
    cpu_results['SF_vis'] = [SF_cpu(vis[i]) for i in range(B)]
    cpu_results['SD_vis'] = [SD_cpu(vis[i]) for i in range(B)]
    cpu_results['PSNR'] = [PSNR_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['MSE'] = [MSE_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['VIF'] = [VIF_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['CC'] = [CC_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['SCD'] = [SCD_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['Qabf'] = [Qabf_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['Nabf'] = [Nabf_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['MI'] = [MI_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_results['AG'] = [AG_cpu(vis[i]) for i in range(B)]
    cpu_results['SSIM'] = [SSIM_cpu(vis[i], ir[i], fused[i]) for i in range(B)]
    cpu_time = time.time() - start

    
    start = time.time()
    vis_t = torch.tensor(vis, dtype=torch.float32, device=device)
    ir_t = torch.tensor(ir, dtype=torch.float32, device=device)
    fused_t = torch.tensor(fused, dtype=torch.float32, device=device)

    gpu_results = {}
    gpu_results['EN_vis'] = EN_function_batch(vis_t).cpu().numpy()
    gpu_results['SF_vis'] = SF_function_batch(vis_t).cpu().numpy()
    gpu_results['SD_vis'] = SD_function_batch(vis_t).cpu().numpy()
    gpu_results['PSNR'] = PSNR_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['MSE'] = MSE_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['VIF'] = VIF_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['CC'] = CC_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['SCD'] = SCD_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['Qabf'] = Qabf_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['Nabf'] = Nabf_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['MI'] = MI_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    gpu_results['AG'] = AG_function_batch(vis_t).cpu().numpy()
    
    gpu_results['SSIM'] = SSIM_function_batch(vis_t, ir_t, fused_t).cpu().numpy()
    
    gpu_time = time.time() - start

    
    tol = 1e-4
    all_ok = True
    diffs = {}
    keys = list(cpu_results.keys())
    for k in keys:
        cpu0 = cpu_results[k][0]
        gpu0 = float(gpu_results[k][0])
        diff = abs(cpu0 - gpu0)
        diffs[k] = diff
        if diff > tol:
            all_ok = False

    print("CPU time (full batch loop): %.4f s" % cpu_time)
    print("GPU time (batch): %.4f s" % gpu_time)
    print("Speedup (CPU/GPU): %.2f x" % (cpu_time / (gpu_time + 1e-12)))
    print("Per-metric diffs for sample 0 (tol=%g):" % tol)
    for k in keys:
        print(" %s: diff=%.6e" % (k, diffs[k]))
    print("All metrics within tol:", all_ok)


if __name__ == "__main__":
    _run_test()
    result = \
    '''
    CPU time (full batch loop): 4.4701 s
    GPU time (batch): 0.5856 s
    Speedup (CPU/GPU): 7.63 x
    Per-metric diffs for sample 0 (tol=1e-05):
    EN_vis: diff=3.153185e-07
    SF_vis: diff=1.845950e-08
    SD_vis: diff=1.331919e-06
    PSNR: diff=6.183915e-08
    MSE: diff=2.574472e-08
    VIF: diff=1.182285e-07
    CC: diff=6.855539e-08
    SCD: diff=1.576207e-07
    Qabf: diff=2.494915e-08
    Nabf: diff=5.410565e-08
    MI: diff=1.079525e-05
    AG: diff=2.934190e-06
    SSIM: diff=1.037610e-06
    All metrics within tol: True
    '''