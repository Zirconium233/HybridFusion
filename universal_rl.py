import torch
import torch.nn as nn
from model.text_xrestormer import XRestormer # no text version 
import numpy as np
from metric.MetricGPU import VIF_function_batch, SSIM_function_batch, Qabf_function_batch

def load_model(checkpoint_path):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path)
    model = XRestormer()
    model.load_state_dict(checkpoint)
    return model

def fspecial_gaussian(shape, sigma):
    """生成二维高斯权重"""
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def inference(model, vi, ir, text, window_size=128, stride=64):
    """使用滑动窗口进行推理，小模型在128*128训练，所以需要滑动窗口的处理"""
    model.eval()
    B, C, H, W = vi.shape
    assert B == 1, "测试时batch size必须为1"

    # 创建输出tensor和权重mask
    fusion_result = torch.zeros_like(vi)
    weight_mask = torch.zeros_like(vi)
    
    # 创建高斯权重
    gaussian_weights = torch.from_numpy(
        np.tile(fspecial_gaussian((window_size, window_size), window_size/4), (C,1,1))
    ).float().to(vi.device)
    with torch.no_grad():
        # 滑动窗口处理
        for h in range(0, H-window_size+1, stride):
            for w in range(0, W-window_size+1, stride):
                vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                
                fusion_patch = model(vi_patch, ir_patch)
                
                fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
        
        # 处理边缘区域
        if H % stride != 0:
            h = H - window_size
            for w in range(0, W-window_size+1, stride):
                vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                
                fusion_patch = model(vi_patch, ir_patch)
                
                fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
        
        if W % stride != 0:
            w = W - window_size
            for h in range(0, H-window_size+1, stride):
                vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                
                fusion_patch = model(vi_patch, ir_patch)
                
                fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
    
        # 加权平均得到最终结果
        fusion_result = fusion_result / (weight_mask + 1e-6)

    return fusion_result

# placeholder
class ModelWrapperWithProb(nn.Module):
    """
    目标是让确定性的输出变成随机的输出，predict的可以当做均值，往上面加方差
    """
    def __init__(self, model: XRestormer, args):
        pass
    def forward(self, args):
        pass
# placeholder
def reward_func(vi, ir, fused, type='auto'):
    """
    计算奖励函数，模型应该可以对任意黑盒奖励进行RL优化，当前考虑先用vif，qabf，ssim等metrics作为优化指标
    vi, ir, fused格式是tensor，注意无监督任务没有label，(B, C, H, W)，如果ir的C==1需要repeat到3
    上面XRestormer model默认数据范围是[0, 1]， type支持 [-1,1] [0,1] [0,255] 3种格式，可以根据实际数据推断
    """
    # process data A B F
    pass
    # metric
    vif = VIF_function_batch(A, B, F) # 返回是tensor
    Qabf = Qabf_function_batch(A, B, F)
    ssim = SSIM_function_batch(A, B, F)
    return (vif + Qabf * 1.5 + ssim) / 3

# 这个后面再实现，先pass
def reward_func_llm(args):
    pass

# 当前直接用生成一个batch，然后算平均差异，因为有随机，所以可以被rl
def calc_value():
    pass

def train_rl():
    pass