from .Metric import *

# expose GPU batch implementations
from .MetricGPU import *

__all__ = [
    # CPU版本指标
    'EN_function',    # 熵(Entropy) - 评估图像信息丰富程度
    'MI_function',    # 互信息(Mutual Information) - 评估两幅图像之间的信息共享程度
    'SF_function',    # 空间频率(Spatial Frequency) - 评估图像细节和纹理特征
    'AG_function',    # 平均梯度(Average Gradient) - 评估图像的清晰度和边缘保持能力
    'SD_function',    # 标准差(Standard Deviation) - 评估图像对比度
    'CC_function',    # 相关系数(Correlation Coefficient) - 评估融合图像与源图像的相似度
    'SCD_function',   # 结构相关性差异(Structure Correlation Divergence) - 评估结构信息保持程度
    'VIF_function',   # 视觉信息保真度(Visual Information Fidelity) - 评估视觉信息保持质量
    'MSE_function',   # 均方误差(Mean Square Error) - 评估融合图像与源图像的差异
    'PSNR_function',  # 峰值信噪比(Peak Signal-to-Noise Ratio) - 评估图像质量
    'Qabf_function',  # 边缘信息保持质量(Edge Preservation Quality) - 评估边缘信息保持程度
    'Nabf_function',  # 非线性边缘保持质量(Nonlinear Edge Preservation Quality) - 评估非线性边缘保持程度
    'SSIM_function',  # 结构相似性(Structural Similarity) - 评估结构信息保持质量
    # 'MS_SSIM_function', # 多尺度结构相似性(Multi-Scale Structural Similarity) - 多尺度评估结构信息

    # GPU batch 版本（以 _batch 结尾）
    'EN_function_batch',
    'MI_function_batch',
    'SF_function_batch',
    'AG_function_batch',
    'SD_function_batch',
    'CC_function_batch',
    'SCD_function_batch',
    'VIF_function_batch',
    'MSE_function_batch',
    'PSNR_function_batch',
    'Qabf_function_batch',
    'Nabf_function_batch',
    'SSIM_function_batch',
    # 'MS_SSIM_function_batch', # 这个的GPU实现存在误差，先不使用
]
