from .Metric import *

# expose GPU batch implementations
from .MetricGPU import *

__all__ = [
    # CPU version metrics
    'EN_function',    # Entropy - evaluates the information richness of an image
    'MI_function',    # Mutual Information - evaluates the information sharing between two images
    'SF_function',    # Spatial Frequency - evaluates image details and texture features 
    'AG_function',    # Average Gradient - evaluates image sharpness and edge preservation capability
    'SD_function',    # Standard Deviation - evaluates image contrast
    'CC_function',    # Correlation Coefficient - evaluates similarity between fused image and source images
    'SCD_function',   # Structure Correlation Divergence - evaluates structural information preservation
    'VIF_function',   # Visual Information Fidelity - evaluates visual information preservation quality
    'MSE_function',   # Mean Square Error - evaluates differences between fused image and source images
    'PSNR_function',  # Peak Signal-to-Noise Ratio - evaluates image quality
    'Qabf_function',  # Edge Preservation Quality - evaluates edge information preservation
    'Nabf_function',  # Nonlinear Edge Preservation Quality - evaluates nonlinear edge preservation
    'SSIM_function',  # Structural Similarity - evaluates structural information preservation quality
    # 'MS_SSIM_function', # Multi-Scale Structural Similarity - multi-scale evaluation of structural information

    # GPU batch versions (ending with _batch)
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
    # 'MS_SSIM_function_batch', # Use ssim instead
]