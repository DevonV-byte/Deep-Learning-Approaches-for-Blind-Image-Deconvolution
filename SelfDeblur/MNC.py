import numpy as np
from scipy.signal import convolve2d

def calculate_mnc(k, k_gt):
    """
    Compute the Maximum of Normalized Convolution (MNC) between 
    two 2D kernels k and k_gt:

        MNC = max( (k ⊗ k_gt) / (||k||_2 * ||k_gt||_2) )

    where '⊗' is 2D convolution.
    """
    k = np.asarray(k, dtype=float)
    k_gt = np.asarray(k_gt, dtype=float)

    # 2D convolution (output shape: "full")
    conv_result = convolve2d(k, k_gt, mode='full')
    
    # L2 norms
    denom = np.linalg.norm(k) * np.linalg.norm(k_gt)
    
    # Return maximum of normalized convolution
    return np.max(conv_result) / denom
