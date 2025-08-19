# src/utils/metrics_utils.py
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 20.0 * math.log10(data_range / math.sqrt(m))

def ssim_img(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    # a, b: uint8 or float images, shape (H, W, C)
    return float(ssim(a, b, data_range=data_range, channel_axis=2))

def nccorr(a: np.ndarray, b: np.ndarray) -> float:
    # Normalized cross-correlation over all pixels/channels
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def uiqi(a: np.ndarray, b: np.ndarray) -> float:
    """
    Universal Image Quality Index (Wang & Bovik, 2002), computed globally.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # Flatten across HWC and compute on overall distribution
    a = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
    b = b.reshape(-1, b.shape[-1]) if b.ndim == 3 else b.reshape(-1, 1)

    # Average over channels
    vals = []
    for ch in range(a.shape[1]):
        ax = a[:, ch]
        bx = b[:, ch]
        mu_a = ax.mean()
        mu_b = bx.mean()
        var_a = ax.var()
        var_b = bx.var()
        cov = ((ax - mu_a) * (bx - mu_b)).mean()

        num = 4 * mu_a * mu_b * cov
        den = (mu_a**2 + mu_b**2) * (var_a + var_b)
        vals.append(0.0 if den == 0 else float(num / den))

    return float(np.mean(vals))

def compute_all(a_uint8: np.ndarray, b_uint8: np.ndarray) -> dict:
    """
    a_uint8: GT image, uint8 [0,255], HxWxC
    b_uint8: Pred image, uint8 [0,255], HxWxC
    """
    return {
        "PSNR": psnr(a_uint8, b_uint8, data_range=255.0),
        "SSIM": ssim_img(a_uint8, b_uint8, data_range=255.0),
        "UIQI": uiqi(a_uint8, b_uint8),
        "NCORR": nccorr(a_uint8, b_uint8),
        "MSE": mse(a_uint8, b_uint8),
    }
