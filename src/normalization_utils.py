import numpy as np

# -------------------------------
# Normalization Functions
# -------------------------------
def minmax_normalize(vertices):
    """Performs Min–Max normalization on 3D vertex coordinates."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    denom = np.where(vmax - vmin == 0, 1, vmax - vmin)
    normalized = (vertices - vmin) / denom
    return normalized, vmin, vmax


def unit_sphere_normalize(vertices):
    """Centers the mesh and scales it to fit within a unit sphere."""
    centroid = vertices.mean(axis=0)
    shifted = vertices - centroid
    max_dist = np.linalg.norm(shifted, axis=1).max()
    if max_dist == 0:
        max_dist = 1.0
    normalized = shifted / max_dist
    return normalized, centroid, max_dist


def zscore_normalize(vertices):
    """(Optional) Z-Score normalization for statistical scaling."""
    mean = vertices.mean(axis=0)
    std = vertices.std(axis=0)
    std[std == 0] = 1.0
    normalized = (vertices - mean) / std
    return normalized, mean, std


# -------------------------------
# Quantization Functions
# -------------------------------
def quantize(normalized, bins=1024):
    """Quantizes normalized coordinates into discrete bins."""
    q = np.floor(normalized * (bins - 1) + 0.5).astype(int)
    return np.clip(q, 0, bins - 1)


def dequantize(q, bins=1024):
    """Converts quantized bins back to normalized floating-point values."""
    return q.astype(float) / (bins - 1)


# -------------------------------
# Error Measurement
# -------------------------------
def compute_errors(original, reconstructed):
    """Computes per-axis MSE and MAE."""
    diff = original - reconstructed
    mse = np.mean(diff ** 2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)
    return mse, mae
