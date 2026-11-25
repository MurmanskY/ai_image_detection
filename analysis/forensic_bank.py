"""固定高通取证滤波器组：独立于 Noiseprint++ 的 24 维残差表示。

三个固定线性滤波器（水平/垂直二阶差分、Laplacian），每个提取 8 维：
4 个分布统计、2 个方向自相关、2 个径向谱能量比例。取与 NP-RSR 相同的
三个 128x128 patch 并平均。该表示是 SRM-inspired 固定特征，不是完整 SRM
模型或已发表检测器的官方复现。
"""

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from analysis.sacr_p0 import patches_for_image

KERNELS = (
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2,
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2,
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 4,
)


def residual8(residual):
    x = residual[16:-16, 16:-16]
    flat = x.ravel()
    if flat.std() < 1e-12:
        stats = [float(flat.mean()), 0.0, 0.0, 0.0]
    else:
        stats = [float(flat.mean()), float(flat.std()), float(skew(flat)),
                 float(kurtosis(flat))]
    for axis in (0, 1):
        a = np.take(x, range(x.shape[axis] - 1), axis=axis)
        b = np.take(x, range(1, x.shape[axis]), axis=axis)
        if a.std() < 1e-12 or b.std() < 1e-12:
            stats.append(0.0)
        else:
            stats.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
    mag2 = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    low = radius < radius.max() / 3
    high = radius >= 2 * radius.max() / 3
    total = mag2.sum() + 1e-12
    stats.extend([float(mag2[low].sum() / total),
                  float(mag2[high].sum() / total)])
    return stats


def highpass24_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.concatenate([
        residual8(cv2.filter2D(gray, cv2.CV_32F, kernel))
        for kernel in KERNELS
    ]).astype(np.float32)


def highpass24_one(job):
    path, view = job
    values = [highpass24_patch(p) for p in patches_for_image(path, view)]
    return np.nan_to_num(np.mean(values, axis=0), nan=0.0,
                         posinf=0.0, neginf=0.0)


def extract_highpass24(paths, view, workers=8):
    jobs = [(p, view) for p in paths]
    if workers <= 1:
        return np.stack([highpass24_one(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(highpass24_one, jobs, chunksize=16))
    return np.stack(rows).astype(np.float32)


def ensure_highpass_cache(out_dir, key, paths, view, workers=8):
    os.makedirs(out_dir, exist_ok=True)
    safe = key.replace("/", "__").replace("::", "--")
    path = os.path.join(out_dir, f"hp24_{safe}_{view}.npy")
    if not os.path.exists(path):
        values = extract_highpass24(paths, view, workers=workers)
        np.save(path, values)
    return path
