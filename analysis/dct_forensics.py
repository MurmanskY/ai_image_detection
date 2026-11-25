"""固定 DCT 频域取证表示 (24 维)，用于表示依赖性复核。

8x8 DCT 的 zigzag 序划分为 8 个频带，每带提取 log 平均幅度、幅度标准差、
log 能量，共 24 维；三个 patch 取均值。无训练、无 Noiseprint 权重。
"""

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.fft import dctn

from analysis.sacr_p0 import patches_for_image

BANDS = ((1, 4), (4, 7), (7, 11), (11, 16),
         (16, 22), (22, 29), (29, 46), (46, 64))


def zigzag_indices():
    coords = sorted(((i, j) for i in range(8) for j in range(8)),
                    key=lambda x: (x[0] + x[1], x[1] if (x[0] + x[1]) % 2 == 0
                                   else x[0]))
    return np.asarray(coords)


ZIGZAG = zigzag_indices()


def dct24_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blocks = gray.reshape(16, 8, 16, 8).transpose(0, 2, 1, 3)
    coeff = dctn(blocks, axes=(-2, -1), norm="ortho")
    flat = coeff[..., ZIGZAG[:, 0], ZIGZAG[:, 1]].reshape(-1, 64)
    feats = []
    for lo, hi in BANDS:
        x = flat[:, lo:hi].ravel()
        feats.extend([float(np.log(np.mean(np.abs(x)) + 1e-12)),
                      float(np.std(x)),
                      float(np.log(np.mean(np.square(x)) + 1e-12))])
    return np.asarray(feats, dtype=np.float32)


def dct24_one(job):
    path, view = job
    values = [dct24_patch(p) for p in patches_for_image(path, view)]
    return np.nan_to_num(np.mean(values, axis=0), nan=0.0,
                         posinf=0.0, neginf=0.0)


def extract_dct24(paths, view, workers=8):
    jobs = [(p, view) for p in paths]
    if workers <= 1:
        return np.stack([dct24_one(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(dct24_one, jobs, chunksize=16))
    return np.stack(rows).astype(np.float32)


def ensure_dct_cache(out_dir, key, paths, view, workers=8):
    os.makedirs(out_dir, exist_ok=True)
    safe = key.replace("/", "__").replace("::", "--")
    path = os.path.join(out_dir, f"dct24_{safe}_{view}.npy")
    if not os.path.exists(path):
        np.save(path, extract_dct24(paths, view, workers=workers))
    return path
