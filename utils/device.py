import contextlib

import torch


def get_device(pref="auto"):
    """根据偏好选择设备。pref: auto / cuda / mps / cpu。"""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device, enabled=True):
    """仅在 CUDA 上启用 AMP；MPS/CPU 下为空上下文（fp32）。"""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda")
    return contextlib.nullcontext()
