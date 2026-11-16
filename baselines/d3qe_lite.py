import os

import torch
import torch.nn as nn
from torchvision import transforms as T

from .base import BaseMethod, METHODS


class D3QELiteModel(nn.Module):
    """D³QE 轻量复现: 冻结 LlamaGen VQ encoder, 用离散 token 统计特征分类。

    特征 (对应 D³QE 的核心观察):
      1. codebook 使用直方图 (16384-d, 归一化) -> 线性投影
      2. 标量统计: token entropy / unique ratio / 量化误差 mean,std
    拼接后接 MLP 输出 logit。
    """

    def __init__(self, vq, codebook_size=16384, hist_dim=128):
        super().__init__()
        self.vq = vq
        for p in self.vq.parameters():
            p.requires_grad = False
        self.codebook_size = codebook_size
        self.hist_proj = nn.Linear(codebook_size, hist_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hist_dim + 4, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            quant, _, info = self.vq.encode(x)
            z_q, z = quant  # z_q: (B,C,H,W), z: (B,H,W,C)
            indices = info[2]  # (B*256,)
            B = x.shape[0]
            qerr = (z_q - z.permute(0, 3, 1, 2)).pow(2).mean(dim=1).flatten(1)
            tokens = indices.view(B, -1)  # (B, 256)
            hist = torch.zeros(B, self.codebook_size, device=x.device)
            hist.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float))
            hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1)
            # 标量统计
            p = hist.clamp(min=1e-12)
            entropy = -(p * p.log()).sum(dim=1)  # (B,)
            unique_ratio = (hist > 0).float().mean(dim=1)
            qerr_mean = qerr.mean(dim=1)
            qerr_std = qerr.std(dim=1)
            scalars = torch.stack(
                [entropy / 10.0, unique_ratio, qerr_mean * 100.0, qerr_std * 100.0],
                dim=1,
            )
        feat = torch.cat([self.hist_proj(hist), scalars], dim=1)
        return self.mlp(feat).squeeze(-1)


def _load_vq(ckpt_path, device="cpu"):
    import sys
    d3qe_dir = os.path.join(
        os.path.dirname(__file__), "..", "third_party", "D3QE")
    sys.path.insert(0, os.path.abspath(d3qe_dir))
    from networks.vq_model import ModelArgs, VQModel

    model = VQModel(ModelArgs())
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)
    return model


@METHODS.register("d3qe_lite")
class D3QELite(BaseMethod):
    """D³QE (ICCV 2025) 轻量复现: 离散 token 分布统计 + 量化误差特征。

    完整 D³QE 还包含 CLIP 语义特征与 D³-Attention 融合, 见 third_party/D3QE。
    需要 LlamaGen VQ 权重 (method.vq_ckpt)。
    """

    def get_transform(self, train):
        return T.Compose([
            T.Resize(256), T.CenterCrop(256), T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def build_model(self):
        ckpt = self.cfg.get("vq_ckpt", "pretrained/vq_ds16_c2i.pt")
        vq = _load_vq(ckpt)
        return D3QELiteModel(vq, codebook_size=vq.config.codebook_size)
