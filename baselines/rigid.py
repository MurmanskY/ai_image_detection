import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from .base import BaseMethod, METHODS
from .cnn_detection import IMAGENET_MEAN, IMAGENET_STD

_DINO_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class RIGIDModel(nn.Module):
    """RIGID (He et al., 2024): 比较原图与加噪图的 DINOv2 特征余弦相似度。

    真实图像对微小扰动更鲁棒 (相似度更高), 生成图像更敏感 (相似度更低)。
    输出 logit = -相似度 * scale, 值越大越倾向判为生成图像。
    """

    def __init__(self, backbone, noise_std=0.05, scale=10.0, n_repeat=1):
        super().__init__()
        self.backbone = backbone
        self.noise_std = noise_std
        self.scale = scale
        self.n_repeat = n_repeat

    def forward(self, x):
        f_clean = F.normalize(self.backbone(x), dim=-1)
        sims = []
        for _ in range(self.n_repeat):
            noisy = x + torch.randn_like(x) * self.noise_std
            f_noisy = F.normalize(self.backbone(noisy), dim=-1)
            sims.append((f_clean * f_noisy).sum(dim=-1))
        sim = torch.stack(sims, dim=0).mean(dim=0)
        return -sim * self.scale


@METHODS.register("rigid")
class RIGID(BaseMethod):
    """RIGID: A Training-Free and Model-Agnostic Framework for Robust
    AI-Generated Image Detection (2024). 无需训练, 直接用 eval.py 评估。"""

    requires_training = False
    score_type = "anomaly"

    def get_transform(self, train):
        return T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def build_model(self):
        arch = self.cfg.get("backbone", "dinov2_vits14")
        backbone = torch.hub.load("facebookresearch/dinov2", arch)
        for p in backbone.parameters():
            p.requires_grad = False
        return RIGIDModel(
            backbone,
            noise_std=self.cfg.get("noise_std", 0.05),
            n_repeat=self.cfg.get("n_repeat", 1),
        )
