import numpy as np
import torch
from torchvision import transforms as T

from models.backbones import build_backbone
from models.classifier import LinearClassifier

from .base import BaseMethod, METHODS


def log_spectrum(img):
    """PIL 图像 -> log 幅度谱 (3, 256, 256)。"""
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log(np.abs(f) + 1e-8)
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    return torch.from_numpy(mag).float().unsqueeze(0).repeat(3, 1, 1)


@METHODS.register("freq_spec")
class FreqSpec(BaseMethod):
    """频域基线: log 幅度谱 + 分类器（Frank et al., ICML 2020 的思路）。"""

    def get_transform(self, train):
        ops = []
        if train:
            ops.append(T.RandomHorizontalFlip())
        ops.append(T.Lambda(log_spectrum))
        return T.Compose(ops)

    def build_model(self):
        backbone, feat_dim = build_backbone(
            self.cfg.get("backbone", "resnet18"),
            pretrained=self.cfg.get("pretrained", True),
        )
        return LinearClassifier(backbone, feat_dim)
