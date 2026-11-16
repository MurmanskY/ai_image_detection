import torch
import torch.nn.functional as F
from torchvision import transforms as T

from models.backbones import build_backbone
from models.classifier import LinearClassifier

from .base import BaseMethod, METHODS


def npr_map(img):
    """NPR (Tan et al., CVPR 2024): x - up2x(down2x(x)), 最近邻插值。

    提取 CNN 生成图像在上采样操作中留下的局部像素关系伪影。
    输入 PIL 图像, 输出 (3, H, W) tensor。
    """
    x = T.ToTensor()(img)
    down = F.interpolate(x.unsqueeze(0), scale_factor=0.5, mode="nearest")
    up = F.interpolate(down, scale_factor=2.0, mode="nearest")
    return (x - up.squeeze(0)) * 4.0  # 放大残差便于网络学习


@METHODS.register("npr")
class NPR(BaseMethod):
    """NPR: Rethinking the Up-Sampling Operations in CNN-based GAI Detection
    (CVPR 2024). NPR 残差图 + ResNet 二分类。"""

    def get_transform(self, train):
        ops = []
        if train:
            ops += [T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                    T.RandomHorizontalFlip()]
        else:
            ops += [T.Resize(256), T.CenterCrop(224)]
        ops.append(T.Lambda(npr_map))
        return T.Compose(ops)

    def build_model(self):
        backbone, feat_dim = build_backbone(
            self.cfg.get("backbone", "resnet50"),
            pretrained=self.cfg.get("pretrained", True),
        )
        return LinearClassifier(backbone, feat_dim)
