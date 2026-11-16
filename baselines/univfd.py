import torch.nn as nn
from torchvision import transforms as T

from models.backbones import build_backbone
from models.classifier import LinearClassifier

from .base import BaseMethod, METHODS

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@METHODS.register("univfd")
class UnivFD(BaseMethod):
    """UnivFD (Ojha et al., CVPR 2023): CLIP:ViT-L/14 冻结特征 + 线性分类头。"""

    def get_transform(self, train):
        ops = [T.Resize(224), T.CenterCrop(224)]
        if train:
            ops = [T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                   T.RandomHorizontalFlip()]
        ops += [T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)]
        return T.Compose(ops)

    def build_model(self):
        backbone, feat_dim = build_backbone(
            self.cfg.get("backbone", "clip_vit_l14"),
            pretrained=True,
        )
        # 冻结 CLIP 特征提取器, 只训练线性头
        for p in backbone.parameters():
            p.requires_grad = False
        model = LinearClassifier(backbone, feat_dim)
        model.backbone.eval()
        return model
