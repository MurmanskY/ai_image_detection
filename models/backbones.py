import torch.nn as nn
from torchvision import models as tvm

_RESNET_WEIGHTS = {
    "resnet18": "IMAGENET1K_V1",
    "resnet34": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
}

_CLIP_ARCHS = {
    "clip_vit_b16": ("ViT-B-16", "laion2b_s34b_b88k"),
    "clip_vit_l14": ("ViT-L-14", "openai"),
}


def build_backbone(name, pretrained=True):
    """返回 (backbone, feat_dim)。backbone 前向输出全局特征向量。"""
    if name in _RESNET_WEIGHTS:
        weights = _RESNET_WEIGHTS[name] if pretrained else None
        m = getattr(tvm, name)(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name in _CLIP_ARCHS:
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "使用 CLIP 骨干需要安装 open_clip_torch: pip install open_clip_torch"
            ) from e
        arch, tag = _CLIP_ARCHS[name]
        m, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
        return m.visual, m.visual.output_dim
    raise ValueError(f"未知骨干网络: {name}")
