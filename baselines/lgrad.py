import importlib.util
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms as T

from models.backbones import build_backbone
from models.classifier import LinearClassifier

from .base import BaseMethod, METHODS
from .cnn_detection import IMAGENET_MEAN, IMAGENET_STD


class LGradModel(nn.Module):
    def __init__(self, lgrad_root, checkpoint, backbone):
        super().__init__()
        package_dir = os.path.join(os.path.abspath(lgrad_root), "img2gad_pytorch", "models")
        spec = importlib.util.spec_from_file_location(
            "lgrad_models", os.path.join(package_dir, "__init__.py"),
            submodule_search_locations=[package_dir],
        )
        lgrad_models = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = lgrad_models
        spec.loader.exec_module(lgrad_models)

        self.discriminator = lgrad_models.build_model(
            gan_type="stylegan",
            module="discriminator",
            resolution=256,
            label_size=0,
            image_channels=3,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.discriminator.load_state_dict(state, strict=True)
        self.discriminator.eval()
        for parameter in self.discriminator.parameters():
            parameter.requires_grad = False
        self.classifier = LinearClassifier(*backbone)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            score = self.discriminator(x)
            gradient = torch.autograd.grad(
                score.sum(), x, create_graph=False, retain_graph=False
            )[0]
        minimum = gradient.flatten(1).amin(dim=1).view(-1, 1, 1, 1)
        maximum = gradient.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        gradient = (gradient - minimum) / (maximum - minimum).clamp(min=1e-8)
        mean = gradient.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = gradient.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return self.classifier((gradient - mean) / std)


@METHODS.register("lgrad")
class LGrad(BaseMethod):
    """LGrad (CVPR 2023): 预训练 StyleGAN 判别器梯度图 + ResNet 分类。"""

    def get_transform(self, train):
        ops = []
        if train:
            ops += [T.RandomResizedCrop(256, scale=(0.75, 1.0)), T.RandomHorizontalFlip()]
        else:
            ops += [T.Resize((256, 256))]
        ops += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        return T.Compose(ops)

    def build_model(self):
        root = os.path.join(os.path.dirname(__file__), "..", "third_party", "LGrad")
        checkpoint = self.cfg.get(
            "checkpoint", "pretrained/stylegan_bedroom256_discriminator.pth"
        )
        backbone, feat_dim = build_backbone(
            self.cfg.get("backbone", "resnet50"),
            pretrained=self.cfg.get("pretrained", True),
        )
        return LGradModel(root, checkpoint, (backbone, feat_dim))
