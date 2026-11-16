import random
from io import BytesIO

from PIL import Image, ImageFilter
from torchvision import transforms as T

from models.backbones import build_backbone
from models.classifier import LinearClassifier

from .base import BaseMethod, METHODS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def jpeg_compress(img, quality):
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


class RandomGaussianBlur:
    """以概率 p 施加随机高斯模糊（模块级类, 保证 dataloader worker 可 pickle）。"""

    def __init__(self, p=0.1, radius_max=1.5):
        self.p = p
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0, self.radius_max))
            )
        return img


class RandomJPEG:
    """以概率 p 施加随机质量 JPEG 压缩。"""

    def __init__(self, p=0.1, q_min=30, q_max=100):
        self.p = p
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, img):
        if random.random() < self.p:
            return jpeg_compress(img, random.randint(self.q_min, self.q_max))
        return img


@METHODS.register("cnn_detection")
class CNNDetection(BaseMethod):
    """Wang et al., CVPR 2020 "CNN-generated images are surprisingly easy to
    spot... for now": ResNet-50 二分类 + 高斯模糊/JPEG 压缩增广。"""

    def get_transform(self, train):
        ops = []
        if train:
            ops += [
                T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                RandomGaussianBlur(p=0.1),
                RandomJPEG(p=0.1),
            ]
        else:
            ops += [T.Resize(256), T.CenterCrop(224)]
        ops += [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        return T.Compose(ops)

    def build_model(self):
        backbone, feat_dim = build_backbone(
            self.cfg.get("backbone", "resnet50"),
            pretrained=self.cfg.get("pretrained", True),
        )
        return LinearClassifier(backbone, feat_dim)
