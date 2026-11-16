import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms as T

from models.backbones import build_backbone
from models.classifier import LinearClassifier

from .base import BaseMethod, METHODS
from .cnn_detection import IMAGENET_MEAN, IMAGENET_STD


class DIREModel(nn.Module):
    def __init__(self, diffusion_root, checkpoint, backbone, steps=20):
        super().__init__()
        sys.path.insert(0, os.path.abspath(diffusion_root))
        from guided_diffusion.script_util import create_model_and_diffusion

        self.diffusion_model, self.diffusion = create_model_and_diffusion(
            image_size=256,
            class_cond=False,
            learn_sigma=True,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            attention_resolutions="32,16,8",
            dropout=0.0,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing=f"ddim{steps}",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=True,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.diffusion_model.load_state_dict(state)
        self.diffusion_model.eval()
        for parameter in self.diffusion_model.parameters():
            parameter.requires_grad = False
        self.classifier = LinearClassifier(*backbone)

    def forward(self, x):
        with torch.no_grad():
            latent = self.diffusion.ddim_reverse_sample_loop(
                self.diffusion_model, x.shape, noise=x, clip_denoised=True
            )
            reconstruction = self.diffusion.ddim_sample_loop(
                self.diffusion_model, x.shape, noise=latent, clip_denoised=True
            )
            residual = (x - reconstruction).abs() * 2.0
        residual = (residual - residual.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / residual.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return self.classifier(residual)


@METHODS.register("dire")
class DIRE(BaseMethod):
    """DIRE (ICCV 2023): ADM DDIM 重建误差 + ResNet 二分类。"""

    def get_transform(self, train):
        ops = []
        if train:
            ops += [T.RandomResizedCrop(256, scale=(0.75, 1.0)), T.RandomHorizontalFlip()]
        else:
            ops += [T.Resize((256, 256))]
        ops += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        return T.Compose(ops)

    def build_model(self):
        root = os.path.join(os.path.dirname(__file__), "..", "third_party", "DIRE", "guided-diffusion")
        checkpoint = self.cfg.get("checkpoint", "pretrained/256x256_diffusion_uncond.pt")
        backbone, feat_dim = build_backbone(
            self.cfg.get("backbone", "resnet50"),
            pretrained=self.cfg.get("pretrained", True),
        )
        return DIREModel(
            root,
            checkpoint,
            (backbone, feat_dim),
            steps=self.cfg.get("steps", 20),
        )
