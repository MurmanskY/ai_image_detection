import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms as T

from .base import BaseMethod, METHODS


def to_8bit(x):
    return x * 255.0


class ZEDModel(nn.Module):
    def __init__(self, srec_root, checkpoint, statistic="abs_d0", entropy_chunk=16):
        super().__init__()
        sys.path.insert(0, os.path.abspath(srec_root))
        from src import network

        self.compressor = network.Compressor()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.compressor.nets.load_state_dict(state["nets"])
        self.compressor.eval()
        for parameter in self.compressor.parameters():
            parameter.requires_grad = False
        self.statistic = statistic
        self.entropy_chunk = entropy_chunk

    def _entropy(self, logits, target):
        loss_fn = self.compressor.loss_fn
        _, mixture_logits, means, log_scales, _ = loss_fn._extract_non_shared(
            target, logits
        )
        log_weights = torch.log_softmax(mixture_logits, dim=2).unsqueeze(2)
        means = means.unsqueeze(2)
        log_scales = log_scales.unsqueeze(2)
        entropy = torch.zeros_like(target)
        for start in range(0, loss_fn.L, self.entropy_chunk):
            stop = min(start + self.entropy_chunk, loss_fn.L)
            values = torch.arange(
                start, stop, device=logits.device, dtype=logits.dtype
            ).view(1, 1, -1, 1, 1, 1)
            log_cdf = loss_fn.log_cdf(values, values, means, log_scales)
            log_probs = torch.logsumexp(log_weights + log_cdf, dim=3)
            probs = log_probs.exp()
            entropy -= (probs * log_probs).sum(dim=2)
        return entropy

    def _decode_level(self, decoder, x, y, ctx):
        from src import data, network, util

        height, width = x.shape[-2:]
        if not isinstance(ctx, float):
            ctx = ctx[..., :height, :width]
        y_slices = network.group_2x2(y)
        generator = decoder.forward_probs(x, ctx)
        nll_sum = entropy_sum = None
        count = 0
        try:
            for i, y_slice in enumerate(y_slices):
                if i == 0:
                    probs = next(generator)
                else:
                    probs = generator.send(y_slices[i - 1])
                h, w = y_slice.shape[-2:]
                logits = probs.probs[..., :h, :w]
                nll = decoder.loss_fn(y_slice, logits)
                entropy = self._entropy(logits, y_slice)
                if nll_sum is None:
                    nll_sum = nll.sum(dim=(1, 2, 3))
                    entropy_sum = entropy.sum(dim=(1, 2, 3))
                else:
                    nll_sum = nll_sum + nll.sum(dim=(1, 2, 3))
                    entropy_sum = entropy_sum + entropy.sum(dim=(1, 2, 3))
                count += y_slice.numel() // y_slice.shape[0]
        except StopIteration as stop:
            _, ctx = stop.value
        return nll_sum / count, entropy_sum / count, ctx

    def forward(self, x):
        from src import data, util

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            downsampled = data.average_downsamples(x)
            ctx = 0.0
            gaps = []
            for decoder, upsampler, level_x, level_y in zip(
                self.compressor.decs,
                self.compressor.ctx_upsamplers,
                downsampled[::-1],
                downsampled[-2::-1],
            ):
                ctx = upsampler(ctx)
                nll, entropy, ctx = self._decode_level(
                    decoder, level_x, util.tensor_round(level_y), ctx
                )
                gaps.append(nll - entropy)
        d2, d1, d0 = gaps
        statistics = {
            "d0": d0,
            "abs_d0": d0.abs(),
            "delta01": d0 - d1,
            "abs_delta01": (d0 - d1).abs(),
        }
        if self.statistic not in statistics:
            raise ValueError(f"未知 ZED 统计量: {self.statistic}")
        return statistics[self.statistic]


@METHODS.register("zed")
class ZED(BaseMethod):
    """ZED (ECCV 2024): 使用 SReC 的 NLL 与 Entropy 编码代价差检测生成图像。"""

    requires_training = False

    def get_transform(self, train):
        size = self.cfg.get("image_size")
        crop_size = self.cfg.get("crop_size")
        ops = []
        if size is not None:
            ops.append(T.Resize((size, size)))
        if crop_size is not None:
            ops.append(T.CenterCrop((crop_size, crop_size)))
        ops += [T.ToTensor(), T.Lambda(to_8bit)]
        return T.Compose(ops)

    def build_model(self):
        root = os.path.join(os.path.dirname(__file__), "..", "third_party", "SReC")
        checkpoint = self.cfg.get(
            "checkpoint", os.path.join(root, "models", "openimages.pth")
        )
        return ZEDModel(
            root,
            checkpoint,
            statistic=self.cfg.get("statistic", "abs_d0"),
            entropy_chunk=self.cfg.get("entropy_chunk", 16),
        )
