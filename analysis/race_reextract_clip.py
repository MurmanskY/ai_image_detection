"""RACE: 仅重提 CLIP 特征 (修复 batch-first 布局 bug 后的缓存迁移)。

feats.npz 中 dino/prp 键有效保留; clip/clip_pooled 全部重提;
train_real::v1 的 dino 一并重提 (aug seed 改为 md5, 保证 dino/clip 同视图)。
完成后删除 cfe.pt, 重跑 analysis/race_p0.py。

用法: python analysis/race_reextract_clip.py
"""

import json
import os
import sys

import numpy as np
import torch

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.race_p0 import (GENS, OUT_DIR, _norm_clip, clip_depth_cls,  # noqa: E402
                              jpeg95, wait_for_gpu)  # noqa: E402
from analysis.mpid_p0 import wait_for_gpu as _wg  # noqa: E402


@torch.no_grad()
def extract_clip_only(clip, paths, device, jpeg=False, aug=False,
                      batch_size=128, tag=""):
    from PIL import Image
    from torchvision import transforms as T
    F_c, F_p = [], []
    for bi in range(0, len(paths), batch_size):
        xs = []
        for p in paths[bi:bi + batch_size]:
            img = Image.open(p).convert("RGB")
            if aug:
                import hashlib
                torch.manual_seed(int(hashlib.md5(
                    str(p).encode()).hexdigest()[:8], 16))
                tf = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.2, 0.2),
                    T.ToTensor()])
                x01 = tf(img)
            else:
                img = img.resize((224, 224), Image.BICUBIC)
                if jpeg:
                    img = jpeg95(img)
                x01 = T.ToTensor()(img)
            xs.append(_norm_clip(x01))
        dc, pooled = clip_depth_cls(clip, torch.stack(xs).to(device))
        F_c.append(dc.float().cpu())
        F_p.append(pooled.float().cpu())
        if bi == 0:
            print(f"    [{tag}] {len(paths)} 张提取中...", flush=True)
    return torch.cat(F_c).numpy(), torch.cat(F_p).numpy()


def main():
    manifest = json.load(open(os.path.join(OUT_DIR, "manifest.json")))
    feats_path = os.path.join(OUT_DIR, "feats.npz")
    feats = dict(np.load(feats_path))
    np.savez_compressed(feats_path + ".bak", **feats)

    gpu = wait_for_gpu(need_mb=4000)
    device = torch.device(f"cuda:{gpu}")
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    clip = clip.visual.eval().to(device)
    for p in clip.parameters():
        p.requires_grad = False

    # dino v1 重提需要 dino (aug seed 已改为 md5)
    dino = None

    sets = [("train_real", "v0", False, False),
            ("train_real", "v1", False, True),
            ("dev_real", "v0", False, False),
            ("cal_real", "v0", False, False)]
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            sets.append((f"{g}/{cls}", "orig", False, False))
            sets.append((f"{g}/{cls}", "jpeg95", True, False))

    for key, codec, jpeg, aug in sets:
        if key in ("train_real", "dev_real", "cal_real"):
            paths = manifest[key]
        else:
            paths = manifest["test"][key]
        c, pooled = extract_clip_only(clip, paths, device, jpeg=jpeg,
                                      aug=aug, tag=f"{key}/{codec}")
        feats[f"{key}::{codec}::clip"] = c
        feats[f"{key}::{codec}::clip_pooled"] = pooled
        print(f"  {key}/{codec}: clip {c.shape}", flush=True)
        if key == "train_real" and codec == "v1":
            # dino v1 同步重提 (md5 aug 与 clip 一致)
            if dino is None:
                dino = torch.hub.load("facebookresearch/dinov2",
                                      "dinov2_vitl14").eval().to(device)
                for p in dino.parameters():
                    p.requires_grad = False
            from analysis.race_p0 import extract_vit_aug
            dv1 = extract_vit_aug(dino, clip, paths, device, tag="v1-dino")
            feats["train_real::v1::dino"] = dv1["dino"]
            # extract_vit_aug 内部也算了 clip (同样 md5 seed), 以它为准覆盖
            feats["train_real::v1::clip"] = dv1["clip"]

    np.savez_compressed(feats_path, **feats)
    print(f"已更新 {feats_path} (备份 .bak)", flush=True)


if __name__ == "__main__":
    main()
