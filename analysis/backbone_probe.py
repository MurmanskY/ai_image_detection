"""UnivFD 归因分析: 同协议下不同冻结骨干的线性探测对比。

问题: UnivFD (CLIP ViT-L/14 + 线性头) 效果最好, 是 CLIP 特征本身强,
还是"冻结特征 + 线性头"的协议强?

方法: 对多个冻结骨干在完全相同的 train/val/test 上提取特征,
用 LogisticRegression 拟合, 比较 per-generator AUC。
所有 backbone 只用公开预训练权重, 不在 ARForensics 上微调。

用法:
  python analysis/backbone_probe.py --n 500 --out outputs/analysis/backbone_probe.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from datasets.arforensics import ARForensicsDataset  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def make_transform(mean, std):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


BACKBONES = {
    # (loader 参数, 图像均值方差)
    "clip_vit_l14": {"mean": CLIP_MEAN, "std": CLIP_STD},
    "clip_vit_b16": {"mean": CLIP_MEAN, "std": CLIP_STD},
    "resnet50": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "dinov2_vits14": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
}


@torch.no_grad()
def extract(model, ds, device, batch_size=64):
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True)
    feats = []
    for batch in loader:
        f = model(batch["img"].to(device))
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def build(name, device):
    if name.startswith("clip"):
        import open_clip
        arch, tag = {"clip_vit_l14": ("ViT-L-14", "openai"),
                     "clip_vit_b16": ("ViT-B-16", "laion2b_s34b_b88k")}[name]
        m, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
        return m.visual.eval().to(device)
    if name == "resnet50":
        from torchvision import models as tvm
        m = tvm.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
        return m.eval().to(device)
    if name == "dinov2_vits14":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"
                              ).eval().to(device)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--n", type=int, default=500, help="test 每类采样")
    parser.add_argument("--n_train", type=int, default=2000,
                        help="训练特征采样上限 (train 每类)")
    parser.add_argument("--out", default="outputs/analysis/backbone_probe.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = {}
    for name, cfg in BACKBONES.items():
        print(f"===== backbone: {name}")
        transform = make_transform(cfg["mean"], cfg["std"])
        model = build(name, device)

        train_ds = ARForensicsDataset(args.root, "train", transform,
                                      max_per_class=args.n_train)
        val_ds = ARForensicsDataset(args.root, "val", transform)
        ftr = extract(model, train_ds, device)
        ytr = np.array([s[1] for s in train_ds.samples])
        fval = extract(model, val_ds, device)
        yval = np.array([s[1] for s in val_ds.samples])

        clf = LogisticRegression(max_iter=2000).fit(ftr, ytr)
        print(f"  train acc: {clf.score(ftr, ytr):.4f}, "
              f"val acc: {clf.score(fval, yval):.4f}")

        # test: per-generator
        probe = ARForensicsDataset(args.root, "test", transform,
                                   max_per_class=args.n)
        gens = sorted({g for _, l, g in probe.samples if l == 1})
        per_gen = {}
        for gen in gens:
            sub = ARForensicsDataset(args.root, "test", transform,
                                     models=[gen], max_per_class=args.n)
            fte = extract(model, sub, device)
            yte = np.array([s[1] for s in sub.samples])
            scores = clf.predict_proba(fte)[:, 1]
            per_gen[gen] = float(roc_auc_score(yte, scores))
        macro = float(np.mean(list(per_gen.values())))
        results[name] = {"per_generator": per_gen, "macro_auc": macro}
        print(f"  macro AUC: {macro:.4f}; " +
              " ".join(f"{g}={v:.3f}" for g, v in per_gen.items()))

        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
