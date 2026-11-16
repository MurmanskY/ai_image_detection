"""从 HF 镜像构建 PRADA C2I 所需的 ImageNet 真实图子集。

PRADA 官方流程: ImageNet val (50k) -> 每类 10 张共 10,000 张, stratify + random_state=0。
官方 prepare_imagenet.py 基于本地 ImageFolder; 本脚本改用非门控 HF 镜像
Tsomaros/Imagenet-1k_validation, 类别标签与 ImageNet 0-999 对齐。
注意: 由于源文件排序不同, 选出的 10k 子集与官方脚本不完全相同, 但分布一致。

用法:
    python scripts/prepare_imagenet_subset_hf.py --out data/darg/darg/c2i/real
"""
import argparse
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/darg/darg/c2i/real")
    parser.add_argument("--repo", default="Tsomaros/Imagenet-1k_validation")
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()

    ds = load_dataset(args.repo, split="validation")
    labels = ds["label"]
    n_per_class = args.per_class
    train_size = n_per_class * 1000
    indices, _ = train_test_split(
        range(len(ds)), train_size=train_size, stratify=labels,
        random_state=0, shuffle=True,
    )
    os.makedirs(args.out, exist_ok=True)
    for i in tqdm(indices, desc="写出 ImageNet 子集"):
        label = labels[i]
        out_dir = os.path.join(args.out, f"{label:03d}")
        os.makedirs(out_dir, exist_ok=True)
        ds[i]["image"].convert("RGB").save(
            os.path.join(out_dir, f"imagenet_val_{i:05d}.png")
        )
    print(f"完成: {len(indices)} 张 -> {args.out}")


if __name__ == "__main__":
    main()
