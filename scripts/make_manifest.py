"""生成 ARForensics 数据划分 manifest。

当前训练协议说明: 官方 train 分卷未下载, train/val 由官方 validation
(10k real + 10k fake, 仅 LlamaGen) 重划分而来, 命名为
"ARForensics-small-val-derived", 与 D3QE 官方 100k/100k 训练协议不可直接比较。

用法:
    python scripts/make_manifest.py --root data/ARForensics --out splits
"""
import argparse
import csv
import hashlib
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(directory):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            yield os.path.join(directory, name)


def content_id(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read(1 << 20)).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--out", default="splits")
    parser.add_argument("--with-hash", action="store_true",
                        help="计算 content_id (慢, 用于去重审计)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}

    for split in ("train", "val"):
        for class_name, label in (("0_real", 0), ("1_fake", 1)):
            generator = "real" if label == 0 else "LlamaGen"
            for path in iter_images(os.path.join(args.root, split, class_name)):
                rows[split].append((path, label, generator, split))

    test_root = os.path.join(args.root, "test")
    for generator in sorted(os.listdir(test_root)):
        generator_root = os.path.join(test_root, generator)
        if not os.path.isdir(generator_root):
            continue
        for class_name, label in (("0_real", 0), ("1_fake", 1)):
            g = "real" if label == 0 else generator
            for path in iter_images(os.path.join(generator_root, class_name)):
                rows["test"].append((path, label, g, "test"))

    for split, split_rows in rows.items():
        out_path = os.path.join(args.out, f"arforensics_small_{split}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["path", "label", "generator", "source_dataset", "split", "content_id"]
            )
            for path, label, generator, split_name in split_rows:
                cid = content_id(path) if args.with_hash else ""
                writer.writerow(
                    [path, label, generator, "ARForensics-small-val-derived"
                     if split in ("train", "val") else "ARForensics-test",
                     split_name, cid]
                )
        print(f"{out_path}: {len(split_rows)} 行")


if __name__ == "__main__":
    main()
