"""统一数据 Manifest 构建 (deep-research-report-v2 §2.3)。

每张图片一行, 字段:
    sample_id,path,sha256,phash,label,generator,generator_family,source,
    split,original_split,dataset

- `sha256` 为完整文件哈希, `phash` 为感知哈希 (16 hex chars, imagehash);
  计算较慢, 通过 --with-hash 启用。
- 原始数据没有的字段写 "unknown", 不推测补全。
- 官方划分与自定义划分通过 dataset/original_split 字段区分:
  当前 train/val 来自官方 validation 重划分, 标记为
  dataset=ARForensics-small-val-derived-v1, original_split=validation。

用法:
    python -m datasets.manifest --root data/ARForensics --dataset arforensics \
        --out data/manifests/arforensics.csv --with-hash
"""

import argparse
import csv
import hashlib
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

FIELDS = [
    "sample_id", "path", "sha256", "phash", "label", "generator",
    "generator_family", "source", "split", "original_split", "dataset",
]

# 数据集名 -> (fake 样本 generator_family, real 来源说明)
DATASET_META = {
    "arforensics": ("AR", "ImageNet"),
    "darg_t2i": ("AR", "ARForensics-real (RAISE-1k 替代, 见 reproduction note)"),
    "darg_c2i": ("AR", "ImageNet-val HF 镜像子集 (见 reproduction note)"),
    "genimage": ("unknown", "ImageNet"),
}


def iter_images(directory):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            yield os.path.join(directory, name)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def file_phash(path):
    import imagehash
    from PIL import Image
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB")))


def _rows_arforensics(root, with_hash):
    """ARForensics 目录结构: train/val 为 0_real/1_fake;
    test 为 <generator>/0_real|1_fake。"""
    family, source = DATASET_META["arforensics"]
    for split in ("train", "val"):
        for class_name, label in (("0_real", 0), ("1_fake", 1)):
            generator = "real" if label == 0 else "LlamaGen"
            for path in iter_images(os.path.join(root, split, class_name)):
                yield _row(path, root, label, generator, family, source, split,
                           original_split="validation",
                           dataset="ARForensics-small-val-derived-v1",
                           with_hash=with_hash)
    test_root = os.path.join(root, "test")
    for generator in sorted(os.listdir(test_root)):
        generator_root = os.path.join(test_root, generator)
        if not os.path.isdir(generator_root):
            continue
        for class_name, label in (("0_real", 0), ("1_fake", 1)):
            g = "real" if label == 0 else generator
            for path in iter_images(os.path.join(generator_root, class_name)):
                yield _row(path, root, label, g, family, source, "test",
                           original_split="test",
                           dataset="ARForensics-test",
                           with_hash=with_hash)


def _iter_images_recursive(directory):
    if not os.path.isdir(directory):
        return
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                yield os.path.join(dirpath, name)


def _rows_darg(root, with_hash, mode):
    """DARG 目录结构: <mode>/<generator_or_real>/*.png (t2i) 或
    <mode>/<generator_or_real>/<class_id>/*.png (c2i); real 目录为真实图。"""
    family, source = DATASET_META[f"darg_{mode}"]
    mode_root = os.path.join(root, mode)
    for generator in sorted(os.listdir(mode_root)):
        generator_root = os.path.join(mode_root, generator)
        if not os.path.isdir(generator_root):
            continue
        label = 0 if generator == "real" else 1
        g = generator
        for path in _iter_images_recursive(generator_root):
            yield _row(path, root, label, g, family, source, "test",
                       original_split="unknown",
                       dataset=f"DARG-{mode}",
                       with_hash=with_hash)


def _row(path, root, label, generator, family, source, split,
         original_split, dataset, with_hash):
    rel = os.path.relpath(path, root)
    sample_id = hashlib.sha1(rel.encode()).hexdigest()[:16]
    return {
        "sample_id": sample_id,
        "path": rel,
        "sha256": file_sha256(path) if with_hash else "",
        "phash": file_phash(path) if with_hash else "",
        "label": label,
        "generator": generator,
        "generator_family": "real" if label == 0 else family,
        "source": source if label == 0 else "unknown",
        "split": split,
        "original_split": original_split,
        "dataset": dataset,
    }


def build_rows(root, dataset, with_hash=False):
    if dataset == "arforensics":
        yield from _rows_arforensics(root, with_hash)
    elif dataset in ("darg_t2i", "darg_c2i"):
        yield from _rows_darg(root, with_hash, mode=dataset.split("_")[1])
    else:
        raise ValueError(f"未知数据集: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["arforensics", "darg_t2i", "darg_c2i"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-hash", action="store_true",
                        help="计算 sha256/phash (慢, 用于重复审计)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in build_rows(args.root, args.dataset, args.with_hash):
            writer.writerow(row)
            n += 1
            if n % 5000 == 0:
                print(f"  {n} 行...")
    print(f"{args.out}: {n} 行")


if __name__ == "__main__":
    main()
