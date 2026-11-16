"""组织 ARForensics 数据目录。

HF 上的 ARForensics 以压缩包形式发布:
  - train/ARForensics_train_split.zip.aa~ag (35GB, 本项目未下载)
  - val/ARForensics_val.zip (10k LlamaGen fake + 10k ImageNet real)
  - ARForensics_updated_test_part001~004.zip (7 个 AR 模型, 每模型 6k 对)

由于完整 train 分卷太大, 本脚本把 val 集重新划分为:
  data/ARForensics/train/{0_real,1_fake}   (每类前 --n_train 张)
  data/ARForensics/val/{0_real,1_fake}     (每类接下来 --n_val 张)
test 各分卷解压到:
  data/ARForensics/test/<模型>/{0_real,1_fake}

用法:
  python scripts/organize_arforensics.py --hf_snapshot <snapshot路径> \
      --out data/ARForensics [--extract_test]
"""

import argparse
import os
import shutil
import zipfile


def extract_zip(zip_path, dst):
    os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dst)
    print(f"解压完成: {zip_path} -> {dst}")


def find_class_dir(root, cls_name):
    """在 root 下递归寻找名为 cls_name 的目录。"""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == cls_name:
                return os.path.join(dirpath, d)
    return None


def split_val(val_extract_dir, out_dir, n_train, n_val):
    """把 val 集的 0_real/1_fake 重划分为 train/val。"""
    for cls_name in ("0_real", "1_fake"):
        src = find_class_dir(val_extract_dir, cls_name)
        assert src, f"未在 {val_extract_dir} 找到 {cls_name}"
        files = sorted(
            f for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        print(f"{cls_name}: 共 {len(files)} 张")
        for split, subset in (("train", files[:n_train]),
                              ("val", files[n_train:n_train + n_val])):
            dst = os.path.join(out_dir, split, cls_name)
            os.makedirs(dst, exist_ok=True)
            for f in subset:
                os.link(os.path.join(src, f), os.path.join(dst, f))
            print(f"  -> {split}: {len(subset)} 张 (硬链接)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_snapshot", required=True,
                        help="HF snapshot 目录 (包含 val/ 与 test 分卷)")
    parser.add_argument("--out", default="data/ARForensics")
    parser.add_argument("--n_train", type=int, default=6000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--extract_test", action="store_true")
    args = parser.parse_args()

    val_zip = os.path.join(args.hf_snapshot, "val", "ARForensics_val.zip")
    if os.path.exists(val_zip):
        val_dir = os.path.join(args.out, "_val_extracted")
        if not os.path.exists(val_dir):
            extract_zip(val_zip, val_dir)
        split_val(val_dir, args.out, args.n_train, args.n_val)

    if args.extract_test:
        for i in range(1, 5):
            part = os.path.join(
                args.hf_snapshot, f"ARForensics_updated_test_part{i:03d}.zip")
            if os.path.exists(part):
                extract_zip(part, args.out)
            else:
                print(f"跳过 (未下载): {part}")


if __name__ == "__main__":
    main()
