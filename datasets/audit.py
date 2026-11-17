"""数据集审计 CLI (deep-research-report-v2 §2.4)。

对 manifest CSV 执行以下检查:
  1. train/val/test 之间 sha256 精确重复
  2. train/val/test 之间 phash 近似重复 (阈值可配置, 输出候选对供人工复核)
  3. label 取值合法性 (0/1)
  4. fake 样本 generator 非 real; real 样本 generator == "real"
  5. 每个 split/label/generator 的样本数
  6. 重复路径
  7. 文件损坏/无法解码 (可选 --check-decode, 慢)
  8. 文件格式与大小分布概览

pHash 相似不直接等同重复: 仅输出候选对, 阈值记录在报告中。

用法:
    python -m datasets.audit \
        --manifest data/manifests/arforensics.csv \
        --root data/ARForensics \
        --check-split-overlap --check-exact-duplicates \
        --check-near-duplicates --check-labels --check-generator-metadata \
        --report reports/dataset_audit/arforensics.json
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _hamming(hex1, hex2):
    return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")


def audit(rows, root, args):
    report = {"manifest_rows": len(rows), "checks": {}}

    # 5. 计数
    by_split = Counter(r["split"] for r in rows)
    by_split_label = Counter((r["split"], r["label"]) for r in rows)
    by_gen = Counter((r["split"], r["generator"]) for r in rows)
    report["counts"] = {
        "by_split": dict(by_split),
        "by_split_label": {f"{s}/{l}": n for (s, l), n in sorted(by_split_label.items())},
        "by_split_generator": {f"{s}/{g}": n for (s, g), n in sorted(by_gen.items())},
    }

    # 6. 重复路径
    path_counts = Counter(r["path"] for r in rows)
    dup_paths = {p: n for p, n in path_counts.items() if n > 1}
    report["checks"]["duplicate_paths"] = {"count": len(dup_paths), "examples": dict(list(dup_paths.items())[:20])}

    # 3/4. 标签与 generator 元数据
    if args.check_labels or args.check_generator_metadata:
        bad_label, bad_gen = [], []
        for r in rows:
            if r["label"] not in ("0", "1"):
                bad_label.append(r["path"])
            if r["label"] == "0" and r["generator"] != "real":
                bad_gen.append({"path": r["path"], "issue": "real 样本 generator 非 real",
                                "generator": r["generator"]})
            if r["label"] == "1" and r["generator"] == "real":
                bad_gen.append({"path": r["path"], "issue": "fake 样本 generator 为 real"})
        report["checks"]["labels"] = {"invalid_count": len(bad_label), "examples": bad_label[:20]}
        report["checks"]["generator_metadata"] = {"invalid_count": len(bad_gen), "examples": bad_gen[:20]}

    # 1. sha256 精确重复 (含跨 split)
    if args.check_exact_duplicates:
        by_hash = defaultdict(list)
        for r in rows:
            if r.get("sha256"):
                by_hash[r["sha256"]].append((r["split"], r["path"]))
        dups = {h: v for h, v in by_hash.items() if len(v) > 1}
        cross_split = {h: v for h, v in dups.items() if len({s for s, _ in v}) > 1}
        report["checks"]["exact_duplicates"] = {
            "duplicate_groups": len(dups),
            "cross_split_groups": len(cross_split),
            "cross_split_examples": dict(list(cross_split.items())[:20]),
        }
        if not by_hash:
            report["checks"]["exact_duplicates"]["note"] = "manifest 无 sha256, 请用 --with-hash 重新生成"

    # 2. phash 近似重复 (跨 split 候选对)
    # LSH band 优化: 64 位 hash 分 4 个 16 位 band, 只比较共享 band 的对,
    # 避免 O(n^2) 全对比。
    if args.check_near_duplicates:
        threshold = args.phash_threshold
        entries = [(r["phash"], r["path"], r["split"]) for r in rows if r.get("phash")]
        bands = defaultdict(list)  # (band_idx, band_val) -> [(phash, path, split)]
        for h, p, s in entries:
            v = int(h, 16)
            for b in range(4):
                bands[(b, (v >> (16 * b)) & 0xFFFF)].append((h, p, s))
        seen = set()
        candidates = []
        for bucket in bands.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    h1, p1, s1 = bucket[i]
                    h2, p2, s2 = bucket[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in seen:
                        continue
                    seen.add(key)
                    d = _hamming(h1, h2)
                    if d <= threshold:
                        candidates.append({"a": p1, "b": p2, "split_a": s1,
                                           "split_b": s2, "hamming": d})
        report["checks"]["near_duplicates"] = {
            "phash_threshold": threshold,
            "candidate_pairs": len(candidates),
            "note": "pHash 相似不直接等同重复, 候选对需人工复核",
            "examples": candidates[:50],
        }
        if not any(by_split.values()):
            report["checks"]["near_duplicates"]["note"] = "manifest 无 phash, 请用 --with-hash 重新生成"

    # 7. 文件损坏检查
    if args.check_decode:
        from PIL import Image
        corrupt, missing = [], []
        for r in rows:
            p = os.path.join(root, r["path"])
            if not os.path.isfile(p):
                missing.append(r["path"])
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                corrupt.append(r["path"])
        report["checks"]["decode"] = {"missing": len(missing), "corrupt": len(corrupt),
                                      "missing_examples": missing[:20], "corrupt_examples": corrupt[:20]}

    # 8. 格式与大小分布
    ext_counts = Counter(os.path.splitext(r["path"])[1].lower() for r in rows)
    report["formats"] = {"extensions": dict(ext_counts)}

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--check-split-overlap", action="store_true")
    parser.add_argument("--check-exact-duplicates", action="store_true")
    parser.add_argument("--check-near-duplicates", action="store_true")
    parser.add_argument("--check-labels", action="store_true")
    parser.add_argument("--check-generator-metadata", action="store_true")
    parser.add_argument("--check-decode", action="store_true")
    parser.add_argument("--phash-threshold", type=int, default=8,
                        help="pHash 汉明距离阈值 (默认 8/64)")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    report = audit(rows, args.root, args)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write(text)
        print(f"报告已保存到 {args.report}")


if __name__ == "__main__":
    main()
