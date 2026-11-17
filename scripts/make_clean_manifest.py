"""基于 sha256 精确去重生成 clean manifest (deep-research-report-v3 P0-B)。

从原始 manifest 中移除跨 split 精确重复的样本:
- 同一 sha256 出现在多个 split 时, 保留优先级 test > val > train
  (test 是官方固定协议, train 是可变的一方, 删除 train 侧泄漏);
- 输出 clean manifest 与被排除样本清单 (leakage_exclusions.csv)。

用法:
    python scripts/make_clean_manifest.py \
        --manifest data/manifests/arforensics.csv \
        --clean data/manifests/arforensics_clean.csv \
        --excluded data/manifests/leakage_exclusions.csv
"""

import argparse
import csv
from collections import defaultdict

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--excluded", required=True)
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    by_hash = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("sha256"):
            by_hash[r["sha256"]].append(i)

    drop = set()
    for h, idxs in by_hash.items():
        splits = {rows[i]["split"] for i in idxs}
        if len(splits) > 1:
            # 跨 split 重复: 按优先级保留一个, 其余排除
            keep = min(idxs, key=lambda i: SPLIT_PRIORITY.get(rows[i]["split"], 9))
            for i in idxs:
                if i != keep:
                    drop.add(i)

    clean = [r for i, r in enumerate(rows) if i not in drop]
    excluded = [rows[i] for i in sorted(drop)]

    with open(args.clean, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(clean)
    with open(args.excluded, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(excluded)

    print(f"原始 {len(rows)} 行, 排除 {len(excluded)} 行跨 split 精确重复, "
          f"clean {len(clean)} 行")
    for r in excluded:
        print(f"  excluded: {r['split']}/{r['path']} sha256={r['sha256'][:12]}...")


if __name__ == "__main__":
    main()
