"""格式/压缩偏差报告 (deep-research-report-v3 P1-D)。

按 (label, generator) 分组统计图像的编码与尺寸分布, 用于检查 detector
是否可能利用格式差异 (如 real 全 JPEG / fake 全 PNG) 而非生成痕迹。

每组抽样最多 --n 张读取尺寸 (PIL 只读 header); 扩展名与文件大小全量统计。

用法:
    python scripts/format_bias_report.py \
        --manifest data/manifests/arforensics.csv \
        --root data/ARForensics --n 500 \
        --report reports/dataset_audit/format_bias_arforensics.json
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--n", type=int, default=500, help="每组最多读取多少张的尺寸")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["generator"])].append(r["path"])

    report = {}
    for (label, gen), paths in sorted(groups.items()):
        key = f"{'real' if label == '0' else 'fake'}:{gen}"
        ext_counter = Counter(os.path.splitext(p)[1].lower() for p in paths)
        sizes = []
        dims = []
        for p in paths:
            fp = os.path.join(args.root, p)
            try:
                sizes.append(os.path.getsize(fp))
            except OSError:
                pass
        for p in paths[: args.n]:
            fp = os.path.join(args.root, p)
            try:
                with Image.open(fp) as im:
                    dims.append((im.width, im.height, im.format, im.mode))
            except Exception:
                pass
        fmt_counter = Counter(d[2] for d in dims)
        widths = [d[0] for d in dims]
        heights = [d[1] for d in dims]
        modes = Counter(d[3] for d in dims)
        entry = {
            "n_total": len(paths),
            "extensions": dict(ext_counter),
            "decoded_formats_sampled": dict(fmt_counter),
            "color_modes_sampled": dict(modes),
            "n_size_measured": len(sizes),
            "file_size_bytes": {
                "min": min(sizes) if sizes else None,
                "median": sorted(sizes)[len(sizes) // 2] if sizes else None,
                "max": max(sizes) if sizes else None,
            },
            "n_dims_sampled": len(dims),
            "widths": sorted(set(widths))[:10],
            "heights": sorted(set(heights))[:10],
        }
        report[key] = entry
        print(key, json.dumps({k: entry[k] for k in
                               ("n_total", "extensions", "decoded_formats_sampled")}))

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"报告已保存到 {args.report}")


if __name__ == "__main__":
    main()
