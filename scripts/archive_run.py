"""把一次完成的实验归档到 reports/runs/<run_id>/ (deep-research-report-v2 §7.2)。

收集: config.yaml, metrics.json, predictions.csv(.gz), manifest.json,
environment.json, README.md。只复制轻量文件, 不复制 checkpoint。

用法:
    python scripts/archive_run.py \
        --run_id dire_arforensics_val \
        --config configs/dire_arforensics.yaml \
        --results outputs/dire_arforensics/results_val.json \
        --predictions outputs/dire_arforensics/results_val_predictions.csv \
        --method dire --split val \
        --checkpoint outputs/dire_arforensics/best.pth
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def file_count_csv(path):
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset_manifest", default=None)
    parser.add_argument("--out_root", default="reports/runs")
    args = parser.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    if os.path.isdir(run_dir):
        raise SystemExit(f"run 目录已存在, 拒绝覆盖: {run_dir}")
    os.makedirs(run_dir)

    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    shutil.copy(args.results, os.path.join(run_dir, "metrics.json"))
    shutil.copy("environment/system.json", os.path.join(run_dir, "environment.json"))

    sample_count = None
    if args.predictions and os.path.isfile(args.predictions):
        with open(args.predictions, "rb") as fin, \
                gzip.open(os.path.join(run_dir, "predictions.csv.gz"), "wb") as fout:
            shutil.copyfileobj(fin, fout)
        sample_count = file_count_csv(args.predictions)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "UNKNOWN"

    manifest = {
        "run_id": args.run_id,
        "git_commit": commit,
        "method": args.method,
        "split": args.split,
        "seed": None,
        "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint and os.path.isfile(args.checkpoint) else None,
        "sample_count": sample_count,
        "dataset_manifest": args.dataset_manifest,
        "dataset_manifest_sha256": sha256(args.dataset_manifest) if args.dataset_manifest and os.path.isfile(args.dataset_manifest) else None,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(run_dir, "README.md"), "w") as f:
        f.write(f"# Run {args.run_id}\n\n"
                f"- method: {args.method}\n- split: {args.split}\n"
                f"- git: {commit}\n"
                f"- 指标见 metrics.json; 逐样本预测见 predictions.csv.gz\n")

    print(f"已归档到 {run_dir}")


if __name__ == "__main__":
    main()
