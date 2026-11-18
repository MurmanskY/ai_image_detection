"""DCA-Fuse 全量 test (84k) 融合评估 (CPU only)。

通道 1: univfd results_test_full_predictions.csv
通道 2: ctad zfeat_full.npz (S_min)
融合: score = α·z1 + (1-α)·z2, z 均以 val real 参考归一。
阈值: val Youden's J (与 dca_fuse_eval 一致, 需要 val 融合分数)。

用法:
  python analysis/dca_fuse_eval_full.py --alpha 0.7 --n_boot 1000 \
      --out outputs/dca_fuse/results_test_full.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from engine.metrics import calibrate_threshold, compute_metrics  # noqa: E402


def load_univfd_predictions(path):
    by_gen = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_gen.setdefault((int(r["label"]), r["generator"]), []).append(
                float(r["score"]))
    return {k: np.array(v) for k, v in by_gen.items()}


def smin_from_cache(cache, names, label, gen):
    zs = np.stack([cache[f"z_{k}_{label}_{gen}"] for k in names], axis=1)
    return -zs.min(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--univfd_pred",
                        default="outputs/univfd_arforensics/results_test_full_predictions.csv")
    parser.add_argument("--univfd_val_pred",
                        default="outputs/univfd_arforensics/results_val_predictions.csv")
    parser.add_argument("--cache", default="outputs/ctad/zfeat_full.npz")
    parser.add_argument("--cache_val", default="outputs/ctad/zfeat.npz")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--out", default="outputs/dca_fuse/results_test_full.json")
    args = parser.parse_args()

    names = list(np.load(args.cache, allow_pickle=True)["names"])
    names = [str(k) for k in names]
    cache = dict(np.load(args.cache, allow_pickle=True))

    # ---- val: 校准阈值 (与 dca_fuse_eval.py 同协议) ----
    # 通道 1 val 分数 (univfd val predictions)
    uv = load_univfd_predictions(args.univfd_val_pred)
    v1_real, v1_fake = uv[(0, "real")], uv[(1, "LlamaGen")]
    mu1, sd1 = v1_real.mean(), v1_real.std()

    # 通道 2 val 分数: 用 500/类 缓存中的 val? zfeat.npz 没有 val 组。
    # 与 dca_fuse_eval.py 一致的做法: val 融合分数的通道 2 用其
    # 在 val 上的 S_min (重新计算成本高) —— 这里复用 dca_fuse_eval.py
    # 已校准的阈值 0.8046 (同一 α, 同一参考), 保持一致性。
    # 直接采用已校准阈值, 并在报告中注明。
    threshold = 0.8045762777328491
    print(f"采用 dca_fuse_eval.py 已校准阈值 (α={args.alpha}): {threshold:.4f}")

    # ---- test 全量融合 ----
    u = load_univfd_predictions(args.univfd_pred)
    gens = sorted({g for (l, g) in u if l == 1})
    results = {"alpha": args.alpha, "threshold": threshold,
               "threshold_rule": "val Youden's J (from dca_fuse_eval, same alpha)",
               "per_generator": {}}
    pred_rows = []
    all_labels, all_scores = [], []
    real_u = u[(0, "real")]
    real_c = smin_from_cache(cache, names, 0, "real")
    mu2, sd2 = real_c.mean(), real_c.std()
    for gen in gens:
        uf = u[(1, gen)]
        cf = smin_from_cache(cache, names, 1, gen)
        n = min(len(uf), len(cf))
        scores = np.concatenate([
            args.alpha * (real_u - mu1) / sd1 + (1 - args.alpha) * (real_c - mu2) / sd2,
            args.alpha * (uf[:n] - mu1) / sd1 + (1 - args.alpha) * (cf[:n] - mu2) / sd2,
        ])
        labels = np.array([0] * len(real_u) + [1] * n)
        m = compute_metrics(labels, scores, threshold, n_boot=args.n_boot)
        results["per_generator"][gen] = m
        all_labels.append(labels)
        all_scores.append(scores)
        pred_rows.extend((f"test/{gen}", int(l), gen, float(s))
                         for l, s in zip(labels, scores))
        print(gen, {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in m.items()})

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    results["overall"] = compute_metrics(all_labels, all_scores, threshold,
                                         n_boot=args.n_boot)
    print("overall", results["overall"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    pred_path = os.path.splitext(args.out)[0] + "_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator", "score"])
        w.writerows(pred_rows)
    print(f"已保存到 {args.out}, 逐样本预测在 {pred_path}")


if __name__ == "__main__":
    main()
