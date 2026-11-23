"""DCA-Fuse 全量 test (84k) 融合评估。

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
import torch

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "D3QE"))
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.ctad_feasibility import TOKENIZERS  # noqa: E402
from datasets.arforensics import ARForensicsDataset  # noqa: E402
from engine.metrics import (  # noqa: E402
    calibrate_threshold,
    compute_metrics,
    paired_bootstrap_delta,
)


def load_univfd_predictions(path):
    by_gen = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["label"]), r["generator"])
            image_path = os.path.abspath(os.path.normpath(r["path"]))
            group = by_gen.setdefault(key, {})
            if image_path in group:
                raise ValueError(f"duplicate prediction path: {r['path']}")
            group[image_path] = float(r["score"])
    return by_gen


def smin_from_cache(cache, names, label, gen):
    zs = np.stack([cache[f"z_{k}_{label}_{gen}"] for k in names], axis=1)
    return -zs.min(axis=1)


def align_predictions(predictions, key, paths):
    normalized_paths = [os.path.abspath(os.path.normpath(path)) for path in paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError(f"duplicate image paths in CTAD features for {key}")
    scores_by_path = predictions[key]
    if set(scores_by_path) != set(normalized_paths):
        missing = len(set(normalized_paths) - set(scores_by_path))
        unexpected = len(set(scores_by_path) - set(normalized_paths))
        raise ValueError(
            f"prediction/feature path mismatch for {key}: "
            f"{missing} missing, {unexpected} unexpected")
    return np.asarray([scores_by_path[path] for path in normalized_paths])


def compute_validation_scores(root, val_predictions, cache, names, alpha,
                              batch_size, device):
    """Recompute fused validation scores so threshold matches the requested alpha."""
    val_real_paths = list(val_predictions[(0, "real")])
    val_fake_paths = list(val_predictions[(1, "LlamaGen")])
    val_real_u = align_predictions(val_predictions, (0, "real"), val_real_paths)
    val_fake_u = align_predictions(val_predictions, (1, "LlamaGen"), val_fake_paths)
    mu1, sd1 = val_real_u.mean(), val_real_u.std()
    if not np.isfinite(sd1) or sd1 <= 0:
        raise ValueError("UnivFD real validation scores have zero variance")

    qerr_by_label = {0: {}, 1: {}}
    for name, spec in TOKENIZERS.items():
        if name not in names:
            continue
        model = spec["load"](device)
        for label in (0, 1):
            expected_paths = val_real_paths if label == 0 else val_fake_paths
            dataset = ARForensicsDataset(
                root, "val", spec["transform"], max_per_class=1000)
            dataset.samples = [sample for sample in dataset.samples
                               if sample[1] == label]
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=spec.get("batch_size", batch_size),
                shuffle=False, num_workers=4)
            values = []
            paths = []
            for batch in loader:
                values.append(spec["qerr"](model, batch["img"].to(device)))
                paths.extend(os.path.abspath(os.path.normpath(path))
                             for path in batch["path"])
            normalized_expected = [
                os.path.abspath(os.path.normpath(path)) for path in expected_paths]
            if len(set(paths)) != len(paths) or set(paths) != set(normalized_expected):
                raise ValueError(
                    f"{name} validation feature paths do not match UnivFD predictions")
            value_by_path = dict(zip(paths, np.concatenate(values)))
            qerr_by_label[label][name] = np.asarray(
                [value_by_path[path] for path in normalized_expected])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def val_smin(label):
        z_values = np.stack([
            (qerr_by_label[label][name] - float(cache[f"refmean_{name}"])) /
            (float(cache[f"refstd_{name}"]) + 1e-12)
            for name in names
        ], axis=1)
        return -z_values.min(axis=1)

    val_ctad_real = val_smin(0)
    val_ctad_fake = val_smin(1)
    mu2, sd2 = val_ctad_real.mean(), val_ctad_real.std()
    if not np.isfinite(sd2) or sd2 <= 0:
        raise ValueError("CTAD real validation scores have zero variance")
    val_scores = np.concatenate([
        alpha * (val_real_u - mu1) / sd1 +
        (1 - alpha) * (val_ctad_real - mu2) / sd2,
        alpha * (val_fake_u - mu1) / sd1 +
        (1 - alpha) * (val_ctad_fake - mu2) / sd2,
    ])
    val_labels = np.concatenate([
        np.zeros(len(val_real_u), dtype=int),
        np.ones(len(val_fake_u), dtype=int),
    ])
    return val_labels, val_scores, (mu1, sd1, mu2, sd2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--univfd_pred",
                        default="outputs/univfd_arforensics/results_test_full_predictions.csv")
    parser.add_argument("--univfd_val_pred",
                        default="outputs/univfd_arforensics/results_val_predictions.csv")
    parser.add_argument("--cache", default="outputs/ctad/zfeat_full.npz")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--out", default="outputs/dca_fuse/results_test_full.json")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    names = list(np.load(args.cache, allow_pickle=True)["names"])
    names = [str(k) for k in names]
    cache = dict(np.load(args.cache, allow_pickle=True))

    # ---- val: calibrate with scores produced at the requested alpha ----
    uv = load_univfd_predictions(args.univfd_val_pred)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_labels, val_scores, (mu1, sd1, mu2, sd2) = compute_validation_scores(
        args.root, uv, cache, names, args.alpha, args.batch_size, device)
    threshold = calibrate_threshold(val_labels, val_scores)
    print(f"val 校准阈值 (α={args.alpha}): {threshold:.4f}")

    # ---- test 全量融合 ----
    u = load_univfd_predictions(args.univfd_pred)
    gens = sorted({g for (l, g) in u if l == 1})
    results = {"alpha": args.alpha, "threshold": threshold,
               "threshold_rule": "validation Youden's J (recomputed for this alpha)",
               "evaluation_protocol": {
                   "per_generator": "shared real pool vs. that generator's fake pool",
                   "overall": "shared real pool counted once vs. pooled fake images",
                   "score_alignment": "join UnivFD by normalized image path",
               },
               "per_generator": {}}
    pred_rows = []
    probe = ARForensicsDataset(args.root, "test", None)
    real_paths = [path for path, label, _ in probe.samples if label == 0]
    real_u = align_predictions(u, (0, "real"), real_paths)
    real_c = smin_from_cache(cache, names, 0, "real")
    if len(real_u) != len(real_c):
        raise ValueError("UnivFD and CTAD real test features do not align")
    cache_real_paths = cache.get("paths_0_real")
    if cache_real_paths is not None:
        normalized_real_cache_paths = {
            os.path.abspath(os.path.normpath(path)) for path in cache_real_paths
        }
        if normalized_real_cache_paths != {
                os.path.abspath(os.path.normpath(path)) for path in real_paths}:
            raise ValueError("CTAD cache real paths do not match test data")
    test_real_paths = [
        os.path.abspath(os.path.normpath(path)) for path in real_paths]
    real_scores = (args.alpha * (real_u - mu1) / sd1 +
                   (1 - args.alpha) * (real_c - mu2) / sd2)
    pred_rows.extend((path, 0, "real", float(score))
                     for path, score in zip(real_paths, real_scores))
    fake_scores = []
    baseline_fake_scores = []
    for gen in gens:
        fake_paths = [
            path for path, label, source in probe.samples
            if label == 1 and source == gen
        ]
        uf = align_predictions(u, (1, gen), fake_paths)
        cf = smin_from_cache(cache, names, 1, gen)
        if len(uf) != len(cf):
            raise ValueError(f"UnivFD and CTAD test features do not align for {gen}")
        feature_paths = [
            os.path.abspath(os.path.normpath(path)) for path in fake_paths]
        if len(set(feature_paths)) != len(feature_paths):
            raise ValueError(f"duplicate CTAD fake feature paths for {gen}")
        cache_paths = cache.get(f"paths_1_{gen}")
        if cache_paths is not None and {
                os.path.abspath(os.path.normpath(path)) for path in cache_paths
        } != set(feature_paths):
            raise ValueError(f"CTAD cache paths do not match dataset paths for {gen}")
        n = len(uf)
        fused_fake = (args.alpha * (uf - mu1) / sd1 +
                      (1 - args.alpha) * (cf - mu2) / sd2)
        scores = np.concatenate([
            real_scores, fused_fake,
        ])
        labels = np.array([0] * len(real_u) + [1] * n)
        m = compute_metrics(labels, scores, threshold, n_boot=args.n_boot)
        m["paired_auc_delta_vs_univfd"] = paired_bootstrap_delta(
            labels, np.concatenate([real_u, uf]), scores, n_boot=args.n_boot)
        results["per_generator"][gen] = m
        fake_scores.append(fused_fake)
        baseline_fake_scores.append(uf)
        pred_rows.extend((path, 1, gen, float(score))
                         for path, score in zip(fake_paths, fused_fake))
        print(gen, {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in m.items()})

    pooled_fake_scores = np.concatenate(fake_scores)
    all_labels = np.concatenate([
        np.zeros(len(real_scores), dtype=int),
        np.ones(len(pooled_fake_scores), dtype=int),
    ])
    all_scores = np.concatenate([real_scores, pooled_fake_scores])
    baseline_scores = np.concatenate([
        real_u, np.concatenate(baseline_fake_scores),
    ])
    results["evaluation_protocol"]["n_real"] = len(real_scores)
    results["evaluation_protocol"]["n_fake"] = len(pooled_fake_scores)
    results["overall"] = compute_metrics(all_labels, all_scores, threshold,
                                         n_boot=args.n_boot)
    results["overall"]["paired_auc_delta_vs_univfd"] = paired_bootstrap_delta(
        all_labels, baseline_scores, all_scores, n_boot=args.n_boot)
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
