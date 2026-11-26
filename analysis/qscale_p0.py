"""QSCALE P0: signed quantile-function likelihood evidence (SCALE 失败后的机制修正).

逐 token 双侧尾概率会丢失方向和幅度。QSCALE 将每个 scale 的条件/无条件
log-likelihood difference 分布表示为 signed quantile function，使用真实参考
的收缩 Mahalanobis 或固定 kNN 异常分数，再做 model-bank max 校准。
Quantile L2 几何等价于经验 1-Wasserstein 几何（单调变换下）。

沿用 SCALE P0 的相同 split、PRADA 逐图分数和 GO 标准，不用测试集选参。
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.scale_p0 import (DATASETS, FEAT_DIR, GENS, N_FAKE, OUT as SCALE_OUT,  # noqa: E402
                               load_features)

OUT = os.path.join(PROJ, "outputs", "qscale")
SEEDS = (0, 1, 2)
LEVELS = np.linspace(0.025, 0.975, 16)
METHODS = ("qscale_maha", "qscale_knn")


def quantile_features(diff, scale_lengths):
    rows = []
    for values in diff:
        features = []
        start = 0
        for length in scale_lengths:
            features.extend(np.quantile(values[start:start + length], LEVELS))
            start += length
        rows.append(features)
    return np.asarray(rows, dtype=np.float32)


def maha_fit_score(reference, queries):
    mu = reference.mean(axis=0)
    centered = reference - mu
    cov = np.cov(centered.T)
    cov = cov + 0.1 * np.diag(np.diag(cov)) + 1e-6 * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov)
    outputs = []
    for query in queries:
        diff = query - mu
        outputs.append(np.einsum("ij,jk,ik->i", diff, precision, diff))
    return outputs


def knn_fit_score(reference, queries, k=8):
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0) + 1e-6
    ref = (reference - mu) / sd
    outputs = []
    for query in queries:
        q = (query - mu) / sd
        d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        nearest = np.partition(d2, k - 1, axis=1)[:, :k]
        outputs.append(nearest.mean(axis=1))
    return outputs


def robust_scale(values):
    median = np.median(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25) + 1e-6
    return median, iqr


def metric(y, score, n_real, tau):
    fpr, tpr, _ = roc_curve(y, score)
    return {"AUROC": float(roc_auc_score(y, score)),
            "AP": float(average_precision_score(y, score)),
            "TPR@1%FPR": float(tpr[fpr <= 0.01].max())
            if np.any(fpr <= 0.01) else 0.0,
            "FPR": float((score[:n_real] > tau).mean())}


def macro_auc(per_gen, method):
    return float(np.mean([per_gen[g][method]["AUROC"] for g in GENS]))


def main():
    os.makedirs(OUT, exist_ok=True)
    scale_results = json.load(open(os.path.join(SCALE_OUT, "scale_p0.json")))
    results = {"config": {"levels": LEVELS.tolist(), "knn_k": 8,
                          "protocol": "qscale-p0-darg-t2i-300 示例"},
               "seeds": {}}
    for seed in SEEDS:
        print(f"===== QSCALE seed {seed} =====", flush=True)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(1000)
        ref_idx, cal_idx, test_idx = perm[:500], perm[500:750], perm[750:]
        normalized = {method: {
            ds: np.full(1000, -np.inf, dtype=np.float32) for ds in DATASETS}
            for method in METHODS}
        for model in GENS:
            real_feat = load_features("real", model)
            lengths = real_feat.get("scale_lengths") or [
                real_feat["cond_gt_llh_BL"].shape[1]]
            ref_diff = (real_feat["cond_gt_llh_BL"][ref_idx]
                        - real_feat["uncond_gt_llh_BL"][ref_idx]).float().numpy()
            reference = quantile_features(ref_diff, lengths)
            model_scores = {method: {} for method in METHODS}
            for ds in DATASETS:
                feat = real_feat if ds == "real" else load_features(ds, model)
                diff = (feat["cond_gt_llh_BL"]
                        - feat["uncond_gt_llh_BL"]).float().numpy()
                query = quantile_features(diff, lengths)
                model_scores["qscale_maha"][ds] = maha_fit_score(reference, [query])[0]
                model_scores["qscale_knn"][ds] = knn_fit_score(reference, [query])[0]
            for method in METHODS:
                median, iqr = robust_scale(model_scores[method]["real"][cal_idx])
                for ds in DATASETS:
                    z = (model_scores[method][ds] - median) / iqr
                    normalized[method][ds] = np.maximum(normalized[method][ds], z)
            del real_feat
        baseline = np.load(os.path.join(SCALE_OUT, f"scores_seed{seed}.npz"))
        per_gen, saved = {}, {}
        for g in GENS:
            fake_idx = np.random.RandomState(100 + seed).choice(
                1000, N_FAKE, replace=False)
            y = np.array([0] * 250 + [1] * N_FAKE)
            entry = {}
            for method in METHODS:
                tau = conformal_tau(normalized[method]["real"][cal_idx])
                score = np.concatenate([normalized[method]["real"][test_idx],
                                        normalized[method][g][fake_idx]])
                entry[method] = metric(y, score, 250, tau)
                saved[f"{g}::{method}"] = score
            entry["prada"] = metric(y, baseline[f"{g}::prada"], 250,
                                    conformal_tau(normalized["qscale_maha"]["real"][cal_idx]))
            entry["prada"]["FPR"] = scale_results["seeds"][str(seed)][g]["prada"]["FPR"]
            per_gen[g] = entry
        np.savez(os.path.join(OUT, f"scores_seed{seed}.npz"), **saved)
        results["seeds"][str(seed)] = per_gen
        print("  " + "; ".join(
            f"{m}={macro_auc(per_gen, m):.5f}" for m in METHODS + ("prada",)),
            flush=True)

    summary = {}
    for method in METHODS + ("prada",):
        summary[method] = {
            "macro_AUROC": [macro_auc(results["seeds"][str(s)], method)
                            for s in SEEDS],
            "FPR": [float(np.mean([results["seeds"][str(s)][g][method]["FPR"]
                                   for g in GENS])) for s in SEEDS],
        }
    deltas, cis, not_worse = [], [], []
    for seed in SEEDS:
        z = np.load(os.path.join(OUT, f"scores_seed{seed}.npz"))
        paired = {}
        for g in GENS:
            y = np.array([0] * 250 + [1] * N_FAKE)
            paired[g] = (y, z[f"{g}::qscale_maha"], baseline[f"{g}::prada"])
        delta, ci = paired_boot_macro(paired, n_boot=1000, seed=seed)
        deltas.append(macro_auc(results["seeds"][str(seed)], "qscale_maha")
                      - macro_auc(results["seeds"][str(seed)], "prada"))
        cis.append(ci)
        not_worse.append(sum(
            results["seeds"][str(seed)][g]["qscale_maha"]["AUROC"] + 0.005
            >= results["seeds"][str(seed)][g]["prada"]["AUROC"] for g in GENS))
    go = {
        "delta_per_seed": deltas, "CI95_per_seed": cis,
        "not_worse_with_0.005_tolerance": not_worse,
        "crit1_AUROC>=0.98": bool(min(summary["qscale_maha"]["macro_AUROC"]) >= 0.98),
        "crit1_noninferior_0.002": bool(min(deltas) >= -0.002),
        "crit2_CI_lower>-0.005": bool(all(ci[0] > -0.005 for ci in cis)),
        "crit3_FPR": bool(all(0.03 <= x <= 0.07
                              for x in summary["qscale_maha"]["FPR"])),
        "crit4_gens>=3/4": bool(all(x >= 3 for x in not_worse)),
    }
    go["GO"] = bool(go["crit1_AUROC>=0.98"]
                    and go["crit1_noninferior_0.002"]
                    and go["crit2_CI_lower>-0.005"] and go["crit3_FPR"]
                    and go["crit4_gens>=3/4"])
    results["summary"] = summary
    results["go"] = go
    with open(os.path.join(OUT, "qscale_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(go, indent=2), flush=True)
    print(f"已保存 {OUT}/qscale_p0.json", flush=True)


if __name__ == "__main__":
    main()
