"""SCALE P0: Sequence-Calibrated Autoregressive Likelihood Evidence.

PRADA 使用学习得到的 token-wise scoring；SCALE 仅用真实参考的逐位置
经验双侧尾概率，scale 内 Fisher/Higher-Criticism 聚合，再对 model-bank
max score 做独立真实序列级校准。无需 fake 训练，无需 token 独立假设。

P0: DARG T2I 1000 real 三 seed 分 500 reference/250 calibration/250 test；
每生成器 300 fake。主方法 Fisher，HC 为固定消融，不做 test 选择。

非劣 GO (预注册):
  1. SCALE-Fisher macro AUROC>=0.98 且 >= PRADA-0.002;
  2. 相对 PRADA paired ΔAUROC 95%CI 下界 > -0.005;
  3. 每 seed real FPR∈[3%,7%];
  4. 至少3/4生成器相对PRADA退化不超过0.005。
通过后才做 C2I/更多模型验证，否则停止。

用法: python analysis/scale_p0.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from prada.prada import PRADA  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "scale")
FEAT_DIR = os.path.join(PROJ, "outputs", "prada_t2i", "features")
WEIGHT_DIR = os.path.join(PROJ, "third_party", "PRADA", "prada_weights")
GENS = ("infinity_2b", "janus_1b", "llamagen_xlstage2", "switti_1024")
DATASETS = ("real",) + GENS
SEEDS = (0, 1, 2)
N_FAKE = 300
METHODS = ("scale_fisher", "scale_hc", "prada")


def load_features(dataset, model):
    return torch.load(os.path.join(
        FEAT_DIR, f"dataset={dataset}_model={model}.pt"),
        map_location="cpu", weights_only=False)


def two_tail_surprisal(reference, query):
    reference = np.asarray(reference, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    n = len(reference)
    p = np.empty_like(query)
    sorted_ref = np.sort(reference, axis=0)
    for j in range(reference.shape[1]):
        left = np.searchsorted(sorted_ref[:, j], query[:, j], side="left")
        right = np.searchsorted(sorted_ref[:, j], query[:, j], side="right")
        lower = (left + 1) / (n + 1)
        upper = (n - right + 1) / (n + 1)
        p[:, j] = np.minimum(1.0, 2 * np.minimum(lower, upper))
    p = np.maximum(p, 1 / (n + 1))
    return p, -np.log(p)


def aggregate_scales(p, surprisal, scale_lengths, method):
    start = 0
    scores = []
    for length in scale_lengths:
        ps = p[:, start:start + length]
        ss = surprisal[:, start:start + length]
        if method == "fisher":
            scores.append(ss.mean(axis=1))
        elif method == "hc":
            ordered = np.sort(ps, axis=1)
            n = ordered.shape[1]
            expected = np.arange(1, n + 1) / n
            variance = np.maximum(ordered * (1 - ordered) / n, 1e-8)
            hc = ((expected - ordered) / np.sqrt(variance)).max(axis=1)
            scores.append(hc)
        else:
            raise ValueError(method)
        start += length
    assert start == p.shape[1]
    return np.max(np.stack(scores, axis=1), axis=1)


def prada_score(features, model):
    stacked = torch.stack(
        [features["cond_gt_llh_BL"], features["uncond_gt_llh_BL"]], dim=-1).float()
    net = PRADA(n_hidden=16, scale_lengths=features.get("scale_lengths"),
                input_dim=1, learnable_difference=True,
                learnable_stage_weights=True, training_noise=0.1)
    net.load_state_dict(torch.load(os.path.join(
        WEIGHT_DIR, model, "final.pt"), map_location="cpu", weights_only=True))
    net.eval()
    with torch.no_grad():
        return net(stacked).numpy()


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
    results = {"config": {"seeds": SEEDS, "reference": 500, "calibration": 250,
                          "test_real": 250, "test_fake": N_FAKE,
                          "protocol": "scale-p0-darg-t2i-300 示例"},
               "seeds": {}}
    for seed in SEEDS:
        print(f"===== SCALE seed {seed} =====", flush=True)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(1000)
        ref_idx, cal_idx, test_idx = perm[:500], perm[500:750], perm[750:]
        combined = {method: {} for method in METHODS}
        for model in GENS:
            real_feat = load_features("real", model)
            reference = (real_feat["cond_gt_llh_BL"][ref_idx]
                         - real_feat["uncond_gt_llh_BL"][ref_idx]).float().numpy()
            lengths = real_feat.get("scale_lengths") or [
                real_feat["cond_gt_llh_BL"].shape[1]]
            for method in METHODS:
                combined[method]["real"] = np.full(1000, -np.inf, dtype=np.float32)
            for ds in DATASETS:
                feat = real_feat if ds == "real" else load_features(ds, model)
                query = (feat["cond_gt_llh_BL"]
                         - feat["uncond_gt_llh_BL"]).float().numpy()
                p, surprisal = two_tail_surprisal(reference, query)
                for method in ("fisher", "hc"):
                    name = f"scale_{method}"
                    score = aggregate_scales(p, surprisal, lengths, method)
                    if ds not in combined[name]:
                        combined[name][ds] = score
                    else:
                        combined[name][ds] = np.maximum(combined[name][ds], score)
                prada = prada_score(feat, model)
                if ds not in combined["prada"]:
                    combined["prada"][ds] = prada
                else:
                    combined["prada"][ds] = np.maximum(
                        combined["prada"][ds], prada)
            del real_feat
        per_gen, saved = {}, {}
        for g in GENS:
            fake_idx = np.random.RandomState(100 + seed).choice(
                1000, N_FAKE, replace=False)
            n_real, n_fake = len(test_idx), len(fake_idx)
            y = np.array([0] * n_real + [1] * n_fake)
            entry = {}
            for method in METHODS:
                real_score = combined[method]["real"]
                tau = conformal_tau(real_score[cal_idx])
                score = np.concatenate([real_score[test_idx],
                                        combined[method][g][fake_idx]])
                entry[method] = metric(y, score, n_real, tau)
                saved[f"{g}::{method}"] = score
            per_gen[g] = entry
        np.savez(os.path.join(OUT, f"scores_seed{seed}.npz"), **saved)
        results["seeds"][str(seed)] = per_gen
        print("  " + "; ".join(
            f"{m}={macro_auc(per_gen, m):.5f}" for m in METHODS), flush=True)

    summary = {}
    for method in METHODS:
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
            n = len(z[f"{g}::scale_fisher"])
            y = np.array([0] * 250 + [1] * N_FAKE)
            paired[g] = (y, z[f"{g}::scale_fisher"], z[f"{g}::prada"])
        delta, ci = paired_boot_macro(paired, n_boot=1000, seed=seed)
        deltas.append(macro_auc(results["seeds"][str(seed)], "scale_fisher")
                      - macro_auc(results["seeds"][str(seed)], "prada"))
        cis.append(ci)
        not_worse.append(sum(
            results["seeds"][str(seed)][g]["scale_fisher"]["AUROC"] + 0.005
            >= results["seeds"][str(seed)][g]["prada"]["AUROC"] for g in GENS))
    go = {
        "delta_per_seed": deltas, "CI95_per_seed": cis,
        "not_worse_with_0.005_tolerance": not_worse,
        "crit1_AUROC>=0.98": bool(min(summary["scale_fisher"]["macro_AUROC"]) >= 0.98),
        "crit1_noninferior_0.002": bool(min(deltas) >= -0.002),
        "crit2_CI_lower>-0.005": bool(all(ci[0] > -0.005 for ci in cis)),
        "crit3_FPR": bool(all(0.03 <= x <= 0.07
                              for x in summary["scale_fisher"]["FPR"])),
        "crit4_gens>=3/4": bool(all(x >= 3 for x in not_worse)),
    }
    go["GO"] = bool(go["crit1_AUROC>=0.98"]
                    and go["crit1_noninferior_0.002"]
                    and go["crit2_CI_lower>-0.005"] and go["crit3_FPR"]
                    and go["crit4_gens>=3/4"])
    results["summary"] = summary
    results["go"] = go
    with open(os.path.join(OUT, "scale_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(go, indent=2), flush=True)
    print(f"已保存 {OUT}/scale_p0.json", flush=True)


if __name__ == "__main__":
    main()
