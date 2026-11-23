"""PARD 负结果诊断性消融 (post-hoc, 非 GO 判定): 基于缓存特征。

回答: 370 维描述子中哪些子块携带了跨生成器检测信号?
  - 一阶 only / 二阶 only / 全局 CLS 动态 only
  - 按扰动类型 (jpeg/resize/blur/noise/crop)
  - 按层 (2/5/8/11)

用法: python analysis/pard_ablation.py
"""

import json
import os

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

FEATS = os.path.join(os.path.dirname(__file__), "..", "outputs", "pard",
                     "pard_mvp_feats.npz")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "pard",
                   "pard_ablation.json")

GENERATORS = ["Infinity", "Janus_Pro", "LlamaGen", "Open_MAGVIT2", "RAR",
              "Switti", "VAR"]
PERT_TYPES = ["jpeg", "resize", "blur", "noise", "crop"]
LAYERS = [2, 5, 8, 11]
# 布局: 每层 5 类 x 18 维 (s1: 0-7, s2: 8-15, K: 16-17), 共 360; 全局 360-369
# 全局列顺序: jpeg95, jpeg75, resize90, resize75, blur1, blur2, noise05,
#             noise10, crop90, crop70


def block(li, pi):
    off = li * 90 + pi * 18
    return list(range(off, off + 18))


SUBSETS = {
    "full": list(range(370)),
    "first_order": [c for li in range(4) for pi in range(5)
                    for c in block(li, pi)[:16]],
    "second_order": [c for li in range(4) for pi in range(5)
                     for c in block(li, pi)[16:]],
    "global_only": list(range(360, 370)),
    "patch_only": list(range(360)),
}
for pi, pt in enumerate(PERT_TYPES):
    SUBSETS[f"pert_{pt}"] = [c for li in range(4) for c in block(li, pi)]
for li, l in enumerate(LAYERS):
    SUBSETS[f"layer_{l}"] = [c for pi in range(5) for c in block(li, pi)]


def knn_auc(fit, test, k=10):
    k = min(k, len(fit))
    nn = NearestNeighbors(n_neighbors=k).fit(fit)
    d, _ = nn.kneighbors(test)
    return d.mean(axis=1)


def maha(fit, test):
    mu = fit.mean(axis=0)
    lw = LedoitWolf().fit(fit - mu)
    d = test - mu
    return np.einsum("ij,jk,ik->i", d, lw.precision_, d)


def main():
    data = dict(np.load(FEATS))
    ref = data["ref"]
    fit = ref[: len(ref) // 2]

    results = {}
    for name, cols in SUBSETS.items():
        mu = fit[:, cols].mean(0)
        std = fit[:, cols].std(0) + 1e-8
        fit_z = (fit[:, cols] - mu) / std
        per_gen, per_gen_m = {}, {}
        for gen in GENERATORS:
            rz = (data[f"{gen}_real"][:, cols] - mu) / std
            fz = (data[f"{gen}_fake"][:, cols] - mu) / std
            labels = np.array([0] * len(rz) + [1] * len(fz))
            s_k = np.concatenate([knn_auc(fit_z, rz), knn_auc(fit_z, fz)])
            per_gen[gen] = float(roc_auc_score(labels, s_k))
            s_m = np.concatenate([maha(fit_z, rz), maha(fit_z, fz)])
            per_gen_m[gen] = float(roc_auc_score(labels, s_m))
        results[name] = {
            "knn_macro": float(np.mean(list(per_gen.values()))),
            "knn_worst": float(np.min(list(per_gen.values()))),
            "maha_macro": float(np.mean(list(per_gen_m.values()))),
            "knn_per_gen": per_gen,
        }
        print(f"{name:16s} knn_macro={results[name]['knn_macro']:.4f} "
              f"knn_worst={results[name]['knn_worst']:.4f} "
              f"maha_macro={results[name]['maha_macro']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {OUT}")


if __name__ == "__main__":
    main()
