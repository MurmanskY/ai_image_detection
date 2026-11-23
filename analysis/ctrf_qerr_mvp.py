"""CTRF P0-A MVP: 跨 tokenizer 关系结构检测 (v5 指南主线)。

核心假设 (与 DCA-Fuse 的 min-z 标量不同): 真实图像在多个 tokenizer 下的
qerr 统计存在稳定的跨 tokenizer 依赖结构; AR 生成过程破坏这一联合几何。

流程 (与 v5 算法一致):
  1. 边际秩归一化: 每维 qerr 用 real reference-fit 集的经验 CDF 转 u,
     再经 Φ^-1 转 normal score g (对单调重标定不敏感)
  2. 关系描述子 r(x): 成对差 g_k - g_l (K=5 -> 10 维) + 边际 g (5 维)
  3. real-only 异常评分: shrinkage Mahalanobis / kNN 距离
  4. 阈值/校准: calibration 集 (val real 另一部分) 上 conformal p 值
  5. 对比: S_min (现 CTAD) vs S_rel, per-generator AUC

数据: 复用 outputs/ctad/zfeat.npz 的 test qerr; val real qerr 重新计算
(分 fit/cal 两半)。

用法:
  python analysis/ctrf_qerr_mvp.py --out outputs/ctrf/qerr_relational_mvp.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "D3QE"))
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.ctad_feasibility import TOKENIZERS, compute_qerr_all  # noqa: E402
from datasets.arforensics import ARForensicsDataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402


def empirical_cdf_transform(fit_vals, vals):
    """经验 CDF u = rank/(n+1), 再经 normal-score 变换 g = Φ^-1(u)。"""
    fit_sorted = np.sort(fit_vals)
    n = len(fit_sorted)
    u = np.searchsorted(fit_sorted, vals, side="right") / (n + 1)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u)


def relational_features(g):
    """g: (N, K) -> [g (K) | pairwise diffs (K*(K-1)/2)]"""
    n, k = g.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    diffs = np.stack([g[:, i] - g[:, j] for i, j in pairs], axis=1)
    return np.concatenate([g, diffs], axis=1)


def mahalanobis_scores(fit_r, test_r):
    mu = fit_r.mean(axis=0)
    lw = LedoitWolf().fit(fit_r - mu)
    d = test_r - mu
    return d @ lw.get_precision() @ d.T if False else np.einsum(
        "ij,jk,ik->i", d, lw.precision_, d)


def knn_scores(fit_r, test_r, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(fit_r)
    dists, _ = nn.kneighbors(test_r)
    return dists.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/ctad/zfeat.npz")
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--n_ref", type=int, default=1000)
    parser.add_argument("--out", default="outputs/ctrf/qerr_relational_mvp.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = dict(np.load(args.cache, allow_pickle=True))
    names = [str(k) for k in cache["names"]]
    names = [n for n in names if n in TOKENIZERS]

    # ---- 1. val real qerr (reference fit + calibration), 分两半 ----
    val_qerr = {}
    for name, spec in TOKENIZERS.items():
        if name not in names:
            continue
        print(f"== tokenizer: {name} (val real)")
        model = spec["load"](device)
        ds = ARForensicsDataset(args.root, "val", spec["transform"],
                                max_per_class=args.n_ref)
        ds.samples = [s for s in ds.samples if s[1] == 0]
        loader = torch.utils.data.DataLoader(
            ds, batch_size=spec.get("batch_size", 16), shuffle=False,
            num_workers=4)
        vals = []
        for batch in loader:
            vals.append(spec["qerr"](model, batch["img"].to(device)))
        val_qerr[name] = np.concatenate(vals)
        del model
        torch.cuda.empty_cache()

    n_ref = len(val_qerr[names[0]])
    fit_idx = np.arange(n_ref // 2)
    cal_idx = np.arange(n_ref // 2, n_ref)

    # ---- 2. 每 tokenizer 的经验 CDF 变换 (fit 集为参考) ----
    def to_g(qerr_dict, label, gen):
        return np.stack([empirical_cdf_transform(val_qerr[k][fit_idx],
                                                 qerr_dict[k])
                         for k in names], axis=1)

    g_fit = np.stack([empirical_cdf_transform(val_qerr[k][fit_idx],
                                              val_qerr[k][fit_idx])
                      for k in names], axis=1)
    r_fit = relational_features(g_fit)

    # calibration 集 (real-only conformal 阈值)
    g_cal = np.stack([empirical_cdf_transform(val_qerr[k][fit_idx],
                                              val_qerr[k][cal_idx])
                      for k in names], axis=1)
    r_cal = relational_features(g_cal)

    scorers = {
        "mahalanobis": lambda r: mahalanobis_scores(r_fit, r),
        "knn": lambda r: knn_scores(r_fit, r),
    }
    cal_scores = {name: fn(r_cal) for name, fn in scorers.items()}

    # ---- 3. test 逐生成器评估 ----
    gens = sorted({g.split(":", 1)[1] for g in cache["groups"]
                   if g.startswith("1:")})
    results = {"names": names, "n_ref_fit": len(fit_idx),
               "n_ref_cal": len(cal_idx), "per_generator": {}}

    test_real_z = np.stack([cache[f"z_{k}_0_real"] for k in names], axis=1)
    for gen in gens:
        test_fake_z = np.stack([cache[f"z_{k}_1_{gen}"] for k in names],
                               axis=1)
        # 注意: cache 中是 z 分数; 关系 MVP 用 qerr 原值经经验 CDF 变换,
        # 需原值 = z*std+mean
        q_real = test_real_z * np.array([cache[f"refstd_{k}"] for k in names]) \
            + np.array([cache[f"refmean_{k}"] for k in names])
        q_fake = test_fake_z * np.array([cache[f"refstd_{k}"] for k in names]) \
            + np.array([cache[f"refmean_{k}"] for k in names])
        g_real = to_g(dict(zip(names, q_real.T)), 0, "real")
        g_fake = to_g(dict(zip(names, q_fake.T)), 1, gen)
        r_real, r_fake = relational_features(g_real), relational_features(g_fake)
        labels = np.array([0] * len(r_real) + [1] * len(r_fake))

        entry = {}
        for sname, fn in scorers.items():
            s = np.concatenate([fn(r_real), fn(r_fake)])
            entry[sname] = float(roc_auc_score(labels, s))
        # S_min 对照 (同一组样本)
        s_min = np.concatenate([-test_real_z.min(axis=1),
                                -test_fake_z.min(axis=1)])
        entry["s_min"] = float(roc_auc_score(labels, s_min))
        # conformal p 值示例 (mahalanobis): 报告在 FPR=1% 的 operating point
        results["per_generator"][gen] = entry
        print(gen, json.dumps(entry))

    macro = {}
    for sname in list(scorers) + ["s_min"]:
        aucs = [v[sname] for v in results["per_generator"].values()]
        macro[sname] = {"mean": float(np.mean(aucs)),
                        "min": float(np.min(aucs))}
    results["macro"] = macro
    print(json.dumps(macro, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
