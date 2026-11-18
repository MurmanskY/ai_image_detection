"""CTAD 消融实验 (纯 CPU, 基于 zfeat.npz 缓存)。

消融维度:
  1. tokenizer 子集: 全部组合中选取代表 (单通道、家族对、去掉中性 llamagen 通道、全部)
  2. 聚合方式: min_z (S_min), top2_mean (两个最低 z 的均值取负), mean_low (所有负 z 的均值取负)

用法:
  python analysis/ctad_ablation.py --cache outputs/ctad/zfeat.npz \
      --out outputs/ctad/ablation.json
"""

import argparse
import itertools
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def agg_scores(z, mode):
    if mode == "min":
        return -z.min(axis=1)
    if mode == "top2_mean":
        return -np.sort(z, axis=1)[:, :2].mean(axis=1)
    if mode == "mean_low":
        return -np.minimum(z, 0).mean(axis=1)
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/ctad/zfeat.npz")
    parser.add_argument("--out", default="outputs/ctad/ablation.json")
    args = parser.parse_args()

    d = dict(np.load(args.cache, allow_pickle=True))
    names = [str(k) for k in d["names"]]
    gens = sorted({g.split(":", 1)[1] for g in d["groups"] if g.startswith("1:")})

    subsets = {
        "all": names,
        "no_llamagen": [k for k in names if k != "llamagen"],
        "single": [[k] for k in names],
        "pairs": [list(p) for p in itertools.combinations(names, 2)],
    }
    modes = ["min", "top2_mean", "mean_low"]

    def zs(label, gen, keys):
        return np.stack([d[f"z_{k}_{label}_{gen}"] for k in keys], axis=1)

    results = {"subsets": {}, "modes": {}}

    # 1. 聚合方式消融 (all tokenizers)
    for mode in modes:
        aucs = {}
        for gen in gens:
            sr = agg_scores(zs(0, "real", names), mode)
            sf = agg_scores(zs(1, gen, names), mode)
            y = np.array([0] * len(sr) + [1] * len(sf))
            aucs[gen] = float(roc_auc_score(y, np.concatenate([sr, sf])))
        results["modes"][mode] = {
            "per_generator": aucs,
            "mean": float(np.mean(list(aucs.values()))),
            "min": float(np.min(list(aucs.values()))),
        }
        print(f"mode={mode}: mean={results['modes'][mode]['mean']:.3f} "
              f"min={results['modes'][mode]['min']:.3f}")

    # 2. tokenizer 子集消融 (min 聚合)
    for subset_name, keys in [("all", subsets["all"]),
                              ("no_llamagen", subsets["no_llamagen"])]:
        aucs = {}
        for gen in gens:
            sr = agg_scores(zs(0, "real", keys), "min")
            sf = agg_scores(zs(1, gen, keys), "min")
            y = np.array([0] * len(sr) + [1] * len(sf))
            aucs[gen] = float(roc_auc_score(y, np.concatenate([sr, sf])))
        results["subsets"][subset_name] = {
            "per_generator": aucs,
            "mean": float(np.mean(list(aucs.values()))),
            "min": float(np.min(list(aucs.values()))),
        }
        print(f"subset={subset_name}: mean={results['subsets'][subset_name]['mean']:.3f} "
              f"min={results['subsets'][subset_name]['min']:.3f}")

    for keys in subsets["single"]:
        k = keys[0]
        aucs = {}
        for gen in gens:
            sr = -zs(0, "real", keys)[:, 0]
            sf = -zs(1, gen, keys)[:, 0]
            y = np.array([0] * len(sr) + [1] * len(sf))
            aucs[gen] = float(roc_auc_score(y, np.concatenate([sr, sf])))
        results["subsets"][f"single_{k}"] = {
            "per_generator": aucs,
            "mean": float(np.mean(list(aucs.values()))),
            "min": float(np.min(list(aucs.values()))),
        }
        print(f"single={k}: mean={results['subsets'][f'single_{k}']['mean']:.3f}")

    for keys in subsets["pairs"]:
        tag = "+".join(keys)
        aucs = {}
        for gen in gens:
            sr = agg_scores(zs(0, "real", keys), "min")
            sf = agg_scores(zs(1, gen, keys), "min")
            y = np.array([0] * len(sr) + [1] * len(sf))
            aucs[gen] = float(roc_auc_score(y, np.concatenate([sr, sf])))
        results["subsets"][f"pair_{tag}"] = {
            "per_generator": aucs,
            "mean": float(np.mean(list(aucs.values()))),
            "min": float(np.min(list(aucs.values()))),
        }
        print(f"pair={tag}: mean={results['subsets'][f'pair_{tag}']['mean']:.3f} "
              f"min={results['subsets'][f'pair_{tag}']['min']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
