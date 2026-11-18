"""CTAD 实验 B: LOGO (leave-one-generator-out) 学习组合评估。

在实验 A 的 z 特征 (每 tokenizer 的 qerr z 分数) 上:
  - 特征: [z_1..z_K, min_z, spread_z] per image
  - 协议: 用 6 个生成器的 fake + real 训练轻量分类器, 测试留出的第 7 个,
          轮换全部 7 个生成器 (LOGO); real 参考/校准只用 val real。
  - 分类器: LogisticRegression 与浅层 MLP (sklearn)
  - 同时报告训练-free 的 S_min 作为对照

z 特征缓存到 outputs/ctad/zfeat.npz, 重复运行时可 --load_cache 跳过特征计算。

用法:
  python analysis/ctad_logo.py --n 500 --out outputs/ctad/logo.json
  python analysis/ctad_logo.py --load_cache outputs/ctad/zfeat.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "D3QE"))
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.ctad_feasibility import TOKENIZERS, compute_qerr_all  # noqa: E402
from datasets.arforensics import ARForensicsDataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402


def build_zfeat(root, n, n_ref, batch_size, device, cache_path):
    probe = ARForensicsDataset(root, "test", T.Compose([T.ToTensor()]),
                               max_per_class=n)
    groups = sorted({(label, gen) for _, label, gen in probe.samples})
    names = list(TOKENIZERS)

    ref = {}
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (val real 参考)")
        model = spec["load"](device)
        ds = ARForensicsDataset(root, "val", spec["transform"],
                                max_per_class=n_ref)
        ds.samples = [s for s in ds.samples if s[1] == 0]
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4)
        vals = []
        for batch in loader:
            vals.append(spec["qerr"](model, batch["img"].to(device)))
        ref[name] = np.concatenate(vals)
        del model
        torch.cuda.empty_cache()

    data = {"groups": np.array([f"{l}:{g}" for l, g in groups]),
            "names": np.array(names)}
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (test)")
        qerr = compute_qerr_all(name, spec, groups, root, n, batch_size, device)
        for (label, gen) in groups:
            z = (qerr[(label, gen)] - ref[name].mean()) / (ref[name].std() + 1e-12)
            data[f"z_{name}_{label}_{gen}"] = z
        data[f"refmean_{name}"] = ref[name].mean()
        data[f"refstd_{name}"] = ref[name].std()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **data)
    print(f"z 特征已缓存到 {cache_path}")
    return data


def gen_features(data, names, label, gen):
    zs = np.stack([data[f"z_{k}_{label}_{gen}"] for k in names], axis=1)
    return np.concatenate(
        [zs, zs.min(axis=1, keepdims=True), zs.std(axis=1, keepdims=True)], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--n_ref", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--load_cache", default=None)
    parser.add_argument("--out", default="outputs/ctad/logo.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = args.load_cache or "outputs/ctad/zfeat.npz"
    if args.load_cache and os.path.isfile(args.load_cache):
        data = dict(np.load(args.load_cache, allow_pickle=True))
        print(f"已加载缓存 {args.load_cache}")
    else:
        data = build_zfeat(args.root, args.n, args.n_ref,
                           args.batch_size, device, cache)

    names = [str(k) for k in data["names"]]
    gens = sorted({g.split(":", 1)[1] for g in data["groups"] if g.startswith("1:")})
    real_gen = "real"

    results = {"per_generator": {}, "n_per_class": args.n}
    for held in gens:
        # LOGO: 训练集 = 除 held 外全部生成器
        xtr, ytr = [], []
        for gen in gens:
            if gen == held:
                continue
            xtr.append(gen_features(data, names, 1, gen))
            ytr.append(np.ones(len(data[f"z_{names[0]}_1_{gen}"])))
        xr = gen_features(data, names, 0, real_gen)
        xtr.append(xr)
        ytr.append(np.zeros(len(xr)))
        xtr, ytr = np.concatenate(xtr), np.concatenate(ytr)

        xte_fake = gen_features(data, names, 1, held)
        xte = np.concatenate([xr, xte_fake])
        yte = np.array([0] * len(xr) + [1] * len(xte_fake))

        entry = {}
        for clf_name, clf in [
            ("lr", LogisticRegression(max_iter=2000)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                                  random_state=42)),
        ]:
            scaler = StandardScaler().fit(xtr)
            clf.fit(scaler.transform(xtr), ytr)
            scores = clf.predict_proba(scaler.transform(xte))[:, 1]
            entry[clf_name] = {"auc": float(roc_auc_score(yte, scores))}

        # 训练-free 对照: S_min
        s_real = -gen_features(data, names, 0, real_gen)[:, len(names)]
        s_fake = -gen_features(data, names, 1, held)[:, len(names)]
        entry["S_min_trainfree"] = {
            "auc": float(roc_auc_score(yte, np.concatenate([s_real, s_fake])))}

        results["per_generator"][held] = entry
        print(held, json.dumps(entry))

    for clf_name in ("lr", "mlp", "S_min_trainfree"):
        aucs = [v[clf_name]["auc"] for v in results["per_generator"].values()]
        results.setdefault("macro", {})[clf_name] = {
            "auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs))}
    print(json.dumps(results["macro"], indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
