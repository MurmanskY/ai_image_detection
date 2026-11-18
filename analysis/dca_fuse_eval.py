"""DCA-Fuse 最终评估: 双通道融合检测器的定型 test 评估。

通道 1: UnivFD (冻结 CLIP ViT-L/14 + 线性头, outputs/univfd_arforensics/best.pth)
通道 2: CTAD S_min (5 个 VQ tokenizer 的 qerr min-z, 免训练)
融合: score = α · z1 + (1-α) · z2, z 均以 val real 参考归一; α 先验固定 0.7。

协议: 阈值只用 val (Youden's J on fused val scores); test 只评估一次。
输出 overall + per-generator 指标 (AUC/AP/TPR@FPR/FPR@95TPR) 与
predictions.csv, 可直接归档。

用法:
  python analysis/dca_fuse_eval.py --n 500 --alpha 0.7 \
      --out outputs/dca_fuse/results_test.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "D3QE"))
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.ctad_feasibility import TOKENIZERS  # noqa: E402
from datasets.arforensics import ARForensicsDataset  # noqa: E402
from datasets.build import build_dataset  # noqa: E402
from engine.evaluator import predict_scores  # noqa: E402
from engine.metrics import calibrate_threshold, compute_metrics  # noqa: E402
from utils.config import load_config  # noqa: E402
from baselines import build_method  # noqa: E402


def smin_from_cache(cache, names, label, gen):
    zs = np.stack([cache[f"z_{k}_{label}_{gen}"] for k in names], axis=1)
    return -zs.min(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/univfd_arforensics.yaml")
    parser.add_argument("--checkpoint",
                        default="outputs/univfd_arforensics/best.pth")
    parser.add_argument("--cache", default="outputs/ctad/zfeat.npz")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--out", default="outputs/dca_fuse/results_test.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    method = build_method(cfg.method)

    # ---------- 通道 1: UnivFD 在 val 与 test(sub500) 上的分数 ----------
    def channel1(split):
        ds = build_dataset(cfg.dataset, split, method.get_transform(train=False))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.eval.get("batch_size", 128), shuffle=False,
            num_workers=4, pin_memory=True)
        model = method.build_model()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        labels, scores, gens, paths = predict_scores(
            model, loader, device, cfg.train.get("amp", True), method)
        return labels, scores, gens

    # val 全量 (1000+1000)
    vl, vs, _ = channel1("val")

    # test sub500: 直接用已存在的 sub500 预测 (与 ctad 采样一致的协议)
    import csv
    u = {}
    with open("outputs/univfd_arforensics/results_test_sub500_predictions.csv") as f:
        for r in csv.DictReader(f):
            u.setdefault((r["label"], r["generator"]), []).append(float(r["score"]))

    cache = dict(np.load(args.cache, allow_pickle=True))
    names = list(TOKENIZERS)  # 只用保留的通道 (缓存可含被剔除通道, 忽略)

    # ---------- 融合参考统计 (val) ----------
    # 通道 1 归一化用 val real; 通道 2 归一化用 val real 的 S_min 分布;
    # 阈值规则: val 融合分数上的 Youden's J, test 冻结一次评估。

    # 通道 1 归一化: val real 统计 (vs 里 label==0 的部分)
    vs = np.asarray(vs)
    vl = np.asarray(vl)
    mu1, sd1 = vs[vl == 0].mean(), vs[vl == 0].std()

    # 通道 2 val 分数: 重新在 val 上快速计算 5 个 tokenizer 的 qerr
    from analysis.ctad_feasibility import compute_qerr_all
    val_groups = [(0, "real"), (1, "LlamaGen")]
    z_val = {}
    for name, spec in TOKENIZERS.items():
        model = spec["load"](device)
        out = {}
        mu_ref, sd_ref = float(cache[f"refmean_{name}"]), float(cache[f"refstd_{name}"])
        for (label, gen) in val_groups:
            ds = ARForensicsDataset("data/ARForensics", "val", spec["transform"],
                                    max_per_class=1000)
            ds.samples = [s for s in ds.samples if s[1] == label]
            loader = torch.utils.data.DataLoader(
                ds, batch_size=spec.get("batch_size", 16), shuffle=False,
                num_workers=4)
            vals = []
            for batch in loader:
                vals.append(spec["qerr"](model, batch["img"].to(device)))
            q = np.concatenate(vals)
            out[(label, gen)] = q
        z_val[name] = out
        del model
        torch.cuda.empty_cache()

    def val_smin(label):
        zs = np.stack([(z_val[k][(label, "real" if label == 0 else "LlamaGen")]
                        - float(cache[f"refmean_{k}"])) /
                       (float(cache[f"refstd_{k}"]) + 1e-12) for k in names], axis=1)
        return -zs.min(axis=1)

    vs2_real, vs2_fake = val_smin(0), val_smin(1)
    mu2, sd2 = vs2_real.mean(), vs2_real.std()

    val_fused = np.concatenate([
        args.alpha * (vs[vl == 0] - mu1) / sd1 + (1 - args.alpha) * (vs2_real - mu2) / sd2,
        args.alpha * (vs[vl == 1] - mu1) / sd1 + (1 - args.alpha) * (vs2_fake - mu2) / sd2,
    ])
    val_labels = np.array([0] * (vl == 0).sum() + [1] * (vl == 1).sum())
    threshold = calibrate_threshold(val_labels, val_fused)
    print(f"val 校准阈值 (α={args.alpha}): {threshold:.4f}")

    # ---------- test 评估 ----------
    test_real_u = np.array(u[("0", "real")])
    test_real_c = smin_from_cache(cache, names, 0, "real")
    results = {"alpha": args.alpha, "threshold": float(threshold),
               "threshold_rule": "val Youden's J", "per_generator": {}}

    gens = sorted({g for (l, g) in u if l == "1"})
    all_labels, all_scores = [], []
    pred_rows = []
    for gen in gens:
        uf = np.array(u[("1", gen)])
        cf = smin_from_cache(cache, names, 1, gen)
        n = min(len(uf), len(cf))
        scores = np.concatenate([
            args.alpha * (test_real_u - mu1) / sd1 + (1 - args.alpha) * (test_real_c - mu2) / sd2,
            args.alpha * (uf[:n] - mu1) / sd1 + (1 - args.alpha) * (cf[:n] - mu2) / sd2,
        ])
        labels = np.array([0] * len(test_real_u) + [1] * n)
        m = compute_metrics(labels, scores, threshold, n_boot=args.n_boot)
        results["per_generator"][gen] = m
        all_labels.append(labels)
        all_scores.append(scores)
        pred_rows.extend(
            (f"test/{gen}/{'0_real' if l == 0 else '1_fake'}", int(l), gen, float(s))
            for l, s in zip(labels, scores)
        )
        print(gen, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in m.items()}))
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    results["overall"] = compute_metrics(all_labels, all_scores, threshold,
                                         n_boot=args.n_boot)
    print("overall", json.dumps(results["overall"], indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    pred_path = os.path.splitext(args.out)[0] + "_predictions.csv"
    import csv as _csv
    with open(pred_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["path", "label", "generator", "score"])
        w.writerows(pred_rows)
    print(f"已保存到 {args.out}, 逐样本预测在 {pred_path}")


if __name__ == "__main__":
    main()
