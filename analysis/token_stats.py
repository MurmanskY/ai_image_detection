"""D³QE 核心统计分析 v2 (调研文档建议的阶段二实验 + 审查报告 P1 扩展)。

在 v1 (codebook 使用频率 / per-image token entropy / quantization error) 基础上新增:
  - effective vocabulary size = exp(H), top-k mass, Gini coefficient
  - 2D transition statistics: 水平/垂直条件熵 H(t_next|t_cur), bigram 互信息
  - quantization error 的 mean/std/skew/kurtosis
  - real vs fake 聚合直方图的 JS divergence
  - 效应量 (Cohen's d, Cliff's delta) 与 bootstrap AUROC 置信区间
  - per-image 特征导出 CSV (供后续统计显著性分析 / logistic regression)

用法:
  python analysis/token_stats.py \
      --root data/ARForensics --split test --models LlamaGen VAR \
      --vq_ckpt pretrained/vq_ds16_c2i.pt --n 500 --out outputs/analysis
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image  # noqa: F401  (dataset 依赖)
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "D3QE"))

from networks.vq_model import ModelArgs, VQModel  # noqa: E402

from datasets.arforensics import ARForensicsDataset  # noqa: E402

GRID = 16  # 256px / ds16 = 16x16 token grid


def load_vq(ckpt_path, device):
    model = VQModel(ModelArgs())
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_transform():
    return T.Compose([
        T.Resize(256), T.CenterCrop(256), T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def encode_tokens(vq, img, device):
    """返回 (token_indices (b*256,), 每个 token 位置的量化误差)。"""
    quant, _, info = vq.encode(img.to(device))
    z_q, z = quant  # z_q: (B,C,H,W), z: (B,H,W,C)
    indices = info[2]
    qerr = (z_q - z.permute(0, 3, 1, 2)).pow(2).mean(dim=1).flatten(1)
    return indices.cpu(), qerr.cpu()


def entropy_of(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def gini(counts):
    """非零计数的 Gini 系数 (越大越集中)。"""
    x = np.sort(counts[counts > 0].astype(np.float64))
    n = len(x)
    if n <= 1 or x.sum() == 0:
        return 0.0
    return float((2.0 * np.sum((np.arange(n) + 1) * x) / (n * x.sum())) - (n + 1) / n)


def pair_stats(a, b):
    """给定配对序列 (a, b), 返回条件熵 H(b|a) 与互信息 I(a;b)。"""
    ua, ca = np.unique(a, return_counts=True)
    ha = entropy_of(ca)
    ub, cb = np.unique(b, return_counts=True)
    hb = entropy_of(cb)
    # a, b 值域相同 (codebook < 2^15), 用 (a << 15) + b 编码配对
    joint = (a.astype(np.int64) << 15) + b.astype(np.int64)
    _, cj = np.unique(joint, return_counts=True)
    hj = entropy_of(cj)
    return hj - ha, ha + hb - hj


def transition_features(tokens):
    """tokens: (256,) -> 水平/垂直条件熵与互信息。"""
    g = tokens.reshape(GRID, GRID)
    hc, hmi = pair_stats(g[:, :-1].ravel(), g[:, 1:].ravel())
    vc, vmi = pair_stats(g[:-1, :].ravel(), g[1:, :].ravel())
    return hc, hmi, vc, vmi


def moments(x):
    """mean, std, skew, kurtosis (Fisher)。"""
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std()
    if s == 0:
        return m, s, 0.0, 0.0
    z = (x - m) / s
    return float(m), float(s), float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def js_divergence(c1, c2):
    """两组聚合直方图的 Jensen-Shannon divergence (自然对数)。"""
    p = c1 / max(c1.sum(), 1)
    q = c2 / max(c2.sum(), 1)
    m = 0.5 * (p + q)
    mask = m > 0
    js = 0.5 * np.sum(np.where(p[mask] > 0, p[mask] * np.log(p[mask] / m[mask]), 0))
    js += 0.5 * np.sum(np.where(q[mask] > 0, q[mask] * np.log(q[mask] / m[mask]), 0))
    return float(js)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def cliffs_delta(x, y):
    """基于 Mann-Whitney U 的 Cliff's delta, 正值表示 x 偏大。"""
    from scipy.stats import mannwhitneyu
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def bootstrap_auc(labels, scores, n_boot=1000, seed=42):
    """labels: 0=real 1=fake; 返回 (auc, lo95, hi95)。"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    auc = roc_auc_score(labels, scores)
    return float(auc), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


FEATURES = [
    "token_entropy", "unique_ratio", "gini",
    "h_trans_entropy", "h_bigram_mi", "v_trans_entropy", "v_bigram_mi",
    "qerr_mean", "qerr_std", "qerr_skew", "qerr_kurtosis",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="*", default=None,
                        help="test 划分下的模型子集, 默认全部")
    parser.add_argument("--vq_ckpt", default="pretrained/vq_ds16_c2i.pt")
    parser.add_argument("--n", type=int, default=500, help="每类最多采样图像数")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")
    vq = load_vq(args.vq_ckpt, device)
    codebook_size = vq.config.codebook_size

    dataset = ARForensicsDataset(
        args.root, args.split, get_transform(),
        models=args.models, max_per_class=args.n,
    )
    print(f"共 {len(dataset)} 张图像")

    groups = {}
    for _, label, gen in dataset.samples:
        groups.setdefault((label, gen), []).append(None)

    counts = {k: np.zeros(codebook_size, dtype=np.int64) for k in groups}
    feats = {k: {f: [] for f in FEATURES} for k in groups}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )
    done = 0
    for batch in loader:
        indices, qerr = encode_tokens(vq, batch["img"], device)
        B = batch["img"].shape[0]
        tokens = indices.view(B, -1).numpy()
        qerr = qerr.view(B, -1).numpy()
        for i in range(B):
            key = (int(batch["label"][i]), batch["generator"][i])
            uniq, c = np.unique(tokens[i], return_counts=True)
            counts[key][uniq] += c
            f = feats[key]
            f["token_entropy"].append(entropy_of(c))
            f["unique_ratio"].append(len(uniq) / tokens[i].size)
            f["gini"].append(gini(c))
            hc, hmi, vc, vmi = transition_features(tokens[i])
            f["h_trans_entropy"].append(hc)
            f["h_bigram_mi"].append(hmi)
            f["v_trans_entropy"].append(vc)
            f["v_bigram_mi"].append(vmi)
            m, s, sk, ku = moments(qerr[i])
            f["qerr_mean"].append(m)
            f["qerr_std"].append(s)
            f["qerr_skew"].append(sk)
            f["qerr_kurtosis"].append(ku)
        done += B
        if done % 200 < args.batch_size:
            print(f"  {done}/{len(dataset)}")

    # per-image 特征 CSV
    import csv
    csv_path = os.path.join(args.out, "per_image_features.csv")
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["label", "generator"] + FEATURES)
        for (label, gen) in groups:
            n_img = len(feats[(label, gen)]["token_entropy"])
            for i in range(n_img):
                w.writerow([label, gen] + [feats[(label, gen)][f][i] for f in FEATURES])
    print(f"per-image 特征已保存到 {csv_path}")

    # 汇总统计
    summary = {}
    real_keys = [k for k in groups if k[0] == 0]
    real_key = real_keys[0] if real_keys else None
    for (label, gen) in groups:
        key_name = f"{'real' if label == 0 else 'fake'}:{gen}"
        c = counts[(label, gen)]
        used = int((c > 0).sum())
        p = c / max(c.sum(), 1)
        h = entropy_of(c)
        entry = {
            "n_images": len(feats[(label, gen)]["token_entropy"]),
            "codebook_usage": used,
            "codebook_usage_ratio": used / codebook_size,
            "hist_entropy": h,
            "effective_vocab": float(np.exp(h)),
            "top10_freq_mass": float(np.sort(p)[-10:].sum()),
            "top100_freq_mass": float(np.sort(p)[-100:].sum()),
        }
        for f in FEATURES:
            v = feats[(label, gen)][f]
            entry[f"{f}_mean"] = float(np.mean(v))
            entry[f"{f}_std"] = float(np.std(v))

        # real vs fake 对比: JS divergence / 效应量 / 单特征 AUROC
        if label == 1 and real_key is not None:
            entry["js_div_vs_real"] = js_divergence(counts[real_key], c)
            entry["effect_vs_real"] = {}
            entry["univariate_auc_vs_real"] = {}
            fr, ff = feats[real_key], feats[(label, gen)]
            for f in FEATURES:
                entry["effect_vs_real"][f] = {
                    "cohens_d": cohens_d(ff[f], fr[f]),
                    "cliffs_delta": cliffs_delta(ff[f], fr[f]),
                }
                labels = np.array([0] * len(fr[f]) + [1] * len(ff[f]))
                scores = np.array(fr[f] + ff[f])
                auc, lo, hi = bootstrap_auc(labels, scores, args.n_boot)
                entry["univariate_auc_vs_real"][f] = {
                    "auc": auc, "ci95": [lo, hi],
                    "higher_is_fake": bool(auc >= 0.5),
                }
        summary[key_name] = entry
        print(key_name, json.dumps(entry, indent=2))

    with open(os.path.join(args.out, "token_stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 绘图: codebook 频率曲线 + token entropy + qerr + 水平 transition 熵
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        for (label, gen) in groups:
            name = f"{'real' if label == 0 else 'fake'}:{gen}"
            c = counts[(label, gen)]
            p = np.sort(c / max(c.sum(), 1))[::-1]
            axes[0].plot(np.arange(len(p)), p, label=name, alpha=0.8)
            axes[1].hist(feats[(label, gen)]["token_entropy"], bins=40, alpha=0.5,
                         label=name, density=True)
            axes[2].hist(feats[(label, gen)]["qerr_mean"], bins=40, alpha=0.5,
                         label=name, density=True)
            axes[3].hist(feats[(label, gen)]["h_trans_entropy"], bins=40, alpha=0.5,
                         label=name, density=True)
        axes[0].set_yscale("log")
        axes[0].set_title("Codebook frequency (sorted, log)")
        axes[1].set_title("Per-image token entropy")
        axes[2].set_title("Per-image quantization error (mean)")
        axes[3].set_title("Horizontal transition conditional entropy")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "token_stats.png"), dpi=150)
        print(f"图表已保存到 {os.path.join(args.out, 'token_stats.png')}")
    except ImportError:
        print("matplotlib 不可用, 跳过绘图")


if __name__ == "__main__":
    main()
