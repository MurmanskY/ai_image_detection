"""PARD P0 MVP: 扰动感知的局部取证响应动力学 (v6 指南主线)。

核心假设: 真实图像与 AR 生成图像在受控扰动下的冻结 VFM 局部 (patch) 特征
响应动力学不同。与 RIGID (全局 CLS 相似度) 的区别: 多扰动类型/强度、
patch 级一阶/二阶响应、空间统计, 且检测为 real-only 异常建模 (不训练分类器)。

设计 (与 v6 §2.2/2.3 对齐的 MVP 简化):
  骨干: DINOv2 ViT-S/14 冻结 (与 RIGID 配置一致, 保证对照是"响应建模"差异,
        而非骨干差异)。中间层 [2,5,8,11], norm=True, 取 patch tokens + CLS。
  扰动 (5 类 x 2 强度):
    jpeg95/jpeg75, resize90/resize75, blur1/blur2 (PIL 高斯半径),
    noise05/noise10 (归一化张量加高斯噪声; noise05 与 RIGID 设置一致),
    crop90/crop70 (中心裁剪后复原)
  描述子 Ψ (每层 x 每类扰动):
    弱强度 s1: per-patch ΔL2 的 mean/std + per-patch cos 的 mean/std
               + ΔL2 四象限均值 (空间) = 8 维
    强强度 s2: 同 8 维 + 二阶响应 K=Δ_s2-Δ_s1 的 per-patch 范数 mean/std = 10 维
    共 4 层 x 5 类 x 18 = 360 维; 另加全局 CLS cos 动态特征 10 维 -> 370 维
  评分: fit 集 z-score 标准化后, LedoitWolf Mahalanobis / kNN(k=10)
  参考: ARForensics val real 1000 张, 前 500 fit / 后 500 cal (cal 本轮仅记录,
        conformal 后续做)
  测试: 每生成器 500 real + 500 fake (seed=0 子采样, 标注 subsample-500)
  对照: RIGID 分数 (-cos(CLS_orig, CLS_noise05)) 在同一批图像上直接计算,
        保证同子集同骨干公平比较

预注册 GO 标准 (相对同子集 RIGID):
  1. macro AUROC 提升 >= 0.03
  2. 最差生成器 AUROC 提升 >= 0.05
  3. 增益不只来自单个生成器 (>=4/7 生成器不差于 RIGID)
  未达标则按 v6 停止条件放弃扩大规模。

用法:
  python analysis/pard_mvp.py --out outputs/pard/pard_mvp.json
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageFilter
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms as T

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LAYERS = [2, 5, 8, 11]
GENERATORS = ["Infinity", "Janus_Pro", "LlamaGen", "Open_MAGVIT2", "RAR",
              "Switti", "VAR"]

# (name, type, strength)
VARIANTS = [
    ("orig", None, None),
    ("jpeg95", "jpeg", 95), ("jpeg75", "jpeg", 75),
    ("resize90", "resize", 0.90), ("resize75", "resize", 0.75),
    ("blur1", "blur", 1.0), ("blur2", "blur", 2.0),
    ("noise05", "noise", 0.05), ("noise10", "noise", 0.10),
    ("crop90", "crop", 0.90), ("crop70", "crop", 0.70),
]
PERT_TYPES = ["jpeg", "resize", "blur", "noise", "crop"]

_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_to_tensor = T.ToTensor()


def apply_pil_perturb(img, ptype, strength):
    if ptype == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(strength))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if ptype == "resize":
        w, h = img.size
        small = img.resize((max(1, int(w * strength)), max(1, int(h * strength))),
                           Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if ptype == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(strength)))
    if ptype == "crop":
        w, h = img.size
        cw, ch = int(w * strength), int(h * strength)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h),
                                                                 Image.BICUBIC)
    return img


def list_images(d):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(exts))


def subsample(paths, n, seed=0):
    if len(paths) <= n:
        return paths
    idx = np.random.RandomState(seed).choice(len(paths), n, replace=False)
    return [paths[i] for i in sorted(idx)]


@torch.no_grad()
def extract_features(model, paths, device, batch_size=64, tag=""):
    """返回 feats (N, 370), rigid_sim (N,), valid_paths。"""
    feats_all, rigid_all, kept = [], [], []
    n_batches = (len(paths) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch_paths = paths[bi * batch_size:(bi + 1) * batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB").resize((224, 224),
                                                          Image.BICUBIC)
            except Exception:
                continue
            imgs.append((p, img))
        if not imgs:
            continue
        kept.extend([p for p, _ in imgs])

        # 构造 11 个变体张量
        variant_tensors = []
        for name, ptype, strength in VARIANTS:
            ts = []
            for _, img in imgs:
                im = apply_pil_perturb(img, ptype, strength) if ptype else img
                t = _normalize(_to_tensor(im))
                if ptype == "noise":
                    t = t + torch.randn_like(t) * strength
                ts.append(t)
            variant_tensors.append(torch.stack(ts))

        # 前向: orig + 10 扰动
        outs = {}
        for (name, _, _), vt in zip(VARIANTS, variant_tensors):
            inter = model.get_intermediate_layers(
                vt.to(device), n=LAYERS, norm=True, return_class_token=True)
            outs[name] = [(p.float().cpu(), c.float().cpu()) for p, c in inter]

        B = len(imgs)
        feats = torch.zeros(B, 370)
        col = 0
        for li in range(len(LAYERS)):
            p_orig = outs["orig"][li][0]  # (B, N, C)
            for ptype in PERT_TYPES:
                names2 = [v[0] for v in VARIANTS if v[1] == ptype]
                deltas = {}
                for vname in names2:
                    p_pert = outs[vname][li][0]
                    d_l2 = (p_pert - p_orig).norm(dim=-1)          # (B, N)
                    d_cos = Fn.cosine_similarity(p_pert, p_orig, dim=-1)
                    deltas[vname] = (d_l2, d_cos)
                s1, s2 = names2  # 弱, 强
                for vname, with_k in ((s1, False), (s2, True)):
                    d_l2, d_cos = deltas[vname]
                    q = d_l2.view(B, 4, 64).mean(dim=-1) \
                        if False else quadrant_means(d_l2)
                    vals = [d_l2.mean(1), d_l2.std(1),
                            d_cos.mean(1), d_cos.std(1),
                            q[:, 0], q[:, 1], q[:, 2], q[:, 3]]
                    if with_k:
                        k_norm = (deltas[s2][0] - deltas[s1][0]).abs()
                        vals += [k_norm.mean(1), k_norm.std(1)]
                    for v in vals:
                        feats[:, col] = v
                        col += 1
        # 全局 CLS 动态特征 (10 维)
        cls_orig = Fn.normalize(outs["orig"][-1][1], dim=-1)
        rigid_sim = None
        for name, ptype, _ in VARIANTS[1:]:
            cls_pert = Fn.normalize(outs[name][-1][1], dim=-1)
            sim = (cls_orig * cls_pert).sum(-1)
            feats[:, col] = sim
            col += 1
            if name == "noise05":
                rigid_sim = sim
        feats_all.append(feats)
        rigid_all.append(rigid_sim)
        if bi % 20 == 0:
            print(f"  [{tag}] batch {bi}/{n_batches}", flush=True)
    return (torch.cat(feats_all).numpy(), torch.cat(rigid_all).numpy(), kept)


def quadrant_means(d_l2):
    """d_l2: (B, 256) patch 网格 16x16 -> 四象限均值 (B, 4)。"""
    B = d_l2.shape[0]
    g = d_l2.view(B, 16, 16)
    qs = [g[:, :8, :8].mean(dim=(1, 2)), g[:, :8, 8:].mean(dim=(1, 2)),
          g[:, 8:, :8].mean(dim=(1, 2)), g[:, 8:, 8:].mean(dim=(1, 2))]
    return torch.stack(qs, dim=1)


def mahalanobis_scores(fit, test):
    mu = fit.mean(axis=0)
    lw = LedoitWolf().fit(fit - mu)
    d = test - mu
    return np.einsum("ij,jk,ik->i", d, lw.precision_, d)


def knn_scores(fit, test, k=10):
    k = min(k, len(fit))
    nn = NearestNeighbors(n_neighbors=k).fit(fit)
    dists, _ = nn.kneighbors(test)
    return dists.mean(axis=1)


def aucs_for(feats_real, feats_fake, fit_mu, fit_std, fit_z):
    rz = (feats_real - fit_mu) / fit_std
    fz = (feats_fake - fit_mu) / fit_std
    labels = np.array([0] * len(rz) + [1] * len(fz))
    out = {}
    for sname, fn in (("mahalanobis", mahalanobis_scores),
                      ("knn", knn_scores)):
        s = np.concatenate([fn(fit_z, rz), fn(fit_z, fz)])
        out[sname] = float(roc_auc_score(labels, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ARForensics")
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--n_ref", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/pard/pard_mvp.json")
    ap.add_argument("--feats_out", default="outputs/pard/pard_mvp_feats.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("加载 DINOv2 ViT-S/14 (冻结)...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # ---- 参考集: val real ----
    val_real = subsample(list_images(os.path.join(args.root, "val", "0_real")),
                         args.n_ref, seed=args.seed)
    print(f"参考集 val real: {len(val_real)}", flush=True)
    ref_feats, _, ref_kept = extract_features(model, val_real, device,
                                              args.batch_size, tag="ref")
    n_half = len(ref_feats) // 2
    fit_feats = ref_feats[:n_half]
    fit_mu, fit_std = fit_feats.mean(0), fit_feats.std(0) + 1e-8
    fit_z = (fit_feats - fit_mu) / fit_std

    # ---- 测试: 每生成器 500 real + 500 fake ----
    results = {"config": {"n_per_class": args.n_per_class, "n_ref": args.n_ref,
                          "seed": args.seed, "backbone": "dinov2_vits14",
                          "layers": LAYERS, "feature_dim": 370,
                          "protocol": "subsample-500 (示例, 非全量)"},
               "per_generator": {}}
    feat_cache = {"ref": ref_feats}
    rigid_by_gen = {}
    for gen in GENERATORS:
        gdir = os.path.join(args.root, "test", gen)
        real = subsample(list_images(os.path.join(gdir, "0_real")),
                         args.n_per_class, seed=args.seed)
        fake = subsample(list_images(os.path.join(gdir, "1_fake")),
                         args.n_per_class, seed=args.seed)
        print(f"== {gen}: real {len(real)} fake {len(fake)}", flush=True)
        fr, rr, _ = extract_features(model, real, device, args.batch_size,
                                     tag=f"{gen}/real")
        ff, rf, _ = extract_features(model, fake, device, args.batch_size,
                                     tag=f"{gen}/fake")
        feat_cache[f"{gen}_real"] = fr
        feat_cache[f"{gen}_fake"] = ff

        entry = aucs_for(fr, ff, fit_mu, fit_std, fit_z)
        labels = np.array([0] * len(rr) + [1] * len(rf))
        entry["rigid"] = float(roc_auc_score(labels,
                                             np.concatenate([-rr, -rf])))
        rigid_by_gen[gen] = entry["rigid"]
        results["per_generator"][gen] = entry
        print(gen, json.dumps(entry), flush=True)

    # ---- macro + GO 判定 ----
    macro = {}
    for sname in ("mahalanobis", "knn", "rigid"):
        aucs = [v[sname] for v in results["per_generator"].values()]
        macro[sname] = {"mean": float(np.mean(aucs)),
                        "min": float(np.min(aucs))}
    results["macro"] = macro

    best = max(("mahalanobis", "knn"), key=lambda k: macro[k]["mean"])
    n_not_worse = sum(1 for g, v in results["per_generator"].items()
                      if v[best] >= v["rigid"] - 1e-9)
    go = {
        "scorer": best,
        "crit1_macro_gain>=0.03": macro[best]["mean"] - macro["rigid"]["mean"],
        "crit2_worst_gain>=0.05": macro[best]["min"] - macro["rigid"]["min"],
        "crit3_gens_not_worse>=4": n_not_worse,
    }
    go["GO"] = bool(go["crit1_macro_gain>=0.03"] >= 0.03
                    and go["crit2_worst_gain>=0.05"] >= 0.05
                    and n_not_worse >= 4)
    results["go_criteria"] = go
    print(json.dumps({"macro": macro, "go": go}, indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(args.feats_out,
                        **{k: v for k, v in feat_cache.items()})
    print(f"已保存 {args.out} 与 {args.feats_out}", flush=True)


if __name__ == "__main__":
    main()
