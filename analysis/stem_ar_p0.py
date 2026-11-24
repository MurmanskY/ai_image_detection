"""P0-C: STEM-AR 机制图可迁移性诊断 (v8 §9 P0-C)。

假设: AR 模型逐 token 机制证据 e(k,t) = cond_llh - uncond_llh 的最细尺度
(或 16x16) 空间图, 能被 DINOv2 patch 特征在未见生成器上预测。
若教师空间图不可预测 (或只在 seen generator 可预测), 按 v8 停止 STEM-AR。

token-to-grid 验证 (已通过, 记录于结果):
  infinity_2b: 13 scales, 全为完全平方 (1^2..64^2), 取原生 16x16=256 scale
  switti_1024: 14 scales 全为完全平方, 取原生 16x16=256 scale
  llamagen_xlstage2: 32x32=1024, 2x2 均值池化 -> 16x16 (精确)
  janus_1b: 24x24=576, 面积插值 -> 16x16 (近似, 记录在案)
  光栅 (row-major) 顺序为 VAR/AR 模型的标准生成顺序。

探针 (v8 要求 Ridge/轻量/rank):
  per-cell 闭式 ridge: X (N,384+1) -> Y (N,256), 每个网格 cell 独立线性映射
  (等价于 256 个独立 ridge); λ=1。
对照: 打乱 teacher cell 位置 (固定排列) 的 shuffled control。
协议: LOGO-4, 训练 = 3 个生成器 fake (各自模型的教师图),
      测试 = held-out 生成器 fake; orig 与 jpeg95 分别训练评估。

预注册 GO:
  1. held-out 上 mean per-image Spearman(real) - Spearman(shuffled) >= 0.1
     且 real ρ > 0.2 (macro)
  2. >= 3/4 held-out 生成器 real ρ > 0 (且 > shuffled)
  3. jpeg95 下同样满足 1 (matched codec)

用法: python analysis/stem_ar_p0.py
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.mpid_p0 import (GENS, N_TRAIN, SEED, OUT_DIR, DATA_ROOT,  # noqa: E402
                              list_images, wait_for_gpu)  # noqa: E402

FEAT_DIR = os.path.join(PROJ, "outputs", "prada_t2i", "features")
PATCH_PATH = os.path.join(OUT_DIR, "stem_ar_dino_patches.npz")
OUT_PATH = os.path.join(OUT_DIR, "stem_ar_p0.json")
GRID = 16
LAYERS = [11]


# ---------------- 教师证据图 ----------------

def teacher_map(f, model):
    """返回 (B, 16, 16) 的 e = cond - uncond 证据图。"""
    e = (f["cond_gt_llh_BL"] - f["uncond_gt_llh_BL"]).float().numpy()
    if model in ("infinity_2b", "switti_1024"):
        sl = f["scale_lengths"]
        idx = sl.index(256)
        start = sum(sl[:idx])
        m = e[:, start:start + 256].reshape(-1, 16, 16)
    elif model == "llamagen_xlstage2":
        m = e.reshape(-1, 32, 32)
        m = m.reshape(-1, 16, 2, 16, 2).mean(axis=(2, 4))
    elif model == "janus_1b":
        m = e.reshape(-1, 24, 24)
        t = torch.tensor(m).unsqueeze(1)
        t = torch.nn.functional.interpolate(t, size=(16, 16),
                                            mode="area")
        m = t.squeeze(1).numpy()
    else:
        raise ValueError(model)
    return m


# ---------------- DINOv2 patch 特征 ----------------

@torch.no_grad()
def extract_patches(model, paths, device, jpeg95=False, batch_size=64, tag=""):
    import io
    from PIL import Image
    from torchvision import transforms as T
    from analysis.pard_mvp import IMAGENET_MEAN, IMAGENET_STD
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    out = []
    for bi in range(0, len(paths), batch_size):
        xs = []
        for p in paths[bi:bi + batch_size]:
            img = Image.open(p).convert("RGB").resize((224, 224),
                                                      Image.BICUBIC)
            if jpeg95:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            xs.append(tf(img))
        x = torch.stack(xs).to(device)
        inter = model.get_intermediate_layers(x, n=LAYERS, norm=True,
                                              return_class_token=False)
        out.append(inter[0].float().cpu().half())
        if bi == 0:
            print(f"  [{tag}] 提取中 ({len(paths)} 张)...", flush=True)
    return torch.cat(out).numpy()


def get_patch_features():
    if os.path.exists(PATCH_PATH):
        z = np.load(PATCH_PATH)
        return {k: z[k] for k in z.files}
    gpu = wait_for_gpu()
    device = torch.device(f"cuda:{gpu}")
    print(f"提取 DINOv2 patch 特征 (GPU{gpu})...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    feats = {}
    for ds in ["real"] + GENS:
        paths = list_images(os.path.join(DATA_ROOT, ds))
        feats[f"{ds}::orig"] = extract_patches(model, paths, device,
                                               tag=f"{ds}/orig")
        feats[f"{ds}::jpeg95"] = extract_patches(model, paths, device,
                                                 jpeg95=True,
                                                 tag=f"{ds}/jpeg95")
        print(f"  {ds}: {feats[f'{ds}::orig'].shape}", flush=True)
    np.savez_compressed(PATCH_PATH, **feats)
    return feats


# ---------------- ridge 探针 ----------------

def fit_ridge(X, Y, lam=1.0):
    """对偶形式 ridge (D >> N): 返回 (X1, alpha), w = X1.T @ alpha。"""
    X1 = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    G = X1 @ X1.T + lam * np.eye(len(X1), dtype=X1.dtype)
    alpha = np.linalg.solve(G, Y)
    return X1, alpha


def pred_ridge(model, X):
    X1tr, alpha = model
    X1 = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    return (X1 @ X1tr.T) @ alpha


def mean_spearman(P, Y):
    """逐图 Spearman 的均值。P,Y: (N, 256)"""
    rhos = []
    for i in range(len(Y)):
        r = spearmanr(P[i], Y[i]).statistic
        if not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 教师图 (CPU)
    print("构造教师证据图...", flush=True)
    teachers = {}
    for g in GENS:
        f = torch.load(os.path.join(FEAT_DIR, f"dataset={g}_model={g}.pt"),
                       map_location="cpu", weights_only=False)
        teachers[g] = teacher_map(f, g).reshape(len(f["cond_gt_llh_BL"]), -1)
        print(f"  {g}: teacher map {teachers[g].shape}", flush=True)

    # DINO patch 特征 (GPU)
    feats = get_patch_features()

    rng = np.random.RandomState(SEED)
    splits = {g: rng.permutation(len(teachers[g])) for g in GENS}

    results = {"config": {
        "protocol": "stem-ar-p0-1000示例 (DARG T2I, LOGO-4)",
        "teacher": "own-model finest/16x16 evidence map (cond-uncond)",
        "student_feats": "DINOv2 ViT-S/14 depth-11 patch tokens 256x384",
        "probe": "per-cell closed-form ridge (λ=1), shuffled control",
        "grid_verification": "infinity/switti 原生 16x16; llamagen 32x32->pool; "
                             "janus 24x24->area resize (近似)",
    }, "folds": {}}

    for codec in ("orig", "jpeg95"):
        for held in GENS:
            train_gens = [g for g in GENS if g != held]
            Xtr = np.concatenate(
                [feats[f"{g}::{codec}"][splits[g][:N_TRAIN]]
                 for g in train_gens]).astype(np.float32)
            Ytr = np.concatenate(
                [teachers[g][splits[g][:N_TRAIN]] for g in train_gens])
            Xte = feats[f"{held}::{codec}"][splits[held][N_TRAIN:]] \
                .astype(np.float32)
            Yte = teachers[held][splits[held][N_TRAIN:]]

            # 按 train 统计标准化教师图 (每个 cell)
            mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
            Ytr_z, Yte_z = (Ytr - mu) / sd, (Yte - mu) / sd

            # patch 特征: 不做标准化 (DINO norm 已稳定), 直接闭式解
            W = fit_ridge(Xtr.reshape(len(Xtr), -1), Ytr_z, lam=1.0)
            P = pred_ridge(W, Xte.reshape(len(Xte), -1))
            rho_real = mean_spearman(P, Yte_z)

            # shuffled control: 固定排列打乱 teacher cells
            perm = np.random.RandomState(1).permutation(Ytr_z.shape[1])
            W_s = fit_ridge(Xtr.reshape(len(Xtr), -1), Ytr_z[:, perm],
                            lam=1.0)
            P_s = pred_ridge(W_s, Xte.reshape(len(Xte), -1))
            rho_shuf = mean_spearman(P_s, Yte_z[:, perm])

            key = f"{codec}/{held}"
            results["folds"][key] = {"rho_real": rho_real,
                                     "rho_shuffled": rho_shuf}
            print(f"  {key}: real={rho_real:.4f} shuffled={rho_shuf:.4f}",
                  flush=True)

    # GO 判定
    go = {}
    for codec in ("orig", "jpeg95"):
        rr = [results["folds"][f"{codec}/{g}"]["rho_real"] for g in GENS]
        rs = [results["folds"][f"{codec}/{g}"]["rho_shuffled"] for g in GENS]
        go[codec] = {
            "macro_rho_real": float(np.mean(rr)),
            "macro_rho_shuffled": float(np.mean(rs)),
            "macro_margin": float(np.mean(rr) - np.mean(rs)),
            "folds_positive_and_better": int(
                sum(r > 0 and r > s for r, s in zip(rr, rs))),
        }
    results["go"] = {
        "detail": go,
        "crit1_orig_margin>=0.1_and_rho>0.2": bool(
            go["orig"]["macro_margin"] >= 0.1
            and go["orig"]["macro_rho_real"] > 0.2),
        "crit2_folds>=3": bool(go["orig"]["folds_positive_and_better"] >= 3),
        "crit3_jpeg95_margin>=0.1_and_rho>0.2": bool(
            go["jpeg95"]["macro_margin"] >= 0.1
            and go["jpeg95"]["macro_rho_real"] > 0.2),
    }
    results["go"]["GO"] = bool(results["go"]["crit1_orig_margin>=0.1_and_rho>0.2"]
                               and results["go"]["crit2_folds>=3"]
                               and results["go"]["crit3_jpeg95_margin>=0.1_and_rho>0.2"])
    print(json.dumps(results["go"], indent=2), flush=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
