"""SACR P0 (v11): signal-aware conditional residual modeling.

协议:
  * clean manifest 中约6000 real 按 seeds {0,1,2} 重划 reference/calibration;
    独立 dev_real(约1000) + ARForensics val fake(1000) 只用于选择模型;
    ARForensics test 7 生成器 x 1000/类只用于最终评估。
  * U 由 reference real 的配对反事实差分估计 (k=6), 不做不变投影。
    C 仅评 z_parallel=U^T r; D 评 z_parallel + lambda*z_perp。
  * 条件: 原文件大小 (baseline), 解码像素复杂度, codec, 像素+codec。
    scorer {gaussian,knn,quantile} 与 lambda {0.25,0.5,1} 只在 dev 选择。
  * SACR 的 ESS<50 输出 unsupported/abstain; CARF-A 保持原 M0 fallback。
  * canonical 为统一 JPEG95; 跨 codec 在 Infinity/Open_MAGVIT2/VAR
    300/类 JPEG85/75 缓存上评估, 标为 sacr-p0-1000/cross-300 示例。

预注册 GO (运行前冻结):
  1. validation 选出的 pixel SACR 相对每 seed 较强 CARF-A(full/size_only)
     的三种子平均 macro ΔAUROC >= +0.02, 逐图 paired bootstrap 95%CI 下界>0,
     且三个 seed 各自 Δ>0;
  2. 三种子平均 >=5/7 生成器不退化, 每 seed FPR 在[3%,7%], coverage>=95%;
  3. pixel+codec 在 JPEG85 与 JPEG75 的 macro ΔAUROC 均>0, FPR 在范围内;
  4. 若仅 file-size 条件有效, 或 pixel 条件不通过, GO=false 并停止扩展;
  5. 像素复杂度组合在 validation 与 within-source 审计 AUC 均须<0.80。

用法: python analysis/sacr_p0.py [--seeds 0,1,2] [--skip-cache]
"""

import argparse
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import pywt
import torch
from PIL import Image
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_a import fixed_sigma, paired_boot_macro, wa_kernel  # noqa: E402
from analysis.carf_p0 import (OUT as CARF_OUT, conformal_tau,  # noqa: E402
                              maha_cond, maha_uncond)
from analysis.contra_p0 import OUT as CONTRA_OUT, apply_view, extract_r24  # noqa: E402
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import GENS, OUT_DIR as RACE_DIR, build_noiseprint  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "sacr")
VIEWS = ("qf95", "qf85", "qf75")
U_VIEWS = ("qf85", "qf75", "qf50", "r090", "r075")
DRIFT_GENS = ("Infinity", "Open_MAGVIT2", "VAR")
N_DRIFT = 300
SEEDS = (0, 1, 2)
K_RANK = 6
ESS_MIN = 50
LAMBDAS = (0.25, 0.5, 1.0)
SCORERS = ("gaussian", "knn", "quantile")
COMPLEXITY_NAMES = (
    "grad_s0", "grad_s1", "grad_s2", "local_entropy", "wavelet_hf",
    "dct_entropy", "edge_density",
)
GO_DELTA = 0.02


def clean(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)


def safe(key):
    return key.replace("/", "__").replace("::", "--")


def quality(view):
    return int(view[2:])


def make_manifest():
    import csv
    race = json.load(open(os.path.join(RACE_DIR, "manifest.json")))
    clean_paths = set()
    with open(os.path.join(PROJ, "data", "manifests",
                           "arforensics_clean.csv")) as f:
        for row in csv.DictReader(f):
            clean_paths.add(os.path.normpath(row["path"]))

    def is_clean(path):
        rel = os.path.relpath(path, os.path.join(PROJ, "data", "ARForensics"))
        return os.path.normpath(rel) in clean_paths

    original_pool = race["train_real"] + race["cal_real"]
    pool_source_idx = [i for i, p in enumerate(original_pool) if is_clean(p)]
    pool = [original_pool[i] for i in pool_source_idx]
    dev_real = [p for p in race["dev_real"] if is_clean(p)]
    val_fake_dir = os.path.join(PROJ, "data", "ARForensics", "val", "1_fake")
    dev_fake = [os.path.join(val_fake_dir, f)
                for f in sorted(os.listdir(val_fake_dir))
                if is_clean(os.path.join(val_fake_dir, f))]
    n_ref = min(3500, len(pool))
    n_cal = min(2500, len(pool) - n_ref)
    seeds = {}
    for seed in SEEDS:
        idx = np.random.RandomState(seed).permutation(len(pool))
        seeds[str(seed)] = {
            "ref_idx": idx[:n_ref].tolist(),
            "cal_idx": idx[n_ref:n_ref + n_cal].tolist(),
        }
    manifest = {"pool_real": pool, "pool_source_idx": pool_source_idx,
                "dev_real": dev_real, "dev_fake": dev_fake,
                "test": race["test"], "seeds": seeds,
                "removed_nonclean_pool": len(original_pool) - len(pool),
                "removed_nonclean_dev": len(race["dev_real"]) - len(dev_real)}
    ref = {os.path.realpath(pool[i]) for i in seeds["0"]["ref_idx"]}
    cal = {os.path.realpath(pool[i]) for i in seeds["0"]["cal_idx"]}
    dev = {os.path.realpath(p) for p in dev_real}
    test = {os.path.realpath(p) for g in GENS
            for p in race["test"][f"{g}/0_real"]}
    assert len(pool) >= 5990 and len(dev_fake) >= 990
    assert not (ref & cal) and not ((ref | cal) & dev)
    assert not ((ref | cal | dev) & test)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_or_make_manifest():
    path = os.path.join(OUT, "manifest.json")
    if os.path.exists(path):
        return json.load(open(path))
    return make_manifest()


def ensure_dev_residuals(manifest):
    missing = [k for k in ("dev_real", "dev_fake")
               if not os.path.exists(os.path.join(OUT, f"r24_{k}_qf95.npy"))]
    if not missing:
        return
    gpu = wait_for_gpu(need_mb=4000)
    device = torch.device(f"cuda:{gpu}")
    net = build_noiseprint(device)
    for key in missing:
        r = extract_r24(net, manifest[key], "qf95", device, tag=key)
        np.save(os.path.join(OUT, f"r24_{key}_qf95.npy"), r)
    del net
    torch.cuda.empty_cache()


def load_residuals(manifest):
    data = {"pool": {}, "dev_real": {}, "dev_fake": {}}
    for view in VIEWS:
        if view == "qf95":
            tr = clean(np.load(os.path.join(
                CARF_OUT, "train_real_jpeg95_nprsr.npy")))[:, :, 960:984].mean(1)
            ca = clean(np.load(os.path.join(
                CARF_OUT, "cal_real_jpeg95_nprsr.npy")))[:, :, 960:984].mean(1)
        else:
            tr = clean(np.load(os.path.join(
                CONTRA_OUT, f"r24_train_real::{view}.npy")))
            ca = clean(np.load(os.path.join(
                CONTRA_OUT, f"r24_cal_real::{view}.npy")))
        data["pool"][view] = np.concatenate([tr, ca])[
            np.asarray(manifest["pool_source_idx"])]
    data["dev_real"]["qf95"] = clean(np.load(os.path.join(
        OUT, "r24_dev_real_qf95.npy")))
    data["dev_fake"]["qf95"] = clean(np.load(os.path.join(
        OUT, "r24_dev_fake_qf95.npy")))
    race_nprsr = np.load(os.path.join(RACE_DIR, "nprsr.npz"))
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            data[key] = {"qf95": clean(
                race_nprsr[f"{key}::jpeg95"][:, :, 960:984].mean(1))}
            if g in DRIFT_GENS:
                for view in VIEWS[1:]:
                    data[key][view] = clean(np.load(os.path.join(
                        CONTRA_OUT, f"r24_{g}__{cls}::{view}.npy")))
    return data


def patches_for_image(path, view):
    img = Image.open(path).convert("RGB")
    if min(img.size) < 128:
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = apply_view(img, view)
    w, h = img.size
    if min(w, h) < 128:
        scale = 256 / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        w, h = img.size
    a = np.asarray(img, dtype=np.float32) / 255.0
    cy, cx = h // 2, w // 2
    coords = ((cy - 64, cx - 64), (0, 0), (h - 128, w - 128))
    return [a[max(0, y):max(0, y) + 128,
              max(0, x):max(0, x) + 128] for y, x in coords]


def local_entropy(gray):
    blocks = gray.reshape(4, 32, 4, 32).transpose(0, 2, 1, 3).reshape(-1, 1024)
    vals = []
    for b in blocks:
        hist = np.histogram(b, bins=16, range=(0, 1))[0].astype(np.float64)
        p = hist / max(hist.sum(), 1)
        vals.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
    return float(np.mean(vals))


def dct_entropy(gray):
    blocks = gray.reshape(16, 8, 16, 8).transpose(0, 2, 1, 3)
    coeff = np.abs(dctn(blocks, axes=(-2, -1), norm="ortho"))
    coeff = coeff.reshape(-1, 64)
    coeff[:, 0] = 0
    p = coeff / (coeff.sum(1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return float(ent.mean())


def complexity_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = []
    for sigma in (0, 1, 2):
        x = gray if sigma == 0 else cv2.GaussianBlur(gray, (0, 0), sigma)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        feats.append(float(np.log(mag.mean() + 1e-6)))
    feats.append(local_entropy(gray))
    _, (lh, hl, hh) = pywt.dwt2(gray, "haar")
    hf = (np.square(lh).mean() + np.square(hl).mean()
          + np.square(hh).mean()) / 3
    feats.append(float(np.log(hf + 1e-12)))
    feats.append(dct_entropy(gray))
    edges = cv2.Canny(np.clip(gray * 255, 0, 255).astype(np.uint8), 50, 150)
    feats.append(float((edges > 0).mean()))
    return feats


def extract_complexity(paths, view, tag):
    rows = []
    start = time.time()
    for i, path in enumerate(paths):
        rows.append(np.mean([complexity_patch(p)
                             for p in patches_for_image(path, view)], axis=0))
        if (i + 1) % 500 == 0:
            print(f"    [{tag}] {i + 1}/{len(paths)} "
                  f"({time.time() - start:.0f}s)", flush=True)
    return np.asarray(rows, dtype=np.float32)


def complexity_jobs(manifest):
    jobs = []
    for split, paths in (("pool", manifest["pool_real"]),
                         ("dev_real", manifest["dev_real"]),
                         ("dev_fake", manifest["dev_fake"])):
        for view in (VIEWS if split == "pool" else ("qf95",)):
            jobs.append((split, paths, view))
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            paths = manifest["test"][key]
            jobs.append((key, paths, "qf95"))
            if g in DRIFT_GENS:
                for view in VIEWS[1:]:
                    jobs.append((key, paths[:N_DRIFT], view))
    return jobs


def ensure_complexity(manifest):
    jobs = complexity_jobs(manifest)
    todo = [(k, p, v) for k, p, v in jobs
            if not os.path.exists(os.path.join(OUT, f"complexity_{safe(k)}_{v}.npy"))]
    print(f"像素复杂度缓存: 待提取 {len(todo)}/{len(jobs)} 集", flush=True)
    for key, paths, view in todo:
        x = extract_complexity(paths, view, f"{key}/{view}")
        np.save(os.path.join(OUT, f"complexity_{safe(key)}_{view}.npy"), x)


def load_complexity(manifest):
    out = {}
    for key, _, view in complexity_jobs(manifest):
        out.setdefault(key, {})[view] = clean(np.load(os.path.join(
            OUT, f"complexity_{safe(key)}_{view}.npy")))
    return out


def acquisition_features(paths, variant):
    rows = []
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
        vals = {"size_only": [np.log(os.path.getsize(p) + 1)],
                "full": [np.log(os.path.getsize(p) + 1), np.log(w), np.log(h)]}
        rows.append(vals[variant])
    return np.asarray(rows, dtype=np.float32)


def fit_u(manifest, residuals, ref_idx):
    r0 = residuals["pool"]["qf95"][ref_idx]
    source_idx = np.asarray(manifest["pool_source_idx"])
    ds = []
    for view in U_VIEWS:
        rv = clean(np.load(os.path.join(
            CONTRA_OUT, f"r24_train_real::{view}.npy")))
        cv = clean(np.load(os.path.join(
            CONTRA_OUT, f"r24_cal_real::{view}.npy")))
        rall = np.concatenate([rv, cv])[source_idx]
        ds.append(rall[ref_idx] - r0)
    d = np.concatenate(ds)
    _, singular, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
    return vt[:K_RANK].T, (singular ** 2 / (singular ** 2).sum())[:K_RANK]


def blocks(r, u):
    parallel = r @ u
    perp = r - parallel @ u.T
    return parallel, perp


def kernel_for_condition(c_ref, c_query):
    if c_ref.shape[1] == 0:
        return np.ones((len(c_query), len(c_ref)), dtype=np.float32), 1.0
    mu, sd = c_ref.mean(0), c_ref.std(0) + 1e-8
    rn, qn = (c_ref - mu) / sd, (c_query - mu) / sd
    sigma = fixed_sigma(rn)
    return wa_kernel(rn, qn, sigma), sigma


def gaussian_block(r_ref, r_query, weights):
    base = maha_uncond(r_ref, r_query)
    score, ess = maha_cond(r_ref, r_query, weights, base, ess_min=0)
    score[ess < ESS_MIN] = np.nan
    return score, ess


def knn_block(r_ref, r_query, weights, k_condition=64, k_distance=8):
    mu, sd = r_ref.mean(0), r_ref.std(0) + 1e-6
    ref = (r_ref - mu) / sd
    query = (r_query - mu) / sd
    scores = np.empty(len(query), dtype=np.float32)
    ess = weights.sum(1) ** 2 / (np.square(weights).sum(1) + 1e-12)
    k_condition = min(k_condition, len(ref))
    for i in range(len(query)):
        if ess[i] < ESS_MIN:
            scores[i] = np.nan
            continue
        nn = np.argpartition(weights[i], -k_condition)[-k_condition:]
        d2 = np.square(ref[nn] - query[i]).sum(1)
        scores[i] = np.sqrt(np.partition(d2, k_distance - 1)[k_distance - 1]
                            + 1e-12)
    return scores, ess


def quantile_block(r_ref, r_query, weights, k_condition=64):
    scores = np.empty(len(r_query), dtype=np.float32)
    ess = weights.sum(1) ** 2 / (np.square(weights).sum(1) + 1e-12)
    k_condition = min(k_condition, len(r_ref))
    for i in range(len(r_query)):
        if ess[i] < ESS_MIN:
            scores[i] = np.nan
            continue
        nn = np.argpartition(weights[i], -k_condition)[-k_condition:]
        ref = r_ref[nn]
        mu, sd = ref.mean(0), ref.std(0) + 1e-6
        za = np.abs((ref - mu) / sd)
        zq = np.abs((r_query[i] - mu) / sd)
        tail = (1 + (za >= zq).sum(0)) / (len(nn) + 1)
        scores[i] = -np.log(tail).mean()
    return scores, ess


def sacr_score(r_ref, r_query, weights, u, model, scorer, lam):
    ref_p, ref_o = blocks(r_ref, u)
    query_p, query_o = blocks(r_query, u)
    fn = {"gaussian": gaussian_block, "knn": knn_block,
          "quantile": quantile_block}[scorer]
    sp, ess = fn(ref_p, query_p, weights)
    if model == "C":
        return sp, ess
    so, _ = fn(ref_o, query_o, weights)
    return sp + lam * so, ess


def condition_arrays(name, paths, complexity, view):
    if name == "file_size":
        return acquisition_features(paths, "size_only")
    if name == "pixel":
        return complexity
    if name == "codec":
        return np.full((len(paths), 1), quality(view), dtype=np.float32)
    if name == "pixel+codec":
        return np.column_stack([complexity,
                                np.full(len(paths), quality(view))])
    raise ValueError(name)


def tau_by_complexity(scores, complexity, threshold):
    group = complexity[:, 3] >= threshold
    taus = {}
    for b in (False, True):
        s = scores[group == b]
        s = s[np.isfinite(s)]
        taus[str(int(b))] = conformal_tau(s) if len(s) >= 50 else np.nan
    return taus


def fpr_grouped(scores, complexity, threshold, taus):
    valid = np.isfinite(scores)
    group = complexity[:, 3] >= threshold
    pred = np.zeros(len(scores), dtype=bool)
    for b in (False, True):
        m = valid & (group == b) & np.isfinite(taus[str(int(b))])
        pred[m] = scores[m] > taus[str(int(b))]
    return float(pred[valid].mean()) if valid.any() else np.nan


def metrics(y, scores, real_complexity, threshold, taus, n_real):
    valid = np.isfinite(scores)
    yv, sv = y[valid], scores[valid]
    if len(np.unique(yv)) < 2:
        return {"AUROC": np.nan, "coverage": float(valid.mean())}
    fpr, tpr, _ = roc_curve(yv, sv)
    tpr01 = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0
    return {
        "AUROC": float(roc_auc_score(yv, sv)),
        "AP": float(average_precision_score(yv, sv)),
        "TPR@1%FPR": tpr01,
        "FPR": fpr_grouped(scores[:n_real], real_complexity, threshold, taus),
        "coverage": float(valid.mean()),
        "coverage_real": float(valid[:n_real].mean()),
        "coverage_fake": float(valid[n_real:].mean()),
    }


def leakage_auc(x_real, x_fake):
    x = np.vstack([x_real, x_fake])
    y = np.array([0] * len(x_real) + [1] * len(x_fake))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pred = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pred))


def complexity_audit(manifest, complexities):
    pool = complexities["pool"]["qf95"]
    dev = {"combined": leakage_auc(complexities["dev_real"]["qf95"],
                                   complexities["dev_fake"]["qf95"])}
    per_gen = {}
    for g in GENS:
        real = complexities[f"{g}/0_real"]["qf95"]
        fake = complexities[f"{g}/1_fake"]["qf95"]
        within = leakage_auc(real, fake)
        cross = leakage_auc(pool, fake)
        single = {}
        for j, name in enumerate(COMPLEXITY_NAMES):
            auc = roc_auc_score(np.array([0] * len(real) + [1] * len(fake)),
                                np.concatenate([real[:, j], fake[:, j]]))
            single[name] = float(max(auc, 1 - auc))
        per_gen[g] = {"within_source_combined": within,
                      "cross_source_combined": cross,
                      "within_source_single": single}
    result = {"validation": dev, "per_generator": per_gen,
              "macro_within_source": float(np.mean(
                  [v["within_source_combined"] for v in per_gen.values()])),
              "macro_cross_source": float(np.mean(
                  [v["cross_source_combined"] for v in per_gen.values()])),
              "feature_names": COMPLEXITY_NAMES}
    with open(os.path.join(OUT, "complexity_audit.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("复杂度泄漏审计:", json.dumps({
        "validation": dev["combined"],
        "within_source": result["macro_within_source"],
        "cross_source": result["macro_cross_source"]}), flush=True)
    return result


def build_seed_data(manifest, residuals, complexities, seed):
    split = manifest["seeds"][str(seed)]
    ref_idx = np.asarray(split["ref_idx"])
    cal_idx = np.asarray(split["cal_idx"])
    paths = manifest["pool_real"]
    return {
        "ref_idx": ref_idx, "cal_idx": cal_idx,
        "ref_paths": [paths[i] for i in ref_idx],
        "cal_paths": [paths[i] for i in cal_idx],
        "ref_r": residuals["pool"]["qf95"][ref_idx],
        "cal_r": residuals["pool"]["qf95"][cal_idx],
        "ref_c": complexities["pool"]["qf95"][ref_idx],
        "cal_c": complexities["pool"]["qf95"][cal_idx],
    }


def select_config(r_ref, c_ref_paths, ref_complexity, dev_r, dev_paths,
                  dev_complexity, dev_y, u, condition, complexity_threshold):
    c_ref = condition_arrays(condition, c_ref_paths, ref_complexity, "qf95")
    c_dev = condition_arrays(condition, dev_paths, dev_complexity, "qf95")
    weights, sigma = kernel_for_condition(c_ref, c_dev)
    candidates = []
    for model in ("C", "D"):
        for scorer in SCORERS:
            for lam in (LAMBDAS if model == "D" else (1.0,)):
                score, ess = sacr_score(r_ref, dev_r, weights, u, model,
                                        scorer, lam)
                valid = np.isfinite(score)
                auc = roc_auc_score(dev_y[valid], score[valid])
                candidates.append({
                    "model": model, "scorer": scorer, "lambda": lam,
                    "dev_AUROC": float(auc), "dev_coverage": float(valid.mean()),
                    "sigma": sigma,
                })
                print(f"      {condition}/{model}/{scorer}/λ{lam}: "
                      f"dev={auc:.4f} coverage={valid.mean():.3f}", flush=True)
    eligible = [c for c in candidates if c["dev_coverage"] >= 0.95]
    if not eligible:
        raise RuntimeError(f"{condition}: validation coverage 全部低于 95%")
    best = max(eligible, key=lambda c: (c["dev_AUROC"], c["dev_coverage"]))
    return best, candidates


def evaluate_configuration(r_ref, r_cal, r_query, ref_paths, cal_paths,
                           query_paths, ref_complexity, cal_complexity,
                           query_complexity, view, query_view, u, condition,
                           config, ref_view=None):
    ref_view = ref_view or view
    c_ref = condition_arrays(condition, ref_paths, ref_complexity, ref_view)
    c_cal = condition_arrays(condition, cal_paths, cal_complexity, view)
    c_query = condition_arrays(condition, query_paths, query_complexity, query_view)
    w_cal, sigma = kernel_for_condition(c_ref, c_cal)
    cal_score, _ = sacr_score(r_ref, r_cal, w_cal, u, config["model"],
                              config["scorer"], config["lambda"])
    w_query, _ = kernel_for_condition(c_ref, c_query)
    score, ess = sacr_score(r_ref, r_query, w_query, u, config["model"],
                            config["scorer"], config["lambda"])
    threshold = float(np.median(ref_complexity[:, 3]))
    taus = tau_by_complexity(cal_score, cal_complexity, threshold)
    return score, ess, taus, threshold, sigma


def evaluate_carf(r_ref, r_cal, r_query, ref_paths, cal_paths, query_paths,
                  ref_complexity, cal_complexity, query_complexity, variant):
    a_ref = acquisition_features(ref_paths, variant)
    a_cal = acquisition_features(cal_paths, variant)
    a_query = acquisition_features(query_paths, variant)
    mu, sd = a_ref.mean(0), a_ref.std(0) + 1e-8
    rn = (a_ref - mu) / sd
    sigma = fixed_sigma(rn)
    w_cal = wa_kernel(rn, (a_cal - mu) / sd, sigma)
    w_query = wa_kernel(rn, (a_query - mu) / sd, sigma)
    cal_base = maha_uncond(r_ref, r_cal)
    query_base = maha_uncond(r_ref, r_query)
    cal_score, _ = maha_cond(r_ref, r_cal, w_cal, cal_base)
    score, ess = maha_cond(r_ref, r_query, w_query, query_base)
    threshold = float(np.median(ref_complexity[:, 3]))
    taus = tau_by_complexity(cal_score, cal_complexity, threshold)
    return score, ess, taus, threshold, sigma


def canonical_seed(manifest, residuals, complexities, seed):
    print(f"===== seed {seed}: canonical JPEG95 =====", flush=True)
    sd = build_seed_data(manifest, residuals, complexities, seed)
    u, ev = fit_u(manifest, residuals, sd["ref_idx"])
    print(f"  U explained variance: {ev.round(4).tolist()}", flush=True)
    dev_r = np.concatenate([residuals["dev_real"]["qf95"],
                            residuals["dev_fake"]["qf95"]])
    dev_c = np.concatenate([complexities["dev_real"]["qf95"],
                            complexities["dev_fake"]["qf95"]])
    dev_paths = manifest["dev_real"] + manifest["dev_fake"]
    dev_y = np.array([0] * len(manifest["dev_real"])
                     + [1] * len(manifest["dev_fake"]))
    selections, validation = {}, {}
    for condition in ("file_size", "pixel", "codec"):
        print(f"    validation: {condition}", flush=True)
        best, candidates = select_config(
            sd["ref_r"], sd["ref_paths"], sd["ref_c"], dev_r, dev_paths,
            dev_c, dev_y, u, condition, float(np.median(sd["ref_c"][:, 3])))
        selections[condition] = best
        validation[condition] = candidates
        print(f"    selected {condition}: {best}", flush=True)

    per_gen, saved_by_gen = {}, {}
    for g in GENS:
        saved = {}
        real_key, fake_key = f"{g}/0_real", f"{g}/1_fake"
        r_query = np.concatenate([residuals[real_key]["qf95"],
                                  residuals[fake_key]["qf95"]])
        c_query = np.concatenate([complexities[real_key]["qf95"],
                                  complexities[fake_key]["qf95"]])
        paths = manifest["test"][real_key] + manifest["test"][fake_key]
        y = np.array([0] * 1000 + [1] * 1000)
        entry = {}
        m0_cal = maha_uncond(sd["ref_r"], sd["cal_r"])
        m0 = maha_uncond(sd["ref_r"], r_query)
        threshold = float(np.median(sd["ref_c"][:, 3]))
        taus = tau_by_complexity(m0_cal, sd["cal_c"], threshold)
        entry["M0"] = metrics(y, m0, c_query[:1000], threshold, taus, 1000)
        saved["M0"] = m0
        for variant in ("full", "size_only"):
            name = f"CARF-A-{variant}"
            score, ess, taus, threshold, sigma = evaluate_carf(
                sd["ref_r"], sd["cal_r"], r_query, sd["ref_paths"],
                sd["cal_paths"], paths, sd["ref_c"], sd["cal_c"], c_query,
                variant)
            entry[name] = metrics(y, score, c_query[:1000], threshold, taus, 1000)
            entry[name]["ESS_median"] = float(np.median(ess))
            saved[name] = score
        for condition in ("file_size", "pixel", "codec"):
            config = selections[condition]
            name = f"SACR-{config['model']}-{condition}"
            score, ess, taus, threshold, sigma = evaluate_configuration(
                sd["ref_r"], sd["cal_r"], r_query, sd["ref_paths"],
                sd["cal_paths"], paths, sd["ref_c"], sd["cal_c"], c_query,
                "qf95", "qf95", u, condition, config)
            entry[name] = metrics(y, score, c_query[:1000], threshold, taus, 1000)
            entry[name]["ESS_median"] = float(np.median(ess))
            entry[name]["scorer"] = config["scorer"]
            entry[name]["lambda"] = config["lambda"]
            saved[name] = score
        per_gen[g] = entry
        saved_by_gen[g] = saved
        print(f"  {g}: " + " ".join(
            f"{k}={v['AUROC']:.3f}" for k, v in entry.items()), flush=True)
    np.savez(os.path.join(OUT, f"scores_seed{seed}.npz"),
             **{f"{g}::{name}": score for g, saved in saved_by_gen.items()
                for name, score in saved.items()})
    return {"seed": seed, "selections": selections, "validation": validation,
            "u_explained_variance": ev.tolist(), "per_generator": per_gen}


def macro(result, method):
    vals = [result["per_generator"][g][method]["AUROC"] for g in GENS]
    return float(np.nanmean(vals))


def bootstrap_delta_samples(canonical, method_a, method_b, n_boot=1000):
    samples = []
    for seed_result in canonical:
        per_gen = {}
        for g in GENS:
            e = seed_result["per_generator"][g]
            z = np.load(os.path.join(OUT, f"scores_seed{seed_result['seed']}.npz"))
            a, b = z[f"{g}::{method_a}"], z[f"{g}::{method_b}"]
            valid = np.isfinite(a) & np.isfinite(b)
            y = np.array([0] * 1000 + [1] * 1000)[valid]
            per_gen[g] = (y, a[valid], b[valid])
        rng = np.random.RandomState(1000 + seed_result["seed"])
        ds = []
        for _ in range(n_boot):
            aa, bb = [], []
            for y, a, b in per_gen.values():
                idx = rng.randint(0, len(y), len(y))
                aa.append(roc_auc_score(y[idx], a[idx]))
                bb.append(roc_auc_score(y[idx], b[idx]))
            ds.append(np.mean(aa) - np.mean(bb))
        samples.append(ds)
    return np.asarray(samples)


def cross_codec(manifest, residuals, complexities, canonical):
    print("===== cross-codec JPEG85/75 (300/类示例) =====", flush=True)
    output = {}
    for seed_result in canonical:
        seed = seed_result["seed"]
        sd = build_seed_data(manifest, residuals, complexities, seed)
        u, _ = fit_u(manifest, residuals, sd["ref_idx"])
        pixel_selection = seed_result["selections"]["pixel"]
        codec_selection = seed_result["selections"]["codec"]
        seed_out = {}
        for view in VIEWS[1:]:
            ref_idx, cal_idx = sd["ref_idx"], sd["cal_idx"]
            r_ref_same = residuals["pool"][view][ref_idx]
            r_cal = residuals["pool"][view][cal_idx]
            c_ref_same = complexities["pool"][view][ref_idx]
            c_cal = complexities["pool"][view][cal_idx]
            per_gen, saved_by_gen = {}, {}
            for g in DRIFT_GENS:
                saved = {}
                real_key, fake_key = f"{g}/0_real", f"{g}/1_fake"
                r_query = np.concatenate([residuals[real_key][view],
                                          residuals[fake_key][view]])
                c_query = np.concatenate([complexities[real_key][view],
                                          complexities[fake_key][view]])
                paths = (manifest["test"][real_key][:N_DRIFT]
                         + manifest["test"][fake_key][:N_DRIFT])
                y = np.array([0] * N_DRIFT + [1] * N_DRIFT)
                entry = {}
                for variant in ("full", "size_only"):
                    name = f"CARF-A-{variant}"
                    score, _, taus, threshold, _ = evaluate_carf(
                        sd["ref_r"], r_cal, r_query, sd["ref_paths"],
                        sd["cal_paths"], paths, sd["ref_c"], c_cal, c_query,
                        variant)
                    entry[name] = metrics(y, score, c_query[:N_DRIFT],
                                          threshold, taus, N_DRIFT)
                    saved[name] = score
                for condition in ("pixel", "pixel+codec", "codec"):
                    same = condition != "pixel"
                    selection = codec_selection if condition == "codec" \
                        else pixel_selection
                    name = f"SACR-{selection['model']}-{condition}"
                    score, ess, taus, threshold, _ = evaluate_configuration(
                        r_ref_same if same else sd["ref_r"], r_cal, r_query,
                        sd["ref_paths"], sd["cal_paths"], paths,
                        c_ref_same if same else sd["ref_c"], c_cal, c_query,
                        view, view, u, condition, selection,
                        ref_view=view if same else "qf95")
                    entry[name] = metrics(y, score, c_query[:N_DRIFT],
                                          threshold, taus, N_DRIFT)
                    entry[name]["coverage"] = float(np.isfinite(score).mean())
                    saved[name] = score
                per_gen[g] = entry
                saved_by_gen[g] = saved
            np.savez(os.path.join(OUT, f"cross_scores_seed{seed}_{view}.npz"),
                     **{f"{g}::{name}": score
                        for g, saved in saved_by_gen.items()
                        for name, score in saved.items()})
            seed_out[view] = per_gen
            print(f"  seed{seed}/{view}: " + "; ".join(
                f"{m}={np.mean([per_gen[g][m]['AUROC'] for g in DRIFT_GENS]):.4f}"
                for m in next(iter(per_gen.values()))), flush=True)
        output[str(seed)] = seed_out
    return output


def summarize(canonical, cross, audit):
    summary = {"seeds": {}, "cross_codec": cross}
    for result in canonical:
        seed = result["seed"]
        methods = list(next(iter(result["per_generator"].values())))
        summary["seeds"][str(seed)] = {
            m: {"macro_AUROC": macro(result, m),
                "worst_AUROC": float(np.nanmin([
                    result["per_generator"][g][m]["AUROC"] for g in GENS])),
                "macro_FPR": float(np.nanmean([
                    result["per_generator"][g][m]["FPR"] for g in GENS])),
                "coverage": float(np.nanmean([
                    result["per_generator"][g][m]["coverage"] for g in GENS]))}
            for m in methods}
        summary["seeds"][str(seed)]["selections"] = result["selections"]

    candidates = []
    for result in canonical:
        names = [n for n in result["per_generator"][GENS[0]]
                 if n.startswith("SACR-") and n.endswith("-pixel")]
        assert len(names) == 1
        candidates.append(names[0])
    deltas, ci_lows, baselines, aligned = [], [], [], []
    for result, candidate in zip(canonical, candidates):
        carf = max(("CARF-A-full", "CARF-A-size_only"),
                   key=lambda m: macro(result, m))
        samples = bootstrap_delta_samples([result], candidate, carf)[0]
        aligned.append(samples)
        deltas.append(macro(result, candidate) - macro(result, carf))
        ci_lows.append(float(np.percentile(samples, 2.5)))
        baselines.append(carf)
    mean_samples = np.asarray(aligned).mean(0)
    ci = [float(np.percentile(mean_samples, 2.5)),
          float(np.percentile(mean_samples, 97.5))]
    seed_fprs = [summary["seeds"][str(result["seed"])][candidate]["macro_FPR"]
                 for result, candidate in zip(canonical, candidates)]
    coverages = [summary["seeds"][str(result["seed"])][candidate]["coverage"]
                 for result, candidate in zip(canonical, candidates)]
    not_worse = []
    for result, candidate, baseline in zip(canonical, candidates, baselines):
        not_worse.append(sum(
            result["per_generator"][g][candidate]["AUROC"] + 1e-9
            >= result["per_generator"][g][baseline]["AUROC"] for g in GENS))
    go = {
        "candidate_per_seed": candidates, "baselines_per_seed": baselines,
        "delta_per_seed": deltas, "delta_mean": float(np.mean(deltas)),
        "delta_CI95": ci, "ci_low_per_seed": ci_lows,
        "fpr_per_seed": seed_fprs, "coverage_per_seed": coverages,
        "not_worse_per_seed": not_worse,
        "crit1_delta>=0.02": bool(np.mean(deltas) >= GO_DELTA),
        "crit1_CI_lower>0": bool(ci[0] > 0),
        "crit1_all_seeds_positive": bool(all(d > 0 for d in deltas)),
        "crit2_not_worse>=5": bool(np.mean(not_worse) >= 5),
        "crit2_FPR_each_seed": bool(all(0.03 <= x <= 0.07 for x in seed_fprs)),
        "crit2_coverage>=0.95": bool(all(x >= 0.95 for x in coverages)),
    }
    cross_deltas = {}
    cross_ok = True
    for view in VIEWS[1:]:
        ds, ds_codec, fprs = [], [], []
        for result, candidate, baseline in zip(canonical, candidates, baselines):
            per_gen = cross[str(result["seed"])][view]
            name = candidate.replace("-pixel", "-pixel+codec")
            codec_name = (f"SACR-{result['selections']['codec']['model']}"
                          "-codec")
            ds.append(float(np.mean([
                per_gen[g][name]["AUROC"] - per_gen[g][baseline]["AUROC"]
                for g in DRIFT_GENS])))
            ds_codec.append(float(np.mean([
                per_gen[g][name]["AUROC"] - per_gen[g][codec_name]["AUROC"]
                for g in DRIFT_GENS])))
            fprs.append(float(np.nanmean([
                per_gen[g][name]["FPR"] for g in DRIFT_GENS])))
        cross_deltas[view] = {"delta_per_seed": ds, "fpr_per_seed": fprs,
                              "delta_vs_codec_only_per_seed": ds_codec}
        cross_ok &= bool(np.mean(ds) > 0 and all(0.03 <= x <= 0.07 for x in fprs))
    go["cross_codec"] = cross_deltas
    go["crit3_cross_codec"] = bool(cross_ok)
    go["crit4_not_file_size_only"] = bool(all(
        candidate.endswith("-pixel") for candidate in candidates))
    go["crit5_complexity_leakage<0.80"] = bool(
        audit["macro_within_source"] < 0.80
        and audit["validation"]["combined"] < 0.80)
    go["GO"] = bool(go["crit1_delta>=0.02"] and go["crit1_CI_lower>0"]
                    and go["crit1_all_seeds_positive"]
                    and go["crit2_not_worse>=5"] and go["crit2_FPR_each_seed"]
                    and go["crit2_coverage>=0.95"] and go["crit3_cross_codec"]
                    and go["crit4_not_file_size_only"]
                    and go["crit5_complexity_leakage<0.80"])
    summary["go"] = go
    summary["complexity_audit"] = {
        "macro_within_source": audit["macro_within_source"],
        "macro_cross_source": audit["macro_cross_source"],
        "validation": audit["validation"],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    global SEEDS
    SEEDS = tuple(int(x) for x in args.seeds.split(","))
    os.makedirs(OUT, exist_ok=True)
    manifest = load_or_make_manifest()
    if set(SEEDS) != {0, 1, 2}:
        raise ValueError("P0 GO 需要完整 --seeds 0,1,2；调试请直接调用函数")
    if not args.skip_cache:
        ensure_dev_residuals(manifest)
        ensure_complexity(manifest)
    residuals = load_residuals(manifest)
    complexities = load_complexity(manifest)
    audit = complexity_audit(manifest, complexities)
    canonical = [canonical_seed(manifest, residuals, complexities, seed)
                 for seed in SEEDS]
    with open(os.path.join(OUT, "canonical.json"), "w") as f:
        json.dump({"config": {
                              "seeds": SEEDS,
                              "n_ref": len(manifest["seeds"]["0"]["ref_idx"]),
                              "n_cal": len(manifest["seeds"]["0"]["cal_idx"]),
                              "n_dev_real": len(manifest["dev_real"]),
                              "n_dev_fake": len(manifest["dev_fake"]),
                              "n_test_per_class": 1000, "k_rank": K_RANK,
                              "ess_min": ESS_MIN, "lambdas": LAMBDAS,
                              "scorers": SCORERS},
                   "runs": canonical}, f, indent=2)
    cross = cross_codec(manifest, residuals, complexities, canonical)
    summary = summarize(canonical, cross, audit)
    results = {"protocol": "sacr-p0-1000 + cross-300 示例",
               "go_thresholds": {"delta": GO_DELTA, "fpr": [0.03, 0.07],
                                 "coverage": 0.95, "not_worse": 5},
               "summary": summary, "canonical": canonical}
    with open(os.path.join(OUT, "sacr_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary["go"], indent=2), flush=True)
    print(f"已保存 {OUT}/sacr_p0.json", flush=True)


if __name__ == "__main__":
    main()
