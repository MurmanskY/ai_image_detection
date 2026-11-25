"""CARF-A 协议修复版 (v10 第一部分 §三 代码审计的 5 项修复)。

修复内容 (相对 analysis/carf_p0.py / carf_p1.py):
  1. 固定带宽: σ 只在 reference 上估计 (参考库内部距离中位数), 冻结后
     用于 calibration 与所有 test generator — 消除 transductive/batch-
     adaptive scoring; calibration 与 test 评分函数完全一致
  2. 独立校准: M2/M3 各自使用自身 calibration scores 计算阈值
     (carf_p1.py 曾用 M2 tau 给 M3)
  3. 真 Mondrian: 按分辨率桶 (短边<512 / >=512) 分组校准阈值并分别
     报告 FPR 与覆盖率
  4. 逐图分数保存 + 逐图 paired bootstrap ΔAUROC (生成器内重采样,
     macro 平均); 不再只做 7 点 cluster bootstrap
  5. 采集变量消融: full [size,w,h] / 仅宽高 / 仅文件大小 /
     按像素归一化文件大小 (log(size/(w·h)), w, h); 并做条件变量
     标签泄漏审计 (a 特征本身的逻辑回归 AUROC)

GO (预注册, 协议 A / r24):
  1. M2 vs M0 逐图 paired ΔAUROC > 0 (95% CI 不含 0)
  2. real FPR ∈ [3%, 7%] (修复校准后的实测值)
  3. >= 5/7 生成器不退化

数据: 与 CARF P0 同 manifest 与缓存 (无新特征提取, 纯评分协议修复)。

用法: python analysis/carf_a.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_p0 import (OUT as CARF_OUT, acq_features,  # noqa: E402
                              conformal_tau, maha_cond, maha_uncond)
from analysis.race_p0 import GENS, OUT_DIR as RACE_DIR  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "carf_a")
ESS_MIN = 50


def load_data(manifest):
    race_feats = dict(np.load(os.path.join(RACE_DIR, "feats.npz")))
    race_nprsr = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                  for k, v in np.load(os.path.join(RACE_DIR,
                                                   "nprsr.npz")).items()}
    data = {}
    for key in ("train_real", "cal_real"):
        j = np.nan_to_num(np.load(os.path.join(
            CARF_OUT, f"{key}_jpeg95_nprsr.npy")), nan=0.0)
        data[key] = {"r24": j[:, :, 960:984].mean(1),
                     "c": np.load(os.path.join(CARF_OUT,
                                               f"{key}_jpeg95_clip.npy")),
                     "paths": manifest[key]}
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            data[key] = {
                "r24": race_nprsr[f"{key}::jpeg95"][:, :, 960:984].mean(1),
                "c": race_feats[f"{key}::jpeg95::clip_pooled"],
                "paths": manifest["test"][key]}
    return data


def acq_variants(paths):
    """采集条件变量的 4 个消融版本。"""
    from PIL import Image
    rows = []
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
        size = os.path.getsize(p)
        rows.append([np.log(size + 1), np.log(w), np.log(h),
                     np.log(size / (w * h) + 1e-9)])
    rows = np.array(rows, dtype=np.float32)
    return {
        "full": rows[:, [0, 1, 2]],          # [log size, log w, log h]
        "wh_only": rows[:, [1, 2]],          # 仅宽高
        "size_only": rows[:, [0]],           # 仅文件大小
        "size_per_px": rows[:, [3, 1, 2]],   # 归一化文件大小 + 宽高
    }


def fixed_sigma(a_ref_n):
    """σ 只在 reference 上估计: 参考库内部两两距离中位数 (采样 1000)。"""
    idx = np.random.RandomState(0).choice(len(a_ref_n),
                                          min(1000, len(a_ref_n)),
                                          replace=False)
    sub = a_ref_n[idx]
    d = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    mask = ~np.eye(len(idx), dtype=bool)
    return float(np.median(d[mask])) + 1e-6


def wa_kernel(a_ref_n, a_test_n, sig_a):
    da = ((a_test_n[:, None, :] - a_ref_n[None, :, :]) ** 2).sum(-1)
    return np.exp(-da / (2 * sig_a))


def paired_boot_macro(per_gen, n_boot=1000, seed=0):
    """per_gen: {gen: (y, s1, s2)}; 返回 Δmacro 的均值与 95% CI。"""
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        aucs1, aucs2 = [], []
        for g, (y, s1, s2) in per_gen.items():
            idx = rng.randint(0, len(y), len(y))
            if y[idx].sum() in (0, len(idx)):
                continue
            aucs1.append(roc_auc_score(y[idx], s1[idx]))
            aucs2.append(roc_auc_score(y[idx], s2[idx]))
        deltas.append(np.mean(aucs1) - np.mean(aucs2))
    return float(np.mean(deltas)), [float(np.percentile(deltas, 2.5)),
                                    float(np.percentile(deltas, 97.5))]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(RACE_DIR, "manifest.json")))
    data = load_data(manifest)
    ref, cal = data["train_real"], data["cal_real"]

    # ---- 采集变量准备 (4 消融版本, 参考库统计标准化) ----
    av = {k: acq_variants(ref["paths"]) for k in ("full",)}
    ref_av = acq_variants(ref["paths"])
    cal_av = acq_variants(cal["paths"])

    # ---- 5. 条件变量标签泄漏审计 ----
    audit = {}
    for vname in ref_av:
        Xr = ref_av[vname]
        Xc = cal_av[vname]
        X_real = np.vstack([Xr, Xc])
        for g in GENS:
            Xf = acq_variants(data[f"{g}/1_fake"]["paths"])[vname]
            X = np.vstack([X_real, Xf])
            y = np.array([0] * len(X_real) + [1] * len(Xf))
            p = cross_val_predict(LogisticRegression(max_iter=1000), X, y,
                                  cv=5, method="predict_proba")[:, 1]
            audit.setdefault(vname, {})[g] = float(roc_auc_score(y, p))
    print("条件变量泄漏审计 (越低越好, 0.5=无泄漏):",
          json.dumps({k: round(float(np.mean(list(v.values()))), 3)
                      for k, v in audit.items()}), flush=True)

    # ---- 主评估: 逐消融版本, 固定带宽 ----
    results = {"config": {
        "protocol": "carf-a-fixed (协议 A JPEG95, 固定带宽, 独立校准, "
                    "Mondrian, 逐图 bootstrap)",
        "fixes": ["fixed_sigma_on_reference", "own_calibration_per_model",
                  "true_mondrian_resolution_buckets", "per_image_scores",
                  "acq_variants_ablation"],
    }, "acq_leakage_audit": audit, "variants": {}}

    for vname in ("full", "wh_only", "size_only", "size_per_px"):
        a_ref = ref_av[vname]
        a_cal = cal_av[vname]
        a_mu, a_sd = a_ref.mean(0), a_ref.std(0) + 1e-8
        a_ref_n, a_cal_n = (a_ref - a_mu) / a_sd, (a_cal - a_mu) / a_sd
        sig_a = fixed_sigma(a_ref_n)

        # 校准 (各模型独立阈值; Mondrian 分辨率桶)
        r_ref, r_cal = ref["r24"], cal["r24"]
        W_cal = wa_kernel(a_ref_n, a_cal_n, sig_a)
        m0_cal = maha_uncond(r_ref, r_cal)
        m2_cal, _ = maha_cond(r_ref, r_cal, W_cal, m0_cal)
        from PIL import Image
        bucket_cal = np.array(
            [0 if min(Image.open(p).size) < 512 else 1
             for p in cal["paths"]])

        def tau_mondrian(s_cal, bucket):
            return {b: conformal_tau(s_cal[bucket == b])
                    for b in np.unique(bucket)}

        tau = {"M0": {"pooled": conformal_tau(m0_cal),
                      "mondrian": tau_mondrian(m0_cal, bucket_cal)},
               "M2": {"pooled": conformal_tau(m2_cal),
                      "mondrian": tau_mondrian(m2_cal, bucket_cal)}}

        vres = {"sigma_a": sig_a, "per_generator": {}}
        per_gen_scores = {}
        for g in GENS:
            te_r, te_f = data[f"{g}/0_real"], data[f"{g}/1_fake"]
            r_te = np.concatenate([te_r["r24"], te_f["r24"]])
            y = np.array([0] * len(te_r["r24"]) + [1] * len(te_f["r24"]))
            a_te = np.concatenate([acq_variants(te_r["paths"])[vname],
                                   acq_variants(te_f["paths"])[vname]])
            a_te_n = (a_te - a_mu) / a_sd
            W_te = wa_kernel(a_ref_n, a_te_n, sig_a)
            m0 = maha_uncond(r_ref, r_te)
            m2, ess = maha_cond(r_ref, r_te, W_te, m0)
            n_real = len(te_r["r24"])
            bucket_r = np.array(
                [0 if min(Image.open(p).size) < 512 else 1
                 for p in te_r["paths"]])

            entry = {}
            for name, s in (("M0", m0), ("M2", m2)):
                sr = s[:n_real]
                e = {"AUROC": float(roc_auc_score(y, s)),
                     "FPR_pooled": float((sr > tau[name]["pooled"]).mean())}
                mf = {}
                for b, tb in tau[name]["mondrian"].items():
                    m = bucket_r == b
                    if m.sum() > 0:
                        mf[str(b)] = float((sr[m] > tb).mean())
                e["FPR_mondrian"] = mf
                entry[name] = e
            entry["M2_ESS_median"] = float(np.median(ess))
            vres["per_generator"][g] = entry
            per_gen_scores[g] = (y, m2, m0)
            # 保存逐图分数
        np.savez(os.path.join(OUT, f"scores_{vname}.npz"),
                 **{f"{g}::{k}": v for g, (y, m2, m0) in
                    per_gen_scores.items()
                    for k, v in (("y", y), ("m2", m2), ("m0", m0))})

        auc_m0 = np.array([vres["per_generator"][g]["M0"]["AUROC"]
                           for g in GENS])
        auc_m2 = np.array([vres["per_generator"][g]["M2"]["AUROC"]
                           for g in GENS])
        fpr_m2 = np.array([vres["per_generator"][g]["M2"]["FPR_pooled"]
                           for g in GENS])
        d_mean, d_ci = paired_boot_macro(per_gen_scores)
        vres["macro"] = {
            "M0": float(auc_m0.mean()), "M2": float(auc_m2.mean()),
            "delta_M2-M0": d_mean, "delta_CI95": d_ci,
            "FPR_M2_pooled": float(fpr_m2.mean()),
            "gens_not_worse": int((auc_m2 >= auc_m0 - 1e-9).sum()),
        }
        results["variants"][vname] = vres
        print(f"  [{vname}] M0={auc_m0.mean():.4f} M2={auc_m2.mean():.4f} "
              f"Δ={d_mean:+.4f} CI{d_ci} FPR={fpr_m2.mean():.3f} "
              f"not_worse={vres['macro']['gens_not_worse']}/7", flush=True)

    # ---- GO (full 变体) ----
    v = results["variants"]["full"]["macro"]
    results["go"] = {
        "crit1_delta>0_CI": bool(v["delta_M2-M0"] > 0
                                 and v["delta_CI95"][0] > 0),
        "crit2_FPR_in_range": bool(0.03 <= v["FPR_M2_pooled"] <= 0.07),
        "crit3_gens_not_worse>=5": bool(v["gens_not_worse"] >= 5),
        "delta": v["delta_M2-M0"], "ci": v["delta_CI95"],
        "fpr": v["FPR_M2_pooled"],
    }
    results["go"]["GO"] = bool(results["go"]["crit1_delta>0_CI"]
                               and results["go"]["crit2_FPR_in_range"]
                               and results["go"]["crit3_gens_not_worse>=5"])
    print(json.dumps(results["go"], indent=2), flush=True)
    with open(os.path.join(OUT, "carf_a.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {OUT}/carf_a.json", flush=True)


if __name__ == "__main__":
    main()
