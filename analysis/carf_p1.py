"""CARF P1-lite (v9 §9 P1 缩规模版): 3000/类 + 超参消融。

P0 (carf_p0.py) 通过后:
  1. 协议 A (全 JPEG95 codec-matched) 扩大到 3000/类 (carf-p1-3000 示例),
     仅 r24, 仅 M0/M2/M3 (M1 已证明≈M0, 不再重复计算)
  2. 超参消融 (P0 的 1000/类缓存上, 免费): 核带宽 σ×{0.5,1,2},
     ESS_MIN {20,50,100}, 收缩 λ {0.05,0.1,0.5}
  3. GO 复核: 同 P0 标准

与 P0 的关系: train(3500)/cal(2500) 参考与校准集不变 (复用 P0 缓存);
test 扩到 3000/类 (seed 0, 包含 P0 的 1000 子集)。

用法: python analysis/carf_p1.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_p0 import (ESS_MIN, OUT as CARF_OUT, acq_features,  # noqa: E402
                              conformal_tau, kernel_weights, maha_cond,
                              maha_uncond)
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import (GENS, OUT_DIR as RACE_DIR,  # noqa: E402
                              build_noiseprint, nprsr_descriptors)

OUT = os.path.join(PROJ, "outputs", "carf_p1")
N_TEST = 3000
SEED = 0
AR_ROOT = os.path.join(PROJ, "data", "ARForensics")


def ensure_test_caches(manifest_test):
    """test 各组 jpeg95 NP-RSR r24/r960 与 CLIP pooled 缓存。"""
    os.makedirs(OUT, exist_ok=True)
    keys = []
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            keys.append(f"{g}__{cls}")
    missing = [k for k in keys
               if not (os.path.exists(os.path.join(OUT, f"{k}_nprsr.npy"))
                       and os.path.exists(os.path.join(OUT, f"{k}_clip.npy")))]
    if not missing:
        return
    gpu = wait_for_gpu(need_mb=4000)
    device = torch.device(f"cuda:{gpu}")
    net = build_noiseprint(device)
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    clip = clip.visual.eval().to(device)
    for p in clip.parameters():
        p.requires_grad = False
    from analysis.race_reextract_clip import extract_clip_only
    for k in missing:
        paths = manifest_test[k]
        print(f"  提取 {k} jpeg95 ({len(paths)} 张)...", flush=True)
        d = nprsr_descriptors(net, paths, device, jpeg=True, tag=k)
        np.save(os.path.join(OUT, f"{k}_nprsr.npy"), d.astype(np.float32))
        _, pooled = extract_clip_only(clip, paths, device, jpeg=True, tag=k)
        np.save(os.path.join(OUT, f"{k}_clip.npy"), pooled.astype(np.float32))
    del net, clip
    torch.cuda.empty_cache()


def run_protocol_A(ref, cal, test_sets, sig_scale=1.0, ess_min=ESS_MIN,
                   lam_cov=None):
    """M0/M2/M3 (r24), 返回 per-gen AUROC 与 FPR。"""
    import analysis.carf_p0 as carf
    if lam_cov is not None:
        carf.LAM_COV = lam_cov
    a_ref = acq_features(ref["paths"])
    a_cal = acq_features(cal["paths"])
    a_mu, a_sd = a_ref.mean(0), a_ref.std(0) + 1e-8
    a_ref_n, a_cal_n = (a_ref - a_mu) / a_sd, (a_cal - a_mu) / a_sd

    def kernels(c_ref, c_te, a_te_n):
        from analysis.carf_p0 import unit
        dc = 1 - unit(c_te) @ unit(c_ref).T
        da = ((a_te_n[:, None, :] - a_ref_n[None, :, :]) ** 2).sum(-1)
        sig_c = (np.median(dc) + 1e-6) * sig_scale
        sig_a = (np.median(da) + 1e-6) * sig_scale
        Wc = np.exp(-dc / (2 * sig_c))
        Wa = np.exp(-da / (2 * sig_a))
        return Wc, Wa, Wc * Wa

    r_ref, r_cal = ref["r24"], cal["r24"]
    _, Wa_cal, _ = kernels(ref["c"], cal["c"], a_cal_n)
    m0_cal = maha_uncond(r_ref, r_cal)
    m2_cal, _ = maha_cond(r_ref, r_cal, Wa_cal, m0_cal, ess_min=ess_min)
    tau = {"M0": conformal_tau(m0_cal), "M2": conformal_tau(m2_cal),
           "M3": conformal_tau(m2_cal)}  # M3≈M2, 校准以 M2 tau 近似

    res = {"M0": {}, "M2": {}, "M3": {}}
    for g in GENS:
        te_r, te_f = test_sets[f"{g}__0_real"], test_sets[f"{g}__1_fake"]
        r_te = np.concatenate([te_r["r24"], te_f["r24"]])
        y = np.array([0] * len(te_r["r24"]) + [1] * len(te_f["r24"]))
        a_te_n = (np.concatenate([acq_features(te_r["paths"]),
                                  acq_features(te_f["paths"])])
                  - a_mu) / a_sd
        c_te = np.concatenate([te_r["c"], te_f["c"]])
        Wc, Wa, Wca = kernels(ref["c"], c_te, a_te_n)
        m0 = maha_uncond(r_ref, r_te)
        m2, _ = maha_cond(r_ref, r_te, Wa, m0, ess_min=ess_min)
        m3, _ = maha_cond(r_ref, r_te, Wca, m0, ess_min=ess_min)
        n_real = len(te_r["r24"])
        for name, s in (("M0", m0), ("M2", m2), ("M3", m3)):
            res[name][g] = {"AUROC": float(roc_auc_score(y, s)),
                            "FPR@5%cal": float((s[:n_real] > tau[name]).mean())}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest_race = json.load(open(os.path.join(RACE_DIR, "manifest.json")))

    # 扩大 test 清单 (3000/类, seed 0)
    manifest_test = {}
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            d = os.path.join(AR_ROOT, "test", g, cls)
            files = sorted(os.listdir(d))
            idx = np.random.RandomState(SEED).choice(
                len(files), min(N_TEST, len(files)), replace=False)
            manifest_test[f"{g}__{cls}"] = [
                os.path.join(d, files[i]) for i in sorted(idx)]

    print("P1: 补提 test jpeg95 特征 (3000/类)...", flush=True)
    ensure_test_caches(manifest_test)

    # 装配
    race_nprsr = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                  for k, v in np.load(os.path.join(RACE_DIR,
                                                   "nprsr.npz")).items()}
    race_feats = dict(np.load(os.path.join(RACE_DIR, "feats.npz")))
    ref = {"paths": manifest_race["train_real"],
           "r24": np.nan_to_num(np.load(os.path.join(
               CARF_OUT, "train_real_jpeg95_nprsr.npy")), nan=0.0)[:, :, 960:984].mean(1),
           "c": np.load(os.path.join(CARF_OUT, "train_real_jpeg95_clip.npy"))}
    cal = {"paths": manifest_race["cal_real"],
           "r24": np.nan_to_num(np.load(os.path.join(
               CARF_OUT, "cal_real_jpeg95_nprsr.npy")), nan=0.0)[:, :, 960:984].mean(1),
           "c": np.load(os.path.join(CARF_OUT, "cal_real_jpeg95_clip.npy"))}
    test_sets = {}
    for k, paths in manifest_test.items():
        d = np.nan_to_num(np.load(os.path.join(OUT, f"{k}_nprsr.npy")),
                          nan=0.0)
        test_sets[k] = {"paths": paths,
                        "r24": d[:, :, 960:984].mean(1),
                        "c": np.load(os.path.join(OUT, f"{k}_clip.npy"))}

    print("P1: 协议 A 3000/类评估...", flush=True)
    res = run_protocol_A(ref, cal, test_sets)
    out = {"protocol": "carf-p1-3000示例 (协议 A, r24, M0/M2/M3)",
           "results": res}
    for name in res:
        aucs = [res[name][g]["AUROC"] for g in GENS]
        fprs = [res[name][g]["FPR@5%cal"] for g in GENS]
        out[f"macro_{name}"] = float(np.mean(aucs))
        out[f"fpr_{name}"] = float(np.mean(fprs))
        print(f"  {name}: macro={np.mean(aucs):.4f} worst={np.min(aucs):.4f} "
              f"FPR={np.mean(fprs):.3f}", flush=True)
    auc_m0 = np.array([res["M0"][g]["AUROC"] for g in GENS])
    auc_m3 = np.array([res["M3"][g]["AUROC"] for g in GENS])
    out["go_recheck"] = {
        "M3>M0": bool(auc_m3.mean() > auc_m0.mean()),
        "delta": float(auc_m3.mean() - auc_m0.mean()),
        "gens_not_worse": int((auc_m3 >= auc_m0 - 1e-9).sum()),
        "FPR_in_range": bool(0.03 <= out["fpr_M3"] <= 0.07),
    }
    with open(os.path.join(OUT, "carf_p1.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"已保存 {OUT}/carf_p1.json", flush=True)

    # ---- 超参消融 (P0 规模 1000/类缓存, M2 分量) ----
    print("P1: 超参消融 (P0 规模缓存)...", flush=True)
    test_p0 = {}
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            d = race_nprsr[f"{key}::jpeg95"]
            test_sets_p0 = {
                "paths": manifest_race["test"][key],
                "r24": d[:, :, 960:984].mean(1),
                "c": race_feats[f"{key}::jpeg95::clip_pooled"]}
            test_p0[f"{g}__{cls}"] = test_sets_p0
    abl = {}
    for sig in (0.5, 1.0, 2.0):
        r = run_protocol_A(ref, cal, test_p0, sig_scale=sig)
        aucs = [r["M2"][g]["AUROC"] for g in GENS]
        abl[f"sig_scale={sig}"] = float(np.mean(aucs))
        print(f"  σ×{sig}: M2 macro={np.mean(aucs):.4f}", flush=True)
    for ess in (20, 50, 100):
        r = run_protocol_A(ref, cal, test_p0, ess_min=ess)
        aucs = [r["M2"][g]["AUROC"] for g in GENS]
        abl[f"ess_min={ess}"] = float(np.mean(aucs))
        print(f"  ESS_MIN={ess}: M2 macro={np.mean(aucs):.4f}", flush=True)
    for lam in (0.05, 0.1, 0.5):
        r = run_protocol_A(ref, cal, test_p0, lam_cov=lam)
        aucs = [r["M2"][g]["AUROC"] for g in GENS]
        abl[f"lam_cov={lam}"] = float(np.mean(aucs))
        print(f"  λ={lam}: M2 macro={np.mean(aucs):.4f}", flush=True)
    out["ablation_P0scale_M2"] = abl
    with open(os.path.join(OUT, "carf_p1.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("消融已写入 carf_p1.json", flush=True)


if __name__ == "__main__":
    main()
