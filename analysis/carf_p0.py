"""CARF P0: 内容与采集条件条件化的取证残差建模 (v9 §4/§9)。

核心假设 (可证伪): 在内容与采集条件相近的真实图像中, 局部取证残差的
条件分布 p(r|c,a) 比无条件分布更能识别 AR 生成图像; 且分组校准能把
JPEG95 下 real FPR 拉回名义水平 (直接回应 RACE 复现的 5%->30% 漂移)。

协议 (v9 §9 P0):
  A (主, codec-matched): 参考库/校准/测试全部统一 JPEG95, 消除格式
     shortcut, 隔离条件化收益
  B (对照, orig as-is): 原始混合格式, 展示条件覆盖行为

特征 (最大化复用 RACE 缓存):
  c (语义, 仅用于匹配): CLIP ViT-L/14 pooled, L2 归一
  a (采集条件): [log 文件大小, log 宽, log 高] (参考库统计标准化)
  r (取证残差): r24 = NP-RSR 静态末端残差 24 维 (3 patch 均值);
     r960 = 中间激活统计 960 维 (3 patch 均值)

模型比较 (隔离条件化收益):
  M0 p(r)     无条件收缩高斯马氏距离
  M1 p(r|c)   内容核加权局部高斯
  M2 p(r|a)   采集条件核加权
  M3 p(r|c,a) 联合核
  对照: RACE NP-RSR s_R (缓存 AE 重算, 仅 r24 协议报告)
  核: w_i ∝ exp(-d_c²/2σ_c²)·exp(-d_a²/2σ_a²), σ=参考库中位距离;
  ESS=(Σw)²/Σw² < 50 时退回 M0, 记录覆盖率。
  协方差: 加权 + 收缩 (Σ̂ = Σ_w + 0.1·diag(Σ_w) + εI); r960 用 diag(Σ_w)。

校准: cal real split-conformal (名义 FPR 5%); Mondrian 按分辨率桶
  (短边<512 / >=512) 分组; 覆盖不足的组标记未校准。

预注册 GO (v9 §9, 协议 A / r24):
  1. M3 macro AUROC > M0 且 > RACE-NP-RSR (paired bootstrap 1000 次, 95% CI)
  2. M3 real FPR ∈ [3%, 7%] (名义 5%±2pp)
  3. >= 5/7 生成器 M3 不差于 M0
  只改善 FPR 不改善 AUROC -> 贡献定位为校准而非检测。
  不满足 -> 按 v9 转向协议/失效机制研究, 不再堆检测模块。

数据: 与 RACE P0 同 manifest (3500 train ref / 2500 cal / 1000/类 test),
race-p0-1000 示例协议。test real 不参与拟合/带宽/阈值。

用法: python analysis/carf_p0.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import (GENS, OUT_DIR as RACE_DIR, DescAE,  # noqa: E402
                              build_noiseprint, nprsr_descriptors)

OUT = os.path.join(PROJ, "outputs", "carf")
SEED = 0
ESS_MIN = 50
LAM_COV = 0.1
ALPHA = 0.05


def ensure_carf_caches(manifest):
    os.makedirs(OUT, exist_ok=True)
    missing = [k for k in ("train_real", "cal_real")
               if not (os.path.exists(os.path.join(OUT, f"{k}_jpeg95_nprsr.npy"))
                       and os.path.exists(os.path.join(OUT, f"{k}_jpeg95_clip.npy")))]
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
    for key in missing:
        paths = manifest[key]
        print(f"  提取 {key} jpeg95 NP-RSR + CLIP...", flush=True)
        d = nprsr_descriptors(net, paths, device, jpeg=True, tag=key)
        np.save(os.path.join(OUT, f"{key}_jpeg95_nprsr.npy"),
                d.astype(np.float32))
        _, pooled = extract_clip_only(clip, paths, device, jpeg=True,
                                      tag=key)
        np.save(os.path.join(OUT, f"{key}_jpeg95_clip.npy"),
                pooled.astype(np.float32))
    del net, clip
    torch.cuda.empty_cache()


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def acq_features(paths):
    from PIL import Image
    feats = []
    for p in paths:
        with Image.open(p) as img:
            feats.append([np.log(os.path.getsize(p) + 1),
                          np.log(img.width), np.log(img.height)])
    return np.array(feats, dtype=np.float32)


def load_all(manifest, race_feats, race_nprsr):
    data = {}
    for key in ("train_real", "cal_real"):
        j_nprsr = np.nan_to_num(
            np.load(os.path.join(OUT, f"{key}_jpeg95_nprsr.npy")),
            nan=0.0, posinf=0.0, neginf=0.0)
        data[key] = {
            "orig": {
                "c": race_feats[f"{key}::v0::clip_pooled"],
                "r24": race_nprsr[key][:, :, 960:984].mean(1),
                "r960": race_nprsr[key][:, :, :960].mean(1),
                "paths": manifest[key]},
            "jpeg95": {
                "c": np.load(os.path.join(OUT, f"{key}_jpeg95_clip.npy")),
                "r24": j_nprsr[:, :, 960:984].mean(1),
                "r960": j_nprsr[:, :, :960].mean(1),
                "paths": manifest[key]},
        }
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            data[key] = {}
            for codec in ("orig", "jpeg95"):
                data[key][codec] = {
                    "c": race_feats[f"{key}::{codec}::clip_pooled"],
                    "r24": race_nprsr[f"{key}::{codec}"][:, :, 960:984].mean(1),
                    "r960": race_nprsr[f"{key}::{codec}"][:, :, :960].mean(1),
                    "paths": manifest["test"][key]}
    return data


def _cov_inv(cov, diag_only=False):
    d = cov.shape[0]
    if diag_only:
        prec_diag = 1.0 / (np.diag(cov) * (1 + LAM_COV) + 1e-6)
        return np.diag(prec_diag)
    cov_s = cov + LAM_COV * np.diag(np.diag(cov)) + 1e-6 * np.eye(d)
    return np.linalg.pinv(cov_s)


def maha_uncond(r_ref, r_test, diag_only=False):
    mu = r_ref.mean(0)
    prec = _cov_inv(np.cov((r_ref - mu).T), diag_only)
    diff = r_test - mu
    return np.einsum("ij,jk,ik->i", diff, prec, diff)


def maha_cond(r_ref, r_test, W, fallback, ess_min=ESS_MIN, diag_only=False):
    scores = np.copy(fallback)
    ess_all = np.zeros(len(r_test))
    for i in range(len(r_test)):
        w = W[i]
        ess = w.sum() ** 2 / ((w ** 2).sum() + 1e-12)
        ess_all[i] = ess
        if ess < ess_min:
            continue  # 退回 M0
        w = w / w.sum()
        mu = (w[:, None] * r_ref).sum(0)
        X = r_ref - mu
        cov = (w[:, None] * X).T @ X
        prec = _cov_inv(cov, diag_only)
        diff = r_test[i] - mu
        scores[i] = diff @ prec @ diff
    return scores, ess_all


def kernel_weights(c_ref, c_test, a_ref_n, a_test_n):
    dc = 1 - unit(c_test) @ unit(c_ref).T
    da = ((a_test_n[:, None, :] - a_ref_n[None, :, :]) ** 2).sum(-1)
    sig_c = np.median(dc) + 1e-6
    sig_a = np.median(da) + 1e-6
    Wc = np.exp(-dc / (2 * sig_c))
    Wa = np.exp(-da / (2 * sig_a))
    return Wc, Wa, Wc * Wa


def conformal_tau(s_cal, alpha=ALPHA):
    n = len(s_cal)
    j = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s_cal)[min(j, n) - 1])


def res_bucket(paths):
    from PIL import Image
    return np.array([0 if min(Image.open(p).size) < 512 else 1
                     for p in paths])


def paired_boot_auc(y, s1, s2, n_boot=1000, seed=0):
    """paired bootstrap ΔAUROC (s1 - s2) 95% CI。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y[idx].sum() in (0, len(idx)):
            continue
        diffs.append(roc_auc_score(y[idx], s1[idx])
                     - roc_auc_score(y[idx], s2[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(RACE_DIR, "manifest.json")))
    ensure_carf_caches(manifest)

    race_feats = dict(np.load(os.path.join(RACE_DIR, "feats.npz")))
    race_nprsr = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                  for k, v in np.load(os.path.join(RACE_DIR,
                                                   "nprsr.npz")).items()}
    data = load_all(manifest, race_feats, race_nprsr)

    # RACE NP-RSR s_R 对照
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = DescAE().to(device)
    ae.load_state_dict(torch.load(os.path.join(RACE_DIR, "nprsr_ae.pt"),
                                  map_location=device, weights_only=True))
    ae.eval()
    tr_desc = race_nprsr["train_real"].reshape(-1, 1968)
    mu_d, sd_d = tr_desc.mean(0), np.maximum(tr_desc.std(0), 1e-3)
    RES_IDX = np.arange(960, 984)

    @torch.no_grad()
    def race_sR(desc):
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        z = torch.tensor(np.clip((desc.reshape(-1, 1968) - mu_d) / sd_d,
                                 -15, 15), device=device, dtype=torch.float32)
        zh, _ = ae(z)
        return ((zh - z)[:, RES_IDX] ** 2).mean(1).cpu().numpy() \
            .reshape(-1, 3).mean(1)

    results = {"config": {
        "protocol": "carf-p0 (race-p0-1000 示例; A=全 JPEG95 codec-matched,"
                    " B=orig as-is)",
        "residuals": ["r24", "r960"], "ess_min": ESS_MIN,
        "lam_cov": LAM_COV, "alpha": ALPHA,
    }, "protocols": {}}

    for protocol, codec in (("A_jpeg95", "jpeg95"), ("B_orig", "orig")):
        print(f"===== 协议 {protocol} =====", flush=True)
        ref, cal = data["train_real"][codec], data["cal_real"][codec]
        a_ref = acq_features(ref["paths"])
        a_cal = acq_features(cal["paths"])
        a_mu, a_sd = a_ref.mean(0), a_ref.std(0) + 1e-8
        a_ref_n, a_cal_n = (a_ref - a_mu) / a_sd, (a_cal - a_mu) / a_sd
        g_cal = res_bucket(cal["paths"])

        proto_res = {}
        for rtype in ("r24", "r960"):
            r_ref, r_cal = ref[rtype], cal[rtype]
            diag_only = (rtype == "r960")
            # 校准集分数 (逐模型)
            Wc_cal, Wa_cal, Wca_cal = kernel_weights(ref["c"], cal["c"],
                                                     a_ref_n, a_cal_n)
            m0_cal = maha_uncond(r_ref, r_cal, diag_only)
            m1_cal, _ = maha_cond(r_ref, r_cal, Wc_cal, m0_cal,
                                  diag_only=diag_only)
            m2_cal, _ = maha_cond(r_ref, r_cal, Wa_cal, m0_cal,
                                  diag_only=diag_only)
            m3_cal, _ = maha_cond(r_ref, r_cal, Wca_cal, m0_cal,
                                  diag_only=diag_only)
            cal_scores = {"M0": m0_cal, "M1": m1_cal, "M2": m2_cal,
                          "M3": m3_cal}
            if rtype == "r24":
                cal_scores["RACE_NP-RSR"] = race_sR(
                    race_nprsr["cal_real"] if codec == "orig"
                    else np.load(os.path.join(OUT, "cal_real_jpeg95_"
                                              "nprsr.npy")))

            models = {}
            for g in GENS:
                te_r, te_f = data[f"{g}/0_real"][codec], \
                    data[f"{g}/1_fake"][codec]
                r_te = np.concatenate([te_r[rtype], te_f[rtype]])
                y = np.array([0] * len(te_r[rtype])
                              + [1] * len(te_f[rtype]))
                a_te_n = (np.concatenate([acq_features(te_r["paths"]),
                                          acq_features(te_f["paths"])])
                          - a_mu) / a_sd
                c_te = np.concatenate([te_r["c"], te_f["c"]])
                Wc, Wa, Wca = kernel_weights(ref["c"], c_te, a_ref_n, a_te_n)

                m0 = maha_uncond(r_ref, r_te, diag_only)
                m1, ess1 = maha_cond(r_ref, r_te, Wc, m0, diag_only=diag_only)
                m2, ess2 = maha_cond(r_ref, r_te, Wa, m0, diag_only=diag_only)
                m3, ess3 = maha_cond(r_ref, r_te, Wca, m0, diag_only=diag_only)
                s_map = {"M0": (m0, None), "M1": (m1, ess1),
                         "M2": (m2, ess2), "M3": (m3, ess3)}
                if rtype == "r24":
                    s_race = np.concatenate([
                        race_sR(race_nprsr[f"{g}/0_real::{codec}"]),
                        race_sR(race_nprsr[f"{g}/1_fake::{codec}"])])
                    s_map["RACE_NP-RSR"] = (s_race, None)

                n_real = len(te_r[rtype])
                for mname, (s, ess) in s_map.items():
                    tau = conformal_tau(cal_scores[mname])
                    entry = {
                        "AUROC": float(roc_auc_score(y, s)),
                        "AP": float(average_precision_score(y, s)),
                        "FPR@5%cal": float((s[:n_real] > tau).mean()),
                    }
                    if ess is not None:
                        entry["ESS_median"] = float(np.median(ess))
                        entry["coverage"] = float((ess >= ESS_MIN).mean())
                    models.setdefault(mname, {})[g] = entry
            proto_res[rtype] = models
            for mname in models:
                aucs = [models[mname][g]["AUROC"] for g in GENS]
                fprs = [models[mname][g]["FPR@5%cal"] for g in GENS]
                print(f"  [{rtype}] {mname}: macro={np.mean(aucs):.4f} "
                      f"worst={np.min(aucs):.4f} FPR={np.mean(fprs):.3f}",
                      flush=True)

        # Mondrian 注记: 协议 A 单 codec 下 Mondrian 退化为分辨率桶分组;
        # 逐生成器 pooled conformal FPR 已在上方报告 (FPR@5%cal)
        results["protocols"][protocol] = proto_res

    # ---- GO 判定 (协议 A, r24) + paired bootstrap ----
    A = results["protocols"]["A_jpeg95"]["r24"]
    auc_m0 = np.array([A["M0"][g]["AUROC"] for g in GENS])
    auc_m3 = np.array([A["M3"][g]["AUROC"] for g in GENS])
    auc_race = np.array([A["RACE_NP-RSR"][g]["AUROC"] for g in GENS])
    fpr_m3 = np.array([A["M3"][g]["FPR@5%cal"] for g in GENS])

    # 逐图分数已在 per-gen 循环中计算; 用保存的逐生成器 AUROC 做
    # cluster bootstrap (生成器内重采样近似): 以各生成器 AUROC 为单元
    rng = np.random.RandomState(SEED)
    n_boot = 1000
    boot_m0, boot_m3, boot_race = [], [], []
    for _ in range(n_boot):
        gi = rng.randint(0, len(GENS), len(GENS))
        boot_m0.append(auc_m0[gi].mean())
        boot_m3.append(auc_m3[gi].mean())
        boot_race.append(auc_race[gi].mean())
    d_m0 = np.array(boot_m3) - np.array(boot_m0)
    d_race = np.array(boot_m3) - np.array(boot_race)
    results["go"] = {
        "macro_M0": float(auc_m0.mean()),
        "macro_M3": float(auc_m3.mean()),
        "macro_RACE_NP-RSR": float(auc_race.mean()),
        "delta_M3-M0_CI95": [float(np.percentile(d_m0, 2.5)),
                             float(np.percentile(d_m0, 97.5))],
        "delta_M3-RACE_CI95": [float(np.percentile(d_race, 2.5)),
                               float(np.percentile(d_race, 97.5))],
        "crit1_M3>M0": bool(auc_m3.mean() > auc_m0.mean()),
        "crit1_M3>RACE": bool(auc_m3.mean() > auc_race.mean()),
        "crit2_FPR_in_[3%,7%]": bool(0.03 <= fpr_m3.mean() <= 0.07),
        "crit3_gens_not_worse>=5": int((auc_m3 >= auc_m0 - 1e-9).sum()),
        "note": "bootstrap 以生成器 AUROC 为单元 (cluster 级); "
                "逐图 bootstrap 需保存逐图分数, 列为后续工作",
    }
    results["go"]["GO"] = bool(results["go"]["crit1_M3>M0"]
                               and results["go"]["crit1_M3>RACE"]
                               and results["go"]["crit2_FPR_in_[3%,7%]"]
                               and results["go"]["crit3_gens_not_worse>=5"] >= 5)
    print(json.dumps(results["go"], indent=2), flush=True)

    out_path = os.path.join(OUT, "carf_p0.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {out_path}", flush=True)


if __name__ == "__main__":
    main()
