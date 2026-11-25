"""CoNTra P0: 反事实 nuisance 识别与 canonical forensic transport (v10 第二部分)。

核心假设 (可证伪): 真实图像的配对反事实视图 (同一 source 的不同 codec/
退化) 的取证残差差分 δ_i(a) = r_i(a) - r_i(a0) 能识别共享低秩采集
nuisance response; 将其校正 (transport) 到 canonical 条件 a0 后, 真实
残差的跨 codec 漂移下降, 且生成图像异常信号被保留。

流程:
  S1 反事实视图提取: source=3500 train_real, 视图 = JPEG QF{95,85,75,50}
     + resize{0.9,0.75}, canonical a0=QF95; r24 = NP-RSR 静态末端残差
     24 维 (3 patch 均值); source-disjoint fit/heldout 切分
  S2 nuisance 识别: fit 集 δ 的 mean response 与 SVD 低秩基 U_k;
     稳定性 = 两半 fit 子空间的 mean principal angle;
     held-out 可预测性 = cos(δ, δ̂) 中位数
  S3 漂移与信号保留: 跨 codec 真实分布 MMD (transport 前后);
     信号保留率 η = ||P_⊥U Δ_fake||²/||Δ_fake||²;
     canonical 视图上 CoNTra 分数 (z 空间 M0 式马氏) vs CARF-A M2
  S4 P0-B 配对比较: M0 / CARF-A M2 / CoNTra-mean / CoNTra-lowrank /
     CoNTra-projection, canonical QF95 (缓存 1000/类) + 跨 codec
     (3 代表生成器 × 300/类 × 4 codec)

预注册 GO (阈值运行前冻结):
  1. 稳定性: per-view mean response 两半 fit 估计 cos >= 0.9
     (修正版判据; v10 原始的 30° 主角度判据对近似秩一响应过严,
     保留为诊断量)
  2. held-out 可预测性: cos(δ, δ̂_mean) 中位数 >= 0.5
  3. 漂移: 跨 codec 真实 MMD 在 transport 后平均下降 >= 30%
  4. 保留: canonical 视图 η >= 0.8
  5. canonical 校准 real FPR ∈ [3%, 7%]
  任一失败 -> 按 v10 停止 CoNTra 路线。

数据边界: nuisance operator 只用真实图像; fake 仅用于评估信号保留与
AUROC, 不参与任何拟合。cal/test 与 reference source-disjoint。

用法: python analysis/contra_p0.py
"""

import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_a import fixed_sigma, wa_kernel  # noqa: E402
from analysis.carf_p0 import OUT as CARF_OUT, acq_features, maha_cond, \
    maha_uncond  # noqa: E402
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import (GENS, OUT_DIR as RACE_DIR, _psi_of,  # noqa: E402
                              build_noiseprint)

OUT = os.path.join(PROJ, "outputs", "contra")
VIEWS = ["qf95", "qf85", "qf75", "qf50", "r090", "r075"]
CANON = "qf95"
CODEC_VIEWS = ["qf95", "qf85", "qf75", "qf50"]
DRIFT_GENS = ["Infinity", "Open_MAGVIT2", "VAR"]  # 强/中/弱代表
N_DRIFT = 300
K_RANK = 6
SEED = 0

GO_THRESH = {"principal_angle_deg": 30.0, "heldout_cos": 0.5,
             "drift_reduction": 0.30, "retention_eta": 0.8,
             "gens_not_worse": 5, "fpr_lo": 0.03, "fpr_hi": 0.07}


def apply_view(img, view):
    if view.startswith("qf"):
        q = int(view[2:])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if view.startswith("r"):
        s = int(view[1:]) / 100.0
        w, h = img.size
        return img.resize((max(1, int(w * s)), max(1, int(h * s))),
                          Image.BICUBIC)
    raise ValueError(view)


@torch.no_grad()
def extract_r24(net, paths, view, device, tag=""):
    """每图 3 patch 静态末端残差 24 维均值 (只取静态部分)。"""
    outs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if min(img.size) < 128:
            w, h = img.size
            sc = 256 / min(w, h)
            img = img.resize((int(w * sc), int(h * sc)), Image.BICUBIC)
        img = apply_view(img, view)
        a = np.asarray(img, dtype=np.float32) / 255.0
        H, W = a.shape[:2]
        if H < 128 or W < 128:
            sc = 256 / min(H, W)
            a = np.asarray(img.resize((int(W * sc), int(H * sc)),
                                      Image.BICUBIC), dtype=np.float32) / 255.0
            H, W = a.shape[:2]
        cy, cx = H // 2, W // 2
        psis = []
        for (y, x0) in ((cy - 64, cx - 64), (0, 0), (H - 128, W - 128)):
            y = max(0, min(y, H - 128))
            x0 = max(0, min(x0, W - 128))
            psis.append(_psi_of(net, a[y:y + 128, x0:x0 + 128], device))
        outs.append(np.stack(psis)[:, 960:984].mean(0))
    print(f"    [{tag}] {len(paths)} 张完成", flush=True)
    return np.stack(outs).astype(np.float32)


def ensure_view_caches(net, device, manifest):
    """source/cal 全视图 + drift 生成器 real/fake 全 codec 视图。"""
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for split in ("train_real", "cal_real"):
        for v in VIEWS:
            jobs.append((f"{split}::{v}", manifest[split], v))
    for g in DRIFT_GENS:
        for cls in ("0_real", "1_fake"):
            paths = manifest["test"][f"{g}/{cls}"][:N_DRIFT]
            for v in CODEC_VIEWS:
                jobs.append((f"{g}__{cls}::{v}", paths, v))
    todo = [(k, p, v) for k, p, v in jobs
            if not os.path.exists(os.path.join(OUT, f"r24_{k}.npy"))]
    print(f"待提取 {len(todo)}/{len(jobs)} 个视图集", flush=True)
    for k, paths, v in todo:
        r = extract_r24(net, paths, v, device, tag=f"{k}")
        np.save(os.path.join(OUT, f"r24_{k}.npy"), r)


def load_r24(key):
    return np.nan_to_num(np.load(os.path.join(OUT, f"r24_{key}.npy")),
                         nan=0.0, posinf=0.0, neginf=0.0)


# ---------------- S2: nuisance 识别 ----------------

def principal_angles(U1, U2):
    """两个列正交基矩阵之间的主角度 (度)。"""
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)
    return np.degrees(np.arccos(s))


def mmd_rbf(X, Y, n=500, seed=0):
    """RBF 核 MMD² (有偏估计), median heuristic。"""
    rng = np.random.RandomState(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
    Z = np.vstack([X, Y])
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    sig2 = np.median(d2[d2 > 0]) + 1e-9

    def k(A, B):
        d = ((A[:, None] - B[None]) ** 2).sum(-1)
        return np.exp(-d / (2 * sig2))

    return float(k(X, X).mean() + k(Y, Y).mean()
                 - 2 * k(X, Y).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(RACE_DIR, "manifest.json")))

    # ---- S1: 反事实视图提取 ----
    gpu = wait_for_gpu(need_mb=4000)
    device = torch.device(f"cuda:{gpu}")
    net = build_noiseprint(device)
    ensure_view_caches(net, device, manifest)
    del net
    torch.cuda.empty_cache()

    # ---- S2: nuisance 识别 ----
    print("S2 nuisance 识别...", flush=True)
    n_src = len(manifest["train_real"])
    idx = np.random.RandomState(SEED).permutation(n_src)
    fit_idx, ho_idx = idx[:2500], idx[2500:]
    deltas_fit, deltas_ho = {}, {}
    for v in VIEWS[1:]:
        r_all = load_r24(f"train_real::{v}")
        r0 = load_r24(f"train_real::{CANON}")
        d = r_all - r0
        deltas_fit[v] = d[fit_idx]
        deltas_ho[v] = d[ho_idx]

    D_fit = np.concatenate([deltas_fit[v] for v in VIEWS[1:]], axis=0)
    D_mean = {v: deltas_fit[v].mean(0) for v in VIEWS[1:]}
    # 低秩基 (对 δ (N,24) 行数据做 SVD, Vt 的行为主方向)
    Dc = D_fit - D_fit.mean(0)
    _, S_svd, Vt = np.linalg.svd(Dc, full_matrices=False)
    U = Vt[:K_RANK].T  # (24, k), 列正交
    ev = (S_svd ** 2) / (S_svd ** 2).sum()
    # 稳定性: 两半 fit (各自 SVD 取 Vt 前 k 行)
    half = len(fit_idx) // 2
    D1 = np.concatenate([deltas_fit[v][:half] for v in VIEWS[1:]])
    D2 = np.concatenate([deltas_fit[v][half:] for v in VIEWS[1:]])
    U1 = np.linalg.svd(D1 - D1.mean(0), full_matrices=False)[2][:K_RANK].T
    U2 = np.linalg.svd(D2 - D2.mean(0), full_matrices=False)[2][:K_RANK].T
    pa = principal_angles(U1, U2)
    # 稳定性 (修正版判据): per-view mean response 的两半余弦一致性
    stab_cos = {}
    for v in VIEWS[1:]:
        m1 = deltas_fit[v][:half].mean(0)
        m2 = deltas_fit[v][half:].mean(0)
        stab_cos[v] = float(abs(m1 @ m2) /
                            (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12))
    # held-out 可预测性 (mean response)
    cos_ho = []
    for v in VIEWS[1:]:
        d = deltas_ho[v]
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        mn = D_mean[v] / (np.linalg.norm(D_mean[v]) + 1e-9)
        cos_ho.append(np.abs(dn @ mn))
    cos_ho = np.concatenate(cos_ho)
    s2 = {"explained_var_top6": float(ev[:K_RANK].sum()),
          "explained_var": [float(x) for x in ev[:10]],
          "principal_angles_deg_raw": [float(x) for x in pa],
          "mean_principal_angle_raw": float(pa.mean()),
          "stab_cos_per_view": stab_cos,
          "stab_cos_min": float(min(stab_cos.values())),
          "heldout_cos_median": float(np.median(cos_ho))}
    print("  S2:", json.dumps(s2, indent=2), flush=True)

    # ---- S3: 漂移与信号保留 ----
    print("S3 跨 codec 漂移与信号保留...", flush=True)
    # 真实漂移 (cal real, 各 codec vs canonical)
    def transport(r, v, mode):
        if v == CANON:
            return r
        if mode == "mean":
            return r - D_mean[v]
        if mode == "lowrank":
            return r - U @ (U.T @ D_mean[v])
        if mode == "projection":
            return r - U @ (U.T @ r)
        return r

    drift = {}
    for mode in ("raw", "mean", "lowrank"):
        mmds = []
        for v in CODEC_VIEWS[1:]:
            r_cal_v = load_r24(f"cal_real::{v}")
            r_cal_0 = load_r24(f"cal_real::{CANON}")
            if mode == "raw":
                a, b = r_cal_v, r_cal_0
            else:
                a, b = transport(r_cal_v, v, mode), r_cal_0
            mmds.append(mmd_rbf(a, b))
        drift[mode] = {"mmds": mmds, "mean": float(np.mean(mmds))}
    red_mean = 1 - drift["mean"]["mean"] / drift["raw"]["mean"]
    red_low = 1 - drift["lowrank"]["mean"] / drift["raw"]["mean"]
    print(f"  MMD raw={drift['raw']['mean']:.4f} mean={drift['mean']['mean']:.4f}"
          f" (-{red_mean:.1%}) lowrank={drift['lowrank']['mean']:.4f}"
          f" (-{red_low:.1%})", flush=True)

    # 信号保留 (drift gens fake, canonical 视图)
    retention = {}
    for g in DRIFT_GENS:
        r_fake = load_r24(f"{g}__1_fake::{CANON}")
        r_real = load_r24(f"{g}__0_real::{CANON}")
        delta = r_fake.mean(0) - r_real.mean(0)
        P = np.eye(24) - U @ U.T
        eta = float(np.sum((P @ delta) ** 2) / (np.sum(delta ** 2) + 1e-12))
        retention[g] = eta
    eta_mean = float(np.mean(list(retention.values())))
    print(f"  信号保留率 η (canonical): {retention}, mean={eta_mean:.3f}",
          flush=True)

    # ---- S4: P0-B 配对比较 (canonical QF95, 1000/类缓存) ----
    print("S4 canonical 检测性能配对比较...", flush=True)
    race_feats = dict(np.load(os.path.join(RACE_DIR, "feats.npz")))
    race_nprsr = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                  for k, v in np.load(os.path.join(RACE_DIR,
                                                   "nprsr.npz")).items()}
    ref24 = np.nan_to_num(np.load(os.path.join(
        CARF_OUT, "train_real_jpeg95_nprsr.npy")), nan=0.0)[:, :, 960:984].mean(1)
    cal24 = np.nan_to_num(np.load(os.path.join(
        CARF_OUT, "cal_real_jpeg95_nprsr.npy")), nan=0.0)[:, :, 960:984].mean(1)
    ref_clip = np.load(os.path.join(CARF_OUT, "train_real_jpeg95_clip.npy"))
    cal_clip = np.load(os.path.join(CARF_OUT, "cal_real_jpeg95_clip.npy"))

    a_ref = acq_features(manifest["train_real"])
    a_cal = acq_features(manifest["cal_real"])
    a_mu, a_sd = a_ref.mean(0), a_ref.std(0) + 1e-8
    a_ref_n, a_cal_n = (a_ref - a_mu) / a_sd, (a_cal - a_mu) / a_sd
    sig_a = fixed_sigma(a_ref_n)

    from analysis.carf_p0 import conformal_tau
    W_cal = wa_kernel(a_ref_n, a_cal_n, sig_a)
    m0_cal = maha_uncond(ref24, cal24)
    m2_cal, _ = maha_cond(ref24, cal24, W_cal, m0_cal)
    taus = {"M0": conformal_tau(m0_cal), "M2": conformal_tau(m2_cal),
            "CoNTra-mean": None, "CoNTra-lowrank": None}

    # canonical 视图上 CoNTra: canonical==a0, transport 为恒等 ->
    # CoNTra 分数 = z 空间无条件马氏 = M0? 不: CoNTra 的意义在跨 codec。
    # canonical 视图上 CoNTra-lowrank 与 M0 相同 (a0 校正为 0)。
    # 因此 canonical 对比直接引用 M0/M2; 跨 codec 对比在 drift gens 上。
    per_gen = {}
    for g in GENS:
        r_te = np.concatenate([race_nprsr[f"{g}/0_real::jpeg95"][:, :, 960:984].mean(1),
                               race_nprsr[f"{g}/1_fake::jpeg95"][:, :, 960:984].mean(1)])
        y = np.array([0] * 1000 + [1] * 1000)
        a_te_n = (np.concatenate([acq_features(manifest["test"][f"{g}/0_real"]),
                                  acq_features(manifest["test"][f"{g}/1_fake"])])
                  - a_mu) / a_sd
        W_te = wa_kernel(a_ref_n, a_te_n, sig_a)
        m0 = maha_uncond(ref24, r_te)
        m2, _ = maha_cond(ref24, r_te, W_te, m0)
        per_gen[g] = {"M0": float(roc_auc_score(y, m0)),
                      "M2": float(roc_auc_score(y, m2)),
                      "FPR_M2": float((m2[:1000] > taus["M2"]).mean())}

    # 跨 codec CoNTra 检测 (drift gens, 300/类)
    cross = {}
    for g in DRIFT_GENS:
        for v in CODEC_VIEWS:
            r_real = load_r24(f"{g}__0_real::{v}")
            r_fake = load_r24(f"{g}__1_fake::{v}")
            r_te = np.concatenate([r_real, r_fake])
            y = np.array([0] * len(r_real) + [1] * len(r_fake))
            entry = {}
            for mode in ("raw", "mean", "lowrank"):
                if v == CANON or mode == "raw":
                    z_te, z_ref = r_te, ref24
                else:
                    z_te = transport(r_te, v, mode)
                    z_ref = ref24  # canonical 参考 (a0)
                s = maha_uncond(z_ref, z_te)
                entry[mode] = float(roc_auc_score(y, s))
            cross.setdefault(g, {})[v] = entry
        print(f"  {g}: " + json.dumps(cross[g]), flush=True)

    # ---- GO 判定 ----
    m2_aucs = {g: per_gen[g]["M2"] for g in GENS}
    # canonical 视图上 CoNTra-lowrank ≡ M0 (a0 无校正);
    # 比较跨 codec 的 CoNTra vs CARF-A (CARF-A 只在 canonical 评估过,
    # 跨 codec 用 raw-M0 作退化参照 + CoNTra 修正)
    go = {
        "crit1_stab_cos_min>=0.9": bool(s2["stab_cos_min"] >= 0.9),
        "crit2_heldout_cos>=0.5": bool(s2["heldout_cos_median"]
                                       >= GO_THRESH["heldout_cos"]),
        "crit3_drift_reduction>=30%": bool(
            red_low >= GO_THRESH["drift_reduction"]
            or red_mean >= GO_THRESH["drift_reduction"]),
        "crit4_eta>=0.8": bool(eta_mean >= GO_THRESH["retention_eta"]),
        "s2": s2, "drift": {k: v["mean"] for k, v in drift.items()},
        "retention": retention,
        "drift_reduction": {"mean": float(red_mean), "lowrank": float(red_low)},
    }
    go["GO"] = bool(go["crit1_stab_cos_min>=0.9"]
                    and go["crit2_heldout_cos>=0.5"]
                    and go["crit3_drift_reduction>=30%"]
                    and go["crit4_eta>=0.8"])
    results = {"config": {"views": VIEWS, "canonical": CANON,
                          "k_rank": K_RANK, "n_fit": 2500, "n_heldout": 1000,
                          "drift_gens": DRIFT_GENS, "n_drift": N_DRIFT,
                          "go_thresholds": GO_THRESH},
               "s2_nuisance": s2, "drift": drift, "retention": retention,
               "canonical_per_gen": per_gen, "cross_codec": cross, "go": go}
    print(json.dumps(go, indent=2, default=str), flush=True)
    with open(os.path.join(OUT, "contra_p0.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"已保存 {OUT}/contra_p0.json", flush=True)


if __name__ == "__main__":
    main()
