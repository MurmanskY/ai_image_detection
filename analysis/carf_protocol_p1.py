"""CARF-A 协议与 nuisance–signal coupling 复核 (v12 P0/P1)。

不提出 SACR-v2。固定 CARF-A full 主配置与 size-only 诊断，复核:
  1. ARForensics clean manifest, 3 seeds, NP-RSR 与独立固定高通表示;
  2. QF95-only / same-quality / mixed-codec reference, 逐 codec 独立校准;
  3. DARG T2I (RAISE real + 5 AR generators) 外部小规模验证;
  4. 两种表示上的 nuisance 稳定性、MMD 漂移与 fake 信号保留率;
  5. 文件/分辨率/像素复杂度标签审计、逐图分数与 paired bootstrap。

外部协议: 1000 RAISE real 按 seed 分 400 reference/300 calibration/300 test;
每生成器固定 300 fake。所有视图统一由源图解码后生成。属于
carf-p1-arforensics-1000/cross-300 + darg-t2i-300 示例, 不是全量评估。

用法: python analysis/carf_protocol_p1.py [--workers 8]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_a import fixed_sigma, paired_boot_macro, wa_kernel  # noqa: E402
from analysis.carf_p0 import conformal_tau, maha_cond, maha_uncond  # noqa: E402
from analysis.contra_p0 import OUT as CONTRA_OUT, extract_r24, mmd_rbf  # noqa: E402
from analysis.forensic_bank import ensure_highpass_cache  # noqa: E402
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import GENS, build_noiseprint  # noqa: E402
from analysis.sacr_p0 import (OUT as SACR_OUT, U_VIEWS, VIEWS,  # noqa: E402
                              acquisition_features, complexity_audit,
                              extract_complexity, leakage_auc, load_residuals)

OUT = os.path.join(PROJ, "outputs", "carf_protocol")
DARG_ROOT = os.path.join(PROJ, "data", "darg", "darg", "t2i")
EXT_GENS = ("infinity_2b", "janus_1b", "janus_7b", "llamagen_xlstage2",
            "switti_1024")
DRIFT_GENS = ("Infinity", "Open_MAGVIT2", "VAR")
SEEDS = (0, 1, 2)
REPS = ("np", "hp")
POOLS = ("qf95_only", "same_quality", "mixed_codec")
N_CROSS = 300
K_RANK = 6


def safe(key):
    return key.replace("/", "__").replace("::", "--")


def clean(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)


def external_paths():
    paths = {"real": [os.path.join(DARG_ROOT, "real", f)
                      for f in sorted(os.listdir(os.path.join(DARG_ROOT, "real")))]}
    for g in EXT_GENS:
        root = os.path.join(DARG_ROOT, g)
        paths[g] = [os.path.join(root, f) for f in sorted(os.listdir(root))][:300]
    assert len(paths["real"]) == 1000
    assert all(len(paths[g]) == 300 for g in EXT_GENS)
    return paths


def ensure_caches(manifest, ext_paths, workers):
    hp_jobs = []
    for view in ("qf95",) + U_VIEWS:
        hp_jobs.append(("pool", manifest["pool_real"], view))
        hp_jobs.append(("ext_real", ext_paths["real"], view))
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            hp_jobs.append((key, manifest["test"][key], "qf95"))
            if g in DRIFT_GENS:
                for view in VIEWS[1:]:
                    hp_jobs.append((key, manifest["test"][key][:N_CROSS], view))
    for g in EXT_GENS:
        for view in VIEWS:
            hp_jobs.append((g, ext_paths[g], view))
    todo = [(k, p, v) for k, p, v in hp_jobs
            if not os.path.exists(os.path.join(
                OUT, f"hp24_{safe(k)}_{v}.npy"))]
    print(f"固定高通缓存: {len(todo)}/{len(hp_jobs)} 集待提取", flush=True)
    for key, paths, view in todo:
        print(f"  HP {key}/{view}: {len(paths)}", flush=True)
        ensure_highpass_cache(OUT, key, paths, view, workers=workers)

    np_jobs = []
    for view in ("qf95",) + U_VIEWS:
        np_jobs.append(("ext_real", ext_paths["real"], view))
    for g in EXT_GENS:
        for view in VIEWS:
            np_jobs.append((g, ext_paths[g], view))
    todo = [(k, p, v) for k, p, v in np_jobs
            if not os.path.exists(os.path.join(OUT, f"np_{safe(k)}_{v}.npy"))]
    print(f"外部 NP-RSR 缓存: {len(todo)}/{len(np_jobs)} 集待提取", flush=True)
    if todo:
        gpu = wait_for_gpu(need_mb=4000)
        device = torch.device(f"cuda:{gpu}")
        net = build_noiseprint(device)
        for key, paths, view in todo:
            r = extract_r24(net, paths, view, device, tag=f"{key}/{view}")
            np.save(os.path.join(OUT, f"np_{safe(key)}_{view}.npy"), r)
        del net
        torch.cuda.empty_cache()

    complexity_jobs = ([("ext_real", ext_paths["real"])]
                       + [(g, ext_paths[g]) for g in EXT_GENS])
    for key, paths in complexity_jobs:
        path = os.path.join(OUT, f"complexity_{safe(key)}_qf95.npy")
        if not os.path.exists(path):
            x = extract_complexity(paths, "qf95", key)
            np.save(path, x)


def load_all(manifest, ext_paths):
    data = {"arf": {"np": {}, "hp": {}}, "ext": {"np": {}, "hp": {}}}
    np_arf = load_residuals(manifest)
    for view in ("qf95",) + U_VIEWS:
        if view in np_arf["pool"]:
            data["arf"]["np"].setdefault("pool", {})[view] = np_arf["pool"][view]
        else:
            tr = clean(np.load(os.path.join(
                CONTRA_OUT, f"r24_train_real::{view}.npy")))
            ca = clean(np.load(os.path.join(
                CONTRA_OUT, f"r24_cal_real::{view}.npy")))
            source_idx = np.asarray(manifest["pool_source_idx"])
            data["arf"]["np"].setdefault("pool", {})[view] = np.concatenate(
                [tr, ca])[source_idx]
        data["arf"]["hp"].setdefault("pool", {})[view] = clean(np.load(os.path.join(
            OUT, f"hp24_pool_{view}.npy")))
        data["ext"]["np"].setdefault("real", {})[view] = clean(np.load(os.path.join(
            OUT, f"np_ext_real_{view}.npy")))
        data["ext"]["hp"].setdefault("real", {})[view] = clean(np.load(os.path.join(
            OUT, f"hp24_ext_real_{view}.npy")))
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            for view in VIEWS:
                if view == "qf95" or g in DRIFT_GENS:
                    data["arf"]["np"].setdefault(key, {})[view] = np_arf[key][view]
                    data["arf"]["hp"].setdefault(key, {})[view] = clean(np.load(
                        os.path.join(OUT, f"hp24_{safe(key)}_{view}.npy")))
    for g in EXT_GENS:
        for view in VIEWS:
            data["ext"]["np"].setdefault(g, {})[view] = clean(np.load(os.path.join(
                OUT, f"np_{safe(g)}_{view}.npy")))
            data["ext"]["hp"].setdefault(g, {})[view] = clean(np.load(os.path.join(
                OUT, f"hp24_{safe(g)}_{view}.npy")))
    return data


def split_indices(manifest, scope, seed):
    if scope == "arf":
        split = manifest["seeds"][str(seed)]
        return (np.asarray(split["ref_idx"]), np.asarray(split["cal_idx"]),
                np.arange(1000))
    idx = np.random.RandomState(seed).permutation(1000)
    return idx[:400], idx[400:700], idx[700:]


def paths_for(manifest, ext_paths, scope):
    if scope == "arf":
        return {"real": manifest["pool_real"],
                **{g: manifest["test"][f"{g}/1_fake"] for g in GENS},
                **{f"{g}::real": manifest["test"][f"{g}/0_real"] for g in GENS}}
    return {"real": ext_paths["real"], **{g: ext_paths[g] for g in EXT_GENS}}


def resolution_groups(paths):
    groups = []
    for p in paths:
        with Image.open(p) as img:
            groups.append(int(min(img.size) >= 512))
    return np.asarray(groups)


def resolution_taus(cal_scores, cal_groups):
    taus = {}
    for b in (0, 1):
        s = cal_scores[cal_groups == b]
        taus[str(b)] = conformal_tau(s) if len(s) >= 50 else np.nan
    return taus


def metric(y, scores, n_real, real_groups, taus):
    valid = np.isfinite(scores)
    auc = roc_auc_score(y[valid], scores[valid])
    ap = average_precision_score(y[valid], scores[valid])
    fpr, tpr, _ = roc_curve(y[valid], scores[valid])
    tpr01 = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0
    pred = np.zeros(n_real, dtype=bool)
    supported = np.zeros(n_real, dtype=bool)
    for b in (0, 1):
        m = (real_groups == b) & np.isfinite(taus[str(b)])
        supported |= m
        pred[m] = scores[:n_real][m] > taus[str(b)]
    return {"AUROC": float(auc), "AP": float(ap), "TPR@1%FPR": tpr01,
            "FPR": float(pred[supported].mean()),
            "coverage": float(valid.mean()),
            "calibration_coverage": float(supported.mean())}


def score_models(r_ref, ref_paths, r_cal, cal_paths, r_query, query_paths):
    scores, cal_scores, ess = {}, {}, {}
    scores["M0"] = maha_uncond(r_ref, r_query)
    cal_scores["M0"] = maha_uncond(r_ref, r_cal)
    for variant in ("full", "size_only"):
        name = "CARF-A-full" if variant == "full" else "CARF-A-size_only"
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
        cal_scores[name], _ = maha_cond(r_ref, r_cal, w_cal, cal_base)
        scores[name], ess[name] = maha_cond(r_ref, r_query, w_query, query_base)
    return scores, cal_scores, ess


def evaluate_gens(r_ref, ref_paths, r_cal, cal_paths, tests, save_path=None):
    cal_groups = resolution_groups(cal_paths)
    per_gen, saved = {}, {}
    for g, (r_query, query_paths, n_real) in tests.items():
        y = np.array([0] * n_real + [1] * (len(r_query) - n_real))
        scores, cal_scores, ess = score_models(
            r_ref, ref_paths, r_cal, cal_paths, r_query, query_paths)
        real_groups = resolution_groups(query_paths[:n_real])
        entry = {}
        for name, score in scores.items():
            taus = resolution_taus(cal_scores[name], cal_groups)
            entry[name] = metric(y, score, n_real, real_groups, taus)
            entry[name]["ESS_median"] = float(np.median(ess[name])) \
                if name in ess else None
            saved[f"{g}::{name}"] = score
        per_gen[g] = entry
    if save_path:
        np.savez(save_path, **saved)
    return per_gen


def canonical(data, paths, manifest, scope, rep, seed):
    ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
    real = data[scope][rep]["real"] if scope == "ext" else data[scope][rep]["pool"]
    real_paths = paths["real"]
    r_ref = real["qf95"][ref_idx]
    r_cal = real["qf95"][cal_idx]
    ref_paths = [real_paths[i] for i in ref_idx]
    cal_paths = [real_paths[i] for i in cal_idx]
    gens = GENS if scope == "arf" else EXT_GENS
    tests = {}
    for g in gens:
        if scope == "arf":
            rr = data[scope][rep][f"{g}/0_real"]["qf95"]
            rf = data[scope][rep][f"{g}/1_fake"]["qf95"]
            pr = paths[f"{g}::real"]
            pf = paths[g]
            rr, pr = rr[test_idx], [pr[i] for i in test_idx]
        else:
            rr = data[scope][rep]["real"]["qf95"][test_idx]
            rf = data[scope][rep][g]["qf95"]
            pr = [real_paths[i] for i in test_idx]
            pf = paths[g]
        tests[g] = (np.concatenate([rr, rf]), pr + pf, len(rr))
    save = os.path.join(OUT, f"canonical_{scope}_{rep}_seed{seed}.npz")
    return evaluate_gens(r_ref, ref_paths, r_cal, cal_paths, tests, save)


def reference_pools(data, paths, manifest, scope, rep, seed):
    ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
    real = data[scope][rep]["real"] if scope == "ext" else data[scope][rep]["pool"]
    real_paths = paths["real"]
    cal_paths = [real_paths[i] for i in cal_idx]
    gens = DRIFT_GENS if scope == "arf" else EXT_GENS
    result = {}
    for view in VIEWS[1:]:
        r_cal = real[view][cal_idx]
        result[view] = {}
        for pool in POOLS:
            ref_views = VIEWS if pool == "mixed_codec" else (
                (view,) if pool == "same_quality" else ("qf95",))
            r_ref = np.concatenate([real[v][ref_idx] for v in ref_views])
            ref_paths = [real_paths[i] for v in ref_views for i in ref_idx]
            tests = {}
            for g in gens:
                if scope == "arf":
                    rr = data[scope][rep][f"{g}/0_real"][view][:N_CROSS]
                    rf = data[scope][rep][f"{g}/1_fake"][view][:N_CROSS]
                    pr = paths[f"{g}::real"][:N_CROSS]
                    pf = paths[g][:N_CROSS]
                else:
                    rr = data[scope][rep]["real"][view][test_idx]
                    rf = data[scope][rep][g][view]
                    pr = [real_paths[i] for i in test_idx]
                    pf = paths[g]
                tests[g] = (np.concatenate([rr, rf]), pr + pf, len(rr))
            save = os.path.join(
                OUT, f"pool_{scope}_{rep}_seed{seed}_{view}_{pool}.npz")
            result[view][pool] = evaluate_gens(
                r_ref, ref_paths, r_cal, cal_paths, tests, save)
            print(f"  pool {scope}/{rep}/seed{seed}/{view}/{pool}: " +
                  "; ".join(f"{m}={np.mean([result[view][pool][g][m]['AUROC'] for g in gens]):.4f}"
                            for m in ("M0", "CARF-A-full", "CARF-A-size_only")),
                  flush=True)
    return result


def fit_u(real, ref_idx):
    r0 = real["qf95"][ref_idx]
    deltas = {v: real[v][ref_idx] - r0 for v in U_VIEWS}
    d = np.concatenate(list(deltas.values()))
    _, _, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
    return vt[:K_RANK].T, deltas


def coupling(data, paths, manifest, scope, rep, seed):
    ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
    real = data[scope][rep]["real"] if scope == "ext" else data[scope][rep]["pool"]
    u, deltas = fit_u(real, ref_idx)
    half = len(ref_idx) // 2
    stability = {}
    means = {}
    for v, d in deltas.items():
        m1, m2 = d[:half].mean(0), d[half:].mean(0)
        stability[v] = float(abs(m1 @ m2) /
                             (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12))
        means[v] = d.mean(0)
    drift = {}
    for v in VIEWS[1:]:
        raw = real[v][cal_idx]
        canonical = real["qf95"][cal_idx]
        corrected = raw - means[v]
        raw_mmd = mmd_rbf(raw, canonical)
        corrected_mmd = mmd_rbf(corrected, canonical)
        drift[v] = {"raw": raw_mmd, "mean": corrected_mmd,
                    "reduction": float(1 - corrected_mmd / (raw_mmd + 1e-12))}
    gens = GENS if scope == "arf" else EXT_GENS
    retention = {}
    p_perp = np.eye(real["qf95"].shape[1]) - u @ u.T
    for g in gens:
        if scope == "arf":
            rr = data[scope][rep][f"{g}/0_real"]["qf95"]
            rf = data[scope][rep][f"{g}/1_fake"]["qf95"]
        else:
            rr = real["qf95"][test_idx]
            rf = data[scope][rep][g]["qf95"]
        delta = rf.mean(0) - rr.mean(0)
        retention[g] = float(np.sum((p_perp @ delta) ** 2) /
                             (np.sum(delta ** 2) + 1e-12))
    return {"stability_cos": stability, "min_stability": min(stability.values()),
            "drift": drift, "retention_eta": retention,
            "mean_eta": float(np.mean(list(retention.values())))}


def external_audit(ext_paths, test_idx):
    real_c = clean(np.load(os.path.join(OUT, "complexity_ext_real_qf95.npy")))[test_idx]
    real_paths = [ext_paths["real"][i] for i in test_idx]
    result = {}
    for g in EXT_GENS:
        fake_c = clean(np.load(os.path.join(OUT, f"complexity_{g}_qf95.npy")))
        real_full = acquisition_features(real_paths, "full")
        fake_full = acquisition_features(ext_paths[g], "full")
        result[g] = {
            "pixel": leakage_auc(real_c, fake_c),
            "full": leakage_auc(real_full, fake_full),
            "size_only": leakage_auc(real_full[:, :1], fake_full[:, :1]),
        }
    result["macro"] = {k: float(np.mean([result[g][k] for g in EXT_GENS]))
                       for k in ("pixel", "full", "size_only")}
    return result


def macro_auc(per_gen, method):
    return float(np.mean([v[method]["AUROC"] for v in per_gen.values()]))


def summarize_canonical(canonical_results):
    summary = {}
    for scope, reps in canonical_results.items():
        summary[scope] = {}
        for rep, seeds in reps.items():
            method_summary = {}
            for method in ("M0", "CARF-A-full", "CARF-A-size_only"):
                aucs = [macro_auc(seeds[str(s)], method) for s in SEEDS]
                fprs = [float(np.mean([
                    v[method]["FPR"] for v in seeds[str(s)].values()]))
                    for s in SEEDS]
                method_summary[method] = {
                    "macro_AUROC_per_seed": aucs,
                    "macro_AUROC_mean": float(np.mean(aucs)),
                    "FPR_per_seed": fprs,
                    "worst_AUROC_mean": float(np.mean([
                        min(v[method]["AUROC"] for v in seeds[str(s)].values())
                        for s in SEEDS])),
                }
            deltas, ci_lows, not_worse = [], [], []
            for s in SEEDS:
                per_gen = seeds[str(s)]
                z = np.load(os.path.join(
                    OUT, f"canonical_{scope}_{rep}_seed{s}.npz"))
                paired = {}
                for g in per_gen:
                    n = len(z[f"{g}::M0"]) // 2
                    y = np.array([0] * n + [1] * n)
                    paired[g] = (y, z[f"{g}::CARF-A-full"], z[f"{g}::M0"])
                delta, ci = paired_boot_macro(paired, n_boot=1000, seed=s)
                deltas.append(macro_auc(per_gen, "CARF-A-full")
                              - macro_auc(per_gen, "M0"))
                ci_lows.append(ci[0])
                not_worse.append(sum(
                    per_gen[g]["CARF-A-full"]["AUROC"] + 1e-9
                    >= per_gen[g]["M0"]["AUROC"] for g in per_gen))
            required = 5 if scope == "arf" else 4
            method_summary["conditional_replication"] = {
                "delta_full_M0_per_seed": deltas,
                "CI95_lower_per_seed": ci_lows,
                "not_worse_per_seed": not_worse,
                "pass": bool(all(d > 0 for d in deltas)
                             and all(x > 0 for x in ci_lows)
                             and all(x >= required for x in not_worse)
                             and all(0.03 <= x <= 0.07 for x in
                                     method_summary["CARF-A-full"]["FPR_per_seed"])),
            }
            summary[scope][rep] = method_summary
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(SACR_OUT, "manifest.json")))
    ext_paths = external_paths()
    ensure_caches(manifest, ext_paths, args.workers)
    data = load_all(manifest, ext_paths)
    arf_paths = paths_for(manifest, ext_paths, "arf")
    ext_path_map = paths_for(manifest, ext_paths, "ext")

    canonical_results = {"arf": {"np": {}, "hp": {}},
                         "ext": {"np": {}, "hp": {}}}
    pool_results = {"arf": {"np": {}, "hp": {}},
                    "ext": {"np": {}, "hp": {}}}
    coupling_results = {"arf": {"np": {}, "hp": {}},
                        "ext": {"np": {}, "hp": {}}}
    for scope, paths in (("arf", arf_paths), ("ext", ext_path_map)):
        for rep in REPS:
            for seed in SEEDS:
                print(f"===== {scope}/{rep}/seed{seed} canonical =====", flush=True)
                canonical_results[scope][rep][str(seed)] = canonical(
                    data, paths, manifest, scope, rep, seed)
                print("  " + "; ".join(
                    f"{m}={macro_auc(canonical_results[scope][rep][str(seed)], m):.4f}"
                    for m in ("M0", "CARF-A-full", "CARF-A-size_only")), flush=True)
                pool_results[scope][rep][str(seed)] = reference_pools(
                    data, paths, manifest, scope, rep, seed)
                coupling_results[scope][rep][str(seed)] = coupling(
                    data, paths, manifest, scope, rep, seed)

    _, _, ext_test_idx = split_indices(manifest, "ext", 0)
    audit = external_audit(ext_paths, ext_test_idx)
    summary = summarize_canonical(canonical_results)
    results = {
        "config": {"seeds": SEEDS, "representations": REPS,
                   "reference_pools": POOLS, "arf_test_per_class": 1000,
                   "arf_cross_per_class": N_CROSS,
                   "ext_split": [400, 300, 300], "ext_gens": EXT_GENS,
                   "protocol": "carf-p1-arforensics-1000/cross-300 + darg-t2i-300 示例"},
        "canonical": canonical_results, "reference_pools": pool_results,
        "coupling": coupling_results, "external_condition_audit": audit,
        "summary": summary,
    }
    with open(os.path.join(OUT, "carf_protocol_p1.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("外部条件审计:", json.dumps(audit["macro"], indent=2), flush=True)
    print(f"已保存 {OUT}/carf_protocol_p1.json", flush=True)


if __name__ == "__main__":
    main()
