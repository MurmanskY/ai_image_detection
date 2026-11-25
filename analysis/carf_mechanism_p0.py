"""v13 最小机制实验：反捷径、参考池统计、表示依赖性。

A. 统一 256x256 + JPEG95，条件为重编码文件大小与既有像素复杂度；
   real/fake 一对一最优匹配 (reference 标准化, 预注册 caliper=2.5)，
   比较 M0/CARF-size/CARF-pixel，匹配前后均审计标签 AUC。
B. same-quality vs mixed-codec 的 ESS、coverage 与 proxy-oracle
   bias–variance 分解；oracle 仅用全部真实参考，不使用 fake。
C. NP-RSR/固定高通/固定 DCT 三种表示的 nuisance stability、MMD reduction、
   eta 及 source/image bootstrap CI。只有 stability>=0.9 且 MMD reduction
   >=0.3 时才解释 eta<0.8 为 coupling，否则标记 unidentified。

示例规模: ARF reference/calibration 全量缓存，test 300/类；
DARG T2I 400/300/300。不是 84k 或新的检测方法评测。

用法: python analysis/carf_mechanism_p0.py --workers 8
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_a import fixed_sigma, paired_boot_macro, wa_kernel  # noqa: E402
from analysis.carf_p0 import conformal_tau, maha_cond, maha_uncond  # noqa: E402
from analysis.carf_protocol_p1 import (DRIFT_GENS, EXT_GENS, OUT as P1_OUT,  # noqa: E402
                                       REPS, external_paths, load_all,
                                       split_indices)
from analysis.contra_p0 import apply_view, mmd_rbf  # noqa: E402
from analysis.dct_forensics import ensure_dct_cache  # noqa: E402
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import GENS, _psi_of, build_noiseprint  # noqa: E402
from analysis.sacr_p0 import (OUT as SACR_OUT, U_VIEWS, VIEWS,  # noqa: E402
                              acquisition_features, complexity_patch,
                              leakage_auc, load_residuals)

OUT = os.path.join(PROJ, "outputs", "carf_mechanism")
SEEDS = (0, 1, 2)
N_TEST = 300
CALIPER = 2.5
MIN_MATCH = 50


def safe(key):
    return key.replace("/", "__").replace("::", "--")


def clean(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)


def controlled_image(path):
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    size = buf.tell()
    buf.seek(0)
    return Image.open(buf).convert("RGB"), size


@torch.no_grad()
def extract_controlled(net, paths, device, tag):
    residuals, conditions = [], []
    for i, path in enumerate(paths):
        img, size = controlled_image(path)
        a = np.asarray(img, dtype=np.float32) / 255.0
        coords = ((64, 64), (0, 0), (128, 128))
        psi, comp = [], []
        for y, x in coords:
            patch = a[y:y + 128, x:x + 128]
            psi.append(_psi_of(net, patch, device)[960:984])
            comp.append(complexity_patch(patch))
        residuals.append(np.mean(psi, axis=0))
        conditions.append(np.concatenate([[np.log(size + 1)], np.mean(comp, axis=0)]))
        if (i + 1) % 500 == 0:
            print(f"    [{tag}] {i + 1}/{len(paths)}", flush=True)
    return (clean(np.stack(residuals)),
            clean(np.stack(conditions)))


def controlled_jobs(manifest, ext_paths):
    jobs = [("pool", manifest["pool_real"]), ("ext_real", ext_paths["real"])]
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            jobs.append((key, manifest["test"][key][:N_TEST]))
    for g in EXT_GENS:
        jobs.append((g, ext_paths[g]))
    return jobs


def ensure_controlled(manifest, ext_paths):
    jobs = controlled_jobs(manifest, ext_paths)
    todo = [(k, p) for k, p in jobs
            if not (os.path.exists(os.path.join(OUT, f"ctrl_r_{safe(k)}.npy"))
                    and os.path.exists(os.path.join(OUT, f"ctrl_c_{safe(k)}.npy")))]
    print(f"受控 256/JPEG95 缓存: {len(todo)}/{len(jobs)} 集待提取", flush=True)
    if not todo:
        return
    gpu = wait_for_gpu(need_mb=4000)
    device = torch.device(f"cuda:{gpu}")
    net = build_noiseprint(device)
    for key, paths in todo:
        r, c = extract_controlled(net, paths, device, key)
        np.save(os.path.join(OUT, f"ctrl_r_{safe(key)}.npy"), r)
        np.save(os.path.join(OUT, f"ctrl_c_{safe(key)}.npy"), c)
    del net
    torch.cuda.empty_cache()


def load_controlled(manifest, ext_paths):
    data = {}
    for key, _ in controlled_jobs(manifest, ext_paths):
        data[key] = {
            "r": clean(np.load(os.path.join(OUT, f"ctrl_r_{safe(key)}.npy"))),
            "c": clean(np.load(os.path.join(OUT, f"ctrl_c_{safe(key)}.npy"))),
        }
    return data


def condition_score(r_ref, c_ref, r_cal, c_cal, r_query, c_query):
    scores, cal_scores, ess = {}, {}, {}
    scores["M0"] = maha_uncond(r_ref, r_query)
    cal_scores["M0"] = maha_uncond(r_ref, r_cal)
    for name, columns in (("CARF-size", [0]), ("CARF-pixel", list(range(1, 8)))):
        ar, ac, aq = c_ref[:, columns], c_cal[:, columns], c_query[:, columns]
        mu, sd = ar.mean(0), ar.std(0) + 1e-8
        rn = (ar - mu) / sd
        sigma = fixed_sigma(rn)
        wc = wa_kernel(rn, (ac - mu) / sd, sigma)
        wq = wa_kernel(rn, (aq - mu) / sd, sigma)
        base_c = maha_uncond(r_ref, r_cal)
        base_q = maha_uncond(r_ref, r_query)
        cal_scores[name], _ = maha_cond(r_ref, r_cal, wc, base_c)
        scores[name], ess[name] = maha_cond(r_ref, r_query, wq, base_q)
    return scores, cal_scores, ess


def simple_metric(y, score, n_real, tau):
    fpr, tpr, _ = roc_curve(y, score)
    tpr01 = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0
    return {"AUROC": float(roc_auc_score(y, score)),
            "AP": float(average_precision_score(y, score)),
            "TPR@1%FPR": tpr01,
            "FPR": float((score[:n_real] > tau).mean())}


def match_pairs(real_c, fake_c, ref_c):
    mu, sd = ref_c.mean(0), ref_c.std(0) + 1e-8
    real = (real_c - mu) / sd
    fake = (fake_c - mu) / sd
    distance = np.sqrt(np.maximum(0, ((real[:, None] - fake[None]) ** 2).sum(-1)))
    ri, fi = linear_sum_assignment(distance)
    keep = distance[ri, fi] <= CALIPER
    return ri[keep], fi[keep], distance[ri, fi][keep]


def matching_audit(real_c, fake_c, ref_c):
    ri, fi, distances = match_pairs(real_c, fake_c, ref_c)
    result = {"n_matched": len(ri), "distance_median": float(np.median(distances)),
              "distance_max": float(np.max(distances)) if len(distances) else None,
              "real_idx": ri.tolist(), "fake_idx": fi.tolist()}
    for name, cols in (("size", [0]), ("pixel", list(range(1, 8))),
                       ("all", list(range(8)))):
        result[f"AUC_{name}_before"] = leakage_auc(real_c[:, cols], fake_c[:, cols])
        result[f"AUC_{name}_after"] = leakage_auc(
            real_c[ri][:, cols], fake_c[fi][:, cols]) if len(ri) >= MIN_MATCH else None
    return result


def controlled_experiment(ctrl, manifest, ext_paths):
    audits = {"arf": {}, "ext": {}}
    for scope in ("arf", "ext"):
        ref_idx0, _, _ = split_indices(manifest, scope, 0)
        real_key0 = "pool" if scope == "arf" else "ext_real"
        ref_c0 = ctrl[real_key0]["c"][ref_idx0]
        gens = GENS if scope == "arf" else EXT_GENS
        for g in gens:
            real_key = f"{g}/0_real" if scope == "arf" else "ext_real"
            fake_key = f"{g}/1_fake" if scope == "arf" else g
            if scope == "ext":
                _, _, test_idx = split_indices(manifest, scope, 0)
                real_c = ctrl[real_key]["c"][test_idx]
            else:
                real_c = ctrl[real_key]["c"]
            audits[scope][g] = matching_audit(real_c, ctrl[fake_key]["c"], ref_c0)

    results = {"arf": {}, "ext": {}}
    for scope in ("arf", "ext"):
        gens = GENS if scope == "arf" else EXT_GENS
        real_key = "pool" if scope == "arf" else "ext_real"
        for seed in SEEDS:
            ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
            r_ref, c_ref = ctrl[real_key]["r"][ref_idx], ctrl[real_key]["c"][ref_idx]
            r_cal, c_cal = ctrl[real_key]["r"][cal_idx], ctrl[real_key]["c"][cal_idx]
            per_gen, saved = {}, {}
            for g in gens:
                real = f"{g}/0_real" if scope == "arf" else "ext_real"
                fake = f"{g}/1_fake" if scope == "arf" else g
                if scope == "ext":
                    rr, cr = ctrl[real]["r"][test_idx], ctrl[real]["c"][test_idx]
                else:
                    rr, cr = ctrl[real]["r"], ctrl[real]["c"]
                rf, cf = ctrl[fake]["r"], ctrl[fake]["c"]
                r_query = np.concatenate([rr, rf])
                c_query = np.concatenate([cr, cf])
                scores, cal_scores, ess = condition_score(
                    r_ref, c_ref, r_cal, c_cal, r_query, c_query)
                y = np.array([0] * N_TEST + [1] * N_TEST)
                entry = {}
                for name, score in scores.items():
                    tau = conformal_tau(cal_scores[name])
                    entry[name] = {"all": simple_metric(y, score, N_TEST, tau)}
                    if name in ess:
                        entry[name]["all"]["ESS_median"] = float(np.median(ess[name]))
                        entry[name]["all"]["ESS_coverage"] = float((ess[name] >= 50).mean())
                    audit = audits[scope][g]
                    ri, fi = np.asarray(audit["real_idx"]), np.asarray(audit["fake_idx"])
                    if len(ri) >= MIN_MATCH:
                        idx = np.concatenate([ri, N_TEST + fi])
                        entry[name]["matched"] = simple_metric(
                            y[idx], score[idx], len(ri), tau)
                    else:
                        entry[name]["matched"] = "unsupported"
                    saved[f"{g}::{name}"] = score
                per_gen[g] = entry
            np.savez(os.path.join(OUT, f"controlled_{scope}_seed{seed}.npz"), **saved)
            results[scope][str(seed)] = per_gen
            print(f"  controlled {scope}/seed{seed}: " + "; ".join(
                f"{m}={np.mean([per_gen[g][m]['all']['AUROC'] for g in gens]):.4f}"
                for m in ("M0", "CARF-size", "CARF-pixel")), flush=True)
    return {"runs": results, "matching_audit": audits}


def summarize_controlled(controlled):
    summary = {"arf": {}, "ext": {}}
    for scope in ("arf", "ext"):
        gens = GENS if scope == "arf" else EXT_GENS
        for seed in SEEDS:
            z = np.load(os.path.join(OUT, f"controlled_{scope}_seed{seed}.npz"))
            seed_summary = {}
            for subset in ("all", "matched"):
                methods = {}
                for method in ("M0", "CARF-size", "CARF-pixel"):
                    aucs = []
                    for g in gens:
                        entry = controlled["runs"][scope][str(seed)][g][method][subset]
                        if entry != "unsupported":
                            aucs.append(entry["AUROC"])
                    methods[method] = {"macro_AUROC": float(np.mean(aucs)),
                                       "n_generators": len(aucs)}
                comparisons = {}
                for method in ("CARF-size", "CARF-pixel"):
                    paired = {}
                    for g in gens:
                        entry = controlled["runs"][scope][str(seed)][g][method][subset]
                        if entry == "unsupported":
                            continue
                        a, b = z[f"{g}::{method}"], z[f"{g}::M0"]
                        if subset == "matched":
                            audit = controlled["matching_audit"][scope][g]
                            ri = np.asarray(audit["real_idx"])
                            fi = N_TEST + np.asarray(audit["fake_idx"])
                            idx = np.concatenate([ri, fi])
                            a, b = a[idx], b[idx]
                        y = np.array([0] * (len(a) // 2) + [1] * (len(a) // 2))
                        paired[g] = (y, a, b)
                    delta, ci = paired_boot_macro(paired, n_boot=1000, seed=seed)
                    comparisons[f"{method}-M0"] = {
                        "bootstrap_delta": delta, "CI95": ci,
                        "n_generators": len(paired),
                    }
                seed_summary[subset] = {"methods": methods,
                                        "comparisons": comparisons}
            summary[scope][str(seed)] = seed_summary
    return summary


def separate_matching_analyses(ctrl, manifest, ext_paths):
    """文件大小与像素复杂度分别匹配；caliper 与全条件匹配保持一致。"""
    output = {}
    for variant, columns in (("size", [0]), ("pixel", list(range(1, 8)))):
        audits = {"arf": {}, "ext": {}}
        real_conditions = {"arf": {}, "ext": {}}
        for scope in ("arf", "ext"):
            ref_idx, _, test_idx = split_indices(manifest, scope, 0)
            real_key = "pool" if scope == "arf" else "ext_real"
            ref_c = ctrl[real_key]["c"][ref_idx][:, columns]
            gens = GENS if scope == "arf" else EXT_GENS
            for g in gens:
                real = f"{g}/0_real" if scope == "arf" else "ext_real"
                fake = f"{g}/1_fake" if scope == "arf" else g
                rc = ctrl[real]["c"]
                if scope == "ext":
                    rc = rc[test_idx]
                fc = ctrl[fake]["c"]
                real_conditions[scope][g] = (rc, fc)
                ri, fi, distances = match_pairs(
                    rc[:, columns], fc[:, columns], ref_c)
                audit = {"n_matched": len(ri),
                         "distance_median": float(np.median(distances)),
                         "real_idx": ri.tolist(), "fake_idx": fi.tolist()}
                for name, cols in (("size", [0]), ("pixel", list(range(1, 8))),
                                   ("all", list(range(8)))):
                    audit[f"AUC_{name}_before"] = leakage_auc(
                        rc[:, cols], fc[:, cols])
                    audit[f"AUC_{name}_after"] = leakage_auc(
                        rc[ri][:, cols], fc[fi][:, cols]) \
                        if len(ri) >= MIN_MATCH else None
                audits[scope][g] = audit
        runs = {"arf": {}, "ext": {}}
        summary = {"arf": {}, "ext": {}}
        for scope in ("arf", "ext"):
            gens = GENS if scope == "arf" else EXT_GENS
            for seed in SEEDS:
                z = np.load(os.path.join(OUT, f"controlled_{scope}_seed{seed}.npz"))
                per_gen, paired_by_method = {}, {"CARF-size": {}, "CARF-pixel": {}}
                for g in gens:
                    audit = audits[scope][g]
                    ri = np.asarray(audit["real_idx"])
                    fi = N_TEST + np.asarray(audit["fake_idx"])
                    idx = np.concatenate([ri, fi])
                    y = np.array([0] * len(ri) + [1] * len(ri))
                    per_gen[g] = {}
                    for method in ("M0", "CARF-size", "CARF-pixel"):
                        score = z[f"{g}::{method}"][idx]
                        per_gen[g][method] = {
                            "AUROC": float(roc_auc_score(y, score)),
                            "AP": float(average_precision_score(y, score)),
                        }
                        if method != "M0":
                            paired_by_method[method][g] = (
                                y, score, z[f"{g}::M0"][idx])
                runs[scope][str(seed)] = per_gen
                comparisons = {}
                for method, paired in paired_by_method.items():
                    delta, ci = paired_boot_macro(paired, n_boot=1000, seed=seed)
                    comparisons[f"{method}-M0"] = {
                        "bootstrap_delta": delta, "CI95": ci,
                        "n_generators": len(paired),
                    }
                summary[scope][str(seed)] = {
                    "macro_AUROC": {
                        method: float(np.mean([
                            per_gen[g][method]["AUROC"] for g in gens]))
                        for method in ("M0", "CARF-size", "CARF-pixel")},
                    "comparisons": comparisons,
                }
        output[variant] = {"matching_audit": audits, "runs": runs,
                           "summary": summary}
    return output


def weighted_mean(r_ref, c_ref, r_query, c_query):
    mu, sd = c_ref.mean(0), c_ref.std(0) + 1e-8
    rn, qn = (c_ref - mu) / sd, (c_query - mu) / sd
    sigma = fixed_sigma(rn)
    w = wa_kernel(rn, qn, sigma)
    w = w / (w.sum(1, keepdims=True) + 1e-12)
    ess = w.sum(1) ** 2 / (np.square(w).sum(1) + 1e-12)
    return w @ r_ref, ess


def reference_statistics(data, paths, manifest, scope):
    real_key = "pool" if scope == "arf" else "real"
    real_paths = paths["real"]
    output = {}
    for rep in REPS:
        real = data[scope][rep][real_key]
        if scope == "arf":
            query_paths = paths["Infinity::real"][:N_TEST]
            query_r = {v: data[scope][rep]["Infinity/0_real"][v][:N_TEST]
                       for v in VIEWS[1:]}
        else:
            _, _, test_idx = split_indices(manifest, scope, 0)
            query_paths = [real_paths[i] for i in test_idx]
            query_r = {v: real[v][test_idx] for v in VIEWS[1:]}
        query_c = acquisition_features(query_paths, "size_only")
        output[rep] = {}
        for view in VIEWS[1:]:
            oracle_mean, _ = weighted_mean(
                real[view], acquisition_features(real_paths, "size_only"),
                query_r[view], query_c)
            output[rep][view] = {}
            for pool in ("same_quality", "mixed_codec"):
                predictions, ess_all = [], []
                for seed in SEEDS:
                    ref_idx, _, _ = split_indices(manifest, scope, seed)
                    views = VIEWS if pool == "mixed_codec" else (view,)
                    r_ref = np.concatenate([real[v][ref_idx] for v in views])
                    ref_paths = [real_paths[i] for v in views for i in ref_idx]
                    c_ref = acquisition_features(ref_paths, "size_only")
                    pred, ess = weighted_mean(r_ref, c_ref, query_r[view], query_c)
                    predictions.append(pred)
                    ess_all.append(ess)
                pred = np.stack(predictions)
                average = pred.mean(0)
                bias2 = float(np.mean((average - oracle_mean) ** 2))
                variance = float(np.mean(np.var(pred, axis=0)))
                mse = float(np.mean((pred - oracle_mean[None]) ** 2))
                ess = np.concatenate(ess_all)
                output[rep][view][pool] = {
                    "proxy_bias2": bias2, "seed_variance": variance, "proxy_MSE": mse,
                    "decomposition_error": float(abs(mse - bias2 - variance)),
                    "ESS_median": float(np.median(ess)),
                    "ESS_coverage>=50": float((ess >= 50).mean()),
                }
    return output


def ensure_dct(manifest, ext_paths, workers):
    jobs = []
    for view in ("qf95",) + U_VIEWS:
        jobs.append(("pool", manifest["pool_real"], view))
        jobs.append(("ext_real", ext_paths["real"], view))
    for g in DRIFT_GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            jobs.append((key, manifest["test"][key][:N_TEST], "qf95"))
    for g in EXT_GENS:
        jobs.append((g, ext_paths[g], "qf95"))
    todo = [(k, p, v) for k, p, v in jobs
            if not os.path.exists(os.path.join(OUT, f"dct24_{safe(k)}_{v}.npy"))]
    print(f"DCT 表示缓存: {len(todo)}/{len(jobs)} 集待提取", flush=True)
    for key, paths, view in todo:
        print(f"  DCT {key}/{view}", flush=True)
        ensure_dct_cache(OUT, key, paths, view, workers=workers)


def add_dct_data(data, manifest, ext_paths):
    for scope, real_key in (("arf", "pool"), ("ext", "real")):
        data[scope]["dct"] = {real_key: {}}
        for view in ("qf95",) + U_VIEWS:
            cache_key = "pool" if scope == "arf" else "ext_real"
            data[scope]["dct"][real_key][view] = clean(np.load(os.path.join(
                OUT, f"dct24_{cache_key}_{view}.npy")))
    for g in DRIFT_GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            data["arf"]["dct"][key] = {"qf95": clean(np.load(os.path.join(
                OUT, f"dct24_{safe(key)}_qf95.npy")))}
    for g in EXT_GENS:
        data["ext"]["dct"][g] = {"qf95": clean(np.load(os.path.join(
            OUT, f"dct24_{safe(g)}_qf95.npy")))}
    return data


def fit_u(real, ref_idx):
    r0 = real["qf95"][ref_idx]
    deltas = {v: real[v][ref_idx] - r0 for v in U_VIEWS}
    d = np.concatenate(list(deltas.values()))
    _, _, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
    return vt[:6].T, deltas


def bootstrap_stability(deltas, seed, n_boot=200):
    rng = np.random.RandomState(seed)
    n = len(next(iter(deltas.values())))
    values = []
    for _ in range(n_boot):
        idx = rng.permutation(n)
        h1, h2 = idx[:n // 2], idx[n // 2:]
        cosines = []
        for d in deltas.values():
            m1, m2 = d[h1].mean(0), d[h2].mean(0)
            cosines.append(abs(m1 @ m2) /
                           (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12))
        values.append(min(cosines))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_eta(real, fake, u, seed, n_boot=200):
    rng = np.random.RandomState(seed)
    p = np.eye(real.shape[1]) - u @ u.T
    values = []
    for _ in range(n_boot):
        rr = real[rng.randint(0, len(real), len(real))]
        ff = fake[rng.randint(0, len(fake), len(fake))]
        delta = ff.mean(0) - rr.mean(0)
        values.append(np.sum((p @ delta) ** 2) / (np.sum(delta ** 2) + 1e-12))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def coupling_ci(data, manifest, scope, rep, seed):
    ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
    real_key = "pool" if scope == "arf" else "real"
    real = data[scope][rep][real_key]
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
        raw, corrected = real[v][cal_idx], real[v][cal_idx] - means[v]
        canonical = real["qf95"][cal_idx]
        raw_mmd = mmd_rbf(raw, canonical)
        corrected_mmd = mmd_rbf(corrected, canonical)
        reduction = float(1 - corrected_mmd / (raw_mmd + 1e-12))
        rng = np.random.RandomState(100 + seed)
        boots = []
        for _ in range(100):
            idx = rng.randint(0, len(cal_idx), len(cal_idx))
            rm = mmd_rbf(raw[idx], canonical[idx])
            cm = mmd_rbf(corrected[idx], canonical[idx])
            boots.append(1 - cm / (rm + 1e-12))
        drift[v] = {"reduction": reduction,
                    "CI95": [float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))]}
    retention = {}
    gens = DRIFT_GENS if scope == "arf" else EXT_GENS
    for g in gens:
        if scope == "arf":
            rr = data[scope][rep][f"{g}/0_real"]["qf95"][:N_TEST]
            ff = data[scope][rep][f"{g}/1_fake"]["qf95"][:N_TEST]
        else:
            rr = real["qf95"][test_idx]
            ff = data[scope][rep][g]["qf95"]
        p = np.eye(rr.shape[1]) - u @ u.T
        delta = ff.mean(0) - rr.mean(0)
        eta = float(np.sum((p @ delta) ** 2) / (np.sum(delta ** 2) + 1e-12))
        retention[g] = {"eta": eta, "CI95": bootstrap_eta(rr, ff, u, seed)}
    return {"min_stability": min(stability.values()),
            "stability_CI95": bootstrap_stability(deltas, seed),
            "drift": drift, "retention": retention,
            "mean_eta": float(np.mean([v["eta"] for v in retention.values()]))}


def summarize_coupling(coupling):
    summary = {}
    for scope, reps in coupling.items():
        summary[scope] = {}
        for rep, runs in reps.items():
            stability = [runs[str(s)]["min_stability"] for s in SEEDS]
            reductions = {v: [runs[str(s)]["drift"][v]["reduction"] for s in SEEDS]
                          for v in VIEWS[1:]}
            identifiable = bool(min(stability) >= 0.9 and all(
                min(values) >= 0.3 for values in reductions.values()))
            eta_upper = [v["CI95"][1] for s in SEEDS
                         for v in runs[str(s)]["retention"].values()]
            summary[scope][rep] = {
                "stability_per_seed": stability, "MMD_reduction": reductions,
                "mean_eta_per_seed": [runs[str(s)]["mean_eta"] for s in SEEDS],
                "identifiable": identifiable,
                "coupling": bool(identifiable and max(eta_upper) < 0.8),
                "interpretation": "coupling" if identifiable and max(eta_upper) < 0.8
                                  else "unidentified" if not identifiable
                                  else "no_coupling_evidence",
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(SACR_OUT, "manifest.json")))
    ext_paths = external_paths()
    ensure_controlled(manifest, ext_paths)
    ctrl = load_controlled(manifest, ext_paths)
    controlled = controlled_experiment(ctrl, manifest, ext_paths)
    controlled["summary"] = summarize_controlled(controlled)
    controlled["separate_matching"] = separate_matching_analyses(
        ctrl, manifest, ext_paths)

    data = load_all(manifest, ext_paths)
    paths = {"real": manifest["pool_real"],
             "Infinity::real": manifest["test"]["Infinity/0_real"]}
    reference_stats = {"arf": reference_statistics(data, paths, manifest, "arf")}
    ext_path_map = {"real": ext_paths["real"]}
    reference_stats["ext"] = reference_statistics(data, ext_path_map, manifest, "ext")

    ensure_dct(manifest, ext_paths, args.workers)
    data = add_dct_data(data, manifest, ext_paths)
    coupling = {"arf": {"np": {}, "hp": {}, "dct": {}},
                "ext": {"np": {}, "hp": {}, "dct": {}}}
    for scope in ("arf", "ext"):
        for rep in ("np", "hp", "dct"):
            for seed in SEEDS:
                print(f"  coupling {scope}/{rep}/seed{seed}", flush=True)
                coupling[scope][rep][str(seed)] = coupling_ci(
                    data, manifest, scope, rep, seed)
    summary = summarize_coupling(coupling)
    results = {
        "config": {"seeds": SEEDS, "n_test_per_class": N_TEST,
                   "matching_caliper": CALIPER, "min_match": MIN_MATCH,
                   "controlled_view": "resize 256x256 BICUBIC + JPEG95",
                   "protocol": "v13 matched-analysis + reference-statistics + representation-ci 示例"},
        "controlled": controlled, "reference_statistics": reference_stats,
        "coupling": coupling, "coupling_summary": summary,
    }
    with open(os.path.join(OUT, "carf_mechanism_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"已保存 {OUT}/carf_mechanism_p0.json", flush=True)


if __name__ == "__main__":
    main()
