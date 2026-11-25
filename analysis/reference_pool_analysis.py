"""v15 参考池统计分析：ESS 模拟、校准 CI、MSE–AUROC 关系。

不开发新检测器、不调整阈值。复用 S3R checkpoint 与逐图分数。
"""

import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_protocol_p1 import EXT_GENS, external_paths, split_indices  # noqa: E402
from analysis.race_p0 import GENS  # noqa: E402
from analysis.s3r_p0 import (EVAL_VIEWS, OUT as S3R_OUT, SEEDS, VIEWS,  # noqa: E402
                             load_checkpoint, load_data, moments_many,
                             normalized_weights, scores_from_moments,
                             source_gaussian)
from analysis.sacr_p0 import OUT as SACR_OUT  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "reference_analysis")


def ess_values(w_views):
    flat = w_views.reshape(-1)
    view_ess = float(flat.sum() ** 2 / (np.square(flat).sum() + 1e-12))
    cluster = w_views.sum(axis=1)
    source_ess = float(cluster.sum() ** 2 / (np.square(cluster).sum() + 1e-12))
    return view_ess, source_ess


def simulate_ess(n_mc=500):
    rows = []
    for n_source in (10, 50, 100, 500):
        for n_views in (1, 2, 4, 8):
            for rho in (0.0, 0.5, 0.9, 1.0):
                for weighting in ("uniform", "skewed"):
                    rng = np.random.RandomState(
                        n_source * 1000 + n_views * 100 + int(rho * 10)
                        + (weighting == "skewed"))
                    mse_naive, mse_source = [], []
                    view_ess, source_ess = [], []
                    for _ in range(n_mc):
                        shared = rng.normal(size=(n_source, 1))
                        independent = rng.normal(size=(n_source, n_views))
                        views = (np.sqrt(rho) * shared
                                 + np.sqrt(1 - rho) * independent)
                        if weighting == "uniform":
                            w = np.ones((n_source, n_views))
                        else:
                            w = rng.dirichlet(np.full(n_views, 0.2),
                                              size=n_source)
                        source_summary = views.mean(axis=1)
                        naive = np.average(views.reshape(-1),
                                           weights=w.reshape(-1))
                        cluster = w.mean(axis=1)
                        source = np.average(source_summary, weights=cluster)
                        mse_naive.append(naive ** 2)
                        mse_source.append(source ** 2)
                        ve, se = ess_values(w)
                        view_ess.append(ve)
                        source_ess.append(se)
                    rows.append({
                        "n_source": n_source, "n_views": n_views, "rho": rho,
                        "weighting": weighting,
                        "naive_view_ESS": float(np.mean(view_ess)),
                        "source_ESS": float(np.mean(source_ess)),
                        "naive_MSE": float(np.mean(mse_naive)),
                        "source_MSE": float(np.mean(mse_source)),
                        "naive_MSE_x_viewESS": float(np.mean(mse_naive)
                                                        * np.mean(view_ess)),
                        "source_MSE_x_sourceESS": float(np.mean(mse_source)
                                                          * np.mean(source_ess)),
                    })
    return rows


def real_weight_invariance(data, manifest, scope):
    ref_idx, cal_idx, _ = split_indices(manifest, scope, 0)
    n_val = 500 if scope == "arf" else 100
    fit_idx = ref_idx[n_val:]
    real_key = "pool" if scope == "arf" else "ext_real"
    real = data[real_key]
    r = np.stack([real[v][0][fit_idx] for v in VIEWS], axis=1)
    c = np.stack([real[v][1][fit_idx] for v in VIEWS], axis=1)
    query_c = real["qf85"][1][cal_idx[:100]]
    w = normalized_weights(c.reshape(-1, 1), query_c).reshape(
        len(query_c), len(fit_idx), len(VIEWS))
    max_mu = max_cov = max_source_ess = 0.0
    view_ratios, source_ratios = [], []
    for i in range(len(query_c)):
        mu0, cov0, ess0 = source_gaussian(r, w[i])
        r_dup = np.tile(r, (1, 2, 1))
        w_dup = np.tile(w[i], (1, 2))
        mu1, cov1, ess1 = source_gaussian(r_dup, w_dup)
        max_mu = max(max_mu, float(np.max(np.abs(mu0 - mu1))))
        max_cov = max(max_cov, float(np.max(np.abs(cov0 - cov1))))
        max_source_ess = max(max_source_ess, abs(ess0 - ess1))
        ve0, se0 = ess_values(w[i])
        ve1, se1 = ess_values(w_dup)
        view_ratios.append(ve1 / ve0)
        source_ratios.append(se1 / se0)
    return {"queries": len(query_c), "max_mu_diff": max_mu,
            "max_cov_diff": max_cov, "max_source_ESS_diff": max_source_ess,
            "view_ESS_ratio_mean": float(np.mean(view_ratios)),
            "source_ESS_ratio_mean": float(np.mean(source_ratios))}


def conformal_details(cal_score, cal_ess, alpha=0.05):
    supported = cal_score[cal_ess >= 50]
    ordered = np.sort(supported)
    n = len(ordered)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    index = min(rank, n) - 1
    tau = float(ordered[index])
    next_gap = float(ordered[index + 1] - ordered[index]) \
        if index + 1 < n else None
    return {"tau": tau, "n": n, "rank": rank, "next_gap": next_gap,
            "cal_score": cal_score, "cal_ess": cal_ess}


def bootstrap_fpr(groups, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    per_group = {}
    macro = []
    for name, flags in groups.items():
        flags = np.asarray(flags, dtype=bool)
        samples = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, len(flags), len(flags))
            samples[b] = flags[idx].mean()
        per_group[name] = {"FPR": float(flags.mean()),
                           "CI95": [float(np.percentile(samples, 2.5)),
                                    float(np.percentile(samples, 97.5))]}
        macro.append(samples)
    macro_samples = np.mean(macro, axis=0)
    macro_result = {"FPR": float(np.mean([v["FPR"] for v in per_group.values()])),
                    "CI95": [float(np.percentile(macro_samples, 2.5)),
                             float(np.percentile(macro_samples, 97.5))]}
    return per_group, macro_result


def calibration_analysis(data, manifest, s3r_results):
    output = {"arf": {}, "ext": {}}
    for scope in ("arf", "ext"):
        real_key = "pool" if scope == "arf" else "ext_real"
        real = data[real_key]
        gens = GENS if scope == "arf" else EXT_GENS
        for seed in SEEDS:
            ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
            n_val = 500 if scope == "arf" else 100
            fit_idx = ref_idx[n_val:]
            cal_idx = cal_idx[:500]
            r_ref = np.stack([real[v][0][fit_idx] for v in VIEWS], axis=1)
            c_ref = np.stack([real[v][1][fit_idx] for v in VIEWS], axis=1)
            output[scope][str(seed)] = {}
            for view in EVAL_VIEWS:
                view_idx = VIEWS.index(view)
                r_cal, c_cal = real[view][0][cal_idx], real[view][1][cal_idx]
                mus, covs, ess = moments_many(
                    r_ref, c_ref, r_cal, c_cal, view_idx, "same")
                cal_score = scores_from_moments(r_cal, mus, covs)
                details = conformal_details(cal_score, ess)
                tau = details["tau"]
                z = np.load(os.path.join(
                    S3R_OUT, f"scores_{scope}_seed{seed}_{view}.npz"))
                groups, curves = {}, {}
                for g in gens:
                    real_score = z[f"{g}::same"][:300]
                    real_ess = z[f"{g}::same::ess"][:300]
                    groups[g] = real_score[real_ess >= 50] > tau
                    curves[g] = {}
                    for threshold in (20, 50, 100, 200):
                        supported = real_ess >= threshold
                        curves[g][str(threshold)] = {
                            "coverage": float(supported.mean()),
                            "flags": (real_score[supported] > tau).tolist(),
                        }
                per_gen, macro = bootstrap_fpr(groups, seed=seed)
                curve_output = {}
                for threshold in (20, 50, 100, 200):
                    flag_groups = {g: curves[g][str(threshold)]["flags"]
                                   for g in gens}
                    _, curve_macro = bootstrap_fpr(flag_groups, seed=seed + threshold)
                    curve_output[str(threshold)] = {
                        "FPR": curve_macro["FPR"], "CI95": curve_macro["CI95"],
                        "coverage": float(np.mean([
                            curves[g][str(threshold)]["coverage"] for g in gens])),
                    }
                output[scope][str(seed)][view] = {
                    "calibration": {k: v for k, v in details.items()
                                    if not k.startswith("cal_")},
                    "per_generator": per_gen, "macro": macro,
                    "risk_coverage": curve_output,
                }
                np.savez(os.path.join(
                    OUT, f"calibration_{scope}_seed{seed}_{view}.npz"),
                    cal_score=cal_score, cal_ess=ess, tau=tau)
    return output


def performance_error_relation(s3r_results):
    rows = []
    for scope in ("arf", "ext"):
        gens = GENS if scope == "arf" else EXT_GENS
        for seed in SEEDS:
            for view in EVAL_VIEWS:
                checkpoint = os.path.join(
                    S3R_OUT, f"checkpoint_{scope}_seed{seed}_{view}.npz")
                _, query, oracle = load_checkpoint(checkpoint)
                run = s3r_results["runs"][scope][str(seed)][view]["run"]
                for method in ("same", "mixed", "source_mixed"):
                    mse = float(np.mean((query[method]["mu"] - oracle) ** 2))
                    auc = float(np.nanmean([
                        run["per_generator"][g][method]["AUROC"] for g in gens]))
                    rows.append({"scope": scope, "seed": seed, "view": view,
                                 "method": method, "MSE": mse,
                                 "source_ESS_median": float(np.median(
                                     query[method]["ess"])), "macro_AUROC": auc})

    def correlations(subset):
        mse = np.array([r["MSE"] for r in subset])
        auc = np.array([r["macro_AUROC"] for r in subset])
        ess = np.array([r["source_ESS_median"] for r in subset])
        return {
            "n": len(subset),
            "MSE_vs_AUROC_spearman": float(spearmanr(mse, auc).statistic),
            "MSE_vs_AUROC_pearson": float(pearsonr(mse, auc).statistic),
            "ESS_vs_AUROC_spearman": float(spearmanr(ess, auc).statistic),
            "ESS_vs_MSE_spearman": float(spearmanr(ess, mse).statistic),
        }

    output = {"rows": rows, "global": correlations(rows), "groups": {}}
    for scope in ("arf", "ext"):
        for view in EVAL_VIEWS:
            subset = [r for r in rows if r["scope"] == scope and r["view"] == view]
            output["groups"][f"{scope}/{view}"] = correlations(subset)
    return output


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(SACR_OUT, "manifest.json")))
    ext_paths = external_paths()
    data = load_data(manifest, ext_paths)
    s3r_results = json.load(open(os.path.join(S3R_OUT, "s3r_p0.json")))
    simulation = simulate_ess()
    invariance = {scope: real_weight_invariance(data, manifest, scope)
                  for scope in ("arf", "ext")}
    calibration = calibration_analysis(data, manifest, s3r_results)
    relation = performance_error_relation(s3r_results)
    results = {"protocol": "v15 source-ESS/calibration/MSE-AUROC analysis",
               "simulation": simulation, "real_weight_invariance": invariance,
               "calibration": calibration, "performance_error_relation": relation,
               "unknown_codec": "100% abstain"}
    with open(os.path.join(OUT, "v15_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"real_weight_invariance": invariance,
                      "correlations": relation["global"]}, indent=2), flush=True)
    print(f"已保存 {OUT}/v15_analysis.json", flush=True)


if __name__ == "__main__":
    main()
