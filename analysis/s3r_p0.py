"""S3R P0: Source-aware Same-quality Reference Modeling (v14).

唯一方法开发路线。ARF 3000 fit + 500 real validation + 500 calibration；
DARG 300 fit + 100 validation + 300 calibration；test 均 300/类。
hierarchical shrinkage gamma 只由真实 validation sources 选择。
比较 same-quality、naive mixed、source-weighted mixed、hierarchical S3R。
所有视图为统一 256x256 后的 JPEG95/85/75；条件为重编码 log 文件大小。

预注册 GO:
  1. S3R proxy-MSE 显著低于 mixed (source bootstrap CI<0)，且不超过 same
     的 1.10 倍;
  2. 在 QF85/QF75 上，S3R 相对每 seed 较强 same/mixed baseline 的
     macro ΔAUROC>0 且逐图 paired CI 下界>0；至少 5/7 ARF、4/5 DARG
     生成器不退化;
  3. 每个已知质量组 FPR∈[3%,7%]、coverage≥95%；未知 codec 直接 abstain;
  4. source-level ESS 不因复制同一 source 的变换视图而虚增（单元测试）。
任一核心检测标准失败即停止 S3R，不开展全量或新增模块。

用法: python analysis/s3r_p0.py
"""

import io
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
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.carf_protocol_p1 import EXT_GENS, external_paths, split_indices  # noqa: E402
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.race_p0 import GENS, _psi_of, build_noiseprint  # noqa: E402
from analysis.sacr_p0 import OUT as SACR_OUT  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "s3r")
VIEWS = ("qf95", "qf85", "qf75")
EVAL_VIEWS = ("qf85", "qf75")
SEEDS = (0, 1, 2)
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
METHODS = ("same", "mixed", "source_mixed", "s3r")
N_TEST = 300
ESS_MIN = 50
MSE_TOLERANCE = 1.10


def safe(key):
    return key.replace("/", "__").replace("::", "--")


def clean(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)


def controlled_image(path, view):
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)
    quality = int(view[2:])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    size = buf.tell()
    buf.seek(0)
    return Image.open(buf).convert("RGB"), size


@torch.no_grad()
def extract_view(net, paths, view, device, tag):
    residuals, conditions = [], []
    for i, path in enumerate(paths):
        img, size = controlled_image(path, view)
        a = np.asarray(img, dtype=np.float32) / 255.0
        psi = []
        for y, x in ((64, 64), (0, 0), (128, 128)):
            psi.append(_psi_of(net, a[y:y + 128, x:x + 128], device)[960:984])
        residuals.append(np.mean(psi, axis=0))
        conditions.append(np.log(size + 1))
        if (i + 1) % 500 == 0:
            print(f"    [{tag}/{view}] {i + 1}/{len(paths)}", flush=True)
    return clean(np.stack(residuals)), clean(np.stack(conditions))


def jobs(manifest, ext_paths):
    result = [("pool", manifest["pool_real"]), ("ext_real", ext_paths["real"])]
    for g in GENS:
        for cls in ("0_real", "1_fake"):
            key = f"{g}/{cls}"
            result.append((key, manifest["test"][key][:N_TEST]))
    for g in EXT_GENS:
        result.append((g, ext_paths[g]))
    return result


def ensure_caches(manifest, ext_paths):
    todo = []
    for key, paths in jobs(manifest, ext_paths):
        for view in EVAL_VIEWS:
            if not (os.path.exists(os.path.join(OUT, f"r_{safe(key)}_{view}.npy"))
                    and os.path.exists(os.path.join(OUT, f"c_{safe(key)}_{view}.npy"))):
                todo.append((key, paths, view))
    print(f"S3R controlled views: {len(todo)} 集待提取", flush=True)
    if not todo:
        return
    gpu = wait_for_gpu(need_mb=4000)
    device = torch.device(f"cuda:{gpu}")
    net = build_noiseprint(device)
    for key, paths, view in todo:
        r, c = extract_view(net, paths, view, device, key)
        np.save(os.path.join(OUT, f"r_{safe(key)}_{view}.npy"), r)
        np.save(os.path.join(OUT, f"c_{safe(key)}_{view}.npy"), c)
    del net
    torch.cuda.empty_cache()


def load_data(manifest, ext_paths):
    data = {}
    for key, _ in jobs(manifest, ext_paths):
        data[key] = {}
        ctrl_r = os.path.join(PROJ, "outputs", "carf_mechanism",
                              f"ctrl_r_{safe(key)}.npy")
        ctrl_c = os.path.join(PROJ, "outputs", "carf_mechanism",
                              f"ctrl_c_{safe(key)}.npy")
        data[key]["qf95"] = (clean(np.load(ctrl_r)),
                             clean(np.load(ctrl_c))[:, 0])
        for view in EVAL_VIEWS:
            r = clean(np.load(os.path.join(OUT, f"r_{safe(key)}_{view}.npy")))
            c = clean(np.load(os.path.join(OUT, f"c_{safe(key)}_{view}.npy")))
            data[key][view] = (r, c)
    return data


def normalized_weights(c_ref, c_query):
    c_ref = np.asarray(c_ref).reshape(-1, 1)
    c_query = np.asarray(c_query).reshape(-1, 1)
    mu, sd = c_ref.mean(0), c_ref.std(0) + 1e-8
    rn, qn = (c_ref - mu) / sd, (c_query - mu) / sd
    sigma = fixed_sigma(rn)
    return wa_kernel(rn, qn, sigma)


def weighted_gaussian(r, w):
    v = w / (w.sum() + 1e-12)
    mu = v @ r
    x = r - mu
    cov = (v[:, None] * x).T @ x
    ess = float(1 / (np.square(v).sum() + 1e-12))
    return mu, cov, ess


def source_gaussian(r_views, w_views):
    cluster_sum = w_views.sum(axis=1, keepdims=True)
    within = w_views / (cluster_sum + 1e-12)
    summaries = (within[..., None] * r_views).sum(axis=1)
    cluster_weight = w_views.mean(axis=1)
    mu, between, ess = weighted_gaussian(summaries, cluster_weight)
    v = cluster_weight / (cluster_weight.sum() + 1e-12)
    centered = r_views - summaries[:, None, :]
    within_cov = np.einsum("is,isj,isk->jk", v[:, None] * within,
                           centered, centered)
    return mu, between + within_cov, ess


def moments(r_views, w_views, pool, view_idx, gamma=0.0):
    if pool == "same":
        return weighted_gaussian(r_views[:, view_idx], w_views[:, view_idx])
    if pool == "mixed":
        mu, cov, _ = weighted_gaussian(
            r_views.reshape(-1, r_views.shape[-1]), w_views.reshape(-1))
        cluster = w_views.sum(axis=1)
        source_ess = float(cluster.sum() ** 2
                           / (np.square(cluster).sum() + 1e-12))
        return mu, cov, source_ess
    if pool == "source_mixed":
        return source_gaussian(r_views, w_views)
    if pool == "s3r":
        mu0, cov0, _ = weighted_gaussian(r_views[:, view_idx], w_views[:, view_idx])
        mu1, cov1, _ = source_gaussian(r_views, w_views)
        v0 = w_views[:, view_idx] / (w_views[:, view_idx].sum() + 1e-12)
        cluster_weight = w_views.mean(axis=1)
        v1 = cluster_weight / (cluster_weight.sum() + 1e-12)
        combined = (1 - gamma) * v0 + gamma * v1
        ess = float(1 / (np.square(combined).sum() + 1e-12))
        return ((1 - gamma) * mu0 + gamma * mu1,
                (1 - gamma) * cov0 + gamma * cov1, ess)
    raise ValueError(pool)


def moments_many(r_ref, c_ref, r_query, c_query, view_idx, pool, gamma=0.0):
    n_views = r_ref.shape[1]
    same_weights = None
    if pool in ("same", "s3r"):
        same_weights = np.zeros((len(r_query),) + c_ref.shape, dtype=np.float32)
        same_weights[:, :, view_idx] = normalized_weights(
            c_ref[:, view_idx], c_query)
    if pool != "same":
        flat = c_ref.reshape(-1, 1)
        weights = normalized_weights(flat, c_query).reshape(
            len(r_query), len(r_ref), n_views)
    else:
        weights = same_weights
    mus, covs, ess = [], [], []
    for i in range(len(r_query)):
        if pool == "s3r":
            mu0, cov0, _ = weighted_gaussian(
                r_ref[:, view_idx], same_weights[i, :, view_idx])
            mu1, cov1, _ = source_gaussian(r_ref, weights[i])
            v0 = same_weights[i, :, view_idx] \
                / (same_weights[i, :, view_idx].sum() + 1e-12)
            cluster_weight = weights[i].mean(axis=1)
            v1 = cluster_weight / (cluster_weight.sum() + 1e-12)
            combined = (1 - gamma) * v0 + gamma * v1
            mu = (1 - gamma) * mu0 + gamma * mu1
            cov = (1 - gamma) * cov0 + gamma * cov1
            e = float(1 / (np.square(combined).sum() + 1e-12))
        else:
            mu, cov, e = moments(r_ref, weights[i], pool, view_idx, gamma)
        mus.append(mu)
        covs.append(cov)
        ess.append(e)
    return np.stack(mus), np.stack(covs), np.asarray(ess)


def precision(cov):
    return np.linalg.inv(cov + 0.1 * np.diag(np.diag(cov))
                         + 1e-6 * np.eye(cov.shape[0]))


def scores_from_moments(r, mus, covs):
    scores = np.empty(len(r), dtype=np.float32)
    for i in range(len(r)):
        diff = r[i] - mus[i]
        scores[i] = diff @ precision(covs[i]) @ diff
    return scores


def select_gamma(r_ref, c_ref, r_val, c_val, view_idx):
    mu0, cov0, _ = moments_many(r_ref, c_ref, r_val, c_val, view_idx, "same")
    mu1, cov1, _ = moments_many(r_ref, c_ref, r_val, c_val, view_idx,
                                "source_mixed")
    values = []
    for gamma in GAMMAS:
        mus = (1 - gamma) * mu0 + gamma * mu1
        covs = (1 - gamma) * cov0 + gamma * cov1
        nll = []
        for i in range(len(r_val)):
            cov = covs[i] + 0.1 * np.diag(np.diag(covs[i])) \
                + 1e-6 * np.eye(covs[i].shape[0])
            sign, logdet = np.linalg.slogdet(cov)
            diff = r_val[i] - mus[i]
            nll.append(logdet + diff @ np.linalg.inv(cov) @ diff)
        values.append(float(np.mean(nll)))
    best = int(np.argmin(values))
    return GAMMAS[best], dict(zip([str(g) for g in GAMMAS], values))


def metric(y, score, ess, n_real, tau):
    supported = ess >= ESS_MIN
    valid = supported & np.isfinite(score)
    result = {"coverage": float(valid.mean())}
    if valid.sum() < 50 or len(np.unique(y[valid])) < 2:
            result.update({"AUROC": np.nan, "AP": np.nan, "TPR@1%FPR": np.nan,
                           "FPR": np.nan})
            return result
    fpr, tpr, _ = roc_curve(y[valid], score[valid])
    result.update({
        "AUROC": float(roc_auc_score(y[valid], score[valid])),
        "AP": float(average_precision_score(y[valid], score[valid])),
        "TPR@1%FPR": float(tpr[fpr <= 0.01].max())
        if np.any(fpr <= 0.01) else 0.0,
        "FPR": float((score[:n_real][valid[:n_real]] > tau).mean()),
        "ESS_median": float(np.median(ess)),
    })
    return result


def risk_coverage(y, score, ess, n_real, tau):
    curve = []
    for threshold in (20, 50, 100, 200):
        supported = ess >= threshold
        curve.append({"ESS_min": threshold, "coverage": float(supported.mean()),
                      "FPR": float((score[:n_real][supported[:n_real]] > tau).mean())
                      if supported[:n_real].any() else np.nan})
    return curve


def oracle_mean(real, ref_idx, query, view):
    r_ref, c_ref = real[view][0][ref_idx], real[view][1][ref_idx, None]
    mu, _, _ = moments_many(r_ref[:, None], c_ref, query[0], query[1][:, None],
                            0, "same")
    return mu


def run_seed_view(data, manifest, ext_paths, scope, seed, view):
    ref_idx, cal_idx, test_idx = split_indices(manifest, scope, seed)
    real_key = "pool" if scope == "arf" else "ext_real"
    real = data[real_key]
    n_val = 500 if scope == "arf" else 100
    val_idx, fit_idx = ref_idx[:n_val], ref_idx[n_val:]
    cal_idx = cal_idx[:500]
    r_ref = np.stack([real[v][0][fit_idx] for v in VIEWS], axis=1)
    c_ref = np.stack([real[v][1][fit_idx] for v in VIEWS], axis=1)
    view_idx = VIEWS.index(view)
    r_val, c_val = real[view][0][val_idx], real[view][1][val_idx]
    gamma, gamma_nll = select_gamma(r_ref, c_ref, r_val, c_val, view_idx)
    r_cal, c_cal = real[view][0][cal_idx], real[view][1][cal_idx]
    cal_moments = {}
    for method in METHODS:
        cal_moments[method] = moments_many(
            r_ref, c_ref, r_cal, c_cal, view_idx, method, gamma)
    cal_scores = {m: scores_from_moments(r_cal, *cal_moments[m][:2])
                  for m in METHODS}
    taus = {m: conformal_tau(cal_scores[m][cal_moments[m][2] >= ESS_MIN])
            for m in METHODS}
    gens = GENS if scope == "arf" else EXT_GENS
    per_gen, saved = {}, {}
    for g in gens:
        real_test = f"{g}/0_real" if scope == "arf" else "ext_real"
        fake_test = f"{g}/1_fake" if scope == "arf" else g
        rr = data[real_test][view][0]
        cr = data[real_test][view][1]
        if scope == "ext":
            rr, cr = rr[test_idx], cr[test_idx]
        rf, cf = data[fake_test][view]
        r_query = np.concatenate([rr, rf])
        c_query = np.concatenate([cr, cf])
        y = np.array([0] * len(rr) + [1] * len(rf))
        entry = {}
        for method in METHODS:
            mus, covs, ess = moments_many(
                r_ref, c_ref, r_query, c_query, view_idx, method, gamma)
            score = scores_from_moments(r_query, mus, covs)
            entry[method] = metric(y, score, ess, len(rr), taus[method])
            entry[method]["risk_coverage"] = risk_coverage(
                y, score, ess, len(rr), taus[method])
            saved[f"{g}::{method}"] = score
            saved[f"{g}::{method}::ess"] = ess
        per_gen[g] = entry
    np.savez(os.path.join(OUT, f"scores_{scope}_seed{seed}_{view}.npz"), **saved)

    oracle_ref, oracle_cal, query_idx = split_indices(manifest, scope, 0)
    if scope == "arf":
        query = data["Infinity/0_real"][view]
        oracle_idx = np.arange(len(real[view][0]))
    else:
        query = (real[view][0][query_idx], real[view][1][query_idx])
        oracle_idx = np.concatenate([oracle_ref, oracle_cal])
    query_stats = {}
    for method in METHODS:
        mus, _, ess = moments_many(r_ref, c_ref, query[0], query[1],
                                   view_idx, method, gamma)
        query_stats[method] = {"mu": mus, "ess": ess}
    oracle = oracle_mean(real, oracle_idx, query, view)
    return {"gamma": gamma, "gamma_validation_nll": gamma_nll,
            "per_generator": per_gen}, query_stats, oracle


def macro_auc(run, method, gens):
    return float(np.nanmean([run["per_generator"][g][method]["AUROC"]
                             for g in gens]))


def bootstrap_detection(runs, scope):
    gens = GENS if scope == "arf" else EXT_GENS
    output = {}
    for view in EVAL_VIEWS:
        deltas, cis, not_worse = [], [], []
        for seed in SEEDS:
            run = runs[str(seed)][view]["run"]
            baseline = max(("same", "mixed"),
                           key=lambda m: macro_auc(run, m, gens))
            z = np.load(os.path.join(OUT, f"scores_{scope}_seed{seed}_{view}.npz"))
            paired = {}
            for g in gens:
                a = z[f"{g}::s3r"]
                b = z[f"{g}::{baseline}"]
                ea = z[f"{g}::s3r::ess"]
                eb = z[f"{g}::{baseline}::ess"]
                valid = (ea >= ESS_MIN) & (eb >= ESS_MIN)
                y = np.array([0] * N_TEST + [1] * N_TEST)[valid]
                paired[g] = (y, a[valid], b[valid])
            delta, ci = paired_boot_macro(paired, n_boot=1000, seed=seed)
            deltas.append(macro_auc(run, "s3r", gens)
                          - macro_auc(run, baseline, gens))
            cis.append(ci)
            not_worse.append(sum(
                run["per_generator"][g]["s3r"]["AUROC"] + 1e-9
                >= run["per_generator"][g][baseline]["AUROC"] for g in gens))
        output[view] = {"delta_per_seed": deltas, "CI95_per_seed": cis,
                        "not_worse_per_seed": not_worse,
                        "pass": bool(all(d > 0 for d in deltas)
                                     and all(ci[0] > 0 for ci in cis))}
    return output


def summarize_query_stats(query_stats, oracles, scope):
    output = {}
    for view in EVAL_VIEWS:
        output[view] = {}
        oracle = oracles[view]
        for method in METHODS:
            mus = np.stack([query_stats[str(s)][view][method]["mu"]
                            for s in SEEDS])
            average = mus.mean(0)
            bias2 = float(np.mean((average - oracle) ** 2))
            variance = float(np.mean(np.var(mus, axis=0)))
            mse = float(np.mean((mus - oracle[None]) ** 2))
            per_query_mse = np.mean((mus - oracle[None]) ** 2, axis=2).mean(0)
            output[view][method] = {"bias2": bias2, "variance": variance,
                                    "MSE": mse, "per_query_MSE": per_query_mse}
        rng = np.random.RandomState(0)
        delta = output[view]["s3r"]["per_query_MSE"] \
            - output[view]["mixed"]["per_query_MSE"]
        boots = [delta[rng.randint(0, len(delta), len(delta))].mean()
                 for _ in range(1000)]
        output[view]["S3R-Mixed_MSE"] = {
            "delta": float(delta.mean()),
            "CI95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
        }
        output[view]["S3R_vs_same_tolerance"] = bool(
            output[view]["s3r"]["MSE"]
            <= MSE_TOLERANCE * output[view]["same"]["MSE"])
        for method in METHODS:
            del output[view][method]["per_query_MSE"]
    return output


def summarize(runs, query_stats, oracles):
    summary = {}
    for scope in ("arf", "ext"):
        detection = bootstrap_detection(runs[scope], scope)
        query = summarize_query_stats(query_stats[scope], oracles[scope], scope)
        fprs, coverages = [], []
        gens = GENS if scope == "arf" else EXT_GENS
        for seed in SEEDS:
            for view in EVAL_VIEWS:
                run = runs[scope][str(seed)][view]["run"]
                fprs.append(float(np.nanmean([
                    run["per_generator"][g]["s3r"]["FPR"] for g in gens])))
                coverages.append(float(np.nanmean([
                    run["per_generator"][g]["s3r"]["coverage"] for g in gens])))
        mse_pass = all(query[v]["S3R-Mixed_MSE"]["CI95"][1] < 0
                       and query[v]["S3R_vs_same_tolerance"] for v in EVAL_VIEWS)
        detection_pass = all(detection[v]["pass"] for v in EVAL_VIEWS)
        not_worse_pass = all(
            (5 if scope == "arf" else 4) <= min(detection[v]["not_worse_per_seed"])
            for v in EVAL_VIEWS)
        calibration_pass = all(0.03 <= x <= 0.07 for x in fprs) \
            and all(x >= 0.95 for x in coverages)
        summary[scope] = {
            "detection": detection, "query_statistics": query,
            "FPR": fprs, "coverage": coverages,
            "criteria": {"MSE": mse_pass, "detection": detection_pass,
                         "not_worse": not_worse_pass,
                         "calibration": calibration_pass},
        }
        summary[scope]["GO"] = bool(mse_pass and detection_pass
                                    and not_worse_pass and calibration_pass)
    summary["GO"] = bool(summary["arf"]["GO"] and summary["ext"]["GO"])
    summary["unknown_codec"] = "abstain / unsupported (100% by protocol)"
    return summary


def save_checkpoint(path, run, query, oracle):
    payload = {"run_json": json.dumps(run), "oracle": oracle}
    for method in METHODS:
        payload[f"{method}_mu"] = query[method]["mu"]
        payload[f"{method}_ess"] = query[method]["ess"]
    np.savez(path, **payload)


def load_checkpoint(path):
    z = np.load(path)
    run = json.loads(z["run_json"].item())
    query = {m: {"mu": z[f"{m}_mu"], "ess": z[f"{m}_ess"]}
             for m in METHODS}
    return run, query, z["oracle"]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(SACR_OUT, "manifest.json")))
    ext_paths = external_paths()
    ensure_caches(manifest, ext_paths)
    data = load_data(manifest, ext_paths)
    runs = {"arf": {}, "ext": {}}
    query_stats = {"arf": {}, "ext": {}}
    oracles = {"arf": {}, "ext": {}}
    for scope in ("arf", "ext"):
        for seed in SEEDS:
            runs[scope][str(seed)] = {}
            query_stats[scope][str(seed)] = {}
            for view in EVAL_VIEWS:
                print(f"===== S3R {scope}/seed{seed}/{view} =====", flush=True)
                checkpoint = os.path.join(
                    OUT, f"checkpoint_{scope}_seed{seed}_{view}.npz")
                if os.path.exists(checkpoint):
                    run, query, oracle = load_checkpoint(checkpoint)
                else:
                    run, query, oracle = run_seed_view(
                        data, manifest, ext_paths, scope, seed, view)
                    save_checkpoint(checkpoint, run, query, oracle)
                runs[scope][str(seed)][view] = {"run": run}
                query_stats[scope][str(seed)][view] = query
                oracles[scope][view] = oracle
                gens = GENS if scope == "arf" else EXT_GENS
                print("  " + "; ".join(
                    f"{m}={macro_auc(run, m, gens):.4f}" for m in METHODS),
                    flush=True)
    summary = summarize(runs, query_stats, oracles)
    serializable_stats = {
        scope: {seed: {view: {m: {"ess": query_stats[scope][seed][view][m]["ess"].tolist()}
                              for m in METHODS}
                       for view in EVAL_VIEWS}
                for seed in map(str, SEEDS)}
        for scope in ("arf", "ext")}
    results = {"config": {"fit_sources": {"arf": 3000, "ext": 300},
                          "validation_sources": {"arf": 500, "ext": 100},
                          "calibration_sources": {"arf": 500, "ext": 300},
                          "test_per_class": N_TEST,
                          "views": VIEWS, "gammas": GAMMAS, "ess_min": ESS_MIN,
                          "mse_tolerance": MSE_TOLERANCE,
                          "protocol": "s3r-p0-300 source-aware controlled views 示例"},
               "runs": runs, "query_source_ESS": serializable_stats,
               "summary": summary}
    with open(os.path.join(OUT, "s3r_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"已保存 {OUT}/s3r_p0.json", flush=True)


if __name__ == "__main__":
    main()
