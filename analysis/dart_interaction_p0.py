"""DART 显式生成循环交互与 cycle residual 的严格配对 P0。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
os.environ.setdefault("DART_N_REAL", "1000")
os.environ.setdefault("DART_REF_REAL", "300")
os.environ.setdefault("DART_SELECT_REAL", "200")
os.environ.setdefault("DART_CAL_REAL", "200")

from analysis import dart_external_family_p0 as external  # noqa: E402
from analysis import dart_source_control_p0 as source  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_interaction import (METHODS, PAIR_BASELINE, CycleResidualizer,  # noqa: E402
                                       parameter_count, transform_images)
from analysis.dart_p0 import train_dart  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_interaction")
SEEDS = tuple(int(x) for x in os.environ.get("DART_INTERACTION_SEEDS", "0,1,2").split(","))
PROTOCOLS = ("darg", "arf_current")
DARG_TARGETS = tuple(external.TARGET_TO_DARG)


def score_path(protocol, target, method, seed):
    return os.path.join(OUT, f"scores_{protocol}_{target}_{method}_seed{seed}.npz")


def fit_residualizers(data, bank, sources, ref_idx, select_idx):
    output = {}
    for model in bank:
        reference = [data[model]["real", src][i] for src in sources for i in ref_idx]
        selection = [data[model]["real", src][i] for src in sources for i in select_idx]
        output[model] = CycleResidualizer().fit(reference, selection)
    return output


def raw_cycle_scores(residualizers, bank, data, target, eval_source,
                     cal_idx, test_idx, fake_test, aggregation="max"):
    real_cal, real_test, fake_test_scores = [], [], []
    for model in bank:
        residualizer = residualizers[model]
        real_images = data[model]["real", eval_source]
        fake_images = data[model]["target", eval_source]
        real_cal.append([residualizer.residual([image]).mean() for image in
                         np.asarray(real_images, dtype=object)[cal_idx]])
        real_test.append([residualizer.residual([image]).mean() for image in
                          np.asarray(real_images, dtype=object)[test_idx]])
        fake_test_scores.append([residualizer.residual([image]).mean() for image in
                                 np.asarray(fake_images, dtype=object)[fake_test]])
    cal = source.aggregate(np.stack(real_cal, axis=1), aggregation)
    test = np.concatenate([source.aggregate(np.stack(real_test, axis=1), aggregation),
                           source.aggregate(np.stack(fake_test_scores, axis=1), aggregation)])
    tau = conformal_tau(cal)
    y = np.array([0] * external.N_TEST_REAL + [1] * external.N_TEST_FAKE)
    return source.family.c2i.metric(y, test, external.N_TEST_REAL, tau), test, cal, tau


def evaluate_fold(protocol, target, seed, arf, darg):
    condition = "darg_transfer" if protocol == "darg" else "current"
    sources = source.real_sources(condition)
    eval_source = "darg" if protocol == "darg" else "arf"
    ref_idx, select_idx, cal_idx, test_idx, train_idx, val_idx, fake_test = external.fixed_splits(seed)
    bank = external.available_models(target)
    fake_sources = source.training_fake_sources(condition, target)
    data = {model: source.normalize_model(arf, darg, model, sources, ref_idx,
                                          fake_sources, target) for model in bank}
    fold = {"bank": bank, "real_sources": sources, "eval_source": eval_source,
            "seeds": {str(seed): {}}}
    for method in METHODS:
        residualizers = fit_residualizers(data, bank, sources, ref_idx, select_idx) \
            if method in ("lr_cycle", "full_cycle") else None
        transformed = {}
        for model in bank:
            transformed[model] = {key: transform_images(images, method,
                                                        None if residualizers is None else residualizers[model])
                                  for key, images in data[model].items()}
        train_data, val_data = source.build_training_data(
            transformed, bank, sources, fake_sources, ref_idx, select_idx, train_idx, val_idx)
        dart = train_dart(train_data, val_data, seed)
        stats = source.selection_statistics(dart, transformed, bank, sources, select_idx)
        metrics, saved = source.evaluate_target(
            dart, transformed, bank, stats, target, eval_source,
            select_idx, cal_idx, test_idx, fake_test, "max")
        entry = {method: metrics["max"], "prada": metrics["prada"]}
        if method == "lr_cycle":
            # Raw residual diagnostics are computed from normalized pre-detector images.
            raw, raw_test, raw_cal, raw_tau = raw_cycle_scores(
                residualizers, bank, data, target, eval_source, cal_idx, test_idx, fake_test)
            entry["cycle_raw"] = raw
            entry["residualizer_selection_R2"] = {
                model: residualizers[model].selection_r2 for model in bank}
            saved.update({"cycle_raw_test": raw_test, "cycle_raw_cal": raw_cal,
                          "cycle_raw_threshold": raw_tau})
        fold["seeds"].setdefault(str(seed), {}).update(entry)
        np.savez(score_path(protocol, target, method, seed), **saved)
        print(f"{protocol} target={target} seed={seed} method={method} "
              f"AUROC={metrics['max']['AUROC']:.4f} FPR={metrics['max']['FPR']:.4f}",
              flush=True)
    return fold


def summarize(results):
    summary = {}
    for protocol in PROTOCOLS:
        targets = DARG_TARGETS if protocol == "darg" else external.TARGETS
        protocol_result = {}
        for method in METHODS + ("cycle_raw", "prada"):
            per_seed = []
            for seed in SEEDS:
                values = []
                for target in targets:
                    entry = results["folds"][f"{protocol}::{target}"]["seeds"][str(seed)]
                    if method == "prada":
                        value = entry["prada"]["AUROC"]
                    elif method == "cycle_raw":
                        if "cycle_raw" not in entry:
                            continue
                        value = entry["cycle_raw"]["AUROC"]
                    else:
                        value = entry[method]["AUROC"]
                    values.append(value)
                per_seed.append(float(np.mean(values)))
            protocol_result[method] = {"macro_AUROC_per_seed": per_seed,
                                       "mean": float(np.mean(per_seed)),
                                       "std": float(np.std(per_seed, ddof=1))}
        summary[protocol] = protocol_result
    return summary


def paired_comparisons(results):
    output = {}
    for protocol in PROTOCOLS:
        targets = DARG_TARGETS if protocol == "darg" else external.TARGETS
        comparisons = {}
        for method in METHODS[2:]:
            baseline = PAIR_BASELINE[method]
            per_seed = {}
            for seed in SEEDS:
                paired = {}
                for target in targets:
                    saved = np.load(score_path(protocol, target, method, seed))
                    base = np.load(score_path(protocol, target, baseline, seed))
                    paired[target] = (saved["y"], saved["max_test"], base["max_test"])
                delta, interval = paired_boot_macro(paired, n_boot=2000,
                                                    seed=19000 + seed)
                per_seed[str(seed)] = {"candidate_minus_baseline": delta, "CI95": interval}
            comparisons[f"{method}_vs_{baseline}"] = per_seed
        output[protocol] = comparisons
    return output


def go_decision(results, comparisons):
    protocol = "arf_current"
    decisions = {}
    for method in METHODS[2:]:
        baseline = PAIR_BASELINE[method]
        deltas = [comparisons[protocol][f"{method}_vs_{baseline}"][str(seed)]["candidate_minus_baseline"]
                  for seed in SEEDS]
        positive_ci = [comparisons[protocol][f"{method}_vs_{baseline}"][str(seed)]["CI95"][0] > 0
                       for seed in SEEDS]
        candidate_tpr, baseline_tpr = [], []
        max_fpr = 0.0
        for target in external.TARGETS:
            for seed in SEEDS:
                entry = results["folds"][f"{protocol}::{target}"]["seeds"][str(seed)]
                candidate_tpr.append(entry[method]["TPR@5%FPR"])
                baseline_tpr.append(entry[baseline]["TPR@5%FPR"])
                max_fpr = max(max_fpr, entry[method]["FPR"])
        criteria = {
            "mean_delta>=0.005": bool(np.mean(deltas) >= 0.005),
            "all_seed_paired_CI_positive": bool(all(positive_ci)),
            "TPR@5_not_worse_by_0.005": bool(np.mean(candidate_tpr) >= np.mean(baseline_tpr) - 0.005),
            "max_FPR<=10%": bool(max_fpr <= 0.10),
        }
        decisions[f"{method}_vs_{baseline}"] = {
            "criteria": criteria, "GO": bool(all(criteria.values())),
            "mean_delta": float(np.mean(deltas)),
            "candidate_TPR@5": float(np.mean(candidate_tpr)),
            "baseline_TPR@5": float(np.mean(baseline_tpr)), "max_FPR": max_fpr,
        }
    return decisions


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    arf = source.torch_load(os.path.join(external.OUT, "packed_cycle.pt"))
    darg = source.torch_load(os.path.join(source.family.OUT, "packed_cycle.pt"))
    results = {"config": {
        "seeds": SEEDS, "protocols": PROTOCOLS, "methods": METHODS,
        "detector_parameter_count": parameter_count(), "cycle_ridge": 1.0,
        "protocol": "strict paired P0; fixed tests; independent calibration; max bank aggregation",
        "parameter_matching": "zero-padding to 25 inputs for every method",
        "go_protocol": "arf_current; mean delta>=.005, all-seed CI>0, TPR@5 noninferior .005, maxFPR<=.10",
        "example": True,
    }, "folds": {}}
    for protocol in PROTOCOLS:
        targets = DARG_TARGETS if protocol == "darg" else external.TARGETS
        for target in targets:
            for seed in SEEDS:
                fold = evaluate_fold(protocol, target, seed, arf, darg)
                results["folds"].setdefault(f"{protocol}::{target}", fold)["seeds"].update(fold["seeds"])
                print(f"completed {protocol}/{target}/seed{seed}", flush=True)
    results["summary"] = summarize(results)
    results["paired"] = paired_comparisons(results)
    results["go"] = go_decision(results, results["paired"])
    with open(os.path.join(OUT, "dart_interaction_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(json.dumps(results["go"], indent=2), flush=True)


def main():
    evaluate()


if __name__ == "__main__":
    main()
