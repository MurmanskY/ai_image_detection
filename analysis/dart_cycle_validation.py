"""Raw generation-cycle residual 的 9-seed 来源、聚合与低 FPR 验证。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_family_p0 as external  # noqa: E402
from analysis import dart_interaction_p0 as interaction  # noqa: E402
from analysis import dart_source_control_p0 as source  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_cycle_validation")
SEEDS = tuple(range(9))
CONDITIONS = ("current", "darg_transfer", "balanced")
AGGREGATIONS = ("max", "mean", "median")


def eval_sources(condition, target):
    if condition == "current":
        return ("arf",)
    return ("arf", "darg") if target in external.TARGET_TO_DARG else ("arf",)


def score_path(condition, target, source_name, seed, aggregation):
    return os.path.join(OUT, f"scores_{condition}_{target}_{source_name}_{aggregation}_seed{seed}.npz")


def summarize(results):
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        source_targets = {}
        for target in external.TARGETS:
            for source_name in eval_sources(condition, target):
                source_targets.setdefault(source_name, []).append(target)
        for source_name, targets in source_targets.items():
            source_result = {}
            for aggregation in AGGREGATIONS:
                per_seed, fprs, tpr1, tpr5 = [], [], [], []
                for seed in SEEDS:
                    entries = [results["folds"][f"{condition}::{target}::{source_name}"]["seeds"][str(seed)]
                               for target in targets]
                    per_seed.append(float(np.mean([x[aggregation]["AUROC"] for x in entries])))
                    fprs.extend(x[aggregation]["FPR"] for x in entries)
                    tpr1.extend(x[aggregation]["TPR@1%FPR"] for x in entries)
                    tpr5.extend(x[aggregation]["TPR@5%FPR"] for x in entries)
                source_result[aggregation] = {
                    "macro_AUROC_per_seed": per_seed, "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed, ddof=1)),
                    "FPR_range": [float(min(fprs)), float(max(fprs))],
                    "TPR@1%FPR": float(np.mean(tpr1)), "TPR@5%FPR": float(np.mean(tpr5)),
                }
            summary[condition][source_name] = source_result
    return summary


def compare_with_detectors(results):
    previous = json.load(open(os.path.join(interaction.OUT, "dart_interaction_p0.json")))
    output = {}
    for baseline in ("lr", "full", "lr_cycle"):
        for aggregation in AGGREGATIONS:
            seed_results = {}
            for seed in interaction.SEEDS:
                paired = {}
                for target in external.TARGETS:
                    current = np.load(score_path("current", target, "arf", seed, aggregation))
                    saved = np.load(interaction.score_path("arf_current", target, baseline, seed))
                    paired[target] = (saved["y"], current["test"], saved["max_test"])
                delta, interval = paired_boot_macro(paired, n_boot=5000, seed=22000 + seed)
                seed_results[str(seed)] = {"raw_minus_baseline": delta, "CI95": interval}
            output[f"raw_{aggregation}_vs_{baseline}"] = seed_results
    return output


def replication_check(results):
    errors = []
    for target in external.TARGETS:
        for seed in interaction.SEEDS:
            new = results["folds"][f"current::{target}::arf"]["seeds"][str(seed)]["max"]["AUROC"]
            saved = np.load(interaction.score_path("arf_current", target, "lr_cycle", seed))
            old = saved["cycle_raw_test"]
            recomputed_old = source.family.c2i.metric(
                saved["y"], old, external.N_TEST_REAL, conformal_tau(saved["cycle_raw_cal"]))["AUROC"]
            errors.append(abs(new - recomputed_old))
    return float(max(errors))


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    arf = source.torch_load(os.path.join(external.OUT, "packed_cycle.pt"))
    darg = source.torch_load(os.path.join(source.family.OUT, "packed_cycle.pt"))
    results = {"config": {
        "seeds": SEEDS, "conditions": CONDITIONS, "aggregations": AGGREGATIONS,
        "targets": external.TARGETS, "models": external.MODELS,
        "protocol": "raw cycle residual; real-only conditional fit; fixed tests; independent calibration",
        "example": True,
    }, "folds": {}}
    for condition in CONDITIONS:
        sources = source.real_sources(condition)
        for target in external.TARGETS:
            bank = external.available_models(target)
            fake_sources = source.training_fake_sources(condition, target)
            for seed in SEEDS:
                ref_idx, select_idx, cal_idx, test_idx, _, _, fake_test = external.fixed_splits(seed)
                data = {model: source.normalize_model(arf, darg, model, sources, ref_idx,
                                                      fake_sources, target) for model in bank}
                residualizers = interaction.fit_residualizers(data, bank, sources, ref_idx, select_idx)
                r2 = {model: residualizers[model].selection_r2 for model in bank}
                for source_name in eval_sources(condition, target):
                    name = f"{condition}::{target}::{source_name}"
                    entry = {"bank": bank, "residualizer_selection_R2": r2, "seeds": {}}
                    seed_result = {"residualizer_selection_R2": r2}
                    saved_outputs = {}
                    for aggregation in AGGREGATIONS:
                        metrics, test, cal, tau = interaction.raw_cycle_scores(
                            residualizers, bank, data, target, source_name,
                            cal_idx, test_idx, fake_test, aggregation)
                        seed_result[aggregation] = metrics
                        path = score_path(condition, target, source_name, seed, aggregation)
                        np.savez(path, y=np.array([0] * external.N_TEST_REAL + [1] * external.N_TEST_FAKE),
                                 test=test, cal=cal, threshold=tau, models=np.array(bank))
                        saved_outputs[aggregation] = (float(test[:external.N_TEST_REAL].mean()),
                                                      float(test[external.N_TEST_REAL:].mean()))
                    seed_result["residual_test_mean_real_fake"] = saved_outputs
                    results["folds"].setdefault(name, {"bank": bank, "seeds": {}})["seeds"][str(seed)] = seed_result
                print(f"{condition} target={target} seed={seed} "
                      f"max={seed_result['max']['AUROC']:.4f}", flush=True)
    results["summary"] = summarize(results)
    results["paired_vs_detectors"] = compare_with_detectors(results)
    results["replication_max_AUROC_error"] = replication_check(results)
    with open(os.path.join(OUT, "dart_cycle_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print("replication error", results["replication_max_AUROC_error"], flush=True)


def main():
    evaluate()


if __name__ == "__main__":
    main()
