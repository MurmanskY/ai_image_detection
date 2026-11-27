"""Cycle residual 的 R / Rhat / R-Rhat 同尺度分解与独立增益检验。"""

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

OUT = os.path.join(PROJ, "outputs", "dart_cycle_decomposition")
SEEDS = tuple(range(9))
COMPONENTS = ("reconstruction", "conditional_prediction", "cycle_residual")
AGGREGATIONS = ("max", "mean", "median")


def component_scores(residualizers, bank, data, target, eval_source,
                     cal_idx, test_idx, fake_test, aggregation, component):
    real_cal, real_test, fake_scores = [], [], []
    for model in bank:
        residualizer = residualizers[model]
        real_images = np.asarray(data[model]["real", eval_source], dtype=object)
        fake_images = np.asarray(data[model]["target", eval_source], dtype=object)
        for images, indices, destination in ((real_images, cal_idx, real_cal),
                                             (real_images, test_idx, real_test),
                                             (fake_images, fake_test, fake_scores)):
            values = []
            for image in images[indices]:
                x, rec = residualizer.design([image])
                if component == "reconstruction":
                    value = rec
                else:
                    predicted = x @ residualizer.coef
                    value = predicted if component == "conditional_prediction" else rec - predicted
                values.append(float(value.mean() / residualizer.residual_scale))
            destination.append(values)
    cal = source.aggregate(np.stack(real_cal, axis=1), aggregation)
    test = np.concatenate([source.aggregate(np.stack(real_test, axis=1), aggregation),
                           source.aggregate(np.stack(fake_scores, axis=1), aggregation)])
    tau = conformal_tau(cal)
    y = np.array([0] * external.N_TEST_REAL + [1] * external.N_TEST_FAKE)
    return source.family.c2i.metric(y, test, external.N_TEST_REAL, tau), test, cal, tau


def score_path(target, seed, aggregation, component):
    return os.path.join(OUT, f"scores_{target}_{aggregation}_{component}_seed{seed}.npz")


def summarize(results):
    summary = {}
    for component in COMPONENTS:
        for aggregation in AGGREGATIONS:
            per_seed, fprs, tpr1, tpr5 = [], [], [], []
            for seed in SEEDS:
                entries = [results["folds"][target]["seeds"][str(seed)] for target in external.TARGETS]
                key = f"{aggregation}::{component}"
                per_seed.append(float(np.mean([x[key]["AUROC"] for x in entries])))
                fprs.extend(x[key]["FPR"] for x in entries)
                tpr1.extend(x[key]["TPR@1%FPR"] for x in entries)
                tpr5.extend(x[key]["TPR@5%FPR"] for x in entries)
            summary[f"{aggregation}::{component}"] = {
                "macro_AUROC_per_seed": per_seed, "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed, ddof=1)), "FPR_range": [min(fprs), max(fprs)],
                "TPR@1%FPR": float(np.mean(tpr1)), "TPR@5%FPR": float(np.mean(tpr5)),
            }
    return summary


def paired_tests():
    output = {}
    for aggregation in AGGREGATIONS:
        for component in ("conditional_prediction", "cycle_residual"):
            seed_results = {}
            for seed in SEEDS:
                paired = {}
                for target in external.TARGETS:
                    candidate = np.load(score_path(target, seed, aggregation, component))
                    baseline = np.load(score_path(target, seed, aggregation, "reconstruction"))
                    paired[target] = (candidate["y"], candidate["test"], baseline["test"])
                delta, interval = paired_boot_macro(paired, n_boot=2000, seed=25000 + seed)
                seed_results[str(seed)] = {"delta_vs_reconstruction": delta, "CI95": interval}
            output[f"{aggregation}::{component}_vs_reconstruction"] = seed_results
    return output


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    arf = source.torch_load(os.path.join(external.OUT, "packed_cycle.pt"))
    darg = source.torch_load(os.path.join(source.family.OUT, "packed_cycle.pt"))
    results = {"config": {
        "seeds": SEEDS, "components": COMPONENTS, "aggregations": AGGREGATIONS,
        "targets": external.TARGETS, "protocol": "current ARF; same real-only residual scale for R/Rhat/residual",
        "example": True,
    }, "folds": {}}
    for target in external.TARGETS:
        bank = external.available_models(target)
        for seed in SEEDS:
            ref_idx, select_idx, cal_idx, test_idx, _, _, fake_test = external.fixed_splits(seed)
            data = {model: source.normalize_model(
                arf, darg, model, ("arf",), ref_idx,
                source.training_fake_sources("current", target), target) for model in bank}
            residualizers = interaction.fit_residualizers(data, bank, ("arf",), ref_idx, select_idx)
            entry = {}
            for aggregation in AGGREGATIONS:
                for component in COMPONENTS:
                    metrics, test, cal, tau = component_scores(
                        residualizers, bank, data, target, "arf", cal_idx, test_idx,
                        fake_test, aggregation, component)
                    entry[f"{aggregation}::{component}"] = metrics
                    np.savez(score_path(target, seed, aggregation, component),
                             y=np.array([0] * external.N_TEST_REAL + [1] * external.N_TEST_FAKE),
                             test=test, cal=cal, threshold=tau, models=np.array(bank))
            results["folds"].setdefault(target, {"bank": bank, "seeds": {}})["seeds"][str(seed)] = entry
            print(f"target={target} seed={seed} complete", flush=True)
    results["summary"] = summarize(results)
    results["paired"] = paired_tests()
    with open(os.path.join(OUT, "dart_cycle_decomposition.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def main():
    evaluate()


if __name__ == "__main__":
    main()
