"""ImageNet-V2 独立外部 calibration/test：区分阈值迁移与外部再校准。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_cycle_robustness as robustness  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric, wilson_interval  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_external_calibration")
SPLITS = tuple(range(5))
N_CAL = 150


def result_path(seed, aggregation, component, split):
    return os.path.join(OUT, f"external_cal_{aggregation}_{component}_seed{seed}_split{split}.npz")


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    results = {"config": {
        "seeds": robustness.SEEDS, "splits": SPLITS, "n_external": 300,
        "external_cal": N_CAL, "external_test": 150,
        "protocol": "disjoint ImageNet-V2 calibration/test; no model or feature retuning",
        "note": "splits measure calibration variability, not independent datasets",
    }, "entries": {}}
    for seed in robustness.SEEDS:
        for aggregation in robustness.AGGREGATIONS:
            for component in robustness.COMPONENTS:
                key = f"{seed}::{aggregation}::{component}"
                saved = np.load(os.path.join(
                    robustness.OUT, f"external_{aggregation}_{component}_seed{seed}.npz"))
                external = saved["external"]
                original_cal = saved["cal"]
                original_tau = conformal_tau(original_cal)
                entries = []
                for split in SPLITS:
                    perm = np.random.RandomState(5000 + split).permutation(len(external))
                    cal_idx, test_idx = perm[:N_CAL], perm[N_CAL:]
                    external_tau = conformal_tau(external[cal_idx])
                    fp_original = int((external[test_idx] > original_tau).sum())
                    fp_recal = int((external[test_idx] > external_tau).sum())
                    per_gen = {}
                    for gen in robustness.c2i.FAKE_GENS:
                        fake_saved = np.load(os.path.join(
                            robustness.OUT, f"scores_original_{aggregation}_{component}_seed{seed}_{gen}.npz"))
                        fake = fake_saved["test"][-100:]
                        y = np.array([0] * len(test_idx) + [1] * len(fake))
                        scores = np.concatenate([external[test_idx], fake])
                        per_gen[gen] = {
                            "original_threshold": metric(y, scores, len(test_idx), original_tau),
                            "external_recalibration": metric(y, scores, len(test_idx), external_tau),
                        }
                    entry = {
                        "split": split,
                        "original_threshold": original_tau,
                        "external_threshold": external_tau,
                        "original_FPR": fp_original / len(test_idx),
                        "original_FPR_CI95": wilson_interval(fp_original, len(test_idx)),
                        "recalibrated_FPR": fp_recal / len(test_idx),
                        "recalibrated_FPR_CI95": wilson_interval(fp_recal, len(test_idx)),
                        "per_generator": per_gen,
                        "original_macro_AUROC": float(np.mean([
                            x["original_threshold"]["AUROC"] for x in per_gen.values()])),
                        "recalibrated_macro_AUROC": float(np.mean([
                            x["external_recalibration"]["AUROC"] for x in per_gen.values()])),
                    }
                    entries.append(entry)
                    np.savez(result_path(seed, aggregation, component, split),
                             external=external, cal_idx=cal_idx, test_idx=test_idx,
                             original_cal=original_cal, original_threshold=original_tau,
                             external_threshold=external_tau)
                results["entries"][key] = entries
    results["summary"] = summarize(results)
    with open(os.path.join(OUT, "dart_external_calibration.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def summarize(results):
    summary = {}
    for aggregation in robustness.AGGREGATIONS:
        for component in robustness.COMPONENTS:
            entries = []
            for seed in robustness.SEEDS:
                entries.extend(results["entries"][f"{seed}::{aggregation}::{component}"])
            summary[f"{aggregation}::{component}"] = {
                "original_FPR_mean": float(np.mean([x["original_FPR"] for x in entries])),
                "original_FPR_range": [min(x["original_FPR"] for x in entries),
                                       max(x["original_FPR"] for x in entries)],
                "recalibrated_FPR_mean": float(np.mean([x["recalibrated_FPR"] for x in entries])),
                "recalibrated_FPR_range": [min(x["recalibrated_FPR"] for x in entries),
                                           max(x["recalibrated_FPR"] for x in entries)],
                "macro_AUROC_mean": float(np.mean([x["recalibrated_macro_AUROC"] for x in entries])),
                "original_recal_AUROC_max_difference": float(max(
                    abs(x["original_macro_AUROC"] - x["recalibrated_macro_AUROC"]) for x in entries)),
            }
    return summary


def main():
    evaluate()


if __name__ == "__main__":
    main()
