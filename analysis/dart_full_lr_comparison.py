"""Full DART 与 DART-LR 的四家族严格 9-seed 配对复核。"""

import json
import os
import sys

import numpy as np
from scipy import stats

os.environ.setdefault("DART_N_REAL", "1000")
os.environ.setdefault("DART_REF_REAL", "300")
os.environ.setdefault("DART_SELECT_REAL", "200")
os.environ.setdefault("DART_CAL_REAL", "200")
PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_c2i_p0 as c2i
from analysis import dart_family_p0 as family
from analysis.carf_a import paired_boot_macro
from analysis.carf_p0 import conformal_tau
from analysis.dart_p0 import ScaleDeepSet

OUT = os.path.join(c2i.OUT, "full_lr_paired_s0_8.json")
SEEDS = tuple(range(9))
TAGS = {
    "full": "exclude_s0_1_2_3_4_5_6_7_8",
    "likelihood_reconstruction": "exclude_likelihood_reconstruction_s0_1_2_3_4_5_6_7_8",
}


def load_run(mode):
    return json.load(open(os.path.join(c2i.OUT, f"dart_family_{TAGS[mode]}.json")))


def scores(mode, target, seed):
    path = os.path.join(
        c2i.OUT, f"scores_dart_family_{TAGS[mode]}_{target}_seed{seed}.npz")
    saved = np.load(path)
    test = np.max(saved["dart_bank_test"], axis=1)
    cal = np.max(saved["dart_bank_cal"], axis=1)
    return saved, test, cal


def main():
    runs = {mode: load_run(mode) for mode in TAGS}
    for run in runs.values():
        assert run["config"]["protocol"] == "strict real 300/200/200/300; fake 100/50/100"
        assert tuple(run["config"]["seeds"]) == SEEDS
        assert run["config"]["pooling"] == "attention"
    results = {"config": {
        "seeds": SEEDS, "targets": family.GENS, "paired_by": "same images, splits, bank and seed",
        "training_epochs": 40, "pooling": "attention", "aggregation": "max",
        "protocol": "strict real 300/200/200/300; fake 100/50/100",
        "parameter_count": {mode: sum(p.numel() for p in ScaleDeepSet(dim).parameters())
                            for mode, dim in (("full", 20), ("likelihood_reconstruction", 19))},
        "note": "seeds measure split/training variability, not independent datasets",
    }, "seeds": {}}
    audit = {"AUROC": 0, "thresholds": 0, "TPR_points": 0}
    for seed in SEEDS:
        paired = {}
        seed_result = {}
        for target in family.GENS:
            y = np.array([0] * c2i.N_TEST_REAL + [1] * 100)
            target_result = {}
            score_values = {}
            for mode in TAGS:
                saved, test, cal = scores(mode, target, seed)
                tau = conformal_tau(cal)
                assert np.isclose(tau, saved["dart_max_threshold"])
                metric = c2i.metric(y, test, c2i.N_TEST_REAL, tau)
                archived = runs[mode]["folds"][target]["seeds"][str(seed)]["dart"]
                assert np.isclose(metric["AUROC"], archived["AUROC"])
                assert metric["FPR"] == archived["FPR"]
                for level in ("0.1", "1", "5", "10"):
                    assert np.isclose(metric[f"TPR@{level}%FPR"], archived[f"TPR@{level}%FPR"])
                audit["AUROC"] += 1
                audit["thresholds"] += 1
                audit["TPR_points"] += 4
                target_result[mode] = metric
                score_values[mode] = test
            paired[target] = (y, score_values["likelihood_reconstruction"], score_values["full"])
            target_result["LR_minus_full_AUROC"] = (
                target_result["likelihood_reconstruction"]["AUROC"] - target_result["full"]["AUROC"])
            seed_result[target] = target_result
        delta, interval = paired_boot_macro(paired, n_boot=5000, seed=7100 + seed)
        seed_result["paired_bootstrap"] = {"LR_minus_full": delta, "CI95": interval}
        results["seeds"][str(seed)] = seed_result
        print(f"seed={seed} paired Δ={delta:+.6f}, CI95={interval}", flush=True)

    summary = {}
    for mode in TAGS:
        macros = [float(np.mean([results["seeds"][str(seed)][target][mode]["AUROC"]
                                 for target in family.GENS])) for seed in SEEDS]
        summary[mode] = {
            "macro_AUROC_per_seed": macros,
            "mean": float(np.mean(macros)), "std": float(np.std(macros, ddof=1)),
            "per_target_mean": {target: float(np.mean([
                results["seeds"][str(seed)][target][mode]["AUROC"] for seed in SEEDS]))
                for target in family.GENS},
            "FPR_per_seed": [float(results["seeds"][str(seed)][family.GENS[0]][mode]["FPR"])
                             for seed in SEEDS],
            "TPR@1%FPR_mean": float(np.mean([
                results["seeds"][str(seed)][target][mode]["TPR@1%FPR"]
                for seed in SEEDS for target in family.GENS])),
            "TPR@5%FPR_mean": float(np.mean([
                results["seeds"][str(seed)][target][mode]["TPR@5%FPR"]
                for seed in SEEDS for target in family.GENS])),
        }
    deltas = [float(np.mean([results["seeds"][str(seed)][target]["LR_minus_full_AUROC"]
                             for target in family.GENS])) for seed in SEEDS]
    sem = stats.sem(deltas)
    interval = stats.t.interval(0.95, len(deltas) - 1,
                                loc=np.mean(deltas), scale=sem)
    summary["paired"] = {
        "LR_minus_full_per_seed": deltas,
        "mean": float(np.mean(deltas)), "std": float(np.std(deltas, ddof=1)),
        "seed_level_t_CI95": [float(x) for x in interval],
        "paired_t_p": float(stats.ttest_rel(
            [summary["likelihood_reconstruction"]["macro_AUROC_per_seed"][i] for i in range(9)],
            [summary["full"]["macro_AUROC_per_seed"][i] for i in range(9)]).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(deltas).pvalue),
        "positive_seeds": int(sum(delta > 0 for delta in deltas)),
        "all_seed_bootstrap_CI_exclude_zero": all(
            results["seeds"][str(seed)]["paired_bootstrap"]["CI95"][0] > 0
            for seed in SEEDS),
    }
    results["summary"] = summary
    results["independent_recompute"] = audit
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {OUT}; independent audit={audit}", flush=True)


if __name__ == "__main__":
    main()
