"""Full DART 与 Full+cycle residual 的五 seed 参数匹配配对实验。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_interaction_p0 as interaction  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_full_residual")
SEEDS = tuple(range(5))
METHODS = ("full", "full_cycle")
PROTOCOLS = ("darg", "arf_current")


def score_path(protocol, target, method, seed):
    return os.path.join(OUT, f"scores_{protocol}_{target}_{method}_seed{seed}.npz")


def summarize(results):
    summary = {}
    for protocol in PROTOCOLS:
        targets = interaction.DARG_TARGETS if protocol == "darg" else interaction.external.TARGETS
        summary[protocol] = {}
        for method in METHODS + ("prada",):
            per_seed = []
            for seed in SEEDS:
                entries = [results["folds"][f"{protocol}::{target}"]["seeds"][str(seed)]
                           for target in targets]
                per_seed.append(float(np.mean([
                    entry["prada"]["AUROC"] if method == "prada" else entry[method]["AUROC"]
                    for entry in entries])))
            summary[protocol][method] = {"macro_AUROC_per_seed": per_seed,
                                         "mean": float(np.mean(per_seed)),
                                         "std": float(np.std(per_seed, ddof=1))}
    return summary


def paired_tests():
    output = {}
    for protocol in PROTOCOLS:
        targets = interaction.DARG_TARGETS if protocol == "darg" else interaction.external.TARGETS
        output[protocol] = {}
        for seed in SEEDS:
            paired = {}
            for target in targets:
                candidate = np.load(score_path(protocol, target, "full_cycle", seed))
                baseline = np.load(score_path(protocol, target, "full", seed))
                paired[target] = (candidate["y"], candidate["max_test"], baseline["max_test"])
            delta, interval = paired_boot_macro(paired, n_boot=2000, seed=31000 + seed)
            output[protocol][str(seed)] = {"full_cycle_minus_full": delta, "CI95": interval}
    return output


def go_decision(results, paired):
    decisions = []
    for seed in SEEDS:
        decisions.append(paired["arf_current"][str(seed)])
    candidate_tpr, baseline_tpr, fprs = [], [], []
    for target in interaction.external.TARGETS:
        for seed in SEEDS:
            entry = results["folds"][f"arf_current::{target}"]["seeds"][str(seed)]
            candidate_tpr.append(entry["full_cycle"]["TPR@5%FPR"])
            baseline_tpr.append(entry["full"]["TPR@5%FPR"])
            fprs.append(entry["full_cycle"]["FPR"])
    deltas = [x["full_cycle_minus_full"] for x in decisions]
    criteria = {
        "mean_delta>=0.005": bool(np.mean(deltas) >= 0.005),
        "all_seed_paired_CI_positive": bool(all(x["CI95"][0] > 0 for x in decisions)),
        "TPR@5_not_worse_by_0.005": bool(np.mean(candidate_tpr) >= np.mean(baseline_tpr) - 0.005),
        "max_FPR<=10%": bool(max(fprs) <= 0.10),
    }
    return {"criteria": criteria, "GO": bool(all(criteria.values())),
            "mean_delta": float(np.mean(deltas)),
            "candidate_TPR@5": float(np.mean(candidate_tpr)),
            "full_TPR@5": float(np.mean(baseline_tpr)), "max_FPR": max(fprs)}


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    interaction.METHODS = METHODS
    interaction.score_path = score_path
    arf = interaction.source.torch_load(os.path.join(interaction.external.OUT, "packed_cycle.pt"))
    darg = interaction.source.torch_load(os.path.join(interaction.source.family.OUT, "packed_cycle.pt"))
    results = {"config": {
        "seeds": SEEDS, "protocols": PROTOCOLS, "methods": METHODS,
        "detector_parameter_count": interaction.parameter_count(),
        "protocol": "Full vs Full+residual; zero-padded 25 inputs; fixed tests; independent calibration",
        "example": True,
    }, "folds": {}}
    for protocol in PROTOCOLS:
        targets = interaction.DARG_TARGETS if protocol == "darg" else interaction.external.TARGETS
        for target in targets:
            for seed in SEEDS:
                fold = interaction.evaluate_fold(protocol, target, seed, arf, darg)
                results["folds"].setdefault(f"{protocol}::{target}", fold)["seeds"].update(fold["seeds"])
    results["summary"] = summarize(results)
    results["paired"] = paired_tests()
    results["go"] = go_decision(results, results["paired"])
    with open(os.path.join(OUT, "dart_full_residual.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(json.dumps(results["go"], indent=2), flush=True)


def main():
    evaluate()


if __name__ == "__main__":
    main()
