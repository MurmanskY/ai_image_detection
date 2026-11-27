"""从交互实验的完整 NPZ 重建 JSON，避免汇总错误导致重训。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_interaction_p0 as experiment  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    results = {"config": {
        "seeds": experiment.SEEDS, "protocols": experiment.PROTOCOLS,
        "methods": experiment.METHODS,
        "detector_parameter_count": experiment.parameter_count(), "cycle_ridge": 1.0,
        "protocol": "strict paired P0; fixed tests; independent calibration; max bank aggregation",
        "parameter_matching": "zero-padding to 25 inputs for every method",
        "go_protocol": "arf_current; mean delta>=.005, all-seed CI>0, TPR@5 noninferior .005, maxFPR<=.10",
        "example": True, "reconstructed_from_npz": True,
    }, "folds": {}}
    for protocol in experiment.PROTOCOLS:
        targets = experiment.DARG_TARGETS if protocol == "darg" else experiment.external.TARGETS
        for target in targets:
            bank = experiment.external.available_models(target)
            sources = ("darg",) if protocol == "darg" else ("arf",)
            fold = {"bank": bank, "real_sources": sources,
                    "eval_source": "darg" if protocol == "darg" else "arf", "seeds": {}}
            for seed in experiment.SEEDS:
                entry = {}
                for method in experiment.METHODS:
                    saved = np.load(experiment.score_path(protocol, target, method, seed))
                    for key in [method, "prada"] + (["cycle_raw"] if method == "lr_cycle" else []):
                        archive_key = "max" if key in experiment.METHODS else key
                        tau = conformal_tau(saved[f"{archive_key}_cal"])
                        entry[key] = metric(saved["y"], saved[f"{archive_key}_test"],
                                            experiment.external.N_TEST_REAL, tau)
                fold["seeds"][str(seed)] = entry
            results["folds"][f"{protocol}::{target}"] = fold
    results["summary"] = experiment.summarize(results)
    results["paired"] = experiment.paired_comparisons(results)
    results["go"] = experiment.go_decision(results, results["paired"])
    with open(os.path.join(experiment.OUT, "dart_interaction_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(json.dumps(results["go"], indent=2), flush=True)


if __name__ == "__main__":
    main()
