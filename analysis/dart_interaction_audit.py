"""独立复算交互实验的逐图指标和阈值。"""

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
    results = json.load(open(os.path.join(experiment.OUT, "dart_interaction_p0.json")))
    counts = {"fold_seeds": 0, "AUROC": 0, "thresholds": 0, "TPR_points": 0}
    max_error = 0.0
    for name, fold in results["folds"].items():
        protocol, target = name.split("::")
        for seed in experiment.SEEDS:
            entry = fold["seeds"][str(seed)]
            for method in experiment.METHODS:
                saved = np.load(experiment.score_path(protocol, target, method, seed))
                methods = [method, "prada"] + (["cycle_raw"] if method == "lr_cycle" else [])
                for key in methods:
                    archive_key = "max" if key in experiment.METHODS else key
                    tau = conformal_tau(saved[f"{archive_key}_cal"])
                    recomputed = metric(saved["y"], saved[f"{archive_key}_test"],
                                        experiment.external.N_TEST_REAL, tau)
                    archived = entry[key]
                    assert np.isclose(tau, saved[f"{archive_key}_threshold"])
                    assert recomputed["FPR"] == archived["FPR"]
                    max_error = max(max_error, abs(recomputed["AUROC"] - archived["AUROC"]))
                    for level in ("0.1", "1", "5", "10"):
                        metric_name = f"TPR@{level}%FPR"
                        assert np.isclose(recomputed[metric_name], archived[metric_name])
                        counts["TPR_points"] += 1
                    counts["AUROC"] += 1
                    counts["thresholds"] += 1
            counts["fold_seeds"] += 1
    assert max_error < 1e-12
    report = {"counts": counts, "max_AUROC_error": max_error, "passed": True}
    with open(os.path.join(experiment.OUT, "interaction_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
