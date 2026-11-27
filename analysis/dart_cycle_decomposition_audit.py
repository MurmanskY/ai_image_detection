"""独立复算 cycle decomposition 的全部指标。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_cycle_decomposition as decomposition  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    results = json.load(open(os.path.join(decomposition.OUT, "dart_cycle_decomposition.json")))
    counts = {"fold_seeds": 0, "AUROC": 0, "thresholds": 0, "TPR_points": 0}
    max_error = 0.0
    for target, fold in results["folds"].items():
        for seed in decomposition.SEEDS:
            entry = fold["seeds"][str(seed)]
            for aggregation in decomposition.AGGREGATIONS:
                for component in decomposition.COMPONENTS:
                    saved = np.load(decomposition.score_path(target, seed, aggregation, component))
                    tau = conformal_tau(saved["cal"])
                    recomputed = metric(saved["y"], saved["test"],
                                        decomposition.external.N_TEST_REAL, tau)
                    archived = entry[f"{aggregation}::{component}"]
                    assert np.isclose(tau, saved["threshold"])
                    assert recomputed["FPR"] == archived["FPR"]
                    max_error = max(max_error, abs(recomputed["AUROC"] - archived["AUROC"]))
                    for level in ("0.1", "1", "5", "10"):
                        key = f"TPR@{level}%FPR"
                        assert np.isclose(recomputed[key], archived[key])
                        counts["TPR_points"] += 1
                    counts["AUROC"] += 1
                    counts["thresholds"] += 1
            counts["fold_seeds"] += 1
    assert max_error < 1e-12
    report = {"counts": counts, "max_AUROC_error": max_error, "passed": True}
    with open(os.path.join(decomposition.OUT, "cycle_decomposition_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
