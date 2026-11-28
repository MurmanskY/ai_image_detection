"""独立复算 Full vs Full+residual 实验。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_full_residual_comparison as comparison  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    results = json.load(open(os.path.join(comparison.OUT, "dart_full_residual.json")))
    counts = {"fold_seeds": 0, "AUROC": 0, "thresholds": 0, "TPR_points": 0}
    max_error = 0.0
    for name, fold in results["folds"].items():
        protocol, target = name.split("::")
        for seed in comparison.SEEDS:
            entry = fold["seeds"][str(seed)]
            for method in comparison.METHODS:
                saved = np.load(comparison.score_path(protocol, target, method, seed))
                for key in (method, "prada"):
                    archive_key = "max" if key in comparison.METHODS else key
                    tau = conformal_tau(saved[f"{archive_key}_cal"])
                    recomputed = metric(saved["y"], saved[f"{archive_key}_test"], 300, tau)
                    archived = entry[key]
                    assert np.isclose(tau, saved[f"{archive_key}_threshold"])
                    assert recomputed["FPR"] == archived["FPR"]
                    max_error = max(max_error, abs(recomputed["AUROC"] - archived["AUROC"]))
                    for level in ("0.1", "1", "5", "10"):
                        name = f"TPR@{level}%FPR"
                        assert np.isclose(recomputed[name], archived[name])
                        counts["TPR_points"] += 1
                    counts["AUROC"] += 1
                    counts["thresholds"] += 1
            counts["fold_seeds"] += 1
    assert max_error < 1e-12
    report = {"counts": counts, "max_AUROC_error": max_error, "passed": True}
    with open(os.path.join(comparison.OUT, "full_residual_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
