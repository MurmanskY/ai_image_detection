"""独立复算来源控制实验，并核对 current 与既有七家族归档一致。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_family_p0 as external  # noqa: E402
from analysis import dart_source_control_p0 as source  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    results = json.load(open(os.path.join(source.OUT, "dart_source_control.json")))
    previous = json.load(open(os.path.join(external.OUT, "dart_external_family.json")))
    counts = {"fold_seeds": 0, "AUROC": 0, "thresholds": 0, "TPR_points": 0}
    max_error = 0.0
    current_replication_error = 0.0
    for name, fold in results["folds"].items():
        condition, target, eval_source = name.split("::")
        for seed in source.SEEDS:
            entry = fold["seeds"][str(seed)]
            saved = np.load(source.score_path(condition, target, eval_source, seed))
            assert tuple(saved["models"]) == tuple(fold["bank"])
            for method in source.AGGREGATIONS + ("selected", "prada"):
                tau = conformal_tau(saved[f"{method}_cal"])
                recomputed = metric(saved["y"], saved[f"{method}_test"],
                                    external.N_TEST_REAL, tau)
                archived = entry[method]
                error = abs(recomputed["AUROC"] - archived["AUROC"])
                max_error = max(max_error, error)
                assert np.isclose(tau, saved[f"{method}_threshold"])
                assert recomputed["FPR"] == archived["FPR"]
                for level in ("0.1", "1", "5", "10"):
                    key = f"TPR@{level}%FPR"
                    assert np.isclose(recomputed[key], archived[key])
                    counts["TPR_points"] += 1
                counts["AUROC"] += 1
                counts["thresholds"] += 1
            assert entry["selected_aggregation"] in source.AGGREGATIONS
            selected = entry["selected_aggregation"]
            assert entry["selected"] == entry[selected]
            if condition == "current":
                bank = external.bank_name(external.available_models(target))
                old = previous["folds"][target]["seeds"][str(seed)]["banks"][bank]["full"]["AUROC"]
                current_replication_error = max(
                    current_replication_error, abs(old - entry["max"]["AUROC"]))
            counts["fold_seeds"] += 1
    assert max_error < 1e-12
    report = {"counts": counts, "max_AUROC_error": max_error,
              "current_vs_previous_max_AUROC_error": current_replication_error,
              "passed": True}
    with open(os.path.join(source.OUT, "source_control_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
