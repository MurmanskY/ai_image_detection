"""从逐图 NPZ 独立复算七家族外部实验的全部归档指标。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_family_p0 as external  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    path = os.path.join(external.OUT, "dart_external_family.json")
    results = json.load(open(path))
    counts = {"target_seed_banks": 0, "AUROC": 0, "thresholds": 0,
              "FPR": 0, "TPR_points": 0}
    max_error = {"AUROC": 0.0, "threshold": 0.0, "FPR": 0.0, "TPR": 0.0}
    for target in external.TARGETS:
        for seed in external.SEEDS:
            banks = results["folds"][target]["seeds"][str(seed)]["banks"]
            for bank, entry in banks.items():
                saved = np.load(os.path.join(
                    external.OUT, external.score_prefix(target, seed, bank) + ".npz"))
                assert tuple(saved["models"]) == tuple(bank.split("+"))
                y = saved["y"]
                assert len(y) == external.N_TEST_REAL + external.N_TEST_FAKE
                for method in external.FEATURES + ("prada",):
                    test = saved[f"{method}_test"]
                    cal = saved[f"{method}_cal"]
                    threshold = conformal_tau(cal)
                    recomputed = metric(y, test, external.N_TEST_REAL, threshold)
                    archived = entry[method]
                    errors = {
                        "AUROC": abs(recomputed["AUROC"] - archived["AUROC"]),
                        "threshold": abs(threshold - float(saved[f"{method}_threshold"])),
                        "FPR": abs(recomputed["FPR"] - archived["FPR"]),
                    }
                    for level in ("0.1", "1", "5", "10"):
                        key = f"TPR@{level}%FPR"
                        error = abs(recomputed[key] - archived[key])
                        max_error["TPR"] = max(max_error["TPR"], error)
                        counts["TPR_points"] += 1
                    for key, error in errors.items():
                        max_error[key] = max(max_error[key], error)
                    counts["AUROC"] += 1
                    counts["thresholds"] += 1
                    counts["FPR"] += 1
                counts["target_seed_banks"] += 1
    assert all(value < 1e-12 for value in max_error.values()), max_error
    report = {"counts": counts, "max_absolute_error": max_error, "passed": True,
              "source": os.path.relpath(path, PROJ)}
    out = os.path.join(external.OUT, "external_family_audit.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
