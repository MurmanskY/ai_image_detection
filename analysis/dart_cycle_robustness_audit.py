"""独立复算 residual 跨 codec 和外部真实校准结果。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_cycle_robustness as robustness  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric, wilson_interval  # noqa: E402


def main():
    results = json.load(open(os.path.join(robustness.OUT, "dart_cycle_robustness.json")))
    counts = {"AUROC": 0, "thresholds": 0, "TPR_points": 0, "external_FPR": 0}
    max_error = 0.0
    for seed in robustness.SEEDS:
        seed_result = results["seeds"][str(seed)]
        for aggregation in robustness.AGGREGATIONS:
            for component in robustness.COMPONENTS:
                key = f"{aggregation}::{component}"
                external = np.load(os.path.join(
                    robustness.OUT, f"external_{aggregation}_{component}_seed{seed}.npz"))
                archived = seed_result["external_FPR"][key]
                tau = conformal_tau(external["cal"])
                fp = int((external["external"] > tau).sum())
                assert np.isclose(tau, external["threshold"])
                assert np.isclose(fp / len(external["external"]), archived["FPR"])
                assert np.allclose(wilson_interval(fp, len(external["external"])), archived["CI95"])
                counts["external_FPR"] += 1
                for codec in robustness.CODECS:
                    archived_codec = seed_result["codecs"][codec][key]
                    aucs = []
                    for gen in robustness.c2i.FAKE_GENS:
                        saved = np.load(os.path.join(
                            robustness.OUT, f"scores_{codec}_{aggregation}_{component}_seed{seed}_{gen}.npz"))
                        recomputed = metric(saved["y"], saved["test"], 300, tau)
                        assert np.isclose(tau, saved["threshold"])
                        archived = archived_codec["per_generator"][gen]
                        assert recomputed["FPR"] == archived["FPR"]
                        max_error = max(max_error, abs(recomputed["AUROC"] - archived["AUROC"]))
                        for level in ("0.1", "1", "5", "10"):
                            name = f"TPR@{level}%FPR"
                            assert np.isclose(recomputed[name], archived[name])
                            counts["TPR_points"] += 1
                        counts["AUROC"] += 1
                        counts["thresholds"] += 1
                        aucs.append(recomputed["AUROC"])
                    assert np.isclose(np.mean(aucs), archived_codec["macro_AUROC"])
    assert max_error < 1e-12
    report = {"counts": counts, "max_AUROC_error": max_error, "passed": True}
    with open(os.path.join(robustness.OUT, "cycle_robustness_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
