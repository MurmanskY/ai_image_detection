"""独立复算外部 calibration/test 实验。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_calibration as calibration  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    results = json.load(open(os.path.join(calibration.OUT, "dart_external_calibration.json")))
    counts = {"entries": 0, "FPR": 0, "AUROC": 0}
    for key, entries in results["entries"].items():
        seed, aggregation, component = key.split("::")
        seed = int(seed)
        for entry in entries:
            saved = np.load(calibration.result_path(seed, aggregation, component, entry["split"]))
            original_tau = conformal_tau(saved["original_cal"])
            external_tau = conformal_tau(saved["external"][saved["cal_idx"]])
            assert np.isclose(original_tau, entry["original_threshold"])
            assert np.isclose(external_tau, entry["external_threshold"])
            test_real = saved["external"][saved["test_idx"]]
            assert np.isclose((test_real > original_tau).mean(), entry["original_FPR"])
            assert np.isclose((test_real > external_tau).mean(), entry["recalibrated_FPR"])
            counts["FPR"] += 2
            for gen in calibration.robustness.c2i.FAKE_GENS:
                fake_saved = np.load(os.path.join(
                    calibration.robustness.OUT,
                    f"scores_original_{aggregation}_{component}_seed{seed}_{gen}.npz"))
                fake = fake_saved["test"][-100:]
                scores = np.concatenate([test_real, fake])
                y = np.array([0] * len(test_real) + [1] * len(fake))
                for method, tau in (("original_threshold", original_tau),
                                    ("external_recalibration", external_tau)):
                    recomputed = metric(y, scores, len(test_real), tau)
                    archived = entry["per_generator"][gen][method]
                    assert np.isclose(recomputed["AUROC"], archived["AUROC"])
                    assert recomputed["FPR"] == archived["FPR"]
                    counts["AUROC"] += 1
            counts["entries"] += 1
    report = {"counts": counts, "passed": True}
    with open(os.path.join(calibration.OUT, "external_calibration_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
