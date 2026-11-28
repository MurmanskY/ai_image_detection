import numpy as np

from analysis import dart_external_calibration as calibration
from analysis.carf_p0 import conformal_tau


def test_external_calibration_is_disjoint_and_only_changes_threshold():
    scores = np.linspace(-1, 1, 300)
    perm = np.random.RandomState(5000).permutation(300)
    cal_idx, test_idx = perm[:calibration.N_CAL], perm[calibration.N_CAL:]
    assert not set(cal_idx) & set(test_idx)
    assert len(cal_idx) == len(test_idx) == 150
    tau = conformal_tau(scores[cal_idx])
    assert 0 <= (scores[test_idx] > tau).mean() <= 1
    assert np.array_equal(scores, np.linspace(-1, 1, 300))
