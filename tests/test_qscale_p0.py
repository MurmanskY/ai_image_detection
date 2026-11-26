import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.qscale_p0 import (LEVELS, knn_fit_score, maha_fit_score,  # noqa: E402
                                quantile_features)


def test_quantile_features_preserve_signed_distribution_shape():
    diff = np.array([[0.0] * 50 + [2.0] * 50])
    features = quantile_features(diff, [50, 50])
    assert features.shape == (1, 2 * len(LEVELS))
    assert features[0, 0] == 0.0
    assert features[0, len(LEVELS)] > 1.0


def test_maha_and_knn_scores_are_larger_for_outlier():
    rng = np.random.RandomState(0)
    reference = rng.normal(size=(100, 4)).astype(np.float32)
    queries = np.array([[0.0] * 4, [10.0] * 4], dtype=np.float32)
    maha = maha_fit_score(reference, [queries])[0]
    knn = knn_fit_score(reference, [queries])[0]
    assert maha[1] > maha[0]
    assert knn[1] > knn[0]
