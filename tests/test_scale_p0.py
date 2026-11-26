import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.scale_p0 import aggregate_scales, two_tail_surprisal  # noqa: E402


def test_two_tail_surprisal_has_small_p_for_extreme_values():
    reference = np.linspace(-1, 1, 101).reshape(-1, 1)
    p, surprisal = two_tail_surprisal(reference, np.array([[0.0], [10.0]]))
    assert p[0, 0] > 0.5
    assert p[1, 0] < 0.03
    assert surprisal[1, 0] > surprisal[0, 0]


def test_two_tail_probability_is_bounded():
    rng = np.random.RandomState(0)
    reference = rng.normal(size=(100, 3))
    query = rng.normal(size=(20, 3))
    p, surprisal = two_tail_surprisal(reference, query)
    assert np.all((p >= 1 / 101) & (p <= 1))
    assert np.isfinite(surprisal).all()


def test_fisher_aggregation_uses_scale_mean_and_max():
    p = np.full((2, 4), 0.5, dtype=np.float32)
    surprisal = -np.log(p)
    surprisal[:, 2:] = -np.log(np.full((2, 2), 0.1))
    score = aggregate_scales(p, surprisal, [2, 2], "fisher")
    np.testing.assert_allclose(score, -np.log(0.1))


def test_hc_aggregation_is_finite():
    rng = np.random.RandomState(1)
    p = rng.uniform(0.001, 1, size=(5, 20)).astype(np.float32)
    score = aggregate_scales(p, -np.log(p), [10, 10], "hc")
    assert score.shape == (5,)
    assert np.isfinite(score).all()
