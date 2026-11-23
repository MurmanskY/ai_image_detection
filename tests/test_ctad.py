import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.ctad_affinity import (  # noqa: E402
    affinity_score,
    conformal_upper_tail_p,
    lower_tail_surprisal,
)


def test_lower_tail_surprisal_is_monotone_and_add_one_smoothed():
    reference = [np.array([1.0, 2.0, 3.0])]
    scores = lower_tail_surprisal(np.array([[0.0], [1.0], [3.0], [4.0]]),
                                  reference)
    assert np.all(np.diff(scores[:, 0]) <= 0)
    assert np.isfinite(scores).all()
    assert scores[0, 0] == -np.log(1 / 4)
    assert scores[-1, 0] == 0


def test_max_affinity_preserves_sparse_extreme_channel():
    reference = [np.arange(10), np.arange(10)]
    q_values = np.array([[0, 5], [5, 0]])
    scores = affinity_score(q_values, reference)
    assert scores[0] == scores[1]
    assert scores[0] > affinity_score(np.array([[5, 5]]), reference)[0]


def test_conformal_p_values_are_finite_sample_and_tie_conservative():
    calibration = np.array([0.0, 1.0, 2.0, 3.0])
    p_values = conformal_upper_tail_p(calibration, np.array([4.0, 2.0, 0.0]))
    np.testing.assert_allclose(p_values, [0.2, 0.6, 1.0])


@pytest.mark.parametrize(
    "values, references",
    [
        (np.ones(3), [np.ones(3)]),
        (np.ones((2, 2)), [np.ones(2)]),
        (np.ones((2, 1)), [np.array([])]),
    ],
)
def test_ctad_rejects_malformed_reference_inputs(values, references):
    with pytest.raises(ValueError):
        lower_tail_surprisal(values, references)


def test_conformal_rejects_empty_calibration():
    with pytest.raises(ValueError):
        conformal_upper_tail_p(np.array([]), np.array([1.0]))
