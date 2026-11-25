import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.reference_pool_analysis import (bootstrap_fpr, conformal_details,  # noqa: E402
                                               ess_values)


def test_source_ess_ignores_full_view_set_duplication():
    w = np.array([[1.0, 2.0], [3.0, 1.0]])
    view1, source1 = ess_values(w)
    view2, source2 = ess_values(np.tile(w, (1, 2)))
    assert view2 > view1
    np.testing.assert_allclose(source1, source2)


def test_conformal_details_uses_finite_sample_rank():
    scores = np.arange(100, dtype=np.float32)
    ess = np.full(100, 100.0)
    details = conformal_details(scores, ess, alpha=0.05)
    assert details["n"] == 100
    assert details["rank"] == 96
    assert details["tau"] == 95.0
    assert details["next_gap"] == 1.0


def test_bootstrap_fpr_is_deterministic_and_tracks_point_estimate():
    groups = {"a": np.array([True] * 5 + [False] * 95),
              "b": np.array([True] * 10 + [False] * 90)}
    per1, macro1 = bootstrap_fpr(groups, n_boot=100, seed=0)
    per2, macro2 = bootstrap_fpr(groups, n_boot=100, seed=0)
    assert per1 == per2 and macro1 == macro2
    assert per1["a"]["CI95"][0] <= per1["a"]["FPR"] <= per1["a"]["CI95"][1]
    assert abs(macro1["FPR"] - 0.075) < 1e-12
