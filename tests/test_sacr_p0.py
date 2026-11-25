import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.sacr_p0 import (blocks, gaussian_block, kernel_for_condition,  # noqa: E402
                              sacr_score, tau_by_complexity)


def test_signal_blocks_reconstruct_residual_without_projection_loss():
    rng = np.random.RandomState(0)
    u = np.linalg.qr(rng.normal(size=(24, 6)))[0]
    r = rng.normal(size=(12, 24)).astype(np.float32)
    parallel, perp = blocks(r, u)
    np.testing.assert_allclose(parallel @ u.T + perp, r, atol=1e-6)
    np.testing.assert_allclose((parallel @ u.T) @ u, parallel, atol=1e-6)


def test_constant_condition_is_unweighted():
    ref = np.ones((100, 2), dtype=np.float32)
    query = np.ones((3, 2), dtype=np.float32)
    weights, _ = kernel_for_condition(ref, query)
    np.testing.assert_array_equal(weights, np.ones((3, 100)))


def test_low_ess_abstains_instead_of_fallback():
    rng = np.random.RandomState(0)
    ref = rng.normal(size=(100, 3)).astype(np.float32)
    query = rng.normal(size=(2, 3)).astype(np.float32)
    weights = np.full((2, 100), 1e-30, dtype=np.float32)
    scores, ess = gaussian_block(ref, query, weights)
    assert np.isnan(scores).all()
    assert (ess < 50).all()


def test_sacr_joint_score_keeps_parallel_component():
    rng = np.random.RandomState(1)
    u = np.eye(24, 6, dtype=np.float32)
    ref = rng.normal(size=(100, 24)).astype(np.float32)
    query = rng.normal(size=(4, 24)).astype(np.float32)
    weights = np.ones((4, 100), dtype=np.float32)
    c, _ = sacr_score(ref, query, weights, u, "C", "gaussian", 1.0)
    d, _ = sacr_score(ref, query, weights, u, "D", "gaussian", 0.5)
    assert np.isfinite(c).all() and np.isfinite(d).all()
    assert not np.allclose(c, d)


def test_group_calibration_has_independent_thresholds():
    scores = np.concatenate([np.linspace(0, 1, 100), np.linspace(10, 20, 100)])
    complexity = np.zeros((200, 7), dtype=np.float32)
    complexity[100:, 3] = 1
    taus = tau_by_complexity(scores, complexity, 0.5)
    assert taus["0"] < 1
    assert 19 < taus["1"] < 20
