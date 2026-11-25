import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.s3r_p0 import (moments, moments_many, normalized_weights,  # noqa: E402
                             source_gaussian, weighted_gaussian)


def test_source_covariance_uses_law_of_total_variance():
    r = np.array([[[0.0], [2.0]]])
    w = np.ones((1, 2))
    mu, cov, ess = source_gaussian(r, w)
    np.testing.assert_allclose(mu, [1.0])
    np.testing.assert_allclose(cov, [[1.0]])
    np.testing.assert_allclose(ess, 1.0)


def test_duplicate_view_set_does_not_change_source_weighted_moments():
    r = np.array([[[0.0], [2.0]], [[10.0], [14.0]]])
    w = np.ones((2, 2))
    mu1, cov1, ess1 = source_gaussian(r, w)
    r_dup = np.tile(r, (1, 2, 1))
    w_dup = np.ones((2, 4))
    mu2, cov2, ess2 = source_gaussian(r_dup, w_dup)
    np.testing.assert_allclose(mu1, mu2)
    np.testing.assert_allclose(cov1, cov2)
    np.testing.assert_allclose(ess1, ess2)


def test_naive_view_ess_inflates_but_source_ess_does_not():
    r = np.array([[[0.0], [2.0]], [[10.0], [14.0]]])
    w = np.ones((2, 2))
    _, _, source_ess1 = moments(r, w, "mixed", 0)
    _, _, view_ess1 = weighted_gaussian(r.reshape(-1, 1), w.reshape(-1))
    r_dup = np.concatenate([r, r[:, 1:2]], axis=1)
    w_dup = np.ones((2, 3))
    _, _, source_ess2 = moments(r_dup, w_dup, "mixed", 0)
    _, _, view_ess2 = weighted_gaussian(r_dup.reshape(-1, 1), w_dup.reshape(-1))
    np.testing.assert_allclose([source_ess1, source_ess2], 2.0)
    np.testing.assert_allclose([view_ess1, view_ess2], [4.0, 6.0])


def test_s3r_endpoints_match_same_and_source_mixed():
    rng = np.random.RandomState(0)
    r = rng.normal(size=(20, 3, 4))
    w = rng.uniform(0.5, 1.5, size=(20, 3))
    same = moments(r, w, "same", 1)
    source = moments(r, w, "source_mixed", 1)
    gamma0 = moments(r, w, "s3r", 1, gamma=0.0)
    gamma1 = moments(r, w, "s3r", 1, gamma=1.0)
    np.testing.assert_allclose(gamma0[0], same[0])
    np.testing.assert_allclose(gamma0[1], same[1])
    np.testing.assert_allclose(gamma1[0], source[0])
    np.testing.assert_allclose(gamma1[1], source[1])


def test_moments_many_preserves_source_view_layout():
    r = np.array([[[0.0], [10.0], [20.0]],
                  [[100.0], [110.0], [120.0]]])
    c = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    mu, _, _ = moments_many(r, c, np.zeros((1, 1)), np.zeros(1), 0,
                            "source_mixed")
    weights = normalized_weights(c.reshape(-1, 1), np.zeros(1)).reshape(1, 2, 3)
    expected, _, _ = moments(r, weights[0], "source_mixed", 0)
    np.testing.assert_allclose(mu[0], expected)


def test_moments_many_same_quality_uses_view_axis():
    rng = np.random.RandomState(2)
    r = rng.normal(size=(10, 3, 4))
    c = rng.normal(size=(10, 3))
    rq = rng.normal(size=(2, 4))
    cq = rng.normal(size=2)
    mu, _, _ = moments_many(r, c, rq, cq, 1, "same")
    weights = np.zeros((2, 10, 3))
    weights[:, :, 1] = normalized_weights(c[:, 1], cq)
    expected, _, _ = moments(r, weights[0], "same", 1)
    np.testing.assert_allclose(mu[0], expected, atol=1e-6, rtol=1e-5)


def test_moments_many_s3r_endpoints_include_component_bandwidths():
    rng = np.random.RandomState(3)
    r = rng.normal(size=(20, 3, 4))
    c = rng.normal(size=(20, 3))
    rq = rng.normal(size=(2, 4))
    cq = rng.normal(size=2)
    for gamma, baseline in ((0.0, "same"), (1.0, "source_mixed")):
        mu, cov, ess = moments_many(r, c, rq, cq, 1, "s3r", gamma)
        expected_mu, expected_cov, expected_ess = moments_many(
            r, c, rq, cq, 1, baseline)
        np.testing.assert_allclose(mu, expected_mu, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(ess, expected_ess, atol=1e-6, rtol=1e-5)
