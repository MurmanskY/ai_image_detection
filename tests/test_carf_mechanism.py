import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.carf_mechanism_p0 import (controlled_image, match_pairs,  # noqa: E402
                                        weighted_mean)
from analysis.dct_forensics import dct24_patch  # noqa: E402


def test_dct_representation_is_finite_and_deterministic():
    rng = np.random.RandomState(0)
    patch = rng.uniform(0, 1, size=(128, 128, 3)).astype(np.float32)
    a, b = dct24_patch(patch), dct24_patch(patch)
    assert a.shape == (24,)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_dct_constant_image_has_no_nan():
    patch = np.full((128, 128, 3), 0.5, dtype=np.float32)
    assert np.isfinite(dct24_patch(patch)).all()


def test_controlled_view_has_fixed_resolution_and_encoded_size(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (80, 120), color=(100, 120, 140)).save(path)
    img, size = controlled_image(str(path))
    assert img.size == (256, 256)
    assert size > 0


def test_matching_identical_conditions_is_perfect():
    c = np.array([[0.0] * 8, [1.0] * 8], dtype=np.float32)
    ri, fi, distances = match_pairs(c, c, c)
    np.testing.assert_array_equal(ri, [0, 1])
    np.testing.assert_array_equal(fi, [0, 1])
    np.testing.assert_allclose(distances, 0)


def test_weighted_mean_constant_condition_uses_all_reference_samples():
    rng = np.random.RandomState(1)
    r = rng.normal(size=(100, 24)).astype(np.float32)
    c = np.ones((100, 1), dtype=np.float32)
    means, ess = weighted_mean(r, c, r[:2], c[:2])
    np.testing.assert_allclose(means, np.repeat(r.mean(0)[None], 2, axis=0),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ess, 100, rtol=1e-5)
