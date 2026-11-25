import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.forensic_bank import KERNELS, highpass24_patch  # noqa: E402
from analysis.carf_protocol_p1 import fit_u  # noqa: E402
from analysis.sacr_p0 import U_VIEWS  # noqa: E402


def test_highpass_filters_have_zero_dc_response():
    for kernel in KERNELS:
        assert abs(float(kernel.sum())) < 1e-7


def test_highpass_representation_is_finite_and_deterministic():
    rng = np.random.RandomState(0)
    patch = rng.uniform(0, 1, size=(128, 128, 3)).astype(np.float32)
    a = highpass24_patch(patch)
    b = highpass24_patch(patch)
    assert a.shape == (24,)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_highpass_constant_patch_is_finite():
    patch = np.full((128, 128, 3), 0.5, dtype=np.float32)
    features = highpass24_patch(patch)
    assert features.shape == (24,)
    assert np.isfinite(features).all()


def test_intervention_subspace_recovers_shared_direction():
    rng = np.random.RandomState(1)
    direction = np.zeros(24, dtype=np.float32)
    direction[0] = 1
    base = rng.normal(size=(100, 24)).astype(np.float32)
    real = {"qf95": base}
    for i, view in enumerate(U_VIEWS, start=1):
        real[view] = base + i * direction
    u, _ = fit_u(real, np.arange(100))
    assert u.shape == (24, 6)
    assert abs(u[0, 0]) > 0.999
