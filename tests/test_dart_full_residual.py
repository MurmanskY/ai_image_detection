import numpy as np

from analysis import dart_full_residual_comparison as comparison
from analysis.dart_interaction import CycleResidualizer, parameter_count, transform_images


def test_full_cycle_keeps_all_features_and_parameter_budget():
    image = np.zeros((2, 20), dtype=np.float32)
    image[:, 18] = [1.0, 2.0]
    image[:, 19] = [3.0, 4.0]
    residualizer = CycleResidualizer().fit([image], [image])
    full = transform_images([image], "full")[0]
    enhanced = transform_images([image], "full_cycle", residualizer)[0]
    assert enhanced.shape == full.shape == (2, 25)
    assert np.array_equal(enhanced[:, :20], full[:, :20])
    assert np.isfinite(enhanced[:, 20].astype(np.float32)).all()
    assert parameter_count() == 1394
    assert comparison.METHODS == ("full", "full_cycle")
