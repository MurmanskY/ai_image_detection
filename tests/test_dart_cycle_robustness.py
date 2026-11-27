import numpy as np

from analysis import dart_cycle_robustness as robustness
from analysis import dart_interaction_p0 as interaction


def test_component_values_decompose_without_fake_training():
    image = np.zeros((2, 20), dtype=np.float32)
    image[:, 19] = [1.0, 2.0]
    residualizer = interaction.CycleResidualizer().fit([image], [image])
    rec = robustness.component_value(residualizer, image, "reconstruction")
    residual = robustness.component_value(residualizer, image, "cycle_residual")
    expected_rec = image[:, 19].mean() / residualizer.residual_scale
    assert np.isclose(rec, expected_rec)
    assert np.isfinite(residual)
    assert np.isclose(rec - residual,
                      (image[:, 19].mean() - (image[:, 19] - residualizer.predict([image])).mean())
                      / residualizer.residual_scale)
