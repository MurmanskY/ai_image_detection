import numpy as np

from analysis import dart_cycle_decomposition as decomposition
from analysis import dart_external_family_p0 as external
from analysis import dart_interaction_p0 as interaction


def test_cycle_component_matches_raw_cycle_score():
    image = np.zeros((2, 20), dtype=np.float32)
    image[:, 19] = [1.0, 2.0]
    residualizer = interaction.CycleResidualizer().fit([image], [image])
    data = {"m": {("real", "arf"): [image] * 1000,
                  ("target", "arf"): [image * 2] * 250}}
    ref, select, cal, test, _, _, fake_test = external.fixed_splits(0)
    for aggregation in decomposition.AGGREGATIONS:
        _, residual, _, _ = decomposition.component_scores(
            {"m": residualizer}, ("m",), data, "Infinity", "arf",
            cal, test, fake_test, aggregation, "cycle_residual")
        _, raw, _, _ = interaction.raw_cycle_scores(
            {"m": residualizer}, ("m",), data, "Infinity", "arf",
            cal, test, fake_test, aggregation)
        assert np.allclose(residual, raw)
