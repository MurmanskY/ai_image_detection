import numpy as np

from analysis import dart_cycle_validation as validation
from analysis import dart_external_family_p0 as external
from analysis import dart_interaction_p0 as interaction


def test_raw_cycle_supports_aggregation_and_fixed_calibration():
    image = np.zeros((2, 20), dtype=np.float32)
    image[:, 19] = [1.0, 2.0]
    residualizer = interaction.CycleResidualizer().fit([image], [image])
    data = {"m": {("real", "arf"): [image] * 1000,
                  ("target", "arf"): [image * 2] * 250}}
    ref, select, cal, test, _, _, fake_test = external.fixed_splits(0)
    for aggregation in validation.AGGREGATIONS:
        metrics, scores, calibration, _ = interaction.raw_cycle_scores(
            {"m": residualizer}, ("m",), data, "Infinity", "arf",
            cal, test, fake_test, aggregation)
        assert len(scores) == 400
        assert len(calibration) == 200
        assert np.isfinite(scores).all()
        assert metrics["FPR"] >= 0


def test_validation_evaluates_source_transfer_in_both_directions():
    assert validation.eval_sources("darg_transfer", "Infinity") == ("arf", "darg")
    assert validation.eval_sources("balanced", "Infinity") == ("arf", "darg")
    assert validation.eval_sources("balanced", "VAR") == ("arf",)
    assert validation.eval_sources("current", "Infinity") == ("arf",)
