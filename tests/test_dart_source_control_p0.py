import numpy as np

from analysis import dart_external_family_p0 as external
from analysis import dart_source_control_p0 as source


def test_source_assignments_and_target_exclusion():
    assert source.real_sources("current") == ("arf",)
    assert source.real_sources("darg_transfer") == ("darg",)
    assert source.real_sources("balanced") == ("arf", "darg")
    for condition in source.CONDITIONS:
        for target in external.TARGETS:
            fake = source.training_fake_sources(condition, target)
            assert target not in fake
            if condition == "balanced":
                assert set(fake) == set(external.DARG_GENS) - {
                    external.TARGET_TO_DARG.get(target)}
                assert all(sources == ("arf", "darg") for sources in fake.values())


def test_reverse_evaluation_only_for_darg_overlaps():
    assert source.eval_sources("arf_same", "Infinity") == ("arf", "darg")
    assert source.eval_sources("balanced", "VAR") == ("arf",)
    assert source.eval_sources("darg_transfer", "Infinity") == ("arf",)
    assert source.target_dataset("Infinity", "darg") == "infinity_2b"


def test_normalization_uses_only_real_reference_and_preserves_target(monkeypatch):
    arf, darg = {}, {}
    for model in external.MODELS[:1]:
        arf["real", model] = [np.full((2, 20), float(i), dtype=np.float32)
                              for i in range(1000)]
        darg["real", model] = [np.full((2, 20), 10.0 + i, dtype=np.float32)
                               for i in range(1000)]
        arf["Infinity", model] = [np.full((2, 20), 100.0, dtype=np.float32)] * 250
        darg["infinity_2b", model] = [np.full((2, 20), 200.0, dtype=np.float32)] * 250
    ref = np.arange(300)
    # No fake training entries are needed to normalize the held-out target.
    result = source.normalize_model(arf, darg, external.MODELS[0], ("arf", "darg"),
                                    ref, {}, "Infinity")
    target = np.asarray(result["target", "arf"][0], dtype=np.float32)
    reference = np.concatenate([np.asarray(arf["real", external.MODELS[0]])[ref],
                                np.asarray(darg["real", external.MODELS[0]])[ref]])
    expected = (100.0 - reference.mean(axis=0)) / (reference.std(axis=0) + 1e-6)
    assert np.all(np.isfinite(target))
    assert np.allclose(target, expected)
    assert not np.shares_memory(target, arf["Infinity", external.MODELS[0]][0])


def test_darg_prada_real_uses_full_1000_images(monkeypatch):
    features = {"cond_gt_llh_BL": np.zeros((1000, 2)),
                "uncond_gt_llh_BL": np.ones((1000, 2))}
    monkeypatch.setattr(source.family, "load_features", lambda ds, model: features)
    monkeypatch.setattr(source.family, "prada_score", lambda feat, model: feat)
    result = source.prada_scores("darg", "real", external.MODELS[0])
    assert len(result["cond_gt_llh_BL"]) == 1000


def test_balanced_maps_darg_generator_names_to_arf_families():
    model = external.MODELS[0]
    arf, darg = {}, {}
    for packed, names in ((arf, ("real", "Janus_Pro", "Infinity")),
                          (darg, ("real", "janus_1b", "infinity_2b"))):
        for name in names:
            count = 1000 if name == "real" else 250
            packed[name, model] = [np.ones((1, 20), dtype=np.float32)] * count
    result = source.normalize_model(
        arf, darg, model, ("arf", "darg"), np.arange(300),
        {"janus_1b": ("arf", "darg")}, "Infinity")
    assert ("fake", "arf", "janus_1b") in result
    assert ("fake", "darg", "janus_1b") in result
