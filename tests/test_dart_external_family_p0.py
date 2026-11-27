import numpy as np

from analysis import dart_external_family_p0 as external


def test_external_targets_are_seven_independent_families():
    assert len(external.TARGETS) == 7
    assert len(set(external.TARGETS)) == 7
    assert set(("Open_MAGVIT2", "RAR", "VAR")) <= set(external.TARGETS)
    assert external.TARGET_ATTRIBUTES["RAR"] == "random_order"
    assert external.TARGET_ATTRIBUTES["Open_MAGVIT2"] == "raster_subtoken"


def test_overlapping_family_is_removed_from_training_and_bank():
    for target, model in external.TARGET_TO_DARG.items():
        assert model not in external.available_models(target)
        assert model not in external.training_generators(target)
    for target in ("Open_MAGVIT2", "RAR", "VAR"):
        assert external.available_models(target) == external.MODELS
        assert external.training_generators(target) == external.DARG_GENS


def test_fixed_splits_are_disjoint_and_keep_test_fixed():
    previous = None
    for seed in external.SEEDS:
        ref, select, cal, test, train, val, fake_test = external.fixed_splits(seed)
        real_parts = (ref, select, cal, test)
        fake_parts = (train, val, fake_test)
        assert [len(x) for x in real_parts] == [300, 200, 200, 300]
        assert [len(x) for x in fake_parts] == [100, 50, 100]
        assert len(np.unique(np.concatenate(real_parts))) == external.N_REAL
        assert len(np.unique(np.concatenate(fake_parts))) == external.N_FAKE
        if previous is not None:
            assert np.array_equal(test, previous[0])
            assert np.array_equal(fake_test, previous[1])
        previous = (test, fake_test)


def test_coverage_contexts_are_shared_only_for_identical_training_information():
    contexts = external.evaluation_contexts(external.COVERAGE_SEEDS[0])
    assert len(contexts) == 43
    assert sum(len(targets) for targets in contexts.values()) == 73
    assert all(len(bank) >= 1 and len(train) >= 3 for train, bank in contexts)
    full_only = external.evaluation_contexts(next(
        seed for seed in external.SEEDS if seed not in external.COVERAGE_SEEDS))
    assert sum(len(targets) for targets in full_only.values()) == len(external.TARGETS)


def test_score_prefix_accepts_tuple_and_serialized_bank():
    bank = ("infinity_2b", "janus_1b")
    expected = "scores_Infinity_seed0_infinity_2b__janus_1b"
    assert external.score_prefix("Infinity", 0, bank) == expected
    assert external.score_prefix("Infinity", 0, external.bank_name(bank)) == expected
