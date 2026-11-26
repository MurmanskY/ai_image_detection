import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.dart_p0 import ScaleDeepSet, batches, scale_tensor  # noqa: E402


def test_scale_tensor_has_quantiles_and_metadata():
    diff = np.arange(100, dtype=np.float32)[None]
    output = scale_tensor(diff, [50, 50])
    assert len(output) == 1
    assert output[0].shape == (2, 18)
    assert output[0][0, -2] == output[0][1, -2]
    assert output[0][0, 0] < output[0][1, 0]
    assert output[0][0, -1] == 0.0 and output[0][1, -1] == 1.0


def test_deepset_handles_variable_scale_counts_and_masks():
    torch.manual_seed(0)
    model = ScaleDeepSet().eval()
    images = [np.random.randn(2, 18).astype(np.float32),
              np.random.randn(5, 18).astype(np.float32)]
    labels = np.array([0, 1])
    x, mask, _ = next(batches(images, labels, 2))
    with torch.no_grad():
        output = model(x, mask)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_c2i_cache_does_not_expand_fake_to_real_count(monkeypatch):
    import analysis.dart_c2i_p0 as c2i

    monkeypatch.setattr(c2i, "N_REAL", 1000)
    assert c2i.sample_count("real") == 1000
    assert c2i.sample_count("var_d16") == c2i.N_IMAGES
    assert c2i.cycle_batch_size("var_d20") == 1
    assert c2i.cycle_batch_size("rar_l") == 1
    assert c2i.cycle_batch_size("llamagen_l256") == c2i.BATCH_SIZE
    assert c2i.rec_batch_size("var_d20") == c2i.BATCH_SIZE
    assert c2i.rec_batch_size("rar_l") == 1


def test_c2i_shards_preserve_original_sample_order(monkeypatch):
    import analysis.dart_c2i_p0 as c2i

    monkeypatch.setattr(c2i, "N_REAL", 1000)
    monkeypatch.setattr(c2i, "SHARD_COUNT", 1)
    expected = c2i.selected_indices(10000, "real")
    monkeypatch.setattr(c2i, "SHARD_COUNT", 2)
    monkeypatch.setattr(c2i, "SHARD_INDEX", 0)
    even = c2i.selected_indices(10000, "real")
    monkeypatch.setattr(c2i, "SHARD_INDEX", 1)
    odd = c2i.selected_indices(10000, "real")
    restored = np.empty(1000, dtype=even.dtype)
    restored[::2], restored[1::2] = even, odd
    assert np.array_equal(restored, expected)


def test_c2i_pack_aligns_oversized_likelihood_cache(monkeypatch):
    import analysis.dart_c2i_p0 as c2i

    features = {
        "cond_gt_llh_BL": torch.ones(1000, 2),
        "uncond_gt_llh_BL": torch.zeros(1000, 2),
        "quant_err": torch.ones(250, 2),
        "rec_err": torch.ones(250),
        "scale_lengths": [2],
    }
    monkeypatch.setattr(c2i, "load_c2i", lambda ds, model: features)
    packed = c2i.pack("var_d16", "var_d20")
    assert len(packed) == c2i.N_IMAGES
    assert all(x.shape == (1, 20) for x in packed)
    monkeypatch.setattr(c2i, "prada_score", lambda feat, model: feat)
    prada_features = c2i.prada_c2i("var_d16", "var_d20")
    assert len(prada_features["cond_gt_llh_BL"]) == c2i.N_IMAGES
    assert len(prada_features["uncond_gt_llh_BL"]) == c2i.N_IMAGES


def test_c2i_fake_train_validation_test_are_disjoint():
    from analysis.dart_c2i_p0 import fake_split

    train, validation, test = fake_split(0)
    assert len(train) == 100 and len(validation) == 50 and len(test) == 100
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)


def test_c2i_real_selection_calibration_and_test_are_disjoint(monkeypatch):
    import analysis.dart_c2i_p0 as c2i

    monkeypatch.setattr(c2i, "N_REAL", 1000)
    monkeypatch.setattr(c2i, "N_REF_REAL", 300)
    monkeypatch.setattr(c2i, "N_SELECT_REAL", 200)
    monkeypatch.setattr(c2i, "N_CAL_REAL", 200)
    parts = c2i.real_split(0)
    assert [len(x) for x in parts] == [300, 200, 200, 300]
    for i, first in enumerate(parts):
        for second in parts[i + 1:]:
            assert not set(first) & set(second)


def test_c2i_metric_has_fpr_interval_and_all_working_points():
    from analysis.dart_c2i_p0 import metric, wilson_interval

    y = np.array([0, 0, 1, 1])
    score = np.array([0.0, 0.2, 0.8, 1.0])
    result = metric(y, score, 2, 0.5)
    assert result["FPR"] == 0.0
    assert result["FPR_CI95"][0] == 0.0
    assert wilson_interval(10, 200)[0] < 0.05 < wilson_interval(10, 200)[1]
    for level in ("0.1", "1", "5", "10"):
        assert f"TPR@{level}%FPR" in result


def test_dart_feature_combinations_keep_expected_columns():
    from analysis.dart_p0 import select_feature_mode

    images = np.asarray([np.arange(40, dtype=np.float32).reshape(2, 20)], dtype=object)
    expected = {"likelihood": 18, "quantization": 3, "reconstruction": 3,
                "likelihood_quantization": 19, "likelihood_reconstruction": 19,
                "quantization_reconstruction": 4, "full": 20,
                "full_no_metadata": 18}
    for mode, width in expected.items():
        selected = select_feature_mode(images, mode)
        assert selected[0].shape == (2, width)
    assert np.array_equal(select_feature_mode(images, "likelihood")[0][:, -1],
                          images[0][:, 17])
    assert np.array_equal(select_feature_mode(images, "reconstruction")[0][:, -1],
                          images[0][:, 19])
    assert np.array_equal(select_feature_mode(images, "quantization")[0],
                          images[0][:, [16, 17, 18]])


def test_dart_pooling_variants_have_identical_parameter_counts():
    models = [ScaleDeepSet(pooling=pooling).eval()
              for pooling in ("attention", "mean", "sum")]
    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1
    x = torch.randn(2, 3, 18)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    with torch.no_grad():
        outputs = [model(x, mask) for model in models]
    assert all(output.shape == (2,) and torch.isfinite(output).all()
               for output in outputs)


def test_fixed_test_split_never_reuses_test_images(monkeypatch):
    import analysis.dart_c2i_p0 as c2i

    monkeypatch.setattr(c2i, "N_REAL", 1000)
    monkeypatch.setattr(c2i, "N_REF_REAL", 300)
    monkeypatch.setattr(c2i, "N_SELECT_REAL", 200)
    monkeypatch.setattr(c2i, "N_CAL_REAL", 200)
    splits = [c2i.fixed_test_split(seed) for seed in range(3)]
    assert np.array_equal(splits[0][3], splits[1][3])
    assert np.array_equal(splits[0][6], splits[2][6])
    for parts in splits:
        ref, select, cal, test, train, val, fake_test = parts
        assert not set(test) & (set(ref) | set(select) | set(cal))
        assert not set(fake_test) & (set(train) | set(val))
        assert not set(train) & set(val)
