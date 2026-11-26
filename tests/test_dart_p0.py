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
