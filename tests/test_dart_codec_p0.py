import numpy as np

from analysis import dart_c2i_p0 as c2i
from analysis import dart_codec_p0 as codec


def configure_real1000(monkeypatch):
    monkeypatch.setattr(c2i, "N_REAL", 1000)
    monkeypatch.setattr(c2i, "N_REF_REAL", 300)
    monkeypatch.setattr(c2i, "N_SELECT_REAL", 200)
    monkeypatch.setattr(c2i, "N_CAL_REAL", 200)
    monkeypatch.setattr(c2i, "N_TEST_REAL", 300)


def test_codec_positions_match_original_test_split(monkeypatch):
    configure_real1000(monkeypatch)
    positions = codec.codec_positions("real")
    expected = codec.test_positions("real", 0)
    assert len(positions) == len(expected) == 300
    assert np.array_equal(np.sort(expected), positions)
    rows = codec.position_rows("real", expected)
    assert np.array_equal(positions[rows], expected)
    assert len(codec.codec_positions("var_d16")) == 100


def test_codec_real_test_is_disjoint_from_train_and_calibration(monkeypatch):
    configure_real1000(monkeypatch)
    perm = np.random.RandomState(0).permutation(c2i.N_REAL)
    test = set(codec.test_positions("real", 0))
    assert not test & set(perm[:700])
    assert len(codec.test_positions("var_d16", 0)) == 100


def test_codec_shards_restore_global_positions(monkeypatch):
    configure_real1000(monkeypatch)
    expected = codec.codec_positions("real")
    monkeypatch.setattr(codec, "SHARD_COUNT", 2)
    monkeypatch.setattr(codec, "SHARD_INDEX", 0)
    even = codec.extraction_positions("real")
    monkeypatch.setattr(codec, "SHARD_INDEX", 1)
    odd = codec.extraction_positions("real")
    restored = np.empty(300, dtype=np.int64)
    restored[::2], restored[1::2] = even, odd
    assert np.array_equal(restored, expected)
