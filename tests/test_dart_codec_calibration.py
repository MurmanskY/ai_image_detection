import importlib

from analysis import dart_codec_calibration as adaptation


def test_compressed_calibration_is_independent_from_test(monkeypatch):
    c2i = adaptation.c2i
    monkeypatch.setattr(c2i, "N_REAL", 1000)
    monkeypatch.setattr(c2i, "N_REF_REAL", 300)
    monkeypatch.setattr(c2i, "N_SELECT_REAL", 200)
    monkeypatch.setattr(c2i, "N_CAL_REAL", 200)
    parts = c2i.fixed_test_split(0)
    assert len(parts[2]) == 200 and len(parts[3]) == 300
    assert not set(parts[2]) & set(parts[3])
    assert not set(parts[2]) & (set(parts[0]) | set(parts[1]))


def test_full_adaptation_cache_and_positions_are_unambiguous(tmp_path, monkeypatch):
    c2i = adaptation.c2i
    for key, value in (("N_REAL", 1000), ("N_REF_REAL", 300),
                       ("N_SELECT_REAL", 200), ("N_CAL_REAL", 200)):
        monkeypatch.setattr(c2i, key, value)
    full = importlib.reload(importlib.import_module(
        "analysis.dart_codec_full_adaptation"))
    ref, select, cal, test, train, val, fake_test = full.SPLIT
    assert len(full.REAL_POSITIONS) == 500
    assert len(full.FAKE_POSITIONS) == 150
    assert not set(full.REAL_POSITIONS) & (set(cal) | set(test))
    assert not set(full.FAKE_POSITIONS) & set(fake_test)
    first = full.cache("real", "var_d20", full.REAL_POSITIONS)
    assert first == full.cache("real", "var_d20", full.REAL_POSITIONS)
    assert first != full.cache("var_d16", "var_d20", full.FAKE_POSITIONS)
