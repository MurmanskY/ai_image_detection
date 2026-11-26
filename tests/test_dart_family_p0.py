import numpy as np

from analysis import dart_family_p0 as family


def test_quantization_alignment_modes():
    lengths = [2, 2]
    direct = np.array([[1.0, 2.0]])
    assert np.array_equal(family.quant_to_scales(direct, 4, lengths), direct)
    token = np.array([[1.0, 3.0, 5.0, 7.0]])
    assert np.array_equal(family.quant_to_scales(token, 4, lengths),
                          np.array([[2.0, 6.0]]))
    global_quant = np.array([[2.0, 4.0, 6.0]])
    assert np.array_equal(family.quant_to_scales(global_quant, 4, lengths),
                          np.array([[4.0, 4.0]]))


def test_family_exclusion_controls_both_training_and_bank(monkeypatch):
    heldout = "infinity_2b"
    assert heldout not in family.allowed_generators(heldout)
    monkeypatch.setattr(family, "BANK_EXCLUDE", True)
    assert heldout not in family.included_models(heldout)
    monkeypatch.setattr(family, "BANK_EXCLUDE", False)
    assert heldout in family.included_models(heldout)


def test_t2i_prada_uses_the_same_fake_sample_count(monkeypatch):
    features = {"cond_gt_llh_BL": np.zeros((1000, 2)),
                "uncond_gt_llh_BL": np.ones((1000, 2))}
    monkeypatch.setattr(family, "load_features", lambda ds, model: features)
    monkeypatch.setattr(family, "prada_score", lambda feat, model: feat)
    result = family.prada_t2i("infinity_2b", "janus_1b")
    assert len(result["cond_gt_llh_BL"]) == family.c2i.N_IMAGES
    assert len(result["uncond_gt_llh_BL"]) == family.c2i.N_IMAGES
