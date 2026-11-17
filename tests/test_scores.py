"""Score contract 单元测试 (deep-research-report-v2 §4.4)。

覆盖: higher_is_fake 方向、logit/anomaly score 处理、NaN 检测。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.evaluator import _score_from_output  # noqa: E402


class _FakeMethod:
    def __init__(self, score_type="logit", higher_is_fake=True):
        self.score_type = score_type
        self.higher_is_fake = higher_is_fake


def test_higher_is_fake_true_no_flip():
    raw = torch.tensor([0.0, 2.0, -2.0])
    out = _score_from_output(raw, _FakeMethod("logit", True))
    assert torch.allclose(out, torch.sigmoid(raw))


def test_higher_is_fake_false_flips():
    raw = torch.tensor([0.5, -0.5])
    out = _score_from_output(raw, _FakeMethod("anomaly", False))
    assert torch.allclose(out, torch.tensor([-0.5, 0.5]))


def test_anomaly_score_no_sigmoid():
    raw = torch.tensor([-9.9, 3.3])
    out = _score_from_output(raw, _FakeMethod("anomaly", True))
    assert torch.allclose(out, raw)


def test_logit_sigmoid_monotonic():
    raw = torch.linspace(-5, 5, 50)
    out = _score_from_output(raw, _FakeMethod("logit", True))
    assert torch.all(out[1:] >= out[:-1])


def test_nan_propagates_for_detection():
    raw = torch.tensor([1.0, float("nan")])
    out = _score_from_output(raw, _FakeMethod("anomaly", True))
    assert torch.isnan(out).sum().item() == 1


def test_default_method_is_logit():
    raw = torch.tensor([0.0])
    out = _score_from_output(raw, None)
    assert torch.allclose(out, torch.tensor([0.5]))


def test_scores_rank_matches_labels():
    """方向语义: fake 样本应得到更高分。"""
    labels = np.array([0, 0, 1, 1])
    raw_real = torch.tensor([-2.0, -1.0])
    raw_fake = torch.tensor([1.0, 2.0])
    out = _score_from_output(torch.cat([raw_real, raw_fake]),
                             _FakeMethod("logit", True)).numpy()
    assert (out[labels == 1] > out[labels == 0].max()).all()
