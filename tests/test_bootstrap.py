"""Bootstrap CI 单元测试。"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.metrics import bootstrap_ci, compute_metrics  # noqa: E402


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 200 + [1] * 200)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)])
    lo, hi = bootstrap_ci(labels, scores, roc_auc_score, n_boot=200)
    auc = roc_auc_score(labels, scores)
    assert lo <= auc <= hi
    assert 0 <= lo < hi <= 1


def test_bootstrap_ci_deterministic():
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.linspace(0, 1, 100)
    a = bootstrap_ci(labels, scores, roc_auc_score, n_boot=100)
    b = bootstrap_ci(labels, scores, roc_auc_score, n_boot=100)
    assert a == b


def test_compute_metrics_n_boot():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    m = compute_metrics(labels, scores, n_boot=100)
    assert "auc_ci95" in m and "ap_ci95" in m and "tpr@0.01fpr_ci95" in m
    lo, hi = m["auc_ci95"]
    assert lo <= m["auc"] <= hi


def test_compute_metrics_no_boot_by_default():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(labels, scores)
    assert "auc_ci95" not in m
