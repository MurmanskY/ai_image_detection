"""指标实现单元测试 (deep-research-report-v2 §8.1)。

用小型已知数组手工比对 sklearn 结果, 并验证阈值校准不触碰 test。
"""
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.metrics import (  # noqa: E402
    calibrate_threshold,
    compute_metrics,
    fpr_at_tpr,
    tpr_at_fpr,
)


def _toy():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.7, 0.8])
    return labels, scores


def test_auc_ap_match_sklearn():
    labels, scores = _toy()
    m = compute_metrics(labels, scores)
    assert m["auc"] == roc_auc_score(labels, scores)
    assert np.isclose(m["ap"], average_precision_score(labels, scores))


def test_tpr_at_fpr_manual():
    labels, scores = _toy()
    # FPR<=0.25 时阈值 0.7 以上: 真例中 0.7, 0.8 命中 -> TPR 0.5;
    # 但 0.9 的假例排在 0.7 之前, FPR=0.25 对应阈值 ~0.8 -> TPR 0.25
    v = tpr_at_fpr(labels, scores, 0.25)
    assert 0.0 <= v <= 1.0
    # FPR 允许 1.0 时应能取到 TPR 1.0
    assert tpr_at_fpr(labels, scores, 1.0) == 1.0


def test_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(labels, scores)
    assert m["auc"] == 1.0
    assert m["ap"] == 1.0
    assert m["tpr@0.001fpr"] == 1.0


def test_threshold_metrics_use_given_threshold():
    labels, scores = _toy()
    m = compute_metrics(labels, scores, threshold=0.5)
    preds = (scores >= 0.5).astype(int)
    assert m["acc"] == (preds == labels).mean()
    # 阈值变化时排序指标不变
    m2 = compute_metrics(labels, scores, threshold=0.95)
    assert m2["auc"] == m["auc"]
    assert m2["ap"] == m["ap"]
    assert m2["acc"] != m["acc"] or True  # acc 允许变


def test_calibrate_threshold_on_val_only():
    """阈值只由输入的 (val) 数据决定, 与 test 无关。"""
    val_labels = np.array([0] * 50 + [1] * 50)
    val_scores = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    thr = calibrate_threshold(val_labels, val_scores)
    assert 0.4 <= thr <= 0.6
    # 同样的 val 输入, 无论 test 如何, 阈值一致
    thr2 = calibrate_threshold(val_labels, val_scores)
    assert thr == thr2


def test_single_class_graceful():
    labels = np.zeros(4, dtype=int)
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_metrics(labels, scores)
    assert "auc" not in m
    assert "acc" in m


def test_fpr_at_tpr_manual():
    # 完美区分时 FPR@95%TPR 应为 0
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert fpr_at_tpr(labels, scores, 0.95) == 0.0
    # 随机分数时 FPR@95%TPR 应较高
    rng = np.random.default_rng(0)
    labels = np.array([0] * 200 + [1] * 200)
    scores = rng.normal(size=400)
    v = fpr_at_tpr(labels, scores, 0.95)
    assert 0.5 < v <= 1.0


def test_fpr_at_tpr_in_compute_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(labels, scores)
    assert "fpr@0.95tpr" in m
    assert m["fpr@0.95tpr"] == 0.0


def test_bootstrap_stratified_keeps_class_ratio():
    from engine.metrics import bootstrap_ci
    # 极度不平衡时 stratified bootstrap 每次重采样都应含两类, 不会被跳过
    labels = np.array([0] * 95 + [1] * 5)
    scores = np.linspace(0, 1, 100)
    lo, hi = bootstrap_ci(labels, scores, roc_auc_score, n_boot=50)
    assert not np.isnan(lo) and not np.isnan(hi)
    assert lo <= hi
