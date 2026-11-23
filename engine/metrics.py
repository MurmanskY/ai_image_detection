import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

FPR_TARGETS = (0.001, 0.01, 0.05)


def tpr_at_fpr(labels, scores, target):
    """在 FPR <= target 的最大 TPR (离散 ROC, 不做插值)。"""
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(fpr <= target)[0]
    return float(tpr[valid[-1]]) if len(valid) else 0.0


def fpr_at_tpr(labels, scores, target=0.95):
    """在 TPR >= target 的最小 FPR (离散 ROC, 不做插值)。"""
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= target)[0]
    return float(fpr[valid[0]]) if len(valid) else 1.0


def calibrate_threshold(labels, scores):
    """在验证集上用 Youden's J 选择 score 空间的决策阈值。"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def bootstrap_ci(labels, scores, metric_fn, n_boot=1000, seed=42, stratified=True):
    """bootstrap 95% CI。

    metric_fn(labels, scores) -> float。返回 (lo, hi)。
    stratified=True (默认): 按类别分层有放回抽样 (各类样本数与原集一致),
    避免普通 sample bootstrap 的类别比例波动; 同一真实图被多个 generator
    复用时仍非完全独立, 需 group bootstrap 时应另传 groups (未实现)。

    样本数过少或重采样后只剩单一类别时跳过该次重采样。
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    idx_classes = [np.where(labels == c)[0] for c in np.unique(labels)]
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([
                ic[rng.integers(0, len(ic), len(ic))] for ic in idx_classes
            ])
        else:
            idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(metric_fn(labels[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_delta(labels, scores_a, scores_b, metric_fn=roc_auc_score,
                           n_boot=1000, seed=42):
    """Estimate metric(B) - metric(A) with a paired, stratified bootstrap.

    Rows in ``scores_a`` and ``scores_b`` must refer to the same examples in
    the same order. Resampling shared row indices preserves that pairing.
    """
    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    if labels.ndim != 1 or scores_a.ndim != 1 or scores_b.ndim != 1:
        raise ValueError("labels and scores must be 1D arrays")
    if not (len(labels) == len(scores_a) == len(scores_b)):
        raise ValueError("paired labels and score arrays must have equal length")
    if len(np.unique(labels)) < 2:
        raise ValueError("paired metric comparison requires both label classes")
    if not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all():
        raise ValueError("paired scores must contain only finite values")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    point_delta = float(metric_fn(labels, scores_b) - metric_fn(labels, scores_a))
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    deltas = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = np.concatenate([
            indices[rng.integers(0, len(indices), len(indices))]
            for indices in class_indices
        ])
        deltas[index] = metric_fn(labels[sample], scores_b[sample]) - metric_fn(
            labels[sample], scores_a[sample])
    return {
        "delta": point_delta,
        "ci95": (float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))),
    }


def compute_metrics(labels, scores, threshold=0.5, n_boot=0):
    """labels: 0/1 数组; scores: 越大越倾向 fake 的分数。

    acc/f1/balanced_acc 使用给定阈值; ap/auc/tpr@fpr 与阈值无关。
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "threshold": float(threshold),
    }
    if len(np.unique(labels)) > 1:
        out["ap"] = float(average_precision_score(labels, scores))
        out["auc"] = float(roc_auc_score(labels, scores))
        for target in FPR_TARGETS:
            key = f"tpr@{target:g}fpr"
            out[key] = tpr_at_fpr(labels, scores, target)
        out["fpr@0.95tpr"] = fpr_at_tpr(labels, scores, 0.95)
        if n_boot > 0:
            out["auc_ci95"] = bootstrap_ci(labels, scores, roc_auc_score, n_boot)
            out["ap_ci95"] = bootstrap_ci(labels, scores, average_precision_score, n_boot)
            out["tpr@0.01fpr_ci95"] = bootstrap_ci(
                labels, scores, lambda l, s: tpr_at_fpr(l, s, 0.01), n_boot)
    return out
