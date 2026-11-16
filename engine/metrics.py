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
    """在 FPR <= target 的最大 TPR。"""
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(fpr <= target)[0]
    return float(tpr[valid[-1]]) if len(valid) else 0.0


def calibrate_threshold(labels, scores):
    """在验证集上用 Youden's J 选择 score 空间的决策阈值。"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def compute_metrics(labels, scores, threshold=0.5):
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
    return out
