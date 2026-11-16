import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(labels, probs, threshold=0.5):
    """labels: 0/1 数组; probs: 预测为 fake 的概率。"""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        out["ap"] = float(average_precision_score(labels, probs))
        out["auc"] = float(roc_auc_score(labels, probs))
    return out
