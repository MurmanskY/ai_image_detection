from collections import defaultdict

import numpy as np
import torch

from utils.device import autocast_ctx

from .metrics import compute_metrics


def _score_from_output(raw, method):
    """按方法的 score 契约把模型输出转成 "越大越 fake" 的分数。

    logit 方法 (BCE 训练): sigmoid 后 0.5 是有意义的阈值;
    anomaly 方法 (training-free raw score): 不做 sigmoid, 阈值需另行校准。
    """
    score = raw.float()
    score_type = getattr(method, "score_type", "logit") if method is not None else "logit"
    if score_type == "logit":
        score = torch.sigmoid(score)
    if method is not None and not getattr(method, "higher_is_fake", True):
        score = -score
    return score


@torch.no_grad()
def predict_scores(model, loader, device, amp=True, method=None):
    """返回 labels, scores, generators, paths (均为 numpy/list)。"""
    model.eval()
    all_labels, all_scores, all_gens, all_paths = [], [], [], []
    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            raw = model(imgs)
        all_scores.append(_score_from_output(raw, method).cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_gens.extend(batch["generator"])
        all_paths.extend(batch["path"])

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    return labels, scores, all_gens, all_paths


@torch.no_grad()
def evaluate(model, loader, device, amp=True, method=None, threshold=0.5):
    """返回 dict: overall 指标 + 每个 generator 的细分指标。"""
    labels, scores, all_gens, _ = predict_scores(model, loader, device, amp, method)
    results = {"overall": compute_metrics(labels, scores, threshold)}

    per_gen = defaultdict(lambda: ([], []))
    for l, s, g in zip(labels, scores, all_gens):
        per_gen[g][0].append(l)
        per_gen[g][1].append(s)
    for g, (l, s) in per_gen.items():
        results[g] = compute_metrics(l, s, threshold)
    return results
