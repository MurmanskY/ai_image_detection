from collections import defaultdict

import numpy as np
import torch

from utils.device import autocast_ctx

from .metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    """返回 dict: overall 指标 + 每个 generator 的细分指标。"""
    model.eval()
    all_labels, all_probs, all_gens = [], [], []
    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits = model(imgs)
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_gens.extend(batch["generator"])

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    results = {"overall": compute_metrics(labels, probs)}

    per_gen = defaultdict(lambda: ([], []))
    for l, p, g in zip(labels, probs, all_gens):
        per_gen[g][0].append(l)
        per_gen[g][1].append(p)
    for g, (l, p) in per_gen.items():
        results[g] = compute_metrics(l, p)
    return results
