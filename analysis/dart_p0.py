"""DART P0: Distribution-Aware Autoregressive Token detector (LOGO).

每个 scale 的 signed likelihood-difference quantile function (16 quantiles)
+ log scale length + scale index，经共享 Deep Sets encoder/attention pooling。
训练严格 leave-one-generator-out：held-out fake 不参与训练、validation 或
提前停止；真实 reference 仅用于输入标准化，真实 calibration 仅用于阈值。

DART 是 QSCALE real-only 失败后的监督式机制检验。三 seed x 4 folds。
GO: macro AUROC>=0.98、>=PRADA-0.02、>=QSCALE-kNN+0.10、至少3/4折
AUROC>=0.90、每 seed FPR∈[3%,7%]。否则停止，不扩展。
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.scale_p0 import (DATASETS, GENS, N_FAKE, OUT as SCALE_OUT,  # noqa: E402
                               load_features)
from analysis.qscale_p0 import LEVELS, OUT as QSCALE_OUT  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart")
SEEDS = (0, 1, 2)
MAX_SCALES = 14


def scale_tensor(diff, scale_lengths, quant_err=None, rec_err=None):
    outputs = []
    for image_index, values in enumerate(diff):
        features = []
        start = 0
        for index, length in enumerate(scale_lengths):
            quantiles = np.quantile(values[start:start + length], LEVELS)
            meta = [np.log10(length) / 4, index / max(1, len(scale_lengths) - 1)]
            if quant_err is not None:
                meta.append(quant_err[image_index, index])
            if rec_err is not None:
                meta.append(rec_err[image_index])
            features.append(np.concatenate([quantiles, meta]))
            start += length
        outputs.append(np.asarray(features, dtype=np.float32))
    return outputs


def pack_model_data(model):
    packed = {}
    for ds in ("real", model):
        feat = load_features(ds, model)
        diff = (feat["cond_gt_llh_BL"]
                - feat["uncond_gt_llh_BL"]).float().numpy()
        lengths = feat.get("scale_lengths") or [diff.shape[1]]
        packed[ds] = scale_tensor(diff, lengths)
    return packed


def normalize_data(data, ref_idx):
    reference = np.concatenate([data["real"][i] for i in ref_idx], axis=0)
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0) + 1e-6
    normalized = {}
    for ds, images in data.items():
        normalized[ds] = np.asarray([(x - mu) / sd for x in images],
                                    dtype=object)
    return normalized


def batches(images, labels, batch_size, rng=None):
    order = np.arange(len(images)) if rng is None else rng.permutation(len(images))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        xs = np.zeros((len(idx), MAX_SCALES, images[0].shape[1]), dtype=np.float32)
        mask = np.zeros((len(idx), MAX_SCALES), dtype=bool)
        for row, image_idx in enumerate(idx):
            x = images[image_idx]
            xs[row, :len(x)] = x
            mask[row, :len(x)] = True
        yield (torch.tensor(xs), torch.tensor(mask),
               torch.tensor(labels[idx], dtype=torch.float32))


class ScaleDeepSet(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(),
                                     nn.Linear(32, 16), nn.GELU())
        self.attention = nn.Linear(16, 1)
        self.classifier = nn.Linear(16, 1)

    def forward(self, x, mask):
        h = self.encoder(x)
        logits = self.attention(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def train_dart(train_data, val_data, seed, epochs=40):
    torch.manual_seed(seed)
    model = ScaleDeepSet(input_dim=train_data[0][0].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_loss = np.inf
    for epoch in range(epochs):
        model.train()
        rng = np.random.RandomState(seed * 1000 + epoch)
        for x, mask, y in batches(*train_data, 128, rng):
            optimizer.zero_grad()
            loss = criterion(model(x, mask), y)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for x, mask, y in batches(*val_data, 256):
                losses.append(criterion(model(x, mask), y).item())
        val_loss = float(np.mean(losses))
        if not np.isfinite(val_loss):
            print(f"[dart] seed{seed} epoch{epoch}: non-finite validation loss, skip",
                  flush=True)
            continue
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


def predict(model, images):
    labels = np.zeros(len(images), dtype=np.int64)
    outputs = []
    with torch.no_grad():
        for x, mask, _ in batches(images, labels, 256):
            outputs.append(model(x, mask).numpy())
    return np.concatenate(outputs)


def metric(y, score, n_real, tau):
    fpr, tpr, _ = roc_curve(y, score)
    return {"AUROC": float(roc_auc_score(y, score)),
            "AP": float(average_precision_score(y, score)),
            "TPR@1%FPR": float(tpr[fpr <= 0.01].max())
            if np.any(fpr <= 0.01) else 0.0,
            "FPR": float((score[:n_real] > tau).mean())}


def main():
    os.makedirs(OUT, exist_ok=True)
    scale_results = json.load(open(os.path.join(SCALE_OUT, "scale_p0.json")))
    qscale_results = json.load(open(os.path.join(QSCALE_OUT, "qscale_p0.json")))
    packed = {model: pack_model_data(model) for model in GENS}
    results = {"config": {"epochs": 40, "batch_size": 128, "levels": LEVELS.tolist(),
                          "protocol": "dart-p0-logo-300 示例"},
               "seeds": {}}
    for seed in SEEDS:
        perm = np.random.RandomState(seed).permutation(1000)
        ref_idx, cal_idx, test_idx = perm[:500], perm[500:750], perm[750:]
        normalized = {model: normalize_data(packed[model], ref_idx)
                      for model in GENS}
        seed_result = {}
        for heldout in GENS:
            print(f"===== DART seed{seed} heldout={heldout} =====", flush=True)
            train_images, train_labels, val_images, val_labels = [], [], [], []
            for model in GENS:
                if model == heldout:
                    continue
                train_images.extend(normalized[model]["real"][ref_idx])
                train_labels += [0] * len(ref_idx)
                train_images.extend(normalized[model][model][i] for i in ref_idx)
                train_labels += [1] * len(ref_idx)
                val_images.extend(normalized[model]["real"][cal_idx])
                val_labels += [0] * len(cal_idx)
                val_images.extend(normalized[model][model][i] for i in cal_idx)
                val_labels += [1] * len(cal_idx)
            train_images = np.asarray(train_images, dtype=object)
            train_labels = np.asarray(train_labels)
            val_images = np.asarray(val_images, dtype=object)
            val_labels = np.asarray(val_labels)
            dart = train_dart((train_images, train_labels),
                              (val_images, val_labels), seed)
            fake_idx = np.random.RandomState(100 + seed).choice(
                1000, N_FAKE, replace=False)
            real_images = [normalized[heldout]["real"][i] for i in test_idx]
            fake_images = [normalized[heldout][heldout][i] for i in fake_idx]
            scores = predict(dart, real_images + fake_images)
            cal_scores = predict(dart, normalized[heldout]["real"][cal_idx])
            tau = conformal_tau(cal_scores)
            y = np.array([0] * 250 + [1] * N_FAKE)
            baseline = np.load(os.path.join(SCALE_OUT, f"scores_seed{seed}.npz"))
            qbaseline = np.load(os.path.join(QSCALE_OUT, f"scores_seed{seed}.npz"))
            entry = {"dart": metric(y, scores, 250, tau),
                     "prada": metric(y, baseline[f"{heldout}::prada"], 250, tau),
                     "qscale_knn": metric(y, qbaseline[f"{heldout}::qscale_knn"],
                                          250, tau)}
            entry["prada"]["FPR"] = scale_results["seeds"][str(seed)][heldout]["prada"]["FPR"]
            entry["qscale_knn"]["FPR"] = qscale_results["seeds"][str(seed)][heldout]["qscale_knn"]["FPR"]
            seed_result[heldout] = entry
            np.savez(os.path.join(OUT, f"scores_seed{seed}_{heldout}.npz"),
                     dart=scores, prada=baseline[f"{heldout}::prada"],
                     qscale_knn=qbaseline[f"{heldout}::qscale_knn"])
            print("  " + "; ".join(f"{m}={entry[m]['AUROC']:.4f}"
                                   for m in entry), flush=True)
        results["seeds"][str(seed)] = seed_result

    macros = {method: [float(np.mean([
        results["seeds"][str(s)][g][method]["AUROC"] for g in GENS]))
        for s in SEEDS] for method in ("dart", "prada", "qscale_knn")}
    fprs = [float(np.mean([results["seeds"][str(s)][g]["dart"]["FPR"]
                           for g in GENS])) for s in SEEDS]
    folds = [results["seeds"][str(s)][g]["dart"]["AUROC"]
             for s in SEEDS for g in GENS]
    go = {
        "macro_AUROC": macros, "FPR": fprs,
        "crit1_AUROC>=0.98": bool(min(macros["dart"]) >= 0.98),
        "crit1_within_PRADA_0.02": bool(all(
            macros["dart"][i] >= macros["prada"][i] - 0.02 for i in range(3))),
        "crit2_vs_QSCALE+0.10": bool(all(
            macros["dart"][i] >= macros["qscale_knn"][i] + 0.10 for i in range(3))),
        "crit3_folds>=0.90": bool(sum(x >= 0.90 for x in folds) >= 9),
        "crit4_FPR": bool(all(0.03 <= x <= 0.07 for x in fprs)),
    }
    go["GO"] = bool(go["crit1_AUROC>=0.98"]
                    and go["crit1_within_PRADA_0.02"]
                    and go["crit2_vs_QSCALE+0.10"] and go["crit3_folds>=0.90"]
                    and go["crit4_FPR"])
    results["summary"] = {"macro_AUROC": macros, "FPR": fprs}
    results["go"] = go
    with open(os.path.join(OUT, "dart_p0.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(go, indent=2), flush=True)
    print(f"已保存 {OUT}/dart_p0.json", flush=True)


if __name__ == "__main__":
    main()
