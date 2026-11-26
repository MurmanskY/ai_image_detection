"""DART 完整循环特征的严格 T2I 未知家族留出实验。"""

import json
import os
import sys

import numpy as np
import torch

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_c2i_p0 as c2i
from analysis.carf_p0 import conformal_tau
from analysis.dart_p0 import (predict, scale_tensor, select_feature_mode,  # noqa: E402
                              train_dart)
from analysis.scale_p0 import GENS, load_features, prada_score

OUT = os.path.join(c2i.OUT, "family_holdout")
MODELS = tuple(GENS)
DATASETS = ("real",) + tuple(GENS)
SEEDS = tuple(int(x) for x in os.environ.get("DART_FAMILY_SEEDS", "0,1,2").split(","))
BANK_EXCLUDE = os.environ.get("DART_FAMILY_BANK_EXCLUDE", "1") == "1"
TRAINING_MODE = os.environ.get("DART_FAMILY_TRAINING", "shared")
FEATURE_MODE = os.environ.get("DART_FEATURE_MODE", "full")
POOLING = os.environ.get("DART_POOLING", "attention")
AGGREGATIONS = ("max", "mean", "median")


def quant_to_scales(quant, diff_length, lengths):
    if quant.shape[1] == len(lengths):
        return quant
    if quant.shape[1] == diff_length:
        return np.stack([row.reshape(len(lengths), -1).mean(axis=1) for row in quant])
    return np.repeat(quant.mean(axis=1, keepdims=True), len(lengths), axis=1)


def pack(ds, model):
    feat = load_features(ds, model)
    n = c2i.sample_count(ds)
    diff = (feat["cond_gt_llh_BL"][:n] - feat["uncond_gt_llh_BL"][:n]).float().numpy()
    lengths = feat.get("scale_lengths") or [diff.shape[1]]
    quant = feat["quant_err"][:n].float().numpy()
    quant = quant_to_scales(quant, diff.shape[1], lengths)
    rec = feat["aeroblade"][:n].float().numpy()
    return scale_tensor(diff, lengths, quant, rec)


def prada_t2i(ds, model):
    feat = dict(load_features(ds, model))
    n = c2i.sample_count(ds)
    for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL"):
        feat[key] = feat[key][:n]
    return prada_score(feat, model)


def included_models(heldout):
    return tuple(model for model in MODELS
                 if not BANK_EXCLUDE or model != heldout)


def allowed_generators(heldout):
    return tuple(gen for gen in GENS if gen != heldout)


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    packed_path = os.path.join(OUT, "packed_cycle.pt")
    if os.path.exists(packed_path):
        packed = torch.load(packed_path, map_location="cpu", weights_only=False)
    else:
        packed = {(ds, model): pack(ds, model)
                  for ds in DATASETS for model in MODELS}
        torch.save(packed, packed_path + ".tmp")
        os.replace(packed_path + ".tmp", packed_path)
    if FEATURE_MODE != "full":
        packed = {key: select_feature_mode(images, FEATURE_MODE)
                  for key, images in packed.items()}
    mode = "family-and-bank-excluded" if BANK_EXCLUDE else "heldout-fake-reference-available"
    run_tag = ("exclude" if BANK_EXCLUDE else "available") + (
        "" if TRAINING_MODE == "shared" else f"_{TRAINING_MODE}") + (
        "" if FEATURE_MODE == "full" else f"_{FEATURE_MODE}") + (
        "" if POOLING == "attention" else f"_{POOLING}") + (
        "" if SEEDS == (0, 1, 2) else "_s" + "_".join(map(str, SEEDS)))
    results = {"config": {"models": MODELS, "generators": GENS, "seeds": SEEDS,
                          "mode": mode, "training_mode": TRAINING_MODE,
                          "features": FEATURE_MODE, "pooling": POOLING,
                          "protocol": "strict real 300/200/200/300; fake 100/50/100"},
               "folds": {}}
    for heldout in GENS:
        models = included_models(heldout)
        generators = allowed_generators(heldout)
        results["folds"][heldout] = {"included_models": models, "seeds": {}}
        for seed in SEEDS:
            ref_idx, select_idx, cal_idx, test_idx = c2i.real_split(seed)
            train_idx, val_idx, fake_test = c2i.fake_split(seed)
            normalized = {}
            for model in models:
                inputs = {"real": packed["real", model],
                          "heldout": packed[heldout, model]}
                inputs.update({gen: packed[gen, model] for gen in generators})
                normalized[model] = c2i.normalize_data(inputs, ref_idx)
            darts = {}
            if TRAINING_MODE == "shared":
                train_images, train_labels, val_images, val_labels = [], [], [], []
                for model in models:
                    train_images.extend(normalized[model]["real"][i] for i in ref_idx)
                    train_labels.extend([0] * len(ref_idx))
                    val_images.extend(normalized[model]["real"][i] for i in select_idx)
                    val_labels.extend([0] * len(select_idx))
                    for gen in generators:
                        train_images.extend(normalized[model][gen][i] for i in train_idx)
                        train_labels.extend([1] * len(train_idx))
                        val_images.extend(normalized[model][gen][i] for i in val_idx)
                        val_labels.extend([1] * len(val_idx))
                darts["shared"] = train_dart(
                    (np.asarray(train_images, dtype=object), np.asarray(train_labels)),
                    (np.asarray(val_images, dtype=object), np.asarray(val_labels)), seed,
                    pooling=POOLING)
            elif TRAINING_MODE == "model_specific":
                if not BANK_EXCLUDE:
                    raise ValueError("model_specific family holdout requires BANK_EXCLUDE=1")
                for model in models:
                    train_data = (
                        np.concatenate([normalized[model]["real"][ref_idx],
                                        normalized[model][model][train_idx]]),
                        np.array([0] * len(ref_idx) + [1] * len(train_idx)))
                    val_data = (
                        np.concatenate([normalized[model]["real"][select_idx],
                                        normalized[model][model][val_idx]]),
                        np.array([0] * len(select_idx) + [1] * len(val_idx)))
                    darts[model] = train_dart(train_data, val_data, seed, pooling=POOLING)
            else:
                raise ValueError(f"unknown training mode: {TRAINING_MODE}")
            dart_bank_real, dart_bank_fake = [], []
            prada_bank_real, prada_bank_fake = [], []
            for model in models:
                dart = darts["shared"] if TRAINING_MODE == "shared" else darts[model]
                normalized_real = normalized[model]["real"]
                normalized_fake = normalized[model]["heldout"]
                scores = predict(dart, normalized_real)
                median = np.median(scores[select_idx])
                iqr = (np.percentile(scores[select_idx], 75)
                       - np.percentile(scores[select_idx], 25) + 1e-6)
                dart_bank_real.append((scores - median) / iqr)
                dart_bank_fake.append((predict(dart, normalized_fake) - median) / iqr)
                prada_bank_real.append(prada_t2i("real", model))
                prada_bank_fake.append(prada_t2i(heldout, model))
            banks = {
                "dart": (np.stack(dart_bank_real, axis=1), np.stack(dart_bank_fake, axis=1)),
                "prada": (np.stack(prada_bank_real, axis=1), np.stack(prada_bank_fake, axis=1))}
            y = np.array([0] * c2i.N_TEST_REAL + [1] * 100)
            variants = {}
            saved = {}
            for method, (bank_real, bank_fake) in banks.items():
                bank_test = np.concatenate([bank_real[test_idx], bank_fake[fake_test]], axis=0)
                bank_cal = bank_real[cal_idx]
                variants[method] = {}
                for aggregation in AGGREGATIONS:
                    reduce = {"max": np.max, "mean": np.mean, "median": np.median}[aggregation]
                    score = reduce(bank_test, axis=1)
                    cal = reduce(bank_cal, axis=1)
                    tau = conformal_tau(cal)
                    variants[method][aggregation] = c2i.metric(
                        y, score, c2i.N_TEST_REAL, tau)
                    saved[f"{method}_{aggregation}_threshold"] = tau
                saved[f"{method}_bank_test"] = bank_test
                saved[f"{method}_bank_cal"] = bank_cal
            result = {"dart": variants["dart"]["max"],
                      "prada": variants["prada"]["max"], "aggregations": variants}
            results["folds"][heldout]["seeds"][str(seed)] = result
            prefix = f"scores_dart_family_{run_tag}_{heldout}_seed{seed}"
            np.savez(os.path.join(c2i.OUT, prefix + ".npz"),
                     n_real=c2i.N_TEST_REAL, models=np.array(models), **saved)
            print(f"{mode} heldout={heldout} seed={seed} "
                  f"DART={result['dart']['AUROC']:.4f} PRADA={result['prada']['AUROC']:.4f} "
                  f"FPR={result['dart']['FPR']:.4f}", flush=True)
    for method in ("dart", "prada"):
        values = [results["folds"][heldout]["seeds"][str(seed)][method]["AUROC"]
                  for heldout in GENS for seed in SEEDS]
        results.setdefault("summary", {})[method] = {
            "macro_AUROC": float(np.mean(values)),
            "per_fold_mean": {heldout: float(np.mean([
                results["folds"][heldout]["seeds"][str(seed)][method]["AUROC"]
                for seed in SEEDS])) for heldout in GENS}}
    results["summary"]["aggregations"] = {}
    for method in ("dart", "prada"):
        for aggregation in AGGREGATIONS:
            values = [results["folds"][heldout]["seeds"][str(seed)]
                      ["aggregations"][method][aggregation]["AUROC"]
                      for heldout in GENS for seed in SEEDS]
            results["summary"]["aggregations"][f"{method}_{aggregation}"] = {
                "macro_AUROC": float(np.mean(values)),
                "per_fold_mean": {heldout: float(np.mean([
                    results["folds"][heldout]["seeds"][str(seed)]
                    ["aggregations"][method][aggregation]["AUROC"]
                    for seed in SEEDS])) for heldout in GENS}}
    results["summary"]["FPR"] = {
        str(seed): float(results["folds"][GENS[0]]["seeds"][str(seed)]["dart"]["FPR"])
        for seed in SEEDS}
    name = f"dart_family_{run_tag}.json"
    with open(os.path.join(c2i.OUT, name), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    evaluate()
