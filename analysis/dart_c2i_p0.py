"""DART C2I 小规模模型特异 P0/P1（5 models x real/4 generators）。

保留 ImageFolder 全部类别映射后 Subset，避免条件标签错配。默认每类 250 张，
可用 DART_N_REAL 单独扩大真实集；fake 训练/验证/测试固定为互斥 100/50/100。
DART 每个模型独立训练，PRADA 使用官方预训练评分函数。
GO: macro AUROC>=0.85、相对 PRADA +0.02、4/4 生成器不退化、FPR 3%-7%。
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_p0 import (normalize_data, predict, scale_tensor,  # noqa: E402
                              train_dart)
from analysis.mpid_p0 import wait_for_gpu  # noqa: E402
from analysis.scale_p0 import prada_score  # noqa: E402
from prada.metrics import aeroblade  # noqa: E402
from prada.wrappers import get_wrapper  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_c2i")
FEAT_DIR = os.path.join(OUT, "features")
DATA_ROOT = os.path.join(PROJ, "data", "darg", "darg", "c2i")
ALL_MODELS = ("var_d20", "var_d30", "llamagen_b256", "llamagen_l256", "rar_l")
MODELS = tuple(m for m in ALL_MODELS if m in os.environ.get(
    "DART_MODELS", ",".join(ALL_MODELS)).split(","))
FAKE_GENS = ("var_d16", "llamagen_b256", "rar_b", "hmar_d16")
DATASETS = ("real",) + FAKE_GENS
SEEDS = tuple(int(x) for x in os.environ.get(
    "DART_SEEDS", "0,1,2").split(","))
N_IMAGES = 250
N_REAL = int(os.environ.get("DART_N_REAL", str(N_IMAGES)))
N_REF_REAL = int(os.environ.get("DART_REF_REAL", "100"))
N_CAL_REAL = int(os.environ.get("DART_CAL_REAL", "50"))
N_TEST_REAL = N_REAL - N_REF_REAL - N_CAL_REAL
BATCH_SIZE = int(os.environ.get("DART_BATCH_SIZE", "4"))
NUM_WORKERS = int(os.environ.get("DART_NUM_WORKERS", "2"))
EXTRACT_ONLY = os.environ.get("DART_EXTRACT_ONLY", "0") == "1"
SHARD_INDEX = int(os.environ.get("DART_SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("DART_SHARD_COUNT", "1"))


def sample_count(ds):
    return N_REAL if ds == "real" else N_IMAGES


def fake_split(seed):
    perm = np.random.RandomState(100 + seed).permutation(N_IMAGES)
    return perm[:100], perm[100:150], perm[150:]


def acquire_gpu():
    if "DART_GPU" in os.environ:
        device_id = int(os.environ["DART_GPU"])
    else:
        device_id = wait_for_gpu(need_mb=12000)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[device] model worker uses GPU{device_id}", flush=True)


def selected_indices(n, ds):
    idx = np.random.RandomState(0).choice(n, sample_count(ds), replace=False)
    return idx[SHARD_INDEX::SHARD_COUNT]


def make_loader(wrapper, ds, batch_size=1):
    dataset = ImageFolder(os.path.join(DATA_ROOT, ds), transform=wrapper.transform)
    idx = selected_indices(len(dataset), ds)
    return DataLoader(Subset(dataset, idx), batch_size=batch_size,
                      num_workers=NUM_WORKERS, pin_memory=True)


def cycle_batch_size(model):
    return BATCH_SIZE if model.startswith("llamagen") else 1


def rec_batch_size(model):
    return BATCH_SIZE if model.startswith(("llamagen", "var")) else 1


def load_model_wrapper(model):
    previous = os.getcwd()
    os.chdir(os.path.join(PROJ, "third_party", "PRADA"))
    try:
        return get_wrapper(model, checkpoints_root=os.path.join(
            PROJ, "third_party", "PRADA", "model_weights"))
    finally:
        os.chdir(previous)


@torch.no_grad()
def extract_model(model):
    acquire_gpu()
    wrapper = load_model_wrapper(model)
    for ds in DATASETS:
        path = os.path.join(FEAT_DIR, f"dataset={ds}_model={model}.pt")
        n_images = sample_count(ds)
        if os.path.exists(path):
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if len(cached["cond_gt_llh_BL"]) >= n_images:
                continue
        output_path = path if SHARD_COUNT == 1 else (
            f"{path}.shard{SHARD_COUNT}_{SHARD_INDEX}.pt")
        if SHARD_COUNT > 1 and os.path.exists(output_path):
            continue
        loader = make_loader(wrapper, ds, batch_size=1)
        features = []
        for image, condition in loader:
            image = image.to(wrapper.device, non_blocking=True)
            if isinstance(condition, torch.Tensor):
                condition = condition.to(wrapper.device, non_blocking=True)
            gt = wrapper.get_gt_idx(image_B3HW=image)
            logits = wrapper.get_logits(gt_idx_BL=gt["gt_idx_BL"],
                                        condition_B=condition)
            llh = wrapper.get_gt_llh(logits_BLX=logits, gt_idx_BL=gt["gt_idx_BL"])
            features.append({k: v.cpu() for k, v in llh.items()})
        output = {}
        for key in features[0]:
            output[key] = torch.cat([x[key] for x in features])
        if hasattr(wrapper, "scale_lengths"):
            output["scale_lengths"] = wrapper.scale_lengths
        torch.save(output, output_path + ".tmp")
        os.replace(output_path + ".tmp", output_path)
        print(f"  extracted {ds}/{model} shard {SHARD_INDEX}/{SHARD_COUNT}",
              flush=True)
    del wrapper
    torch.cuda.empty_cache()


def ensure_features():
    os.makedirs(FEAT_DIR, exist_ok=True)
    for model in MODELS:
        if all(cached_has(ds, model, "cond_gt_llh_BL") for ds in DATASETS):
            continue
        extract_model(model)


def cached_has(ds, model, key):
    features = torch.load(os.path.join(FEAT_DIR, f"dataset={ds}_model={model}.pt"),
                          map_location="cpu", weights_only=False)
    return key in features and len(features[key]) >= sample_count(ds)


@torch.no_grad()
def ensure_quant_err(model):
    if all(cached_has(ds, model, "quant_err") for ds in DATASETS):
        return
    acquire_gpu()
    wrapper = load_model_wrapper(model)
    for ds in DATASETS:
        path = os.path.join(FEAT_DIR, f"dataset={ds}_model={model}.pt")
        features = torch.load(path, map_location="cpu", weights_only=False)
        if cached_has(ds, model, "quant_err"):
            continue
        loader = make_loader(wrapper, ds, batch_size=cycle_batch_size(model))
        values = []
        for image, _ in loader:
            image = image.to(wrapper.device, non_blocking=True)
            rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
            values.append(-rec["quant_err_BL"].cpu())
        features["quant_err"] = torch.cat(values)
        torch.save(features, path)
        print(f"  quant_err {ds}/{model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


@torch.no_grad()
def ensure_rec_err(model):
    if all(cached_has(ds, model, "rec_err") for ds in DATASETS):
        return
    acquire_gpu()
    wrapper = load_model_wrapper(model)
    for ds in DATASETS:
        path = os.path.join(FEAT_DIR, f"dataset={ds}_model={model}.pt")
        features = torch.load(path, map_location="cpu", weights_only=False)
        if cached_has(ds, model, "rec_err"):
            continue
        loader = make_loader(wrapper, ds, batch_size=rec_batch_size(model))
        values = []
        for image, _ in loader:
            image = image.to(wrapper.device, non_blocking=True)
            rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
            values.append(-aeroblade(
                img1=image, img2=rec["rec_B3HW"], net_type="vgg",
                reduction="none",
                normalize=wrapper.range_after_transform == (0, 1)).cpu().reshape(-1))
        features["rec_err"] = torch.cat(values)
        torch.save(features, path)
        print(f"  rec_err {ds}/{model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


def load_c2i(ds, model):
    return torch.load(os.path.join(FEAT_DIR, f"dataset={ds}_model={model}.pt"),
                      map_location="cpu", weights_only=False)


def pack(ds, model):
    feat = load_c2i(ds, model)
    n = sample_count(ds)
    diff = (feat["cond_gt_llh_BL"][:n]
            - feat["uncond_gt_llh_BL"][:n]).float().numpy()
    lengths = feat.get("scale_lengths") or [diff.shape[1]]
    quant_err = feat["quant_err"][:n].float().numpy()
    quant_err = np.stack([row.reshape(len(lengths), -1).mean(axis=1)
                          for row in quant_err])
    return scale_tensor(diff, lengths, quant_err,
                        feat["rec_err"][:n].float().numpy())


def prada_c2i(ds, model):
    feat = load_c2i(ds, model)
    n = sample_count(ds)
    feat = dict(feat)
    for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL"):
        feat[key] = feat[key][:n]
    return prada_score(feat, model)


def metric(y, score, n_real, tau):
    fpr, tpr, _ = roc_curve(y, score)
    return {"AUROC": float(roc_auc_score(y, score)),
            "AP": float(average_precision_score(y, score)),
            "TPR@1%FPR": float(tpr[fpr <= 0.01].max())
            if np.any(fpr <= 0.01) else 0.0,
            "FPR": float((score[:n_real] > tau).mean())}


def main():
    ensure_features()
    if SHARD_COUNT > 1:
        print(f"  completed likelihood shard {SHARD_INDEX}/{SHARD_COUNT}", flush=True)
        return
    for model in MODELS:
        ensure_quant_err(model)
        ensure_rec_err(model)
    if EXTRACT_ONLY:
        print(f"  completed feature worker: {','.join(MODELS)}", flush=True)
        return
    packed = {(ds, model): pack(ds, model)
              for ds in DATASETS for model in MODELS}
    result_name = os.environ.get("DART_RESULT", "dart_c2i_p0.json")
    score_prefix = f"scores_{os.path.splitext(result_name)[0]}"
    results = {"config": {"models": MODELS, "generators": FAKE_GENS,
                          "n_images": N_IMAGES, "n_real": N_REAL,
                          "real_ref": N_REF_REAL, "real_cal": N_CAL_REAL,
                          "real_test": N_TEST_REAL,
                          "protocol": f"dart-c2i-real{N_REAL}-fake{N_IMAGES} 示例"},
               "seeds": {}}
    for seed in SEEDS:
        real_perm = np.random.RandomState(seed).permutation(N_REAL)
        ref_idx = real_perm[:N_REF_REAL]
        cal_idx = real_perm[N_REF_REAL:N_REF_REAL + N_CAL_REAL]
        test_idx = real_perm[N_REF_REAL + N_CAL_REAL:]
        train_fake = {"var_d20": "var_d16", "var_d30": "var_d16",
                      "llamagen_b256": "llamagen_b256",
                      "llamagen_l256": "llamagen_b256", "rar_l": "rar_b"}
        normalized = {}
        for model in MODELS:
            inputs = {"real": packed["real", model],
                      "fake": packed[train_fake[model], model]}
            inputs.update({ds: packed[ds, model] for ds in FAKE_GENS})
            normalized[model] = normalize_data(inputs, ref_idx)
        combined_dart = {ds: np.full(N_REAL if ds == "real" else N_IMAGES,
                                     -np.inf, dtype=np.float32)
                         for ds in DATASETS}
        combined_prada = {ds: np.full(N_REAL if ds == "real" else N_IMAGES,
                                      -np.inf, dtype=np.float32)
                          for ds in DATASETS}
        for model in MODELS:
            train_fake, val_fake, _ = fake_split(seed)
            model_dart = train_dart(
                (np.concatenate([normalized[model]["real"][ref_idx],
                                 normalized[model]["fake"][train_fake]]),
                 np.array([0] * N_REF_REAL + [1] * 100)),
                (np.concatenate([normalized[model]["real"][cal_idx],
                                 normalized[model]["fake"][val_fake]]),
                 np.array([0] * N_CAL_REAL + [1] * 50)), seed)
            model_scores = {}
            for ds in DATASETS:
                model_scores[ds] = predict(model_dart, normalized[model][ds])
            median = np.median(model_scores["real"][cal_idx])
            iqr = (np.percentile(model_scores["real"][cal_idx], 75)
                   - np.percentile(model_scores["real"][cal_idx], 25) + 1e-6)
            for ds in DATASETS:
                z_score = (model_scores[ds] - median) / iqr
                if not np.isfinite(z_score).all():
                    raise ValueError(f"non-finite DART score: seed{seed} {model}/{ds}")
                combined_dart[ds] = np.maximum(combined_dart[ds], z_score)
                combined_prada[ds] = np.maximum(
                    combined_prada[ds], prada_c2i(ds, model))
        per_gen = {}
        for g in FAKE_GENS:
            _, _, test_fake = fake_split(seed)
            y = np.array([0] * N_TEST_REAL + [1] * 100)
            dart_score = np.concatenate([combined_dart["real"][test_idx],
                                         combined_dart[g][test_fake]])
            prada = np.concatenate([combined_prada["real"][test_idx],
                                    combined_prada[g][test_fake]])
            tau = conformal_tau(combined_dart["real"][cal_idx])
            prada_tau = conformal_tau(combined_prada["real"][cal_idx])
            per_gen[g] = {"dart": metric(y, dart_score, N_TEST_REAL, tau),
                          "prada": metric(y, prada, N_TEST_REAL, prada_tau)}
            np.savez(os.path.join(OUT, f"{score_prefix}_seed{seed}_{g}.npz"),
                     dart=dart_score, prada=prada, n_real=N_TEST_REAL)
        results["seeds"][str(seed)] = per_gen
        print(f"seed {seed}: " + "; ".join(
            f"{m}={np.mean([per_gen[g][m]['AUROC'] for g in FAKE_GENS]):.4f}"
            for m in ("dart", "prada")), flush=True)
    macro = {m: [float(np.mean([results["seeds"][str(s)][g][m]["AUROC"]
                                for g in FAKE_GENS])) for s in SEEDS]
             for m in ("dart", "prada")}
    fprs = [float(np.mean([results["seeds"][str(s)][g]["dart"]["FPR"]
                           for g in FAKE_GENS])) for s in SEEDS]
    deltas = [macro["dart"][i] - macro["prada"][i] for i in range(len(SEEDS))]
    go = {"macro_AUROC": macro, "FPR": fprs, "delta": deltas,
          "crit1_AUROC>=0.85": bool(min(macro["dart"]) >= 0.85),
          "crit2_delta>=0.02": bool(min(deltas) >= 0.02),
          "crit3_both_gens_not_worse": bool(all(
              results["seeds"][str(s)][g]["dart"]["AUROC"] + 1e-9
              >= results["seeds"][str(s)][g]["prada"]["AUROC"]
              for s in SEEDS for g in FAKE_GENS)),
          "crit4_FPR": bool(all(0.03 <= x <= 0.07 for x in fprs))}
    go["GO"] = bool(go["crit1_AUROC>=0.85"] and go["crit2_delta>=0.02"]
                    and go["crit3_both_gens_not_worse"] and go["crit4_FPR"])
    results["summary"] = {"macro_AUROC": macro, "FPR": fprs}
    results["go"] = go
    with open(os.path.join(OUT, result_name), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(go, indent=2), flush=True)
    print(f"已保存 {OUT}/{result_name}", flush=True)


if __name__ == "__main__":
    main()
