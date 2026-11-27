"""DART 七家族跨数据集留出与 model-bank coverage 实验。

DARG T2I 提供训练 fake，ARForensics corrected test 提供独立真实四划分和
七个测试家族。重叠家族同时从训练 fake 和参考 bank 排除；Open-MAGVIT2、
RAR、VAR 在 DARG T2I 训练集与四个 T2I 参考模型中均不存在。测试位置在
所有 seed 固定，仅重排训练/选择/校准集合。coverage seeds 枚举所有非空
参考模型子集，其余 seeds 只评估目标严格排除后的最大可用 bank。
"""

import gc
import itertools
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis import dart_family_p0 as family  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_p0 import predict, select_feature_mode, train_dart  # noqa: E402
from analysis.scale_p0 import GENS as DARG_GENS  # noqa: E402
from analysis.scale_p0 import prada_score  # noqa: E402
from prada.data import cat_tensor_dicts, get_dataloader  # noqa: E402
from prada.metrics import aeroblade  # noqa: E402
from prada.prompt_extraction import extract_prompts  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_external_family")
DATA_ROOT = os.path.join(PROJ, "data", "dart_external_family")
ARF_ROOT = os.path.join(PROJ, "data", "ARForensics", "test")
FEATURE_DIR = os.path.join(OUT, "features")
PROMPT_DIR = os.path.join(OUT, "extracted_prompts")
MODELS = tuple(DARG_GENS)
TARGETS = ("Infinity", "Janus_Pro", "LlamaGen", "Open_MAGVIT2", "RAR", "Switti", "VAR")
TARGET_TO_DARG = {
    "Infinity": "infinity_2b", "Janus_Pro": "janus_1b",
    "LlamaGen": "llamagen_xlstage2", "Switti": "switti_1024",
}
FEATURES = ("full", "likelihood_reconstruction")
SEEDS = tuple(int(x) for x in os.environ.get("DART_EXTERNAL_SEEDS", "0,1,2,3,4").split(","))
COVERAGE_SEEDS = tuple(int(x) for x in os.environ.get(
    "DART_EXTERNAL_COVERAGE_SEEDS", "0,1,2").split(","))
N_REAL = 1000
N_FAKE = 250
N_TEST_REAL = 300
N_TEST_FAKE = 100
SHARD_SIZE = int(os.environ.get("DART_EXTERNAL_SHARD_SIZE", "100"))
BATCH_SIZE = int(os.environ.get("DART_EXTERNAL_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("DART_EXTERNAL_NUM_WORKERS", "4"))
MODEL_ATTRIBUTES = {
    "infinity_2b": "next_scale", "switti_1024": "next_scale",
    "janus_1b": "raster", "llamagen_xlstage2": "raster",
}
TARGET_ATTRIBUTES = {
    "Infinity": "next_scale", "Switti": "next_scale", "VAR": "next_scale",
    "Janus_Pro": "raster", "LlamaGen": "raster", "RAR": "random_order",
    "Open_MAGVIT2": "raster_subtoken",
}


def image_files(path):
    return sorted(os.path.join(path, name) for name in os.listdir(path)
                  if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))


def selected_files(paths, count, seed):
    if len(paths) < count:
        raise ValueError(f"need {count} images, found {len(paths)}")
    idx = np.random.RandomState(seed).choice(len(paths), count, replace=False)
    return [paths[i] for i in sorted(idx)]


def prepare_data():
    os.makedirs(DATA_ROOT, exist_ok=True)
    sources = {"real": os.path.join(ARF_ROOT, "Infinity", "0_real")}
    sources.update({g: os.path.join(ARF_ROOT, g, "1_fake") for g in TARGETS})
    manifest = {}
    for ds, source in sources.items():
        count = N_REAL if ds == "real" else N_FAKE
        chosen = selected_files(image_files(source), count, 2026)
        dst = os.path.join(DATA_ROOT, ds)
        os.makedirs(dst, exist_ok=True)
        manifest[ds] = []
        for index, src in enumerate(chosen):
            stem, extension = os.path.splitext(os.path.basename(src))
            name = f"{index:04d}_{stem}{extension.lower()}"
            link = os.path.join(dst, name)
            if not os.path.exists(link):
                os.link(src, link)
            manifest[ds].append({"source": os.path.relpath(src, PROJ), "file": name})
        actual = image_files(dst)
        if len(actual) != count:
            raise ValueError(f"prepared {ds}: expected {count}, found {len(actual)}")
    with open(os.path.join(OUT, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"prepared real={N_REAL}, each fake={N_FAKE}, targets={len(TARGETS)}", flush=True)


def ensure_prompts(datasets):
    import csv

    os.makedirs(PROMPT_DIR, exist_ok=True)
    for ds in datasets:
        path = os.path.join(PROMPT_DIR, f"{ds}.csv")
        expected = N_REAL if ds == "real" else N_FAKE
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                valid = reader.fieldnames == ["image", "prompt"] and sum(1 for _ in reader) == expected
            if valid:
                continue
            os.remove(path)
        extract_prompts(os.path.join(DATA_ROOT, ds), path,
                        os.path.join(PROJ, "third_party", "PRADA", "model_weights"))


def feature_path(ds, model):
    return os.path.join(FEATURE_DIR, f"dataset={ds}_model={model}.pt")


def shard_path(ds, model, start):
    return feature_path(ds, model) + f".shard{start:05d}.pt"


def cached_count(ds, model):
    path = feature_path(ds, model)
    if not os.path.exists(path):
        return 0
    features = torch.load(path, map_location="cpu", weights_only=False)
    required = ("cond_gt_llh_BL", "uncond_gt_llh_BL", "quant_err", "aeroblade")
    if not all(key in features for key in required):
        return 0
    return len(features["aeroblade"])


@torch.no_grad()
def extract_model(model, datasets):
    wrapper = family.c2i.load_model_wrapper(model)
    try:
        for ds in datasets:
            expected = N_REAL if ds == "real" else N_FAKE
            if cached_count(ds, model) >= expected:
                continue
            for start in range(0, expected, SHARD_SIZE):
                end = min(start + SHARD_SIZE, expected)
                shard = shard_path(ds, model, start)
                if os.path.exists(shard):
                    continue
                base = get_dataloader(
                    os.path.join(DATA_ROOT, ds), wrapper.transform, "t2i",
                    BATCH_SIZE, NUM_WORKERS,
                    os.path.join(PROMPT_DIR, f"{ds}.csv"))
                loader = DataLoader(Subset(base.dataset, range(start, end)),
                                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                    pin_memory=True)
                rows = []
                for image, condition in loader:
                    image = image.to(wrapper.device, non_blocking=True)
                    gt = wrapper.get_gt_idx(image_B3HW=image)
                    logits = wrapper.get_logits(gt_idx_BL=gt["gt_idx_BL"],
                                                condition_B=condition)
                    llh = wrapper.get_gt_llh(logits_BLX=logits,
                                             gt_idx_BL=gt["gt_idx_BL"])
                    rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
                    llh["quant_err"] = -rec["quant_err_BL"]
                    llh["aeroblade"] = -aeroblade(
                        image, rec["rec_B3HW"], net_type="vgg", reduction="none",
                        normalize=wrapper.range_after_transform == (0, 1))
                    rows.append({key: value.cpu() for key, value in llh.items()})
                output = cat_tensor_dicts(rows)
                if len(output["aeroblade"]) != end - start:
                    raise ValueError(f"incomplete shard {ds}/{model}/{start}")
                torch.save(output, shard + ".tmp")
                os.replace(shard + ".tmp", shard)
                print(f"extracted {ds}/{model} [{start}:{end}]", flush=True)
            shards = [torch.load(shard_path(ds, model, start), map_location="cpu",
                                 weights_only=False)
                      for start in range(0, expected, SHARD_SIZE)]
            output = cat_tensor_dicts(shards)
            if hasattr(wrapper, "scale_lengths"):
                output["scale_lengths"] = wrapper.scale_lengths
            for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL", "quant_err", "aeroblade"):
                if not torch.isfinite(output[key]).all():
                    raise ValueError(f"non-finite features: {ds}/{model}/{key}")
            torch.save(output, feature_path(ds, model) + ".tmp")
            os.replace(feature_path(ds, model) + ".tmp", feature_path(ds, model))
            print(f"completed {ds}/{model}: {expected}", flush=True)
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def pack_external(ds, model):
    features = torch.load(feature_path(ds, model), map_location="cpu", weights_only=False)
    count = N_REAL if ds == "real" else N_FAKE
    diff = (features["cond_gt_llh_BL"][:count]
            - features["uncond_gt_llh_BL"][:count]).float().numpy()
    lengths = features.get("scale_lengths") or [diff.shape[1]]
    quant = features["quant_err"][:count].float().numpy()
    quant = family.quant_to_scales(quant, diff.shape[1], lengths)
    rec = features["aeroblade"][:count].float().numpy()
    return family.c2i.scale_tensor(diff, lengths, quant, rec)


def fixed_splits(seed):
    real = np.random.RandomState(0).permutation(N_REAL)
    fake = np.random.RandomState(100).permutation(N_FAKE)
    if seed == 0:
        return real[:300], real[300:500], real[500:700], real[700:], fake[:100], fake[100:150], fake[150:]
    rng = np.random.RandomState(seed)
    real_pool = rng.permutation(real[:700])
    fake_pool = np.random.RandomState(100 + seed).permutation(fake[:150])
    return (real_pool[:300], real_pool[300:500], real_pool[500:], real[700:],
            fake_pool[:100], fake_pool[100:], fake[150:])


def available_models(target):
    excluded = TARGET_TO_DARG.get(target)
    return tuple(model for model in MODELS if model != excluded)


def training_generators(target):
    excluded = TARGET_TO_DARG.get(target)
    return tuple(gen for gen in DARG_GENS if gen != excluded)


def bank_subsets(models):
    return tuple(subset for size in range(1, len(models) + 1)
                 for subset in itertools.combinations(models, size))


def evaluation_contexts(seed):
    contexts = {}
    for target in TARGETS:
        models = available_models(target)
        banks = bank_subsets(models) if seed in COVERAGE_SEEDS else (models,)
        train_gens = training_generators(target)
        for bank in banks:
            key = (train_gens, bank)
            contexts.setdefault(key, []).append(target)
    return contexts


def bank_name(bank):
    return "+".join(bank)


def prada_external(ds, model):
    features = torch.load(feature_path(ds, model), map_location="cpu", weights_only=False)
    count = N_REAL if ds == "real" else N_FAKE
    for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL"):
        features[key] = features[key][:count]
    return prada_score(features, model)


def build_training_data(data, models, generators, ref_idx, select_idx, train_idx, val_idx):
    train_images, train_labels, val_images, val_labels = [], [], [], []
    for model in models:
        train_images.extend(data[model]["real"][i] for i in ref_idx)
        train_labels.extend([0] * len(ref_idx))
        val_images.extend(data[model]["real"][i] for i in select_idx)
        val_labels.extend([0] * len(select_idx))
        for gen in generators:
            train_images.extend(data[model][f"darg::{gen}"][i] for i in train_idx)
            train_labels.extend([1] * len(train_idx))
            val_images.extend(data[model][f"darg::{gen}"][i] for i in val_idx)
            val_labels.extend([1] * len(val_idx))
    return ((np.asarray(train_images, dtype=object), np.asarray(train_labels)),
            (np.asarray(val_images, dtype=object), np.asarray(val_labels)))


def paired_summary(results, feature):
    seed_macros = []
    deltas = []
    bootstrap = {}
    for seed in SEEDS:
        full_scores, lr_scores = {}, {}
        full_auc, lr_auc = [], []
        for target in TARGETS:
            bank = bank_name(available_models(target))
            entry = results["folds"][target]["seeds"][str(seed)]["banks"][bank]
            full_auc.append(entry[feature]["AUROC"])
            lr_entry = results["folds"][target]["seeds"][str(seed)]["banks"][bank][
                "likelihood_reconstruction" if feature == "full" else "full"]
            lr_auc.append(lr_entry["AUROC"])
            prefix = score_prefix(target, seed, bank)
            saved = np.load(os.path.join(OUT, prefix + ".npz"))
            full_scores[target] = (
                np.array([0] * N_TEST_REAL + [1] * N_TEST_FAKE),
                saved[f"{feature}_test"], saved["prada_test"])
        macro = float(np.mean(full_auc))
        seed_macros.append(macro)
        deltas.append(float(np.mean(full_auc) - np.mean(lr_auc)))
        if feature == "full":
            for target in TARGETS:
                bank = bank_name(available_models(target))
                saved = np.load(os.path.join(OUT, score_prefix(target, seed, bank) + ".npz"))
                lr_scores[target] = (
                    np.array([0] * N_TEST_REAL + [1] * N_TEST_FAKE),
                    saved["likelihood_reconstruction_test"], saved["full_test"])
            delta, interval = paired_boot_macro(lr_scores, n_boot=2000, seed=9100 + seed)
            bootstrap[str(seed)] = {"LR_minus_full": delta, "CI95": interval}
    return {"macro_AUROC_per_seed": seed_macros,
            "mean_AUROC": float(np.mean(seed_macros)),
            "std_AUROC": float(np.std(seed_macros, ddof=1)),
            "feature_minus_other_per_seed": deltas,
            "paired_bootstrap_LR_minus_full": bootstrap}


def score_prefix(target, seed, bank):
    name = bank if isinstance(bank, str) else bank_name(bank)
    safe_bank = name.replace("+", "__")
    return f"scores_{target}_seed{seed}_{safe_bank}"


def coverage_summary(results):
    rows = []
    additions = []
    for target in TARGETS:
        for seed in COVERAGE_SEEDS:
            entries = results["folds"][target]["seeds"][str(seed)]["banks"]
            for bank, entry in entries.items():
                models = tuple(bank.split("+"))
                for feature in FEATURES:
                    rows.append({"target": target, "seed": seed, "bank": bank,
                                 "feature": feature, "size": len(models),
                                 "paradigm_covered": any(MODEL_ATTRIBUTES[m] == TARGET_ATTRIBUTES[target]
                                                         for m in models),
                                 "AUROC": entry[feature]["AUROC"],
                                 "FPR": entry[feature]["FPR"]})
            for base_size in range(1, len(available_models(target))):
                for base in itertools.combinations(available_models(target), base_size):
                    base_name = bank_name(base)
                    for model in available_models(target):
                        if model in base:
                            continue
                        expanded = tuple(m for m in available_models(target) if m in base + (model,))
                        expanded_name = bank_name(expanded)
                        for feature in FEATURES:
                            delta = (entries[expanded_name][feature]["AUROC"]
                                     - entries[base_name][feature]["AUROC"])
                            additions.append({"target": target, "seed": seed, "feature": feature,
                                              "added": model, "base_size": base_size,
                                              "paradigm_match": MODEL_ATTRIBUTES[model] == TARGET_ATTRIBUTES[target],
                                              "delta_AUROC": delta})
    summary = {}
    for feature in FEATURES:
        subset = [row for row in rows if row["feature"] == feature]
        by_size = {}
        for size in sorted(set(row["size"] for row in subset)):
            values = [row["AUROC"] for row in subset if row["size"] == size]
            by_size[str(size)] = {"n": len(values), "mean_AUROC": float(np.mean(values))}
        by_paradigm = {}
        for covered in (False, True):
            values = [row["AUROC"] for row in subset if row["paradigm_covered"] == covered]
            by_paradigm[str(covered)] = {"n": len(values), "mean_AUROC": float(np.mean(values))}
        matched = [x["delta_AUROC"] for x in additions
                   if x["feature"] == feature and x["paradigm_match"]]
        unmatched = [x["delta_AUROC"] for x in additions
                     if x["feature"] == feature and not x["paradigm_match"]]
        from scipy.stats import spearmanr

        per_target = {}
        for target in TARGETS:
            target_rows = [row for row in subset if row["target"] == target]
            size_trend = {}
            for size in sorted(set(row["size"] for row in target_rows)):
                values = [row["AUROC"] for row in target_rows if row["size"] == size]
                size_trend[str(size)] = float(np.mean(values))
            correlation = spearmanr([row["size"] for row in target_rows],
                                    [row["AUROC"] for row in target_rows]).statistic
            per_target[target] = {"mean_AUROC_by_bank_size": size_trend,
                                  "spearman_size_AUROC_descriptive": float(correlation)}
        summary[feature] = {
            "by_bank_size": by_size, "by_paradigm_coverage": by_paradigm,
            "per_target": per_target,
            "paired_add_paradigm_match": {"n": len(matched), "mean_delta": float(np.mean(matched))},
            "paired_add_paradigm_mismatch": {"n": len(unmatched), "mean_delta": float(np.mean(unmatched))},
            "note": "bank configurations and seeds are not independent datasets",
        }
    return summary


def evaluate():
    packed_path = os.path.join(OUT, "packed_cycle.pt")
    if os.path.exists(packed_path):
        packed = torch.load(packed_path, map_location="cpu", weights_only=False)
    else:
        packed = {(ds, model): pack_external(ds, model)
                  for ds in ("real",) + TARGETS for model in MODELS}
        torch.save(packed, packed_path + ".tmp")
        os.replace(packed_path + ".tmp", packed_path)
    darg_path = os.path.join(family.OUT, "packed_cycle.pt")
    darg = torch.load(darg_path, map_location="cpu", weights_only=False)
    results = {"config": {
        "models": MODELS, "targets": TARGETS, "training_generators": DARG_GENS,
        "seeds": SEEDS, "coverage_seeds": COVERAGE_SEEDS, "features": FEATURES,
        "real": "ARForensics/test/Infinity/0_real fixed common pool",
        "protocol": "DARG fake train; ARForensics real 300/200/200/300; fixed tests",
        "n_real": N_REAL, "n_fake": N_FAKE, "test_real": N_TEST_REAL,
        "test_fake": N_TEST_FAKE, "epochs": 40, "pooling": "attention",
        "aggregation": "max", "target_to_darg": TARGET_TO_DARG,
        "model_paradigm": MODEL_ATTRIBUTES, "target_paradigm": TARGET_ATTRIBUTES,
        "example": True,
    }, "folds": {target: {"seeds": {}} for target in TARGETS}}
    for seed in SEEDS:
        ref_idx, select_idx, cal_idx, test_idx, train_idx, val_idx, fake_test = fixed_splits(seed)
        normalized = {}
        for model in MODELS:
            inputs = {"real": packed["real", model]}
            inputs.update({f"external::{target}": packed[target, model] for target in TARGETS})
            inputs.update({f"darg::{gen}": darg[gen, model] for gen in DARG_GENS})
            normalized[model] = family.c2i.normalize_data(inputs, ref_idx)
        contexts = evaluation_contexts(seed)
        for feature in FEATURES:
            data = {model: {ds: select_feature_mode(images, feature)
                            for ds, images in normalized[model].items()}
                    for model in MODELS}
            for (train_gens, bank), targets in contexts.items():
                train_data, val_data = build_training_data(
                    data, bank, train_gens, ref_idx, select_idx, train_idx, val_idx)
                dart = train_dart(train_data, val_data, seed)
                dart_real, dart_cal = [], []
                prada_real = []
                for model in bank:
                    scores = predict(dart, data[model]["real"])
                    median = np.median(scores[select_idx])
                    iqr = (np.percentile(scores[select_idx], 75)
                           - np.percentile(scores[select_idx], 25) + 1e-6)
                    dart_real.append((scores - median) / iqr)
                    dart_cal.append((scores[cal_idx] - median) / iqr)
                    prada_real.append(prada_external("real", model))
                real_bank = np.stack(dart_real, axis=1)
                prada_real_bank = np.stack(prada_real, axis=1)
                prada_cal = np.max(prada_real_bank[cal_idx], axis=1)
                prada_tau = conformal_tau(prada_cal)
                for target in targets:
                    fake_bank = []
                    for model_index, model in enumerate(bank):
                        scores = predict(dart, data[model][f"external::{target}"])
                        raw_real = predict(dart, data[model]["real"])
                        median = np.median(raw_real[select_idx])
                        iqr = (np.percentile(raw_real[select_idx], 75)
                               - np.percentile(raw_real[select_idx], 25) + 1e-6)
                        fake_bank.append((scores - median) / iqr)
                    fake_bank = np.stack(fake_bank, axis=1)
                    test_real = np.max(real_bank[test_idx], axis=1)
                    test_fake = np.max(fake_bank[fake_test], axis=1)
                    cal = np.max(real_bank[cal_idx], axis=1)
                    tau = conformal_tau(cal)
                    test = np.concatenate([test_real, test_fake])
                    y = np.array([0] * N_TEST_REAL + [1] * N_TEST_FAKE)
                    prada_test = np.concatenate([
                        np.max(prada_real_bank[test_idx], axis=1),
                        np.max(np.stack([prada_external(target, model) for model in bank], axis=1)[fake_test], axis=1)])
                    seed_entry = results["folds"][target]["seeds"].setdefault(str(seed), {"banks": {}})
                    entry = seed_entry["banks"].setdefault(bank_name(bank), {})
                    entry[feature] = family.c2i.metric(y, test, N_TEST_REAL, tau)
                    entry["prada"] = family.c2i.metric(y, prada_test, N_TEST_REAL, prada_tau)
                    prefix = score_prefix(target, seed, bank)
                    save_values = {"y": y, "prada_test": prada_test,
                                   "prada_cal": prada_cal, "prada_threshold": prada_tau,
                                   "models": np.array(bank)}
                    old_path = os.path.join(OUT, prefix + ".npz")
                    if os.path.exists(old_path):
                        old = np.load(old_path)
                        save_values.update({key: old[key] for key in old.files
                                            if key.startswith(FEATURES) or key in (
                                                "full_cal", "full_threshold",
                                                "likelihood_reconstruction_cal",
                                                "likelihood_reconstruction_threshold")})
                    save_values.update({f"{feature}_test": test, f"{feature}_cal": cal,
                                        f"{feature}_threshold": tau})
                    np.savez(old_path, **save_values)
                    print(f"seed={seed} target={target} bank={bank_name(bank)} "
                          f"{feature}={entry[feature]['AUROC']:.4f} "
                          f"PRADA={entry['prada']['AUROC']:.4f}", flush=True)
        with open(os.path.join(OUT, "dart_external_family_partial.json"), "w") as f:
            json.dump(results, f, indent=2)
    results["summary"] = {feature: paired_summary(results, feature) for feature in FEATURES}
    results["coverage_summary"] = coverage_summary(results)
    with open(os.path.join(OUT, "dart_external_family.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(json.dumps(results["coverage_summary"], indent=2), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    stage = os.environ.get("DART_EXTERNAL_STAGE", "evaluate")
    models = tuple(os.environ.get("DART_EXTERNAL_MODELS", ",".join(MODELS)).split(","))
    datasets = tuple(os.environ.get("DART_EXTERNAL_DATASETS", ",".join(("real",) + TARGETS)).split(","))
    if stage == "prepare":
        prepare_data()
    elif stage == "prompts":
        ensure_prompts(datasets)
    elif stage == "features":
        for model in models:
            extract_model(model, datasets)
    elif stage == "evaluate":
        evaluate()
    else:
        raise ValueError(f"unknown stage: {stage}")


if __name__ == "__main__":
    main()
