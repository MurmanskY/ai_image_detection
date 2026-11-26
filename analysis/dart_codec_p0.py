"""DART JPEG95 P2 smoke：原图训练/校准，压缩图测试（默认 seed 0）。"""

import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_c2i_p0 as c2i
from analysis.carf_p0 import conformal_tau
from analysis.dart_p0 import normalize_data, predict, scale_tensor, train_dart
from analysis.scale_p0 import prada_score
from prada.metrics import aeroblade

QUALITY = int(os.environ.get("DART_CODEC_QUALITY", "95"))
CODEC = f"jpeg{QUALITY}"
OUT = os.path.join(c2i.OUT, f"codec_{CODEC}")
SEEDS = tuple(int(x) for x in os.environ.get("DART_CODEC_SEEDS", "0").split(","))
MODELS = c2i.MODELS
DATASETS = c2i.DATASETS
FAKE_GENS = c2i.FAKE_GENS
EXTRACT_ONLY = os.environ.get("DART_EXTRACT_ONLY", "0") == "1"
SHARD_INDEX = int(os.environ.get("DART_SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("DART_SHARD_COUNT", "1"))
MERGE_SHARDS = os.environ.get("DART_CODEC_MERGE", "0") == "1"


def test_positions(ds, seed):
    if ds == "real":
        return c2i.fixed_test_split(seed)[3]
    return c2i.fixed_test_split(seed)[6]


def codec_positions(ds):
    return np.array(sorted(set(np.concatenate([
        test_positions(ds, seed) for seed in SEEDS]))), dtype=np.int64)


def extraction_positions(ds):
    return codec_positions(ds)[SHARD_INDEX::SHARD_COUNT]


def position_rows(ds, positions):
    lookup = {int(p): i for i, p in enumerate(codec_positions(ds))}
    return np.array([lookup[int(p)] for p in positions], dtype=np.int64)


def base_indices(ds):
    dataset = ImageFolder(os.path.join(c2i.DATA_ROOT, ds))
    return np.random.RandomState(0).choice(
        len(dataset), c2i.sample_count(ds), replace=False)


class JPEGDataset(Dataset):
    def __init__(self, ds, transform, positions=None):
        self.dataset = ImageFolder(os.path.join(c2i.DATA_ROOT, ds))
        if positions is None:
            positions = extraction_positions(ds)
        self.indices = base_indices(ds)[positions]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, condition = self.dataset[int(self.indices[index])]
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=QUALITY)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")
        return self.transform(image), condition


def loader(ds, wrapper, batch_size=1, positions=None):
    return DataLoader(JPEGDataset(ds, wrapper.transform, positions=positions),
                      batch_size=batch_size, num_workers=c2i.NUM_WORKERS,
                      pin_memory=True)


def cache_path(ds, model, shard=True):
    path = os.path.join(OUT, f"dataset={ds}_model={model}.pt")
    if shard and SHARD_COUNT > 1:
        return f"{path}.shard{SHARD_COUNT}_{SHARD_INDEX}.pt"
    return path


def save_features(features, path):
    torch.save(features, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def extract_likelihood(model):
    c2i.acquire_gpu()
    wrapper = c2i.load_model_wrapper(model)
    for ds in DATASETS:
        path = cache_path(ds, model)
        features = torch.load(path, map_location="cpu", weights_only=False) \
            if os.path.exists(path) else {"positions": extraction_positions(ds)}
        if "cond_gt_llh_BL" in features:
            continue
        rows = []
        for image, condition in loader(ds, wrapper):
            image = image.to(wrapper.device, non_blocking=True)
            condition = condition.to(wrapper.device, non_blocking=True)
            gt = wrapper.get_gt_idx(image_B3HW=image)
            logits = wrapper.get_logits(gt_idx_BL=gt["gt_idx_BL"], condition_B=condition)
            llh = wrapper.get_gt_llh(logits_BLX=logits, gt_idx_BL=gt["gt_idx_BL"])
            rows.append({k: v.cpu() for k, v in llh.items()})
        for key in rows[0]:
            features[key] = torch.cat([row[key] for row in rows])
        if hasattr(wrapper, "scale_lengths"):
            features["scale_lengths"] = wrapper.scale_lengths
        save_features(features, path)
        print(f"  {CODEC} likelihood {ds}/{model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


@torch.no_grad()
def extract_cycle(model):
    c2i.acquire_gpu()
    wrapper = c2i.load_model_wrapper(model)
    for ds in DATASETS:
        path = cache_path(ds, model)
        features = torch.load(path, map_location="cpu", weights_only=False)
        for key, batch_size in (("quant_err", c2i.cycle_batch_size(model)),
                                ("rec_err", c2i.rec_batch_size(model))):
            if key in features:
                continue
            values = []
            for image, _ in loader(ds, wrapper, batch_size=batch_size):
                image = image.to(wrapper.device, non_blocking=True)
                rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
                if key == "quant_err":
                    values.append(-rec["quant_err_BL"].cpu())
                else:
                    values.append(-aeroblade(
                        img1=image, img2=rec["rec_B3HW"], net_type="vgg",
                        reduction="none", normalize=wrapper.range_after_transform == (0, 1)
                    ).cpu().reshape(-1))
            features[key] = torch.cat(values)
            save_features(features, path)
            print(f"  {CODEC} {key} {ds}/{model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


def merge_shards(model):
    for ds in DATASETS:
        base = cache_path(ds, model, shard=False)
        shards = [torch.load(f"{base}.shard{SHARD_COUNT}_{i}.pt",
                             map_location="cpu", weights_only=False)
                  for i in range(SHARD_COUNT)]
        output = {"positions": codec_positions(ds)}
        for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL", "quant_err", "rec_err"):
            value = torch.empty((len(codec_positions(ds)), *shards[0][key].shape[1:]),
                                dtype=shards[0][key].dtype)
            for i, shard in enumerate(shards):
                assert np.array_equal(shard["positions"], codec_positions(ds)[i::SHARD_COUNT])
                value[i::SHARD_COUNT] = shard[key]
            assert torch.isfinite(value).all()
            output[key] = value
        if "scale_lengths" in shards[0]:
            output["scale_lengths"] = shards[0]["scale_lengths"]
        save_features(output, base)
        print(f"  merged {CODEC} {ds}/{model}", flush=True)


def pack_codec(ds, model):
    feat = torch.load(cache_path(ds, model), map_location="cpu", weights_only=False)
    diff = (feat["cond_gt_llh_BL"] - feat["uncond_gt_llh_BL"]).float().numpy()
    lengths = feat.get("scale_lengths") or [diff.shape[1]]
    quant = feat["quant_err"].float().numpy()
    quant = np.stack([row.reshape(len(lengths), -1).mean(axis=1) for row in quant])
    return scale_tensor(diff, lengths, quant, feat["rec_err"].float().numpy())


def codec_prada(ds, model):
    feat = torch.load(cache_path(ds, model), map_location="cpu", weights_only=False)
    return prada_score(feat, model)


def evaluate():
    original = {(ds, model): c2i.pack(ds, model)
                for ds in DATASETS for model in MODELS}
    compressed = {(ds, model): pack_codec(ds, model)
                  for ds in DATASETS for model in MODELS}
    results = {"config": {"models": MODELS, "seeds": SEEDS, "codec": CODEC,
                          "protocol": "fixed seed0 test; reshuffled strict train/selection/calibration; compressed test"},
               "seeds": {}}
    train_fake = {"var_d20": "var_d16", "var_d30": "var_d16",
                  "llamagen_b256": "llamagen_b256",
                  "llamagen_l256": "llamagen_b256", "rar_l": "rar_b"}
    for seed in SEEDS:
        (ref_idx, select_idx, cal_idx, test_idx,
         train_idx, val_idx, fake_test) = c2i.fixed_test_split(seed)
        normalized, compressed_normalized = {}, {}
        for model in MODELS:
            inputs = {"real": original["real", model],
                      "fake": original[train_fake[model], model]}
            inputs.update({gen: original[gen, model] for gen in FAKE_GENS})
            normalized[model] = normalize_data(inputs, ref_idx)
            ref = np.concatenate([original["real", model][i] for i in ref_idx], axis=0)
            mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
            compressed_normalized[model] = {
                ds: np.asarray([(x - mu) / sd for x in compressed[ds, model]], dtype=object)
                for ds in DATASETS}
        combined = {ds: np.full(len(codec_positions(ds)), -np.inf, dtype=np.float32)
                    for ds in DATASETS}
        combined_prada = {ds: np.full(len(codec_positions(ds)), -np.inf, dtype=np.float32)
                          for ds in DATASETS}
        original_combined = {ds: np.full(c2i.sample_count(ds), -np.inf, dtype=np.float32)
                             for ds in DATASETS}
        original_prada = {ds: np.full(c2i.sample_count(ds), -np.inf, dtype=np.float32)
                          for ds in DATASETS}
        for model in MODELS:
            dart = train_dart(
                (np.concatenate([normalized[model]["real"][ref_idx],
                                 normalized[model]["fake"][train_idx]]),
                 np.array([0] * c2i.N_REF_REAL + [1] * 100)),
                (np.concatenate([normalized[model]["real"][select_idx],
                                 normalized[model]["fake"][val_idx]]),
                 np.array([0] * c2i.N_SELECT_REAL + [1] * 50)), seed)
            real_scores = predict(dart, normalized[model]["real"])
            median = np.median(real_scores[select_idx])
            iqr = (np.percentile(real_scores[select_idx], 75)
                   - np.percentile(real_scores[select_idx], 25) + 1e-6)
            for ds in DATASETS:
                original_scores = predict(dart, normalized[model][ds])
                original_combined[ds] = np.maximum(
                    original_combined[ds], (original_scores - median) / iqr)
                original_prada[ds] = np.maximum(
                    original_prada[ds], c2i.prada_c2i(ds, model))
                scores = predict(dart, compressed_normalized[model][ds])
                combined[ds] = np.maximum(combined[ds], (scores - median) / iqr)
                combined_prada[ds] = np.maximum(combined_prada[ds], codec_prada(ds, model))
        tau = conformal_tau(original_combined["real"][cal_idx])
        prada_tau = conformal_tau(original_prada["real"][cal_idx])
        per_gen = {}
        real_rows = position_rows("real", test_idx)
        for gen in FAKE_GENS:
            fake_idx = test_positions(gen, seed)
            fake_rows = position_rows(gen, fake_idx)
            y = np.array([0] * c2i.N_TEST_REAL + [1] * 100)
            dart_score = np.concatenate([combined["real"][real_rows], combined[gen][fake_rows]])
            prada = np.concatenate([combined_prada["real"][real_rows],
                                    combined_prada[gen][fake_rows]])
            original_dart = np.concatenate([original_combined["real"][test_idx],
                                            original_combined[gen][fake_test]])
            original_prada_score = np.concatenate([original_prada["real"][test_idx],
                                                   original_prada[gen][fake_test]])
            per_gen[gen] = {
                "original": {"dart": c2i.metric(y, original_dart, c2i.N_TEST_REAL, tau),
                             "prada": c2i.metric(y, original_prada_score,
                                                 c2i.N_TEST_REAL, prada_tau)},
                CODEC: {"dart": c2i.metric(y, dart_score, c2i.N_TEST_REAL, tau),
                        "prada": c2i.metric(y, prada, c2i.N_TEST_REAL, prada_tau)}}
            np.savez(os.path.join(
                c2i.OUT, f"scores_dart_{CODEC}_strict_fixed_test_seed{seed}_{gen}.npz"),
                dart=dart_score, prada=prada, n_real=c2i.N_TEST_REAL,
                original_dart=original_dart, original_prada=original_prada_score,
                dart_cal=original_combined["real"][cal_idx],
                prada_cal=original_prada["real"][cal_idx],
                dart_threshold=tau, prada_threshold=prada_tau)
        results["seeds"][str(seed)] = per_gen
        print(f"{CODEC} seed {seed}: " + "; ".join(
            f"{gen}={per_gen[gen][CODEC]['dart']['AUROC']:.4f}" for gen in FAKE_GENS),
            flush=True)
    results["summary"] = {
        "original_macro_AUROC": [float(np.mean([
            results["seeds"][str(seed)][gen]["original"]["dart"]["AUROC"]
            for gen in FAKE_GENS])) for seed in SEEDS],
        "macro_AUROC": [float(np.mean([
            results["seeds"][str(seed)][gen][CODEC]["dart"]["AUROC"]
            for gen in FAKE_GENS])) for seed in SEEDS],
        "FPR": [float(results["seeds"][str(seed)]["var_d16"][CODEC]["dart"]["FPR"])
                for seed in SEEDS]}
    path = os.path.join(c2i.OUT, f"dart_c2i_{CODEC}_strict_fixed_test.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    if MERGE_SHARDS:
        for model in MODELS:
            merge_shards(model)
        return
    for model in MODELS:
        extract_likelihood(model)
        extract_cycle(model)
    if not EXTRACT_ONLY:
        evaluate()


if __name__ == "__main__":
    main()
