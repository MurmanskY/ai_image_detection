"""DART ImageNet-V2 外部真实校准迁移 smoke（默认 seed 0）。"""

import json
import os
import sys
import tarfile

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_c2i_p0 as c2i  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_p0 import normalize_data, predict, scale_tensor, train_dart  # noqa: E402
from analysis.scale_p0 import prada_score  # noqa: E402
from prada.metrics import aeroblade  # noqa: E402

TAR = os.path.join(PROJ, "data", "imagenetv2", "test", "0.tar")
SUBSET = os.path.join(PROJ, "data", "imagenetv2", "subset300")
OUT = os.path.join(c2i.OUT, "imagenetv2")
MODELS = c2i.MODELS
EXTRACT_ONLY = os.environ.get("DART_EXTRACT_ONLY", "0") == "1"
SHARD_INDEX = int(os.environ.get("DART_SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("DART_SHARD_COUNT", "1"))
MERGE_SHARDS = os.environ.get("DART_EXTERNAL_MERGE", "0") == "1"


def selected_members(tar_path=TAR, n=300):
    with tarfile.open(tar_path) as archive:
        names = {m.name for m in archive.getmembers() if m.isfile()}
    prefixes = sorted(name.rsplit(".", 1)[0] for name in names if name.endswith(".webp"))
    prefixes = [p for p in prefixes if f"{p}.cls" in names]
    idx = np.random.RandomState(0).choice(len(prefixes), n, replace=False)
    return [prefixes[i] for i in idx]


def prepare_subset(tar_path=TAR, root=SUBSET, n=300):
    manifest_path = os.path.join(root, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    os.makedirs(root, exist_ok=True)
    for label in range(1000):
        os.makedirs(os.path.join(root, f"{label:03d}"), exist_ok=True)
    prefixes = set(selected_members(tar_path, n))
    manifest = []
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            prefix, extension = member.name.rsplit(".", 1)
            if prefix not in prefixes or extension != "cls":
                continue
            label = int(archive.extractfile(member).read().strip())
            image = archive.extractfile(f"{prefix}.webp").read()
            filename = f"{prefix}.webp"
            with open(os.path.join(root, f"{label:03d}", filename), "wb") as f:
                f.write(image)
            manifest.append({"prefix": prefix, "label": label})
    order = {prefix: i for i, prefix in enumerate(selected_members(tar_path, n))}
    manifest.sort(key=lambda row: order[row["prefix"]])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def positions():
    return np.arange(SHARD_INDEX, 300, SHARD_COUNT, dtype=np.int64)


def cache_path(model, shard=True):
    path = os.path.join(OUT, f"real_model={model}.pt")
    if shard and SHARD_COUNT > 1:
        return f"{path}.shard{SHARD_COUNT}_{SHARD_INDEX}.pt"
    return path


def loader(wrapper, batch_size=1):
    prepare_subset()
    dataset = ImageFolder(SUBSET, transform=wrapper.transform, allow_empty=True)
    indices = positions()
    return DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size,
                      num_workers=c2i.NUM_WORKERS, pin_memory=True)


def save(features, path):
    torch.save(features, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def extract_likelihood(model):
    c2i.acquire_gpu()
    wrapper = c2i.load_model_wrapper(model)
    path = cache_path(model)
    if not os.path.exists(path):
        features = {"positions": positions()}
        rows = []
        for image, condition in loader(wrapper):
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
        save(features, path)
        print(f"  external likelihood {model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


@torch.no_grad()
def extract_cycle(model):
    c2i.acquire_gpu()
    wrapper = c2i.load_model_wrapper(model)
    path = cache_path(model)
    features = torch.load(path, map_location="cpu", weights_only=False)
    for key, batch_size in (("quant_err", c2i.cycle_batch_size(model)),
                            ("rec_err", c2i.rec_batch_size(model))):
        if key in features:
            continue
        values = []
        for image, _ in loader(wrapper, batch_size=batch_size):
            image = image.to(wrapper.device, non_blocking=True)
            rec = wrapper.get_ae_rec_and_quant_error(image_B3HW=image)
            if key == "quant_err":
                values.append(-rec["quant_err_BL"].cpu())
            else:
                values.append(-aeroblade(
                    img1=image, img2=rec["rec_B3HW"], net_type="vgg", reduction="none",
                    normalize=wrapper.range_after_transform == (0, 1)).cpu().reshape(-1))
        features[key] = torch.cat(values)
        save(features, path)
        print(f"  external {key} {model}", flush=True)
    del wrapper
    torch.cuda.empty_cache()


def merge_shards(model):
    base = cache_path(model, shard=False)
    shards = [torch.load(f"{base}.shard{SHARD_COUNT}_{i}.pt",
                         map_location="cpu", weights_only=False) for i in range(SHARD_COUNT)]
    output = {"positions": np.arange(300)}
    for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL", "quant_err", "rec_err"):
        value = torch.empty((300, *shards[0][key].shape[1:]), dtype=shards[0][key].dtype)
        for i, shard in enumerate(shards):
            assert np.array_equal(shard["positions"], np.arange(i, 300, SHARD_COUNT))
            value[i::SHARD_COUNT] = shard[key]
        assert torch.isfinite(value).all()
        output[key] = value
    if "scale_lengths" in shards[0]:
        output["scale_lengths"] = shards[0]["scale_lengths"]
    save(output, base)
    print(f"  merged external {model}", flush=True)


def pack(model):
    feat = torch.load(cache_path(model), map_location="cpu", weights_only=False)
    diff = (feat["cond_gt_llh_BL"] - feat["uncond_gt_llh_BL"]).float().numpy()
    lengths = feat.get("scale_lengths") or [diff.shape[1]]
    quant = feat["quant_err"].float().numpy()
    quant = np.stack([row.reshape(len(lengths), -1).mean(axis=1) for row in quant])
    return scale_tensor(diff, lengths, quant, feat["rec_err"].float().numpy())


def evaluate(seed=0):
    prepare_subset()
    original = {(ds, model): c2i.pack(ds, model)
                for ds in c2i.DATASETS for model in c2i.MODELS}
    external = {model: pack(model) for model in c2i.MODELS}
    real_perm = np.random.RandomState(seed).permutation(c2i.N_REAL)
    ref_idx = real_perm[:c2i.N_REF_REAL]
    cal_idx = real_perm[c2i.N_REF_REAL:c2i.N_REF_REAL + c2i.N_CAL_REAL]
    train_fake = {"var_d20": "var_d16", "var_d30": "var_d16",
                  "llamagen_b256": "llamagen_b256",
                  "llamagen_l256": "llamagen_b256", "rar_l": "rar_b"}
    external_dart = np.full(300, -np.inf, dtype=np.float32)
    external_prada = np.full(300, -np.inf, dtype=np.float32)
    real_cal = np.full(c2i.N_REAL, -np.inf, dtype=np.float32)
    prada_cal = np.full(c2i.N_REAL, -np.inf, dtype=np.float32)
    fake_dart = {gen: np.full(c2i.N_IMAGES, -np.inf, dtype=np.float32)
                 for gen in c2i.FAKE_GENS}
    fake_prada = {gen: np.full(c2i.N_IMAGES, -np.inf, dtype=np.float32)
                  for gen in c2i.FAKE_GENS}
    for model in c2i.MODELS:
        inputs = {"real": original["real", model], "fake": original[train_fake[model], model]}
        normalized = normalize_data(inputs, ref_idx)
        ref = np.concatenate([original["real", model][i] for i in ref_idx], axis=0)
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
        external_normalized = np.asarray([(x - mu) / sd for x in external[model]], dtype=object)
        train_idx, val_idx, _ = c2i.fake_split(seed)
        dart = train_dart(
            (np.concatenate([normalized["real"][ref_idx], normalized["fake"][train_idx]]),
             np.array([0] * c2i.N_REF_REAL + [1] * 100)),
            (np.concatenate([normalized["real"][cal_idx], normalized["fake"][val_idx]]),
             np.array([0] * c2i.N_CAL_REAL + [1] * 50)), seed)
        real_scores = predict(dart, normalized["real"])
        median = np.median(real_scores[cal_idx])
        iqr = (np.percentile(real_scores[cal_idx], 75)
               - np.percentile(real_scores[cal_idx], 25) + 1e-6)
        real_cal = np.maximum(real_cal, (real_scores - median) / iqr)
        prada_cal = np.maximum(prada_cal, c2i.prada_c2i("real", model))
        external_dart = np.maximum(
            external_dart, (predict(dart, external_normalized) - median) / iqr)
        feat = torch.load(cache_path(model), map_location="cpu", weights_only=False)
        external_prada = np.maximum(external_prada, prada_score(feat, model))
        for gen in c2i.FAKE_GENS:
            normalized_fake = np.asarray([(x - mu) / sd for x in original[gen, model]], dtype=object)
            fake_dart[gen] = np.maximum(
                fake_dart[gen], (predict(dart, normalized_fake) - median) / iqr)
            fake_prada[gen] = np.maximum(fake_prada[gen], c2i.prada_c2i(gen, model))
    tau = conformal_tau(real_cal[cal_idx])
    prada_tau = conformal_tau(prada_cal[cal_idx])
    results = {"config": {"seed": seed, "external_real": "ImageNet-V2 matched-frequency shard0",
                          "n_external": 300, "protocol": "original train/calibration; no retuning"},
               "external_FPR": {"dart": float((external_dart > tau).mean()),
                                "prada": float((external_prada > prada_tau).mean())},
               "generators": {}}
    _, _, test_fake = c2i.fake_split(seed)
    for gen in c2i.FAKE_GENS:
        y = np.array([0] * 300 + [1] * 100)
        dart = np.concatenate([external_dart, fake_dart[gen][test_fake]])
        prada = np.concatenate([external_prada, fake_prada[gen][test_fake]])
        results["generators"][gen] = {"dart": c2i.metric(y, dart, 300, tau),
                                      "prada": c2i.metric(y, prada, 300, prada_tau)}
        np.savez(os.path.join(c2i.OUT, f"scores_dart_imagenetv2_seed{seed}_{gen}.npz"),
                 dart=dart, prada=prada, n_real=300)
    results["macro_AUROC"] = {
        method: float(np.mean([results["generators"][gen][method]["AUROC"]
                               for gen in c2i.FAKE_GENS])) for method in ("dart", "prada")}
    path = os.path.join(c2i.OUT, "dart_c2i_imagenetv2_smoke.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


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
