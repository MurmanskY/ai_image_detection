"""Raw cycle residual 的跨 codec 与 ImageNet-V2 外部真实校准验证。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
os.environ.setdefault("DART_N_REAL", "1000")
os.environ.setdefault("DART_REF_REAL", "300")
os.environ.setdefault("DART_SELECT_REAL", "200")
os.environ.setdefault("DART_CAL_REAL", "200")

from analysis import dart_c2i_p0 as c2i  # noqa: E402
from analysis import dart_external_real_p0 as imagenet  # noqa: E402
from analysis import dart_interaction_p0 as interaction  # noqa: E402
from analysis import dart_source_control_p0 as source  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_cycle_robustness")
SEEDS = (0, 1, 2)
CODECS = ("original", "jpeg95", "jpeg85", "jpeg75", "jpeg50")
AGGREGATIONS = ("max", "mean", "median")
COMPONENTS = ("reconstruction", "cycle_residual")


def pack_cached(path):
    import torch

    features = torch.load(path, map_location="cpu", weights_only=False)
    diff = (features["cond_gt_llh_BL"] - features["uncond_gt_llh_BL"]).float().numpy()
    lengths = features.get("scale_lengths") or [diff.shape[1]]
    quant = features["quant_err"].float().numpy()
    quant = np.stack([row.reshape(len(lengths), -1).mean(axis=1) for row in quant])
    rec_key = "rec_err" if "rec_err" in features else "aeroblade"
    return c2i.scale_tensor(diff, lengths, quant, features[rec_key].float().numpy())


def component_value(residualizer, image, component):
    x, rec = residualizer.design([image])
    if component == "reconstruction":
        value = rec
    elif component == "cycle_residual":
        value = rec - x @ residualizer.coef
    else:
        raise ValueError(component)
    return float(value.mean() / residualizer.residual_scale)


def bank_score(residualizers, bank, images, aggregation, component):
    scores = []
    for model in bank:
        values = [component_value(residualizers[model], image, component) for image in images[model]]
        scores.append(values)
    return source.aggregate(np.stack(scores, axis=1), aggregation)


def take(images, indices):
    return np.asarray(images, dtype=object)[indices]


def metric_for(real, fake, cal):
    tau = conformal_tau(cal)
    y = np.array([0] * len(real) + [1] * len(fake))
    return c2i.metric(y, np.concatenate([real, fake]), len(real), tau), tau


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    original = {(ds, model): c2i.pack(ds, model)
                for ds in c2i.DATASETS for model in c2i.MODELS}
    compressed = {}
    for codec in CODECS[1:]:
        root = os.path.join(c2i.OUT, f"codec_{codec}")
        compressed[codec] = {(ds, model): pack_cached(os.path.join(
            root, f"dataset={ds}_model={model}.pt"))
            for ds in c2i.DATASETS for model in c2i.MODELS}
    external_real = {model: pack_cached(imagenet.cache_path(model, shard=False))
                     for model in c2i.MODELS}
    results = {"config": {
        "seeds": SEEDS, "codecs": CODECS, "aggregations": AGGREGATIONS,
        "components": COMPONENTS, "models": c2i.MODELS, "generators": c2i.FAKE_GENS,
        "protocol": "original real reference/selection/calibration; fixed compressed tests; no retuning",
        "external_real": "ImageNet-V2 matched-frequency 300; no threshold retuning",
        "example": True,
    }, "seeds": {}}
    for seed in SEEDS:
        ref_idx, select_idx, cal_idx, test_idx, _, _, fake_test = c2i.fixed_test_split(seed)
        residualizers = {}
        for model in c2i.MODELS:
            residualizers[model] = interaction.CycleResidualizer().fit(
                [original["real", model][i] for i in ref_idx],
                [original["real", model][i] for i in select_idx])
        seed_result = {"external_FPR": {}, "codecs": {}}
        for aggregation in AGGREGATIONS:
            for component in COMPONENTS:
                cal = bank_score(residualizers, c2i.MODELS,
                                 {model: take(original["real", model], cal_idx) for model in c2i.MODELS},
                                 aggregation, component)
                external_score = bank_score(residualizers, c2i.MODELS, external_real,
                                            aggregation, component)
                tau = conformal_tau(cal)
                false_positives = int((external_score > tau).sum())
                seed_result["external_FPR"][f"{aggregation}::{component}"] = {
                    "FPR": float(false_positives / len(external_score)),
                    "CI95": c2i.wilson_interval(false_positives, len(external_score)),
                    "threshold": tau,
                }
                np.savez(os.path.join(OUT, f"external_{aggregation}_{component}_seed{seed}.npz"),
                         external=external_score, cal=cal, threshold=tau)
                for codec in CODECS:
                    packed = original if codec == "original" else compressed[codec]
                    real = bank_score(residualizers, c2i.MODELS,
                                      {model: packed["real", model] if codec != "original"
                                       else take(packed["real", model], test_idx) for model in c2i.MODELS},
                                      aggregation, component)
                    per_gen = {}
                    paired = {}
                    for gen in c2i.FAKE_GENS:
                        fake = bank_score(residualizers, c2i.MODELS,
                                          {model: packed[gen, model] if codec != "original"
                                           else take(packed[gen, model], fake_test) for model in c2i.MODELS},
                                          aggregation, component)
                        metrics, gen_tau = metric_for(real, fake, cal)
                        assert np.isclose(gen_tau, tau)
                        per_gen[gen] = metrics
                        y = np.array([0] * len(real) + [1] * len(fake))
                        paired[gen] = (y, np.concatenate([real, fake]), np.concatenate([real, fake]))
                        np.savez(os.path.join(OUT, f"scores_{codec}_{aggregation}_{component}_seed{seed}_{gen}.npz"),
                                 y=y, test=np.concatenate([real, fake]), cal=cal, threshold=tau)
                    seed_result["codecs"].setdefault(codec, {})[f"{aggregation}::{component}"] = {
                        "per_generator": per_gen,
                        "macro_AUROC": float(np.mean([x["AUROC"] for x in per_gen.values()])),
                        "FPR": float(per_gen["var_d16"]["FPR"]),
                    }
        results["seeds"][str(seed)] = seed_result
        print(f"completed robustness seed={seed}", flush=True)
    results["summary"] = summarize(results)
    results["paired_residual_vs_reconstruction"] = paired_tests(results)
    with open(os.path.join(OUT, "dart_cycle_robustness.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def summarize(results):
    summary = {"codecs": {}, "external_FPR": {}}
    for codec in CODECS:
        for aggregation in AGGREGATIONS:
            for component in COMPONENTS:
                key = f"{aggregation}::{component}"
                entries = [results["seeds"][str(seed)]["codecs"][codec][key] for seed in SEEDS]
                summary["codecs"].setdefault(codec, {})[key] = {
                    "macro_AUROC_per_seed": [x["macro_AUROC"] for x in entries],
                    "mean": float(np.mean([x["macro_AUROC"] for x in entries])),
                    "std": float(np.std([x["macro_AUROC"] for x in entries], ddof=1)),
                    "FPR": [x["FPR"] for x in entries],
                }
    for aggregation in AGGREGATIONS:
        for component in COMPONENTS:
            key = f"{aggregation}::{component}"
            entries = [results["seeds"][str(seed)]["external_FPR"][key] for seed in SEEDS]
            summary["external_FPR"][key] = {"FPR": [x["FPR"] for x in entries],
                                             "CI95": [x["CI95"] for x in entries]}
    return summary


def paired_tests(results):
    output = {}
    for codec in CODECS:
        for aggregation in AGGREGATIONS:
            seed_results = {}
            for seed in SEEDS:
                paired = {}
                for gen in c2i.FAKE_GENS:
                    residual = np.load(os.path.join(
                        OUT, f"scores_{codec}_{aggregation}_cycle_residual_seed{seed}_{gen}.npz"))
                    reconstruction = np.load(os.path.join(
                        OUT, f"scores_{codec}_{aggregation}_reconstruction_seed{seed}_{gen}.npz"))
                    paired[gen] = (residual["y"], residual["test"], reconstruction["test"])
                delta, interval = paired_boot_macro(paired, n_boot=1000, seed=28000 + seed)
                seed_results[str(seed)] = {"delta": delta, "CI95": interval}
            output[f"{codec}::{aggregation}"] = seed_results
    return output


def main():
    evaluate()


if __name__ == "__main__":
    main()
