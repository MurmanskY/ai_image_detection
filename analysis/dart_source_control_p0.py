"""DART 七家族来源混杂控制、聚合选择与低 FPR 复核。

全部复用已缓存特征，不新增网络。当前协议为 ARF real/DARG fake；对照包含
ARF 同源、DARG->ARF、ARF+DARG 来源平衡及 ARF->DARG 反向评估。目标家族
fake 与对应参考模型始终排除。聚合选择只看训练/validation，calibration 独立。
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_family_p0 as external  # noqa: E402
from analysis import dart_family_p0 as family  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_p0 import predict, train_dart  # noqa: E402

OUT = os.path.join(PROJ, "outputs", "dart_source_control")
SEEDS = tuple(int(x) for x in os.environ.get("DART_SOURCE_SEEDS", "0,1,2,3,4").split(","))
AGGREGATIONS = ("max", "mean", "median")
CONDITIONS = ("current", "arf_same", "darg_transfer", "balanced")


def real_sources(condition):
    return {"current": ("arf",), "arf_same": ("arf",),
            "darg_transfer": ("darg",), "balanced": ("arf", "darg")}[condition]


def training_fake_sources(condition, target):
    excluded = external.TARGET_TO_DARG.get(target)
    if condition == "arf_same":
        return {gen: ("arf",) for gen in external.TARGETS if gen != target}
    if condition == "balanced":
        return {gen: ("arf", "darg") for gen in external.DARG_GENS if gen != excluded}
    return {gen: ("darg",) for gen in external.DARG_GENS if gen != excluded}


def eval_sources(condition, target):
    if condition == "arf_same":
        return ("arf", "darg") if target in external.TARGET_TO_DARG else ("arf",)
    if condition == "balanced" and target in external.TARGET_TO_DARG:
        return ("arf", "darg")
    return ("arf",)


def target_dataset(target, source):
    if source == "arf":
        return target
    return external.TARGET_TO_DARG[target]


def normalize_model(arf, darg, model, sources, ref_idx, fake_sources, target):
    reference = []
    for source in sources:
        packed = arf if source == "arf" else darg
        reference.extend(packed["real", model][i] for i in ref_idx)
    reference = np.concatenate(reference, axis=0)
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0) + 1e-6
    data = {}
    for source in ("arf", "darg"):
        packed = arf if source == "arf" else darg
        data["real", source] = np.asarray([(x - mu) / sd for x in packed["real", model]],
                                          dtype=object)
    for gen, gen_sources in fake_sources.items():
        for source in gen_sources:
            packed = arf if source == "arf" else darg
            reverse = {darg_name: target_name
                       for target_name, darg_name in external.TARGET_TO_DARG.items()}
            key = reverse.get(gen, gen) if source == "arf" else gen
            if (key, model) not in packed:
                raise KeyError(f"missing packed features: {source}/{key}/{model}")
            data["fake", source, gen] = np.asarray(
                [(x - mu) / sd for x in packed[key, model]], dtype=object)
    for source in ("arf", "darg"):
        if source == "darg" and target not in external.TARGET_TO_DARG:
            continue
        packed = arf if source == "arf" else darg
        key = target_dataset(target, source)
        data["target", source] = np.asarray(
            [(x - mu) / sd for x in packed[key, model]], dtype=object)
    return data


def build_training_data(data, bank, sources, fake_sources, ref_idx, select_idx, train_idx, val_idx):
    train_images, train_labels, val_images, val_labels = [], [], [], []
    for model in bank:
        for source in sources:
            train_images.extend(data[model]["real", source][i] for i in ref_idx)
            train_labels.extend([0] * len(ref_idx))
            val_images.extend(data[model]["real", source][i] for i in select_idx)
            val_labels.extend([0] * len(select_idx))
        for gen, gen_sources in fake_sources.items():
            for source in gen_sources:
                train_images.extend(data[model]["fake", source, gen][i] for i in train_idx)
                train_labels.extend([1] * len(train_idx))
                val_images.extend(data[model]["fake", source, gen][i] for i in val_idx)
                val_labels.extend([1] * len(val_idx))
    return ((np.asarray(train_images, dtype=object), np.asarray(train_labels)),
            (np.asarray(val_images, dtype=object), np.asarray(val_labels)))


def aggregate(bank, aggregation):
    return {"max": np.max, "mean": np.mean, "median": np.median}[aggregation](bank, axis=1)


def selection_statistics(dart, data, bank, sources, select_idx):
    stats = []
    for model in bank:
        scores = np.concatenate([predict(dart, data[model]["real", source])[select_idx]
                                 for source in sources])
        stats.append((float(np.median(scores)),
                      float(np.percentile(scores, 75) - np.percentile(scores, 25) + 1e-6)))
    return stats


def select_aggregation(dart, data, bank, fake_sources, stats, select_idx, val_idx):
    macro = {aggregation: [] for aggregation in AGGREGATIONS}
    for gen, gen_sources in fake_sources.items():
        for source in gen_sources:
            real_z, fake_z = [], []
            for model, (median, iqr) in zip(bank, stats):
                real_z.append((predict(dart, data[model]["real", source])[select_idx] - median) / iqr)
                fake_z.append((predict(dart, data[model]["fake", source, gen])[val_idx] - median) / iqr)
            y = np.array([0] * len(select_idx) + [1] * len(val_idx))
            for aggregation in AGGREGATIONS:
                score = np.concatenate([aggregate(np.stack(real_z, axis=1), aggregation),
                                        aggregate(np.stack(fake_z, axis=1), aggregation)])
                macro[aggregation].append(roc_auc_score(y, score))
    means = {aggregation: float(np.mean(values)) for aggregation, values in macro.items()}
    selected = max(AGGREGATIONS, key=lambda aggregation: means[aggregation])
    return selected, means


def prada_scores(source, ds, model):
    if source == "arf":
        return external.prada_external(ds, model)
    features = dict(family.load_features(ds, model))
    count = external.N_REAL if ds == "real" else external.N_FAKE
    for key in ("cond_gt_llh_BL", "uncond_gt_llh_BL"):
        features[key] = features[key][:count]
    return family.prada_score(features, model)


def evaluate_target(dart, data, bank, stats, target, source,
                    select_idx, cal_idx, test_idx, fake_test, selected):
    real_z, fake_z, prada_real, prada_fake = [], [], [], []
    for model, (median, iqr) in zip(bank, stats):
        real = predict(dart, data[model]["real", source])
        fake = predict(dart, data[model]["target", source])
        real_z.append((real - median) / iqr)
        fake_z.append((fake - median) / iqr)
        prada_real.append(prada_scores(source, "real", model))
        prada_fake.append(prada_scores(source, target_dataset(target, source), model))
    real_bank = np.stack(real_z, axis=1)
    fake_bank = np.stack(fake_z, axis=1)
    y = np.array([0] * external.N_TEST_REAL + [1] * external.N_TEST_FAKE)
    metrics, saved = {}, {
        "y": y, "models": np.array(bank),
        "bank_real_test": real_bank[test_idx], "bank_fake_test": fake_bank[fake_test],
        "bank_real_cal": real_bank[cal_idx],
    }
    for aggregation in AGGREGATIONS + ("selected",):
        method = selected if aggregation == "selected" else aggregation
        test = np.concatenate([aggregate(real_bank[test_idx], method),
                               aggregate(fake_bank[fake_test], method)])
        cal = aggregate(real_bank[cal_idx], method)
        tau = conformal_tau(cal)
        metrics[aggregation] = family.c2i.metric(y, test, external.N_TEST_REAL, tau)
        saved[f"{aggregation}_test"] = test
        saved[f"{aggregation}_cal"] = cal
        saved[f"{aggregation}_threshold"] = tau
    prada_test = np.concatenate([np.max(np.stack(prada_real, axis=1)[test_idx], axis=1),
                                 np.max(np.stack(prada_fake, axis=1)[fake_test], axis=1)])
    prada_cal = np.max(np.stack(prada_real, axis=1)[cal_idx], axis=1)
    prada_tau = conformal_tau(prada_cal)
    metrics["prada"] = family.c2i.metric(y, prada_test, external.N_TEST_REAL, prada_tau)
    saved.update({"prada_test": prada_test, "prada_cal": prada_cal,
                  "prada_threshold": prada_tau})
    return metrics, saved


def result_name(condition, target, source):
    return f"{condition}::{target}::{source}"


def score_path(condition, target, source, seed):
    name = result_name(condition, target, source).replace(":", "_")
    return os.path.join(OUT, f"scores_{name}_seed{seed}.npz")


def summarize(results):
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        targets = {}
        for target in external.TARGETS:
            for source in eval_sources(condition, target):
                targets.setdefault(source, []).append(target)
        for source, source_targets in targets.items():
            source_result = {}
            for method in AGGREGATIONS + ("selected", "prada"):
                per_seed = []
                for seed in SEEDS:
                    values = [results["folds"][result_name(condition, target, source)]["seeds"][str(seed)][method]["AUROC"]
                              for target in source_targets]
                    per_seed.append(float(np.mean(values)))
                source_result[method] = {"macro_AUROC_per_seed": per_seed,
                                         "mean": float(np.mean(per_seed)),
                                         "std": float(np.std(per_seed, ddof=1))}
            summary[condition][source] = source_result
    return summary


def paired_comparisons(results):
    comparisons = {}
    for condition in CONDITIONS[1:]:
        paired = {}
        for seed in SEEDS:
            per_target = {}
            for target in external.TARGETS:
                current = np.load(score_path("current", target, "arf", seed))
                other = np.load(score_path(condition, target, "arf", seed))
                per_target[target] = (current["y"], other["max_test"], current["max_test"])
            delta, interval = paired_boot_macro(per_target, n_boot=2000, seed=15000 + seed)
            paired[str(seed)] = {"condition_minus_current": delta, "CI95": interval}
        comparisons[condition] = paired
    return comparisons


def evaluate():
    os.makedirs(OUT, exist_ok=True)
    arf = torch_load(os.path.join(external.OUT, "packed_cycle.pt"))
    darg = torch_load(os.path.join(family.OUT, "packed_cycle.pt"))
    results = {"config": {
        "seeds": SEEDS, "conditions": CONDITIONS, "aggregations": AGGREGATIONS,
        "targets": external.TARGETS, "models": external.MODELS,
        "protocol": "source-control-seven-family-example; fixed tests; independent calibration",
        "selection_rule": "max/mean/median selected by training-generator validation macro AUROC",
        "example": True, "n_real_per_source": 1000, "n_fake_per_target": 250,
    }, "folds": {}}
    for seed in SEEDS:
        ref_idx, select_idx, cal_idx, test_idx, train_idx, val_idx, fake_test = external.fixed_splits(seed)
        for target in external.TARGETS:
            bank = external.available_models(target)
            for condition in CONDITIONS:
                sources = real_sources(condition)
                fake_sources = training_fake_sources(condition, target)
                data = {model: normalize_model(arf, darg, model, sources, ref_idx,
                                               fake_sources, target) for model in bank}
                train_data, val_data = build_training_data(
                    data, bank, sources, fake_sources, ref_idx, select_idx, train_idx, val_idx)
                dart = train_dart(train_data, val_data, seed)
                stats = selection_statistics(dart, data, bank, sources, select_idx)
                selected, validation_macro = select_aggregation(
                    dart, data, bank, fake_sources, stats, select_idx, val_idx)
                for source in eval_sources(condition, target):
                    metrics, saved = evaluate_target(
                        dart, data, bank, stats, target, source, select_idx,
                        cal_idx, test_idx, fake_test, selected)
                    name = result_name(condition, target, source)
                    results["folds"].setdefault(name, {"bank": bank, "seeds": {}})["seeds"][str(seed)] = {
                        **metrics, "selected_aggregation": selected,
                        "validation_macro_AUROC": validation_macro}
                    np.savez(score_path(condition, target, source, seed), **saved)
                    print(f"seed={seed} {name} selected={selected} "
                          f"full={metrics['max']['AUROC']:.4f} "
                          f"selected={metrics['selected']['AUROC']:.4f}", flush=True)
        with open(os.path.join(OUT, "dart_source_control_partial.json"), "w") as f:
            json.dump(results, f, indent=2)
    results["summary"] = summarize(results)
    results["paired_vs_current"] = paired_comparisons(results)
    with open(os.path.join(OUT, "dart_source_control.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)


def torch_load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    evaluate()


if __name__ == "__main__":
    main()
