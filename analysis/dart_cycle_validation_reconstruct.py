"""从 NPZ 重建 raw residual 验证，并按来源/seed/model 计算条件 R²。"""

import json
import os
import sys

import numpy as np

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_cycle_validation as validation  # noqa: E402
from analysis.carf_p0 import conformal_tau  # noqa: E402
from analysis.dart_c2i_p0 import metric  # noqa: E402


def main():
    results = {"config": {
        "seeds": validation.SEEDS, "conditions": validation.CONDITIONS,
        "aggregations": validation.AGGREGATIONS, "targets": validation.external.TARGETS,
        "models": validation.external.MODELS,
        "protocol": "raw cycle residual; real-only conditional fit; fixed tests; independent calibration",
        "example": True, "reconstructed_from_npz": True,
    }, "folds": {}, "conditional_R2": {}}
    for condition in validation.CONDITIONS:
        results["conditional_R2"][condition] = {}
        for seed in validation.SEEDS:
            results["conditional_R2"][condition][str(seed)] = {}
    for condition in validation.CONDITIONS:
        sources = validation.source.real_sources(condition)
        for target in validation.external.TARGETS:
            bank = validation.external.available_models(target)
            fake_sources = validation.source.training_fake_sources(condition, target)
            for seed in validation.SEEDS:
                for source_name in validation.eval_sources(condition, target):
                    fold_name = f"{condition}::{target}::{source_name}"
                    entry = {}
                    for aggregation in validation.AGGREGATIONS:
                        saved = np.load(validation.score_path(
                            condition, target, source_name, seed, aggregation))
                        tau = conformal_tau(saved["cal"])
                        entry[aggregation] = metric(saved["y"], saved["test"],
                                                    validation.external.N_TEST_REAL, tau)
                    results["folds"].setdefault(fold_name, {"bank": bank, "seeds": {}})["seeds"][str(seed)] = entry
    # R² depends on real source, seed and reference model, not on held-out target.
    arf = validation.source.torch_load(os.path.join(validation.external.OUT, "packed_cycle.pt"))
    darg = validation.source.torch_load(os.path.join(validation.source.family.OUT, "packed_cycle.pt"))
    for condition in validation.CONDITIONS:
        sources = validation.source.real_sources(condition)
        for seed in validation.SEEDS:
            ref_idx, select_idx, *_ = validation.external.fixed_splits(seed)
            for model in validation.external.MODELS:
                data = validation.source.normalize_model(
                    arf, darg, model, sources, ref_idx, {}, "Infinity")
                residualizer = validation.interaction.CycleResidualizer().fit(
                    [data["real", src][i] for src in sources for i in ref_idx],
                    [data["real", src][i] for src in sources for i in select_idx])
                results["conditional_R2"][condition][str(seed)][model] = residualizer.selection_r2
    results["summary"] = validation.summarize(results)
    results["paired_vs_detectors"] = validation.compare_with_detectors(results)
    results["replication_max_AUROC_error"] = validation.replication_check(results)
    with open(os.path.join(validation.OUT, "dart_cycle_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2), flush=True)
    print("replication error", results["replication_max_AUROC_error"], flush=True)


if __name__ == "__main__":
    main()
