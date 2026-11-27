"""七家族固定测试集的配对显著性、低 FPR 与风险汇总。"""

import json
import os
import sys

import numpy as np
from scipy import stats

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)

from analysis import dart_external_family_p0 as external  # noqa: E402
from analysis.carf_a import paired_boot_macro  # noqa: E402


def main():
    results = json.load(open(os.path.join(external.OUT, "dart_external_family.json")))
    output = {"note": "fixed test images; seeds are training/calibration splits, not independent datasets",
              "seeds": {}, "per_target": {}}
    full_macros, lr_macros, prada_macros = [], [], []
    for seed in external.SEEDS:
        paired_full_prada = {}
        for target in external.TARGETS:
            bank = external.bank_name(external.available_models(target))
            saved = np.load(os.path.join(
                external.OUT, external.score_prefix(target, seed, bank) + ".npz"))
            paired_full_prada[target] = (saved["y"], saved["full_test"], saved["prada_test"])
        delta, interval = paired_boot_macro(paired_full_prada, n_boot=5000, seed=12100 + seed)
        output["seeds"][str(seed)] = {"full_minus_prada": delta, "CI95": interval}
    for target in external.TARGETS:
        bank = external.bank_name(external.available_models(target))
        target_result = {}
        for method in ("full", "likelihood_reconstruction", "prada"):
            entries = [results["folds"][target]["seeds"][str(seed)]["banks"][bank][method]
                       for seed in external.SEEDS]
            target_result[method] = {
                "AUROC_mean": float(np.mean([x["AUROC"] for x in entries])),
                "AUROC_std": float(np.std([x["AUROC"] for x in entries], ddof=1)),
                "FPR_per_seed": [x["FPR"] for x in entries],
                "TPR@1%FPR_mean": float(np.mean([x["TPR@1%FPR"] for x in entries])),
                "TPR@5%FPR_mean": float(np.mean([x["TPR@5%FPR"] for x in entries])),
            }
        output["per_target"][target] = target_result
        full_macros.append(target_result["full"]["AUROC_mean"])
        lr_macros.append(target_result["likelihood_reconstruction"]["AUROC_mean"])
        prada_macros.append(target_result["prada"]["AUROC_mean"])
    lr_minus_full = list(results["summary"]["likelihood_reconstruction"]["feature_minus_other_per_seed"])
    full_minus_prada = [output["seeds"][str(seed)]["full_minus_prada"] for seed in external.SEEDS]
    output["paired"] = {
        "LR_minus_full_mean": float(np.mean(lr_minus_full)),
        "LR_minus_full_seed_t_CI95": [float(x) for x in stats.t.interval(
            0.95, len(lr_minus_full) - 1, loc=np.mean(lr_minus_full), scale=stats.sem(lr_minus_full))],
        "LR_minus_full_wilcoxon_p": float(stats.wilcoxon(lr_minus_full).pvalue),
        "full_better_seed_count": int(sum(x < 0 for x in lr_minus_full)),
        "full_minus_prada_per_seed": full_minus_prada,
        "full_minus_prada_mean": float(np.mean(full_minus_prada)),
        "all_full_prada_bootstrap_CI_positive": all(
            output["seeds"][str(seed)]["CI95"][0] > 0 for seed in external.SEEDS),
    }
    full_fprs = [fpr for x in output["per_target"].values() for fpr in x["full"]["FPR_per_seed"]]
    lr_fprs = [fpr for x in output["per_target"].values()
               for fpr in x["likelihood_reconstruction"]["FPR_per_seed"]]
    output["macro"] = {
        "full": float(np.mean(full_macros)),
        "likelihood_reconstruction": float(np.mean(lr_macros)),
        "prada": float(np.mean(prada_macros)),
        "full_TPR@1%FPR": float(np.mean([
            x["full"]["TPR@1%FPR_mean"] for x in output["per_target"].values()])),
        "full_TPR@5%FPR": float(np.mean([
            x["full"]["TPR@5%FPR_mean"] for x in output["per_target"].values()])),
        "LR_TPR@1%FPR": float(np.mean([
            x["likelihood_reconstruction"]["TPR@1%FPR_mean"] for x in output["per_target"].values()])),
        "LR_TPR@5%FPR": float(np.mean([
            x["likelihood_reconstruction"]["TPR@5%FPR_mean"] for x in output["per_target"].values()])),
        "full_FPR_range": [float(min(full_fprs)), float(max(full_fprs))],
        "LR_FPR_range": [float(min(lr_fprs)), float(max(lr_fprs))],
    }
    out = os.path.join(external.OUT, "external_family_significance.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output["paired"], indent=2), flush=True)
    print(json.dumps(output["macro"], indent=2), flush=True)


if __name__ == "__main__":
    main()
