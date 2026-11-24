"""MPID P0 修正版 (v8 §2.2/§1.2 复核意见): 严格 teacher holdout + 独立 genprobe。

修正点 (相对 analysis/mpid_p0.py):
  1. v8 §2.2: 原 P0 的教师目标 prada_full = max over 全部 4 个 PRADA 模型,
     包含 held-out 生成器对应的模型 -> 教师集合未严格 held-out。
     本版每折重新构造教师目标: T_fold(x) = max over 3 个非 held-out 模型。
     (原始逐模型分数缓存在 outputs/mpid/m pid_p0_targets_permodel.npz)
  2. v8 §1.2: 原版生成器身份探针在训练隐藏特征上拟合并评估 (训练集内),
     本版在独立 test 折 fake 上评估 (3 个训练生成器的 test split)。
  3. 其余协议与 mpid_p0.py 完全一致 (同缓存 CLIP 特征/划分/预算/λ=1)。

用法: python analysis/mpid_p0_strict.py
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.mpid_p0 import (GENS, DATASETS, N_TRAIN, N_TEST, SEED,  # noqa: E402
                              OUT_DIR, list_images, train_student,  # noqa: E402
                              student_scores, DATA_ROOT)  # noqa: E402
from prada.prada import PRADA  # noqa: E402

FEAT_DIR = os.path.join(PROJ, "outputs", "prada_t2i", "features")
WEIGHT_DIR = os.path.join(PROJ, "third_party", "PRADA", "prada_weights")
PERMODEL_PATH = os.path.join(OUT_DIR, "mpid_p0_targets_permodel.npz")
CLIP_PATH = os.path.join(OUT_DIR, "mpid_p0_clip_feats.npz")
OUT_PATH = os.path.join(OUT_DIR, "mpid_p0_strict.json")


def compute_permodel_scores():
    """逐 (dataset, model) 的 PRADA 分数, 供每折自定义教师集合。"""
    scores = {}
    for ds in DATASETS:
        for m in GENS:
            f = torch.load(os.path.join(FEAT_DIR,
                                        f"dataset={ds}_model={m}.pt"),
                           map_location="cpu", weights_only=False)
            stacked = torch.stack(
                [f["cond_gt_llh_BL"], f["uncond_gt_llh_BL"]], dim=-1).float()
            prada = PRADA(n_hidden=16, scale_lengths=f.get("scale_lengths"),
                          input_dim=1, learnable_difference=True,
                          learnable_stage_weights=True, training_noise=0.1)
            prada.load_state_dict(torch.load(
                os.path.join(WEIGHT_DIR, m, "final.pt"), weights_only=True))
            prada.eval()
            with torch.no_grad():
                scores[f"{ds}::{m}"] = prada(stacked).numpy()
    return scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(PERMODEL_PATH):
        scores = dict(np.load(PERMODEL_PATH))
        print("加载缓存逐模型分数", flush=True)
    else:
        print("计算逐模型 PRADA 分数...", flush=True)
        scores = compute_permodel_scores()
        np.savez_compressed(PERMODEL_PATH, **scores)

    z = np.load(CLIP_PATH)
    feats = {ds: {"orig": z[f"{ds}::orig"], "jpeg95": z[f"{ds}::jpeg95"]}
             for ds in DATASETS}

    rng = np.random.RandomState(SEED)
    splits = {}
    for ds in DATASETS:
        idx = rng.permutation(len(feats[ds]["orig"]))
        splits[ds] = {"train": idx[:N_TRAIN], "test": idx[N_TRAIN:]}

    results = {"config": {
        "protocol": "mpid-p0-strict: 严格 teacher holdout + 独立 genprobe "
                    "(v8 §2.2/§1.2 修正), 其余同 mpid_p0",
        "teacher": "PRADA max over 3 non-held-out models (逐折重构)",
    }, "folds": {}}

    for held in GENS:
        train_gens = [g for g in GENS if g != held]
        teacher_models = train_gens  # 修正 1: 教师集合排除 held-out 模型
        tr_ds = ["real"] + train_gens

        def teacher(ds):
            return np.max(np.stack([scores[f"{ds}::{m}"]
                                    for m in teacher_models]), axis=0)

        xtr = np.concatenate([feats[ds]["orig"][splits[ds]["train"]]
                              for ds in tr_ds])
        ytr = np.array([0] * N_TRAIN + [1] * (N_TRAIN * len(train_gens)),
                       dtype=np.float32)
        ttr = np.concatenate([teacher(ds)[splits[ds]["train"]]
                              for ds in tr_ds]).astype(np.float32)

        mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
        tmu, tsd = ttr.mean(), ttr.std() + 1e-8
        xtr_z = (xtr - mu) / sd
        ttr_z = (ttr - tmu) / tsd

        def fold_test(codec):
            xr = (feats["real"][codec][splits["real"]["test"]] - mu) / sd
            xf = (feats[held][codec][splits[held]["test"]] - mu) / sd
            return xr, xf

        xr_o, xf_o = fold_test("orig")
        xr_j, xf_j = fold_test("jpeg95")
        yte = np.array([0] * N_TEST + [1] * N_TEST)

        entry = {"students": {}}
        fake_mask = ytr == 1
        gen_labels_tr = np.concatenate(
            [[i] * N_TRAIN for i in range(len(train_gens))])
        # 修正 2: 独立 test 折上的 genprobe 评估
        xte_fake = np.concatenate(
            [feats[g]["orig"][splits[g]["test"]] for g in train_gens])
        xte_fake_z = (xte_fake - mu) / sd
        gen_labels_te = np.concatenate(
            [[i] * N_TEST for i in range(len(train_gens))])

        for name, use_mech in (("bce_only", False), ("mech_distill", True)):
            m = train_student(xtr_z, ytr, ttr_z, use_mech)
            sres = {}
            for codec, (xr, xf) in (("orig", (xr_o, xf_o)),
                                    ("jpeg95", (xr_j, xf_j))):
                s = np.concatenate([student_scores(m, xr),
                                    student_scores(m, xf)])
                sres[f"auc_{codec}"] = float(roc_auc_score(yte, s))
            _, h_tr = student_scores(m, xtr_z[fake_mask], return_hidden=True)
            _, h_te = student_scores(m, xte_fake_z, return_hidden=True)
            probe = LogisticRegression(max_iter=1000)
            probe.fit(h_tr, gen_labels_tr)
            sres["genprobe_acc_train"] = float(probe.score(h_tr, gen_labels_tr))
            sres["genprobe_acc_test"] = float(probe.score(h_te, gen_labels_te))
            entry["students"][name] = sres
            print(f"  [{held}] {name}: orig={sres['auc_orig']:.4f} "
                  f"jpeg95={sres['auc_jpeg95']:.4f} "
                  f"genprobe_te={sres['genprobe_acc_test']:.3f}", flush=True)

        # P0-1 探针 (严格教师目标, held-out 可预测性)
        entry["teacher_probes"] = {}
        xf_held_tr = (feats[held]["orig"][splits[held]["train"]] - mu) / sd
        # held-out 生成器的"自身教师"分数 (4 模型 max) 与严格教师分数
        t_held_self = np.max(np.stack([scores[f"{held}::{m}"]
                                       for m in GENS]), axis=0)
        t_all = np.concatenate([teacher(ds)[splits[ds]["train"]]
                                for ds in tr_ds]).astype(np.float32)
        tm, ts = t_all.mean(), t_all.std() + 1e-8
        ridge = Ridge(alpha=1.0)
        ridge.fit(xtr_z, (t_all - tm) / ts)
        rho_strict = spearmanr(ridge.predict(xf_held_tr),
                               t_held_self[splits[held]["train"]]).statistic
        entry["teacher_probes"]["prada_full_strict"] = float(rho_strict)
        results["folds"][held] = entry

    def fm(student, key):
        return [results["folds"][g]["students"][student][key] for g in GENS]

    bce_o, dis_o = fm("bce_only", "auc_orig"), fm("mech_distill", "auc_orig")
    bce_j, dis_j = fm("bce_only", "auc_jpeg95"), \
        fm("mech_distill", "auc_jpeg95")
    gains = [d - b for d, b in zip(dis_o, bce_o)]
    rho = [results["folds"][g]["teacher_probes"]["prada_full_strict"]
           for g in GENS]
    results["macro"] = {
        "bce_only": {"orig": float(np.mean(bce_o)),
                     "jpeg95": float(np.mean(bce_j)),
                     "genprobe_test": float(np.mean(
                         fm("bce_only", "genprobe_acc_test")))},
        "mech_distill": {"orig": float(np.mean(dis_o)),
                         "jpeg95": float(np.mean(dis_j)),
                         "genprobe_test": float(np.mean(
                             fm("mech_distill", "genprobe_acc_test")))},
        "gain_per_fold": dict(zip(GENS, gains)),
    }
    results["go"] = {
        "crit1_macro_gain>=0.02": float(np.mean(gains)),
        "crit2_folds_improved>=3": int(sum(g >= 0 for g in gains)),
        "crit3_jpeg95_gain>0": float(np.mean(dis_j) - np.mean(bce_j)),
        "crit4_info_teacher_spearman_median": float(np.median(rho)),
    }
    results["go"]["GO"] = bool(
        results["go"]["crit1_macro_gain>=0.02"] >= 0.02
        and results["go"]["crit2_folds_improved>=3"] >= 3
        and results["go"]["crit3_jpeg95_gain>0"] > 0)
    print(json.dumps({"macro": results["macro"], "go": results["go"]},
                     indent=2), flush=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
