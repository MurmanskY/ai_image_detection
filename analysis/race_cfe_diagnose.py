"""RACE CFE 分量诊断: 冻结检索变体 vs 学习码 (复现失败归因)。

发现: CFE 学习码检索 a(x) AUC 0.23-0.43 (反向), 拖垮融合。
检验: 用冻结 DINOv2 depth-12 / CLIP pooled / 拼接做同样的 KNN 检索,
判断是"真实参考检索思想"问题还是"本轮 CFE 学习码训练"问题。

另附: UnivFD (best.pth head) 在同批缓存特征上的分数, 供 paired 对比。

用法: python analysis/race_cfe_diagnose.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
from analysis.race_p0 import GENS, K_NN, OUT_DIR, knn_mean_cos  # noqa: E402

feats = dict(np.load(os.path.join(OUT_DIR, "feats.npz")))


def unit(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)) \
        .astype(np.float32)


variants = {
    "dino12_frozen": lambda f: unit(f["dino"][:, 1]),
    "clip_pooled_frozen": lambda f: unit(f["clip_pooled"]),
    "concat_frozen": lambda f: unit(np.concatenate(
        [unit(f["dino"][:, 1]), unit(f["clip_pooled"])], axis=1)),
}

bank_feats = {"dino": feats["train_real::v0::dino"],
              "clip_pooled": feats["train_real::v0::clip_pooled"]}

results = {}
for vname, fn in variants.items():
    bank = fn(bank_feats)
    macro = []
    for g in GENS:
        fr = fn({"dino": feats[f"{g}/0_real::orig::dino"],
                 "clip_pooled": feats[f"{g}/0_real::orig::clip_pooled"]})
        ff = fn({"dino": feats[f"{g}/1_fake::orig::dino"],
                 "clip_pooled": feats[f"{g}/1_fake::orig::clip_pooled"]})
        a = np.concatenate([knn_mean_cos(bank, fr), knn_mean_cos(bank, ff)])
        y = np.array([0] * len(fr) + [1] * len(ff))
        auc = float(roc_auc_score(y, a))
        macro.append(auc)
        results.setdefault(vname, {})[g] = auc
    results[vname]["macro"] = float(np.mean(macro))
    print(f"{vname}: macro={np.mean(macro):.4f} "
          f"per-gen={['%.3f' % a for a in macro]}", flush=True)

# UnivFD head on clip_pooled
ck = torch.load(os.path.join(PROJ, "outputs", "univfd_arforensics",
                             "best.pth"), map_location="cpu",
                weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
W = sd["head.weight"].numpy()
b = sd["head.bias"].numpy()
macro = []
for g in GENS:
    sr = feats[f"{g}/0_real::orig::clip_pooled"] @ W.T + b
    sf = feats[f"{g}/1_fake::orig::clip_pooled"] @ W.T + b
    y = np.array([0] * len(sr) + [1] * len(sf))
    auc = float(roc_auc_score(y, np.concatenate([sr, sf])))
    macro.append(auc)
    results.setdefault("univfd_head", {})[g] = auc
results["univfd_head"]["macro"] = float(np.mean(macro))
print(f"univfd_head: macro={np.mean(macro):.4f}", flush=True)

out = os.path.join(OUT_DIR, "race_cfe_diagnose.json")
json.dump(results, open(out, "w"), indent=2)
print(f"已保存 {out}")
