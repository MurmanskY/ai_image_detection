"""CTAD JPEG95 统一重编码稳健性对照 (实验 C 的 CTAD 部分)。

在 data/ARForensics_reencode (全部统一 decode->RGB->JPEG Q95) 上重算
4 个 tokenizer 的 qerr, 用原始 val real 的参考统计量求 z 分数,
评估 S_min 的 per-generator AUC, 与原始格式结果对比。
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "D3QE"))
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.ctad_feasibility import TOKENIZERS, compute_qerr_all  # noqa: E402
from datasets.arforensics import ARForensicsDataset  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "data/ARForensics_reencode"
    n = 500
    probe = ARForensicsDataset(root, "test", T.Compose([T.ToTensor()]),
                               max_per_class=n)
    groups = sorted({(l, g) for _, l, g in probe.samples})
    names = list(TOKENIZERS)

    d = dict(np.load(os.path.join(PROJ, "outputs/ctad/zfeat.npz"),
                     allow_pickle=True))
    ref = {k: (float(d[f"refmean_{k}"]), float(d[f"refstd_{k}"]))
           for k in names}

    z = {}
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (reencode95 test)")
        qerr = compute_qerr_all(name, spec, groups, root, n, 16, device)
        mu, sd = ref[name]
        for (label, gen) in groups:
            z[f"{name}_{label}_{gen}"] = (qerr[(label, gen)] - mu) / (sd + 1e-12)

    out = {}
    for (label, gen) in groups:
        if label == 0:
            continue
        zr = np.stack([z[f"{k}_0_real"] for k in names], axis=1)
        zf = np.stack([z[f"{k}_1_{gen}"] for k in names], axis=1)
        y = np.array([0] * len(zr) + [1] * len(zf))
        out[gen] = float(roc_auc_score(
            y, np.concatenate([-zr.min(axis=1), -zf.min(axis=1)])))
        print(gen, out[gen])
    out["macro_mean"] = float(np.mean(list(out.values())))
    print("macro_mean:", out["macro_mean"])

    path = os.path.join(PROJ, "outputs/ctad/reencode95_smin.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"已保存到 {path}")


if __name__ == "__main__":
    main()
