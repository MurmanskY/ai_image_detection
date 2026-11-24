"""P0-B: DARG T2I 难度与泄漏审计 (v8 §2.3/§9)。

回答: MPID BCE-only 的 LOGO 0.998 是否由协议天花板/泄漏造成?
  1. 跨 train/test 近重复 (pHash, 每个目录内部 800/200 划分之间)
  2. real <-> fake 跨目录近重复 (内容复用)
  3. 格式/元数据/分辨率/文件大小是否可被简单分类器识别 (标签泄漏)
  4. 各生成器图像属性分布概览
  5. 协议注释: real 在 4 折 LOGO 中复用; DARG 原始 prompt 本地不可得,
     用图像级近重复替代 prompt 级检查 (边界记录在结果中)

用法: python analysis/darg_audit.py
"""

import json
import os
import sys
from collections import Counter

import imagehash
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
from analysis.mpid_p0 import GENS, DATASETS, N_TRAIN, SEED, list_images, \
    DATA_ROOT  # noqa: E402

OUT_PATH = os.path.join(PROJ, "outputs", "mpid", "darg_audit.json")
PHASH_THRESH = 6


def hamming(h1, h2):
    return bin(int(str(h1), 16) ^ int(str(h2), 16)).count("1")


def main():
    rng = np.random.RandomState(SEED)
    report = {"protocol": "DARG T2I 难度审计 (v8 §2.3), seed=0 800/200",
              "dirs": {}}

    hashes, meta = {}, {}
    for ds in DATASETS:
        paths = list_images(os.path.join(DATA_ROOT, ds))
        hs, ms = [], []
        for p in paths:
            img = Image.open(p)
            hs.append(imagehash.phash(img.convert("RGB")))
            info = img.info
            ms.append({
                "size": os.path.getsize(p),
                "w": img.width, "h": img.height,
                "mode": img.mode,
                "format": img.format,
                "n_meta": len(info),
                "meta_keys": sorted(info.keys())[:10],
            })
        hashes[ds] = hs
        meta[ds] = ms
        sizes = [m["size"] for m in ms]
        report["dirs"][ds] = {
            "n": len(paths),
            "formats": dict(Counter(m["format"] for m in ms)),
            "modes": dict(Counter(m["mode"] for m in ms)),
            "resolutions": dict(Counter(f"{m['w']}x{m['h']}" for m in ms)),
            "filesize_kb_median": float(np.median(sizes) / 1024),
            "n_meta_keys_dist": dict(Counter(m["n_meta"] for m in ms)),
            "meta_keys_example": ms[0]["meta_keys"],
        }
        print(f"{ds}: {len(paths)} 张, "
              f"formats={report['dirs'][ds]['formats']}, "
              f"res={report['dirs'][ds]['resolutions']}", flush=True)

    # 1. 跨 train/test 近重复 (每个目录内部, 同 seed 划分)
    report["near_dup_train_test"] = {}
    for ds in DATASETS:
        n = len(hashes[ds])
        idx = rng.permutation(n)
        tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
        pairs = 0
        min_d = 64
        for i in tr:
            for j in te:
                d = hamming(hashes[ds][i], hashes[ds][j])
                min_d = min(min_d, d)
                if d <= PHASH_THRESH:
                    pairs += 1
        report["near_dup_train_test"][ds] = {
            "pairs_hamming<=6": pairs, "min_hamming": min_d}
        print(f"near-dup train/test {ds}: pairs={pairs}, min={min_d}",
              flush=True)

    # 2. real <-> fake 跨目录近重复
    report["near_dup_real_fake"] = {}
    real_h = hashes["real"]
    for g in GENS:
        pairs, min_d = 0, 64
        for i, hr in enumerate(real_h):
            for hf in hashes[g]:
                d = hamming(hr, hf)
                min_d = min(min_d, d)
                if d <= PHASH_THRESH:
                    pairs += 1
        report["near_dup_real_fake"][g] = {
            "pairs_hamming<=6": pairs, "min_hamming": min_d}
        print(f"near-dup real/{g}: pairs={pairs}, min={min_d}", flush=True)

    # 3. 元数据/文件属性泄漏分类器
    def feats_of(ds):
        return np.array([[np.log(m["size"]), m["w"], m["h"], m["n_meta"],
                          1.0 if m["mode"] == "RGB" else 0.0]
                         for m in meta[ds]])

    report["metadata_leakage_auroc_cv"] = {}
    Xr = feats_of("real")
    for g in GENS:
        Xf = feats_of(g)
        X = np.vstack([Xr, Xf])
        y = np.array([0] * len(Xr) + [1] * len(Xf))
        clf = LogisticRegression(max_iter=1000)
        p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, p))
        report["metadata_leakage_auroc_cv"][g] = auc
        print(f"metadata leakage AUC real/{g}: {auc:.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"已保存 {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
