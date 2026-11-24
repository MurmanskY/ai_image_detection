"""MPID P0: 机制特权不变蒸馏的可行性验证 (v7 指南 §8.1)。

研究问题: 白盒 AR 机制证据 (PRADA 教师分数) 能否作为训练阶段 privileged
supervision, 蒸馏到推理阶段仅需 RGB 的视觉学生模型, 并在未见生成器上
(LOGO) 提升真假检测?

四个阶段:
  A. 教师目标: 从 outputs/prada_t2i/features/ 缓存离线计算 PRADA 完整分数
     (max over 4 models) 与单特征目标 (gt_llh_diff/icas/quant_err/aeroblade),
     并用官方 auroc_prada_test.csv 做对齐 sanity check。
  B. CLIP ViT-L/14 冻结特征 (orig + JPEG95 重编码), 复用 UnivFD 骨干。
  C. LOGO 学生训练: 4 折, 每折留 1 个生成器完全不参与训练。
     Student A: MLP(768->256->1) BCE-only。
     Student B: 同 trunk + aux head(256->1) 回归教师分数,
                loss = BCE + 1.0 * MSE (λ=1 预注册, 不调参)。
     固定预算: Adam lr 1e-3, wd 1e-4, 30 epochs, batch 256, seed 0。
  D. 探针与 GO 判定:
     P0-1: Ridge 回归探针 (frozen CLIP -> 教师分数), held-out 生成器上的
           Spearman ρ, 比较 5 种机制目标的可迁移性。
     P0-3: 生成器身份线性探针 (学生 256 维隐藏表示 -> 3 类生成器),
           比较 A/B 的 leakage; codec 干预 = 测试折 JPEG95 重编码。

预注册 GO 标准 (v7 §8.1):
  1. LOGO macro AUROC(B) >= macro(A) + 0.02
  2. >= 3/4 折 B >= A
  3. codec JPEG95 下 macro 增益仍为正
  4. (记录) 教师分数在 held-out 生成器上可预测 (Spearman 中位数 > 0.3)
  1&2&3 全部满足才进入 P1; 否则按 v7 停止 MPID 主线。

数据协议: DARG T2I, 每目录 1000 张 (real + 4 gens), 800/200 train/test
(seed 0, per class), 与 PRADA 特征缓存顺序对齐 (sorted iterdir)。
标注: mpid-p0-1000示例 (非 84k 全量)。

用法: python analysis/mpid_p0.py
"""

import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T
from torchvision.datasets.folder import IMG_EXTENSIONS

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from prada.metrics import icas  # noqa: E402
from prada.prada import PRADA  # noqa: E402

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DATA_ROOT = os.path.join(PROJ, "data", "darg", "darg", "t2i")
FEAT_DIR = os.path.join(PROJ, "outputs", "prada_t2i", "features")
WEIGHT_DIR = os.path.join(PROJ, "third_party", "PRADA", "prada_weights")
OUT_DIR = os.path.join(PROJ, "outputs", "mpid")
GENS = ["infinity_2b", "janus_1b", "llamagen_xlstage2", "switti_1024"]
DATASETS = ["real"] + GENS
SINGLE_TARGETS = ["gt_llh_diff", "icas", "quant_err", "aeroblade"]
LAMBDA_MECH = 1.0
SEED = 0
N_TRAIN, N_TEST = 800, 200


# ---------------- A. 教师目标 ----------------

def compute_teacher_targets():
    """返回 targets: dict[dataset] -> dict[target] -> (B,) np.ndarray"""
    scores, singles = {}, {}
    for ds in DATASETS:
        for m in GENS:
            f = torch.load(os.path.join(FEAT_DIR,
                                        f"dataset={ds}_model={m}.pt"),
                           map_location="cpu", weights_only=False)
            stacked = torch.stack(
                [f["cond_gt_llh_BL"], f["uncond_gt_llh_BL"]], dim=-1).float()
            scale_lengths = f.get("scale_lengths")
            prada = PRADA(n_hidden=16, scale_lengths=scale_lengths,
                          input_dim=1, learnable_difference=True,
                          learnable_stage_weights=True, training_noise=0.1)
            prada.load_state_dict(torch.load(
                os.path.join(WEIGHT_DIR, m, "final.pt"), weights_only=True))
            prada.eval()
            with torch.no_grad():
                scores[(ds, m)] = prada(stacked).numpy()

            diff = (f["cond_gt_llh_BL"] - f["uncond_gt_llh_BL"]).float()
            qe = f["quant_err"].float()
            ab = f["aeroblade"].float()
            singles.setdefault("gt_llh_diff", {})[(ds, m)] = diff.mean(1).numpy()
            singles.setdefault("icas", {})[(ds, m)] = icas(diff).mean(1).numpy()
            singles.setdefault("quant_err", {})[(ds, m)] = (
                qe.mean(1).numpy() if qe.ndim == 2 else qe.numpy())
            singles.setdefault("aeroblade", {})[(ds, m)] = (
                ab.mean(1).numpy() if ab.ndim == 2 else ab.numpy())

    targets = {}
    for ds in DATASETS:
        t = {"prada_full": np.max(np.stack([scores[(ds, m)] for m in GENS]),
                                  axis=0)}
        for name in SINGLE_TARGETS:
            t[name] = np.max(
                np.stack([singles[name][(ds, m)] for m in GENS]), axis=0)
        targets[ds] = t
    return targets


def sanity_check_auroc(targets):
    """复算 PRADA 检测 AUROC, 应与官方 auroc_prada_test.csv 一致。"""
    out = {}
    for g in GENS:
        y = np.array([0] * len(targets["real"]["prada_full"])
                     + [1] * len(targets[g]["prada_full"]))
        s = np.concatenate([targets["real"]["prada_full"],
                            targets[g]["prada_full"]])
        out[g] = float(roc_auc_score(y, s))
    return out


# ---------------- B. CLIP 特征 ----------------

_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def list_images(d):
    from pathlib import Path
    d = Path(d)
    return sorted([p for p in d.iterdir() if p.suffix in IMG_EXTENSIONS])


def wait_for_gpu(need_mb=3500, poll_s=240):
    """返回空闲显存足够的 GPU id; 不足则等待 (与其他用户共享服务器)。"""
    while True:
        best_gpu, best_free = 0, -1
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"]).decode()
        for gi, line in enumerate(out.strip().splitlines()):
            total, used = [int(x) for x in line.split(",")]
            if total - used > best_free:
                best_free, best_gpu = total - used, gi
        print(f"[gpu-wait] GPU{best_gpu} 空闲 {best_free}MiB "
              f"(需要 {need_mb}MiB)", flush=True)
        if best_free >= need_mb:
            return best_gpu
        time.sleep(poll_s)


def jpeg95(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def extract_clip_features(backbone, paths, device, batch_size=64, tag=""):
    feats_o, feats_j = [], []
    for bi in range(0, len(paths), batch_size):
        batch_o, batch_j = [], []
        for p in paths[bi:bi + batch_size]:
            img = Image.open(p).convert("RGB")
            batch_o.append(_transform(img))
            batch_j.append(_transform(jpeg95(img)))
        feats_o.append(backbone(torch.stack(batch_o).to(device)).float().cpu())
        feats_j.append(backbone(torch.stack(batch_j).to(device)).float().cpu())
        if bi == 0:
            print(f"  [{tag}] 提取中 ({len(paths)} 张)...", flush=True)
    return torch.cat(feats_o).numpy(), torch.cat(feats_j).numpy()


# ---------------- C. 学生模型 ----------------

class Student(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.det_head = nn.Linear(hidden, 1)
        self.mech_head = nn.Linear(hidden, 1)

    def forward(self, x, with_hidden=False):
        h = self.trunk(x)
        det = self.det_head(h).squeeze(-1)
        mech = self.mech_head(h).squeeze(-1)
        return (det, mech, h) if with_hidden else (det, mech)


def train_student(x, y, t, use_mech, epochs=30, batch=256, lr=1e-3, wd=1e-4):
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Student(dim=x.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    tt = torch.tensor(t, dtype=torch.float32, device=dev)
    n = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            det, mech = model(xt[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(det, yt[idx])
            if use_mech:
                loss = loss + LAMBDA_MECH * nn.functional.mse_loss(mech, tt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def student_scores(model, x, return_hidden=False):
    dev = next(model.parameters()).device
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    det, _, h = model(xt, with_hidden=True)
    if return_hidden:
        return det.cpu().numpy(), h.cpu().numpy()
    return det.cpu().numpy()


# ---------------- 主流程 ----------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets_path = os.path.join(OUT_DIR, "mpid_p0_targets.npz")
    feats_path = os.path.join(OUT_DIR, "mpid_p0_clip_feats.npz")
    out_path = os.path.join(OUT_DIR, "mpid_p0.json")

    # ---- A ----
    if os.path.exists(targets_path):
        z = np.load(targets_path)
        targets = {ds: {k: z[f"{ds}::{k}"] for k in
                        ["prada_full"] + SINGLE_TARGETS} for ds in DATASETS}
        print("A: 加载缓存教师目标", flush=True)
    else:
        print("A: 计算教师目标 (PRADA 缓存离线评分)...", flush=True)
        targets = compute_teacher_targets()
        np.savez_compressed(targets_path,
                            **{f"{ds}::{k}": v for ds, d in targets.items()
                               for k, v in d.items()})
    auroc_check = sanity_check_auroc(targets)
    print("A: PRADA 分数对齐 sanity check (应≈0.984-0.995):",
          json.dumps(auroc_check), flush=True)

    # ---- B ----
    if os.path.exists(feats_path):
        z = np.load(feats_path)
        feats = {ds: {"orig": z[f"{ds}::orig"], "jpeg95": z[f"{ds}::jpeg95"]}
                 for ds in DATASETS}
        print("B: 加载缓存 CLIP 特征", flush=True)
    else:
        gpu = wait_for_gpu()
        device = torch.device(f"cuda:{gpu}")
        print(f"B: 提取 CLIP 特征 (GPU{gpu})...", flush=True)
        from models.backbones import build_backbone
        backbone, _ = build_backbone("clip_vit_l14", pretrained=True)
        backbone.eval().to(device)
        for p in backbone.parameters():
            p.requires_grad = False
        feats = {}
        for ds in DATASETS:
            paths = list_images(os.path.join(DATA_ROOT, ds))
            fo, fj = extract_clip_features(backbone, paths, device, tag=ds)
            feats[ds] = {"orig": fo, "jpeg95": fj}
            print(f"  {ds}: {fo.shape}", flush=True)
        np.savez_compressed(feats_path,
                            **{f"{ds}::{k}": v for ds, d in feats.items()
                               for k, v in d.items()})
        del backbone
        torch.cuda.empty_cache()

    # 对齐检查
    for ds in DATASETS:
        assert len(feats[ds]["orig"]) == len(targets[ds]["prada_full"]), ds

    # ---- C/D: LOGO ----
    rng = np.random.RandomState(SEED)
    splits = {}
    for ds in DATASETS:
        idx = rng.permutation(len(feats[ds]["orig"]))
        splits[ds] = {"train": idx[:N_TRAIN], "test": idx[N_TRAIN:]}

    results = {"config": {
        "protocol": "mpid-p0-1000示例 (DARG T2I, LOGO-4, 非全量)",
        "student": "MLP(768->256->1), BCE vs BCE+MSE(λ=1)",
        "budget": "Adam lr1e-3 wd1e-4, 30ep, batch256, seed0",
        "teacher": "PRADA full score (max over 4 models), 官方预训练评分函数",
        "backbone": "CLIP ViT-L/14 frozen (orig + JPEG95)",
        "auroc_sanity_prada_full": auroc_check,
    }, "folds": {}}

    for held in GENS:
        train_gens = [g for g in GENS if g != held]
        tr_ds = ["real"] + train_gens
        xtr = np.concatenate([feats[ds]["orig"][splits[ds]["train"]]
                              for ds in tr_ds])
        ytr = np.array([0] * N_TRAIN + [1] * (N_TRAIN * len(train_gens)),
                       dtype=np.float32)
        ttr = np.concatenate([targets[ds]["prada_full"][splits[ds]["train"]]
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

        entry = {"students": {}, "teacher_probes": {}}
        fake_mask = ytr == 1
        gen_labels = np.concatenate(
            [[i] * N_TRAIN for i in range(len(train_gens))])
        for name, use_mech in (("bce_only", False), ("mech_distill", True)):
            m = train_student(xtr_z, ytr, ttr_z, use_mech)
            sres = {}
            for codec, (xr, xf) in (("orig", (xr_o, xf_o)),
                                    ("jpeg95", (xr_j, xf_j))):
                s = np.concatenate([student_scores(m, xr),
                                    student_scores(m, xf)])
                sres[f"auc_{codec}"] = float(roc_auc_score(yte, s))
            # P0-3: 生成器身份探针 (train folds 的 3 类 fake 隐藏表示)
            _, h_fake = student_scores(m, xtr_z[fake_mask],
                                       return_hidden=True)
            probe = LogisticRegression(max_iter=1000)
            probe.fit(h_fake, gen_labels)
            sres["genprobe_acc"] = float(probe.score(h_fake, gen_labels))
            entry["students"][name] = sres
            print(f"  [{held}] {name}: orig={sres['auc_orig']:.4f} "
                  f"jpeg95={sres['auc_jpeg95']:.4f} "
                  f"genprobe={sres['genprobe_acc']:.3f}", flush=True)

        # P0-1: 教师分数回归探针 (held-out 生成器可预测性)
        xf_held_tr = (feats[held]["orig"][splits[held]["train"]] - mu) / sd
        for tname in ["prada_full"] + SINGLE_TARGETS:
            t_all = np.concatenate([targets[ds][tname][splits[ds]["train"]]
                                    for ds in tr_ds]).astype(np.float32)
            tm, ts = t_all.mean(), t_all.std() + 1e-8
            ridge = Ridge(alpha=1.0)
            ridge.fit(xtr_z, (t_all - tm) / ts)
            t_held = targets[held][tname][splits[held]["train"]]
            rho = spearmanr(ridge.predict(xf_held_tr), t_held).statistic
            entry["teacher_probes"][tname] = float(rho)

        results["folds"][held] = entry

    # ---- GO 判定 ----
    def fold_metric(student, key):
        return [results["folds"][g]["students"][student][key] for g in GENS]

    bce_o, dis_o = fold_metric("bce_only", "auc_orig"), \
        fold_metric("mech_distill", "auc_orig")
    bce_j, dis_j = fold_metric("bce_only", "auc_jpeg95"), \
        fold_metric("mech_distill", "auc_jpeg95")
    gains = [d - b for d, b in zip(dis_o, bce_o)]
    rho_full = [results["folds"][g]["teacher_probes"]["prada_full"]
                for g in GENS]
    results["macro"] = {
        "bce_only": {"orig": float(np.mean(bce_o)),
                     "jpeg95": float(np.mean(bce_j)),
                     "orig_per_fold": dict(zip(GENS, bce_o))},
        "mech_distill": {"orig": float(np.mean(dis_o)),
                         "jpeg95": float(np.mean(dis_j)),
                         "orig_per_fold": dict(zip(GENS, dis_o))},
        "gain_per_fold": dict(zip(GENS, gains)),
    }
    results["go"] = {
        "crit1_macro_gain>=0.02": float(np.mean(gains)),
        "crit2_folds_improved>=3": int(sum(g >= 0 for g in gains)),
        "crit3_jpeg95_gain>0": float(np.mean(dis_j) - np.mean(bce_j)),
        "crit4_info_teacher_spearman_median": float(np.median(rho_full)),
        "crit4_info_teacher_spearman_per_fold": dict(zip(GENS, rho_full)),
    }
    results["go"]["GO"] = bool(
        results["go"]["crit1_macro_gain>=0.02"] >= 0.02
        and results["go"]["crit2_folds_improved>=3"] >= 3
        and results["go"]["crit3_jpeg95_gain>0"] > 0)
    print(json.dumps({"macro": results["macro"], "go": results["go"]},
                     indent=2), flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {out_path}", flush=True)


if __name__ == "__main__":
    main()
