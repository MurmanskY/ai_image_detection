"""CTAD 实验 A: 跨 tokenizer 亲和力差异的检测可行性验证 (docs/method_design.md)。

核心假设: AR 生成图在其源生成器的 tokenizer 家族下量化误差异常低
(亲和力), 真实图对所有 tokenizer 同等"在流形外"。

对 K 个冻结 VQ tokenizer (LlamaGen / VAR / RAR) 计算每图像量化误差,
用真实验证集拟合参考分布得到 z 分数, 评估:
  S_min    = -min_k z_k       (异常贴合某 tokenizer = fake)
  S_spread = std_k(z_k)       (跨 tokenizer 离散度)
  S_comb   = S_min + S_spread
以及每个 tokenizer 单独 qerr 的 per-generator AUC 作为对照。

用法:
  python analysis/ctad_feasibility.py --root data/ARForensics --n 500 \
      --out outputs/ctad/feasibility.json
"""

import argparse
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

from datasets.arforensics import ARForensicsDataset  # noqa: E402
from analysis.ctad_affinity import (  # noqa: E402
    affinity_score,
    conformal_upper_tail_p,
)

PRADA_WEIGHTS = os.path.join(PROJ, "third_party", "PRADA", "model_weights")


# ---------------- tokenizer 加载与 qerr 计算 ----------------

def load_llamagen_vq(device):
    from networks.vq_model import ModelArgs, VQModel
    model = VQModel(ModelArgs())
    state = torch.load(os.path.join(PROJ, "pretrained", "vq_ds16_c2i.pt"),
                       map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.eval().to(device).requires_grad_(False)


def load_var_vae(device):
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29617")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    from external.VAR.models.vqvae import VQVAE
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=patch_nums).to(device)
    ckpt = os.path.join(PRADA_WEIGHTS, "var", "vae_ch160v4096z32.pth")
    vae.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return vae.eval().requires_grad_(False)


def load_rar_tokenizer(device):
    from external.rar.modeling.titok import PretrainedTokenizer
    ckpt = os.path.join(PRADA_WEIGHTS, "rar", "maskgit-vqgan-imagenet-f16-256.bin")
    return PretrainedTokenizer(ckpt).eval().to(device).requires_grad_(False)


def load_open_magvit2(device):
    sys.path.insert(0, os.path.join(PROJ, "third_party", "SEED-Voken"))
    from src.Open_MAGVIT2.models.lfqgan import VQModel
    ddconfig = {
        "double_z": False, "z_channels": 18, "resolution": 128,
        "in_channels": 3, "out_ch": 3, "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4], "num_res_blocks": 4,
    }
    lossconfig = {"target": "torch.nn.Identity", "params": {}}
    model = VQModel(ddconfig=ddconfig, lossconfig=lossconfig,
                    n_embed=262144, embed_dim=18,
                    sample_minimization_weight=1.0,
                    batch_maximization_weight=1.0)
    ckpt = os.path.join(PROJ, "pretrained", "open_magvit2", "imagenet_256_L.ckpt")
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device).requires_grad_(False)


@torch.no_grad()
def qerr_open_magvit2(model, img):
    h = model.encoder(img)
    (quant, _, _), _ = model.quantize(h, return_loss_breakdown=True)
    return (quant - h).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()


def load_switti_vae(device):
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29617")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    from external.switti.models.vqvae import VQVAEHF
    path = os.path.join(PRADA_WEIGHTS, "switti", "vae")
    return VQVAEHF.from_pretrained(path, reso=512).eval().to(device).requires_grad_(False)


@torch.no_grad()
def qerr_switti(vae, img):
    f = vae.quant_conv(vae.encoder(img))
    f_hat, _, _ = vae.quantize(f)
    return (f_hat - f).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()


def load_infinity_vae(device):
    from external.Infinity.infinity.models.bsq_vae.vae import vae_model
    path = os.path.join(PRADA_WEIGHTS, "infinity", "infinity_vae_d32reg.pth")
    return vae_model(path, "dynamic", 32, 2 ** 32, patch_size=16,
                     encoder_ch_mult=[1, 2, 4, 4, 4],
                     decoder_ch_mult=[1, 2, 4, 4, 4],
                     test_mode=True).to(device).eval().requires_grad_(False)


@torch.no_grad()
def qerr_infinity(vae, img):
    import numpy as _np
    from external.Infinity.infinity.utils.dynamic_resolution import (
        dynamic_resolution_h_w, h_div_w_templates)
    t = h_div_w_templates[_np.argmin(_np.abs(h_div_w_templates - 1.0))]
    ss = [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[t]["1M"]["scales"]]
    h, z, *_ = vae.encode(img, scale_schedule=ss,
                          return_residual_norm_per_scale=False)
    return (z - h).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()


@torch.no_grad()
def qerr_llamagen(model, img):
    quant, _, _ = model.encode(img)
    z_q, z = quant
    return (z_q - z.permute(0, 3, 1, 2)).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()


@torch.no_grad()
def qerr_var(vae, img):
    f = vae.quant_conv(vae.encoder(img))
    f_hat, _, _ = vae.quantize(f)
    return (f_hat - f).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()


@torch.no_grad()
def qerr_rar(tok, img):
    hidden = tok.encoder(img)
    quantized, _, _ = tok.quantize(hidden)
    return (quantized - hidden).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()


TOKENIZERS = {
    "llamagen": {
        "load": load_llamagen_vq,
        "qerr": qerr_llamagen,
        "transform": T.Compose([
            T.Resize(256), T.CenterCrop(256), T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)]),
    },
    "var": {
        "load": load_var_vae,
        "qerr": qerr_var,
        "transform": T.Compose([
            T.Resize(256), T.CenterCrop(256), T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)]),
    },
    "rar": {
        "load": load_rar_tokenizer,
        "qerr": qerr_rar,
        "transform": T.Compose([
            T.Resize(256), T.CenterCrop(256), T.ToTensor(),
            T.Normalize([0.0] * 3, [1.0] * 3)]),  # [0,1]
    },
    "open_magvit2": {
        "load": load_open_magvit2,
        "qerr": qerr_open_magvit2,
        "transform": T.Compose([
            T.Resize(256), T.CenterCrop(256), T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)]),
    },
    "switti": {
        "load": load_switti_vae,
        "qerr": qerr_switti,
        "transform": T.Compose([
            T.Resize(512, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(512), T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)]),
        "batch_size": 4,
    },
    # infinity VAE 通道已剔除: 消融显示其 qerr 对所有组几乎无差异 (单通道
    # AUC 0.534), 中性通道会稀释 min 聚合 (6 通道 0.661 < 5 通道 0.695)。
}


def compute_qerr_all(name, spec, groups, root, n, batch_size, device):
    """返回 {group_key: qerr 数组}。groups: {(label, gen): None} 的键集合。"""
    batch_size = spec.get("batch_size", batch_size)
    model = spec["load"](device)
    out = {}
    for (label, gen) in groups:
        models_arg = None if label == 0 else [gen]
        ds = ARForensicsDataset(root, "test", spec["transform"],
                                models=models_arg, max_per_class=n)
        if label == 0:
            ds.samples = [s for s in ds.samples if s[1] == 0]
        else:
            ds.samples = [s for s in ds.samples if s[1] == 1 and s[2] == gen]
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4)
        vals = []
        for batch in loader:
            vals.append(spec["qerr"](model, batch["img"].to(device)))
        out[(label, gen)] = np.concatenate(vals)
        print(f"  [{name}] {'real' if label == 0 else 'fake'}:{gen} "
              f"n={len(out[(label, gen)])} qerr={out[(label, gen)].mean():.5f}")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--n", type=int, default=500, help="每类采样数")
    parser.add_argument("--n_ref", type=int, default=1000, help="real 参考样本数 (val)")
    parser.add_argument("--target_fpr", type=float, default=0.05,
                        help="conformal p-value operating point")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="outputs/ctad/feasibility.json")
    args = parser.parse_args()
    if not 0 < args.target_fpr < 1:
        raise ValueError("--target_fpr must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 收集测试组: 每生成器 real + fake
    probe = ARForensicsDataset(args.root, "test", T.Compose([T.ToTensor()]),
                               max_per_class=args.n)
    groups = sorted({(label, gen) for _, label, gen in probe.samples})

    # 参考分布: val real
    names = list(TOKENIZERS)
    ref = {}
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (val real 参考)")
        model = spec["load"](device)
        ds = ARForensicsDataset(args.root, "val", spec["transform"],
                                max_per_class=args.n_ref)
        ds.samples = [s for s in ds.samples if s[1] == 0]
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        vals = []
        for batch in loader:
            vals.append(spec["qerr"](model, batch["img"].to(device)))
        ref[name] = np.concatenate(vals)
        print(f"  ref n={len(ref[name])} mean={ref[name].mean():.5f} "
              f"std={ref[name].std():.5f}")
        del model
        torch.cuda.empty_cache()

    # Split the real-only validation reference before fitting empirical tails.
    # The calibration half is never used to estimate tokenizer CDFs.
    ref_lengths = {len(values) for values in ref.values()}
    if len(ref_lengths) != 1:
        raise ValueError("tokenizer reference sets must contain the same images")
    n_ref = ref_lengths.pop()
    if n_ref < 4:
        raise ValueError("at least four real reference images are required")
    ref_order = np.random.default_rng(42).permutation(n_ref)
    n_fit = n_ref // 2
    fit_indices, calibration_indices = ref_order[:n_fit], ref_order[n_fit:]
    ref_fit = [ref[name][fit_indices] for name in names]
    ref_calibration = np.column_stack(
        [ref[name][calibration_indices] for name in names])
    calibration_scores = affinity_score(ref_calibration, ref_fit)

    # 每组 qerr
    qerr = {}
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (test)")
        qerr[name] = compute_qerr_all(
            name, spec, groups, args.root, args.n, args.batch_size, device)

    # z 分数与检测分数
    results = {"n_per_class": args.n, "n_ref": len(ref[names[0]]),
               "ref_stats": {k: {"mean": float(ref[k].mean()),
                                 "std": float(ref[k].std())} for k in names},
               "per_generator": {}}

    real_group = next(g for g in groups if g[0] == 0)
    for (label, gen) in groups:
        if label == 0:
            continue
        z_real = np.stack([(qerr[k][real_group] - ref[k].mean()) /
                           (ref[k].std() + 1e-12) for k in names], axis=1)
        z_fake = np.stack([(qerr[k][(1, gen)] - ref[k].mean()) /
                           (ref[k].std() + 1e-12) for k in names], axis=1)
        q_real = np.column_stack([qerr[k][real_group] for k in names])
        q_fake = np.column_stack([qerr[k][(1, gen)] for k in names])
        affinity_real = affinity_score(q_real, ref_fit)
        affinity_fake = affinity_score(q_fake, ref_fit)
        labels = np.array([0] * len(affinity_real) + [1] * len(affinity_fake))
        affinity = np.concatenate([affinity_real, affinity_fake])
        conformal_p = conformal_upper_tail_p(calibration_scores, affinity)
        is_fake = conformal_p <= args.target_fpr
        conformal_metrics = {
            "affinity_auc": float(roc_auc_score(labels, affinity)),
            "target_fpr": args.target_fpr,
            "real_calibration_size": int(len(calibration_scores)),
            "test_fpr": float(is_fake[labels == 0].mean()),
            "test_tpr": float(is_fake[labels == 1].mean()),
        }
        labels = np.array([0] * len(z_real) + [1] * len(z_fake))
        entry = {}
        for score_name, s_real, s_fake in [
            ("S_min", -z_real.min(axis=1), -z_fake.min(axis=1)),
            ("S_spread", z_real.std(axis=1), z_fake.std(axis=1)),
            ("S_comb", -z_real.min(axis=1) + z_real.std(axis=1),
             -z_fake.min(axis=1) + z_fake.std(axis=1)),
        ]:
            scores = np.concatenate([s_real, s_fake])
            entry[score_name] = {
                "auc": float(roc_auc_score(labels, scores)),
                "real_mean": float(s_real.mean()),
                "fake_mean": float(s_fake.mean()),
            }
        # 单 tokenizer qerr 对照
        entry["per_tokenizer_auc"] = {
            k: float(roc_auc_score(
                labels, np.concatenate([qerr[k][real_group], qerr[k][(1, gen)]])))
            for k in names
        }
        entry["empirical_tail_conformal"] = conformal_metrics
        results["per_generator"][gen] = entry
        print(gen, json.dumps(entry, indent=None))

    # macro 汇总
    for score_name in ("S_min", "S_spread", "S_comb"):
        aucs = [v[score_name]["auc"] for v in results["per_generator"].values()]
        results.setdefault("macro", {})[score_name] = {
            "auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs))}
    for k in names:
        aucs = [v["per_tokenizer_auc"][k] for v in results["per_generator"].values()]
        results["macro"][f"single_{k}"] = {
            "auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs))}
    conformal_aucs = [
        value["empirical_tail_conformal"]["affinity_auc"]
        for value in results["per_generator"].values()
    ]
    results["macro"]["empirical_tail_conformal"] = {
        "auc_mean": float(np.mean(conformal_aucs)),
        "auc_min": float(np.min(conformal_aucs)),
        "reference_fit_size": int(n_fit),
        "real_calibration_size": int(len(calibration_scores)),
        "seed": 42,
    }

    print(json.dumps(results["macro"], indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
