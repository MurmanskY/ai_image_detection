"""D³QE 核心统计分析 (调研文档建议的阶段二实验)。

不训练完整 D³QE, 而是复现其核心观察:
  1. 用 LlamaGen VQ tokenizer (vq_ds16_c2i) 将图像编码为离散 token
  2. 统计真实/生成图像的 codebook 使用频率分布 (长尾 vs 集中)
  3. 计算每张图像的 token entropy 与 quantization error
  4. 对比分布并输出图表与统计摘要

用法:
  python analysis/token_stats.py \
      --root data/ARForensics --split test --models LlamaGen VAR \
      --vq_ckpt pretrained/vq_ds16_c2i.pt --n 500 --out outputs/analysis
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "D3QE"))

from networks.vq_model import ModelArgs, VQModel  # noqa: E402

from datasets.arforensics import ARForensicsDataset  # noqa: E402


def load_vq(ckpt_path, device):
    model = VQModel(ModelArgs())
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_transform():
    return T.Compose([
        T.Resize(256), T.CenterCrop(256), T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def encode_tokens(vq, img, device):
    """返回 (token_indices (b*256,), 每个 token 位置的量化误差)。"""
    quant, _, info = vq.encode(img.to(device))
    z_q, z = quant  # z_q: (B,C,H,W), z: (B,H,W,C)
    indices = info[2]
    qerr = (z_q - z.permute(0, 3, 1, 2)).pow(2).mean(dim=1).flatten(1)
    return indices.cpu(), qerr.cpu()


def entropy_of(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="*", default=None,
                        help="test 划分下的模型子集, 默认全部")
    parser.add_argument("--vq_ckpt", default="pretrained/vq_ds16_c2i.pt")
    parser.add_argument("--n", type=int, default=500, help="每类最多采样图像数")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")
    vq = load_vq(args.vq_ckpt, device)
    codebook_size = vq.config.codebook_size

    dataset = ARForensicsDataset(
        args.root, args.split, get_transform(),
        models=args.models, max_per_class=args.n,
    )
    print(f"共 {len(dataset)} 张图像")

    # 按 (label, generator) 聚合统计
    groups = {}
    for _, label, gen in dataset.samples:
        groups.setdefault((label, gen), []).append(None)

    counts = {k: np.zeros(codebook_size, dtype=np.int64) for k in groups}
    token_ent = {k: [] for k in groups}
    quant_err = {k: [] for k in groups}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )
    done = 0
    for batch in loader:
        indices, qerr = encode_tokens(vq, batch["img"], device)
        B = batch["img"].shape[0]
        tokens = indices.view(B, -1).numpy()
        for i in range(B):
            key = (int(batch["label"][i]), batch["generator"][i])
            uniq, c = np.unique(tokens[i], return_counts=True)
            counts[key][uniq] += c
            token_ent[key].append(entropy_of(c))
            quant_err[key].append(float(qerr.view(B, -1)[i].mean()))
        done += B
        if done % 200 < args.batch_size:
            print(f"  {done}/{len(dataset)}")

    # 汇总统计
    summary = {}
    for (label, gen) in groups:
        key_name = f"{'real' if label == 0 else 'fake'}:{gen}"
        c = counts[(label, gen)]
        used = int((c > 0).sum())
        p = c / max(c.sum(), 1)
        top10 = float(np.sort(p)[-10:].sum())
        top100 = float(np.sort(p)[-100:].sum())
        summary[key_name] = {
            "n_images": len(token_ent[(label, gen)]),
            "codebook_usage": used,
            "codebook_usage_ratio": used / codebook_size,
            "hist_entropy": entropy_of(c),
            "top10_freq_mass": top10,
            "top100_freq_mass": top100,
            "token_entropy_mean": float(np.mean(token_ent[(label, gen)])),
            "token_entropy_std": float(np.std(token_ent[(label, gen)])),
            "quant_error_mean": float(np.mean(quant_err[(label, gen)])),
            "quant_error_std": float(np.std(quant_err[(label, gen)])),
        }
        print(key_name, json.dumps(summary[key_name], indent=2))

    with open(os.path.join(args.out, "token_stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 绘图: codebook 频率曲线 (排序后) + token entropy 直方图 + quant error 直方图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for (label, gen) in groups:
            name = f"{'real' if label == 0 else 'fake'}:{gen}"
            c = counts[(label, gen)]
            p = np.sort(c / max(c.sum(), 1))[::-1]
            axes[0].plot(np.arange(len(p)), p, label=name, alpha=0.8)
            axes[1].hist(token_ent[(label, gen)], bins=40, alpha=0.5,
                         label=name, density=True)
            axes[2].hist(quant_err[(label, gen)], bins=40, alpha=0.5,
                         label=name, density=True)
        axes[0].set_yscale("log")
        axes[0].set_title("Codebook frequency (sorted, log)")
        axes[1].set_title("Per-image token entropy")
        axes[2].set_title("Per-image quantization error")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "token_stats.png"), dpi=150)
        print(f"图表已保存到 {os.path.join(args.out, 'token_stats.png')}")
    except ImportError:
        print("matplotlib 不可用, 跳过绘图")


if __name__ == "__main__":
    main()
