"""CTAD 通道 2 全量 test (84k) qerr 计算并缓存 zfeat_full.npz。

与 ctad_logo.build_zfeat 相同管线, 但 test 不采样 (max_per_class=None),
参考分布沿用 val real (与 ctad_feasibility 一致)。

用法:
  python analysis/ctad_full_test.py --out outputs/ctad/zfeat_full.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

PROJ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "third_party", "D3QE"))
sys.path.insert(0, os.path.join(PROJ, "third_party", "PRADA", "src"))

from analysis.ctad_feasibility import TOKENIZERS, compute_qerr_all  # noqa: E402
from datasets.arforensics import ARForensicsDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ARForensics")
    parser.add_argument("--n_ref", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="outputs/ctad/zfeat_full.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    probe = ARForensicsDataset(args.root, "test", T.Compose([T.ToTensor()]))
    groups = sorted({(label, gen) for _, label, gen in probe.samples})
    names = list(TOKENIZERS)
    print(f"groups: {groups}")

    # 参考: val real (与 zfeat.npz 相同的参考, 直接复算保持自洽)
    ref = {}
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (val real 参考)")
        model = spec["load"](device)
        ds = ARForensicsDataset(args.root, "val", spec["transform"],
                                max_per_class=args.n_ref)
        ds.samples = [s for s in ds.samples if s[1] == 0]
        loader = torch.utils.data.DataLoader(
            ds, batch_size=spec.get("batch_size", args.batch_size),
            shuffle=False, num_workers=4)
        vals = []
        for batch in loader:
            vals.append(spec["qerr"](model, batch["img"].to(device)))
        ref[name] = np.concatenate(vals)
        print(f"  ref mean={ref[name].mean():.5f} std={ref[name].std():.5f}")
        del model
        torch.cuda.empty_cache()

    data = {"groups": np.array([f"{l}:{g}" for l, g in groups]),
            "names": np.array(names)}
    for label, gen in groups:
        data[f"paths_{label}_{gen}"] = np.asarray([
            path for path, sample_label, sample_gen in probe.samples
            if sample_label == label and sample_gen == gen
        ])
    for name, spec in TOKENIZERS.items():
        print(f"== tokenizer: {name} (test 全量)")
        qerr = compute_qerr_all(name, spec, groups, args.root, None,
                                args.batch_size, device)
        for (label, gen) in groups:
            z = (qerr[(label, gen)] - ref[name].mean()) / (ref[name].std() + 1e-12)
            data[f"z_{name}_{label}_{gen}"] = z
        data[f"refmean_{name}"] = ref[name].mean()
        data[f"refstd_{name}"] = ref[name].std()
        np.savez(args.out, **data)  # 每个 tokenizer 后落盘, 断点可续
        print(f"  已增量保存 {args.out}")

    print(f"完成, 已保存到 {args.out}")


if __name__ == "__main__":
    main()
