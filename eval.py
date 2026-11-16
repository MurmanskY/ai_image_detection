import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import ours  # noqa: F401  注册 ours 方法
from baselines import build_method
from datasets.build import build_dataset
from engine.evaluator import evaluate
from utils.config import load_config
from utils.device import get_device
from utils.misc import set_seed


def main():
    parser = argparse.ArgumentParser(description="评估生成图像鉴别器")
    parser.add_argument("--config", required=True, help="实验配置 yaml 路径")
    parser.add_argument("--checkpoint", default=None,
                        help="权重路径 (best.pth/last.pth); training-free 方法可省略")
    parser.add_argument("--split", default="test", help="评估划分: train/val/test")
    parser.add_argument("--out", default=None, help="结果 json 保存路径")
    parser.add_argument("--opts", nargs="*", default=None, help="覆盖配置项")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    set_seed(cfg.get("seed", 42))

    method = build_method(cfg.method)
    test_set = build_dataset(cfg.dataset, args.split, method.get_transform(train=False))
    loader = DataLoader(
        test_set, batch_size=cfg.eval.get("batch_size", 128), shuffle=False,
        num_workers=cfg.train.get("num_workers", 8), pin_memory=True,
    )

    model = method.build_model()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    elif method.requires_training:
        raise SystemExit("该方法需要训练权重, 请通过 --checkpoint 指定")
    device = get_device(cfg.get("device", "auto"))
    model.to(device)

    results = evaluate(model, loader, device, cfg.train.get("amp", True))
    if args.out:
        out_path = args.out
    elif args.checkpoint:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.checkpoint)),
            f"results_{args.split}.json",
        )
    else:
        out_path = os.path.join(cfg.get("work_dir", "outputs"), f"results_{args.split}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
