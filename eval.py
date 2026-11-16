import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

import ours  # noqa: F401  注册 ours 方法
from baselines import build_method
from datasets.build import build_dataset
from engine.evaluator import predict_scores
from engine.metrics import calibrate_threshold
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
    parser.add_argument("--threshold", type=float, default=None,
                        help="决策阈值; 缺省时 logit 方法用 0.5, anomaly 方法在 val 上校准")
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

    amp = cfg.train.get("amp", True)
    threshold = args.threshold
    if threshold is None and getattr(method, "score_type", "logit") != "logit" \
            and args.split != "val":
        # training-free 方法: 在验证集上用 Youden's J 校准阈值, 避免 sigmoid(score)>=0.5 的误用
        val_set = build_dataset(cfg.dataset, "val", method.get_transform(train=False))
        val_loader = DataLoader(
            val_set, batch_size=cfg.eval.get("batch_size", 128), shuffle=False,
            num_workers=cfg.train.get("num_workers", 8), pin_memory=True,
        )
        val_labels, val_scores, _, _ = predict_scores(model, val_loader, device, amp, method)
        threshold = calibrate_threshold(val_labels, val_scores)
        print(f"anomaly 方法在 val 上校准阈值: {threshold:.6f}")
    if threshold is None:
        threshold = 0.5

    from collections import defaultdict

    from engine.metrics import compute_metrics

    labels, scores, gens, paths = predict_scores(model, loader, device, amp, method)
    results = {"overall": compute_metrics(labels, scores, threshold)}
    per_gen = defaultdict(lambda: ([], []))
    for l, s, g in zip(labels, scores, gens):
        per_gen[g][0].append(l)
        per_gen[g][1].append(s)
    for g, (l, s) in per_gen.items():
        results[g] = compute_metrics(l, s, threshold)
    results["score_type"] = getattr(method, "score_type", "logit")
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

    # 保存逐样本预测, 之后无需 GPU 即可重算任意指标
    pred_path = os.path.splitext(out_path)[0] + "_predictions.csv"
    method_name = cfg.method.get("name", "unknown")
    seed = cfg.get("seed", 42)
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["path", "label", "generator", "score", "prediction", "split", "method", "seed"]
        )
        for path, label, gen, score in zip(paths, labels, gens, scores):
            writer.writerow(
                [path, int(label), gen, float(score), int(score >= threshold),
                 args.split, method_name, seed]
            )

    print(json.dumps(results, indent=2))
    print(f"结果已保存到 {out_path}, 逐样本预测在 {pred_path}")


if __name__ == "__main__":
    main()
