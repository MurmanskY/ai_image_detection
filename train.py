import argparse
import os

import ours  # noqa: F401  注册 ours 方法
from baselines import build_method
from datasets.build import build_dataset
from engine.trainer import Trainer
from utils.config import load_config
from utils.misc import set_seed


def main():
    parser = argparse.ArgumentParser(description="训练生成图像鉴别器")
    parser.add_argument("--config", required=True, help="实验配置 yaml 路径")
    parser.add_argument("--opts", nargs="*", default=None,
                        help="覆盖配置项, 例: train.lr 1e-5 dataset.root /data/GenImage")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    set_seed(cfg.get("seed", 42))
    work_dir = cfg.get("work_dir", "outputs/exp")
    os.makedirs(work_dir, exist_ok=True)

    method = build_method(cfg.method)
    model = method.build_model()
    train_set = build_dataset(cfg.dataset, "train", method.get_transform(train=True))
    val_set = build_dataset(cfg.dataset, "val", method.get_transform(train=False))

    trainer = Trainer(model, cfg, work_dir)
    trainer.fit(train_set, val_set)


if __name__ == "__main__":
    main()
