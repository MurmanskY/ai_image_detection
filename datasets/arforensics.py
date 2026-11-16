import os

from .build import DATASETS
from .folder_dataset import RealFakeDataset, make_samples


class ARForensicsDataset(RealFakeDataset):
    """ARForensics 数据集。

    train/val: 扁平的 0_real/1_fake (当前由官方 val 重划分而来, fake 为 LlamaGen);
    test: 每个 generator 一个子目录, 各自含 0_real/1_fake。
    models 参数限定使用哪些 generator (test), 或在 train/val 存在
    per-generator 子目录时做 leave-one-generator-out 选择。
    """

    def __init__(self, root, split, transform, models=None, max_per_class=None,
                 train_fake_generator="LlamaGen"):
        root = os.path.join(root, split)
        generator_dirs = []
        if os.path.isdir(root):
            generator_dirs = [
                name for name in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, name))
                and (os.path.isdir(os.path.join(root, name, "0_real"))
                     or os.path.isdir(os.path.join(root, name, "1_fake")))
            ]
        if generator_dirs:
            samples = []
            for generator in generator_dirs:
                if models is not None and generator not in models:
                    continue
                generator_root = os.path.join(root, generator)
                samples.extend(make_samples(generator_root, {"0_real": 0}, "real", max_per_class))
                samples.extend(make_samples(generator_root, {"1_fake": 1}, generator, max_per_class))
        elif split in ("train", "val"):
            samples = make_samples(root, {"0_real": 0}, "real", max_per_class)
            samples += make_samples(root, {"1_fake": 1}, train_fake_generator, max_per_class)
        else:
            raise ValueError(f"未知 ARForensics 划分: {split}")
        if not samples:
            raise FileNotFoundError(f"未在 {root} 找到 ARForensics 图像")
        super().__init__(samples, transform)


@DATASETS.register("arforensics")
def build_arforensics(cfg, split, transform):
    # 支持 dataset.models 或 dataset.{train,val,test}_generators 配置,
    # 用于 leave-one-generator-out 等跨生成器协议
    models = cfg.get(f"{split}_generators", cfg.get("models"))
    return ARForensicsDataset(
        cfg.root, split, transform, models=models,
        max_per_class=cfg.get("max_per_class"),
        train_fake_generator=cfg.get("train_fake_generator", "LlamaGen"),
    )
