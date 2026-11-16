import os

from .build import DATASETS
from .folder_dataset import RealFakeDataset, make_samples


class ARForensicsDataset(RealFakeDataset):
    def __init__(self, root, split, transform, models=None, max_per_class=None):
        root = os.path.join(root, split)
        if split in ("train", "val"):
            samples = make_samples(root, {"0_real": 0, "1_fake": 1}, "LlamaGen", max_per_class)
            samples = [
                (path, label, "real" if label == 0 else generator)
                for path, label, generator in samples
            ]
        elif split == "test":
            samples = []
            for generator in sorted(os.listdir(root)):
                if models is not None and generator not in models:
                    continue
                generator_root = os.path.join(root, generator)
                if not os.path.isdir(generator_root):
                    continue
                samples.extend(make_samples(generator_root, {"0_real": 0}, "real", max_per_class))
                samples.extend(make_samples(generator_root, {"1_fake": 1}, generator, max_per_class))
        else:
            raise ValueError(f"未知 ARForensics 划分: {split}")
        if not samples:
            raise FileNotFoundError(f"未在 {root} 找到 ARForensics 图像")
        super().__init__(samples, transform)


@DATASETS.register("arforensics")
def build_arforensics(cfg, split, transform):
    return ARForensicsDataset(cfg.root, split, transform, max_per_class=cfg.get("max_per_class"))
