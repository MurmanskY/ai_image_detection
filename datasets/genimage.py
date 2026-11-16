import os

from .build import DATASETS
from .folder_dataset import RealFakeDataset, image_files


@DATASETS.register("genimage")
def build_genimage(cfg, split, transform):
    root = os.path.join(cfg.root, split)
    max_per_class = cfg.get("max_per_class")
    selected = cfg.get("generators", "all")
    samples = []
    for generator in sorted(os.listdir(root)):
        generator_root = os.path.join(root, generator)
        if not os.path.isdir(generator_root) or (selected != "all" and generator not in selected):
            continue
        for class_name, label in (("nature", 0), ("ai", 1)):
            class_root = os.path.join(generator_root, class_name)
            files = image_files(class_root)
            if max_per_class is not None:
                files = files[:max_per_class]
            group = "real" if label == 0 else generator
            samples.extend((path, label, group) for path in files)
    if not samples:
        raise FileNotFoundError(f"未在 {root} 找到 GenImage 图像")
    return RealFakeDataset(samples, transform)
