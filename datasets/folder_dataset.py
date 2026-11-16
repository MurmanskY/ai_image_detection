import os

from PIL import Image
from torch.utils.data import Dataset

from .build import DATASETS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(directory):
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]


class RealFakeDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, generator = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return {"img": image, "label": label, "path": path, "generator": generator}


def make_samples(root, class_map, generator, max_per_class=None):
    samples = []
    for class_name, label in class_map.items():
        files = image_files(os.path.join(root, class_name))
        if max_per_class is not None:
            files = files[:max_per_class]
        samples.extend((path, label, generator) for path in files)
    return samples


@DATASETS.register("folder")
def build_folder_dataset(cfg, split, transform):
    root = os.path.join(cfg.root, split)
    max_per_class = cfg.get("max_per_class")
    samples = []
    if os.path.isdir(root):
        # 若 split 下是 <generator>/{0_real,1_fake} 结构, 按目录名记录 provenance
        generator_dirs = [
            name for name in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, name))
            and (os.path.isdir(os.path.join(root, name, "0_real"))
                 or os.path.isdir(os.path.join(root, name, "1_fake")))
        ]
        if generator_dirs:
            for generator in generator_dirs:
                generator_root = os.path.join(root, generator)
                samples += make_samples(generator_root, {"0_real": 0}, "real", max_per_class)
                samples += make_samples(generator_root, {"1_fake": 1}, generator, max_per_class)
        else:
            samples = make_samples(root, {"0_real": 0}, "real", max_per_class)
            samples += make_samples(
                root, {"1_fake": 1}, cfg.get("generator", "unknown_fake"), max_per_class
            )
    if not samples:
        raise FileNotFoundError(f"未在 {root} 找到图像")
    return RealFakeDataset(samples, transform)
