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
    samples = make_samples(root, {"0_real": 0, "1_fake": 1}, "real", cfg.get("max_per_class"))
    if not samples:
        raise FileNotFoundError(f"未在 {root} 找到图像")
    return RealFakeDataset(samples, transform)
