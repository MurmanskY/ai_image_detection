from utils.registry import Registry

DATASETS = Registry("dataset")


def build_dataset(cfg, split, transform):
    return DATASETS.get(cfg.name)(cfg, split, transform)
