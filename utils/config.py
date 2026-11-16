import yaml


class Config(dict):
    """支持属性访问的递归 dict。"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def load_config(path, opts=None):
    """加载 yaml 配置，并支持命令行 --opts key value 覆盖（key 支持 a.b.c 形式）。"""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if opts:
        _apply_opts(cfg, opts)
    return _to_config(cfg)


def _apply_opts(cfg, opts):
    assert len(opts) % 2 == 0, "--opts 需要成对的 key value"
    for k, v in zip(opts[::2], opts[1::2]):
        node = cfg
        keys = k.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = yaml.safe_load(v)
