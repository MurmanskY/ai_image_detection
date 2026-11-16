from utils.registry import Registry

METHODS = Registry("method")


class BaseMethod:
    """方法接口。每个方法（baseline / ours）通过 @METHODS.register("xxx") 注册，
    并实现:
        get_transform(train) -> callable   图像预处理/增广
        build_model() -> nn.Module         输出单个 logit 的模型
    requires_training=False 的方法（如 RIGID）无需训练与权重即可评估。
    """

    requires_training = True

    def __init__(self, cfg):
        self.cfg = cfg

    def get_transform(self, train):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError


def build_method(cfg):
    return METHODS.get(cfg.name)(cfg)
