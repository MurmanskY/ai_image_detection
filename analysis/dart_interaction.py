"""DART 参数匹配的显式生成循环交互与条件一致性残差。"""

import numpy as np

from analysis.dart_p0 import ScaleDeepSet, select_feature_mode

FEATURE_DIM = 25
METHODS = ("lr", "full", "lr_interaction", "full_interaction", "lr_cycle")
PAIR_BASELINE = {
    "lr": "lr", "full": "full", "lr_interaction": "lr",
    "full_interaction": "full", "lr_cycle": "lr",
}


def likelihood_summaries(image):
    likelihood = image[:, :16]
    return likelihood.mean(axis=1), likelihood.std(axis=1)


def pad_images(images, width=FEATURE_DIM):
    output = []
    for image in images:
        if image.shape[1] > width:
            raise ValueError(f"feature width {image.shape[1]} exceeds {width}")
        padded = np.zeros((image.shape[0], width), dtype=np.float32)
        padded[:, :image.shape[1]] = image
        output.append(padded)
    return np.asarray(output, dtype=object)


class CycleResidualizer:
    """在 real reference 上拟合 E[R|L,Q,metadata]，selection 仅估计残差尺度。"""

    def __init__(self, ridge=1.0):
        self.ridge = float(ridge)
        self.coef = None
        self.residual_scale = None

    @staticmethod
    def design(images):
        rows = []
        targets = []
        for image in images:
            # 16 likelihood quantiles + scale metadata + Q; R is the target.
            rows.append(image[:, list(range(16)) + [16, 17, 18]])
            targets.append(image[:, 19])
        x = np.concatenate(rows).astype(np.float64)
        y = np.concatenate(targets).astype(np.float64)
        return np.concatenate([x, np.ones((len(x), 1))], axis=1), y

    def fit(self, real_reference, real_selection):
        x, y = self.design(real_reference)
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[-1, -1] = 0.0
        self.coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        _, selection_y = self.design(real_selection)
        predicted = self.predict(real_selection)
        residual = selection_y - predicted
        center = np.median(residual)
        self.residual_scale = 1.4826 * np.median(np.abs(residual - center)) + 1e-6
        selection_ss = np.sum((selection_y - selection_y.mean()) ** 2)
        self.selection_r2 = float(1.0 - np.sum(residual ** 2) / (selection_ss + 1e-12))
        return self

    def predict(self, images):
        if self.coef is None:
            raise ValueError("residualizer is not fitted")
        x, _ = self.design(images)
        return x @ self.coef

    def residual(self, images):
        if self.residual_scale is None:
            raise ValueError("residual scale is not fitted")
        x, y = self.design(images)
        residual = y - x @ self.coef
        return residual / self.residual_scale


def transform_images(images, mode, residualizer=None):
    if mode not in METHODS + ("full_cycle",):
        raise ValueError(f"unknown interaction method: {mode}")
    output = []
    for image in images:
        image = np.asarray(image, dtype=np.float32)
        mean_l, std_l = likelihood_summaries(image)
        quant, rec = image[:, 18], image[:, 19]
        if mode in ("lr", "lr_cycle"):
            base = select_feature_mode([image], "likelihood_reconstruction")[0]
        elif mode == "lr_interaction":
            base = select_feature_mode([image], "likelihood_reconstruction")[0]
            interaction = np.stack([mean_l * rec, std_l * rec], axis=1)
            base = np.concatenate([base, interaction], axis=1)
        elif mode in ("full", "full_interaction", "full_cycle"):
            base = image
            if mode == "full_interaction":
                interaction = np.stack([mean_l * rec, std_l * rec, mean_l * quant,
                                        std_l * quant, quant * rec], axis=1)
                base = np.concatenate([base, interaction], axis=1)
        if mode in ("lr_cycle", "full_cycle"):
            if residualizer is None:
                raise ValueError(f"{mode} requires a fitted residualizer")
            residual = residualizer.residual([image])
            base = np.concatenate([base, residual[:, None]], axis=1)
        output.append(np.asarray(base, dtype=np.float32))
    return pad_images(output)


def parameter_count():
    return sum(parameter.numel() for parameter in ScaleDeepSet(FEATURE_DIM).parameters())
