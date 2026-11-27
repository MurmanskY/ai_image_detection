import numpy as np

from analysis.dart_interaction import (FEATURE_DIM, METHODS, CycleResidualizer,
                                       parameter_count, transform_images)


def images():
    first = np.zeros((2, 20), dtype=np.float32)
    first[:, :16] = np.linspace(-1, 1, 16)
    first[:, 16] = [0.1, 0.2]
    first[:, 17] = [0.0, 1.0]
    first[:, 18] = [2.0, 3.0]
    first[:, 19] = [4.0, 5.0]
    return [first]


def test_all_methods_have_identical_detector_parameter_count():
    assert parameter_count() == 1394
    for mode in METHODS:
        residualizer = None
        if mode == "lr_cycle":
            residualizer = CycleResidualizer().fit(images(), images())
        transformed = transform_images([images()[0].astype(object)], mode, residualizer)
        assert transformed[0].shape == (2, FEATURE_DIM)
        assert np.isfinite(transformed[0].astype(np.float32)).all()


def test_explicit_interactions_and_lr_excludes_quantization():
    image = images()[0]
    full = transform_images([image], "full_interaction")[0]
    mean_l = image[:, :16].mean(axis=1)
    std_l = image[:, :16].std(axis=1)
    assert np.allclose(full[:, 20], mean_l * image[:, 19])
    assert np.allclose(full[:, 24], image[:, 18] * image[:, 19])
    changed = image.copy()
    changed[:, 18] *= 100
    lr = transform_images([image], "lr_interaction")[0]
    lr_changed = transform_images([changed], "lr_interaction")[0]
    assert np.array_equal(lr, lr_changed)


def test_cycle_residual_is_conditional_and_scaled():
    real = []
    for index in range(200):
        image = np.zeros((1, 20), dtype=np.float32)
        image[0, :16] = index / 20
        image[0, 18] = index / 40
        image[0, 19] = 2 * image[0, 0] + image[0, 18]
        real.append(image)
    residualizer = CycleResidualizer().fit(real[:100], real[100:])
    predicted = residualizer.predict(real[100:])
    actual = np.array([image[0, 19] for image in real[100:]])
    assert residualizer.selection_r2 > 0.99
    assert np.allclose(predicted, actual, atol=1e-2)
    assert residualizer.residual(real[100:]).shape == (100,)
