import time

import numpy as np

from retrocore.preprocess import bilateral, kuwahara, local_variance, smooth


def noisy_step(h=40, w=60, noise=12.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    img[:, : w // 2] = (60, 80, 100)
    img[:, w // 2:] = (200, 180, 160)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255)


def _flat_var(x):
    g = x.mean(2)
    return local_variance(g, 1)[5:-5, 5:25].mean()


def test_filters_flatten_noise_and_keep_edge():
    src = noisy_step()
    for out in (bilateral(src, 2, 30), kuwahara(src, 2), smooth(src, "bilateral", 0.8),
                smooth(src, "kuwahara", 0.5)):
        assert out.shape == src.shape and out.dtype == np.float32
        assert _flat_var(out) < 0.5 * _flat_var(src)
        # Edge stays sharp: the two columns around the step remain far apart.
        jump = out[:, 31].mean(0) - out[:, 28].mean(0)
        assert (jump > 0.8 * np.array([140, 100, 60])).all()


def test_strength_zero_and_none_are_noops():
    src = noisy_step()
    assert np.allclose(smooth(src, "bilateral", 0.0), src)
    assert np.allclose(smooth(src, "none", 1.0), src)


def test_kuwahara_constant_image():
    src = np.full((10, 12, 3), 77, np.float32)
    assert np.allclose(kuwahara(src, 3), 77)


def test_smoothing_640x400_is_fast():
    src = np.random.default_rng(1).integers(0, 255, (400, 640, 3)).astype(np.float32)
    t = time.perf_counter()
    smooth(src, "bilateral", 1.0)
    smooth(src, "kuwahara", 1.0)
    assert time.perf_counter() - t < 6.0
