import numpy as np

from retrocore.postprocess import (
    apply_outline,
    darkest_index,
    neighbour_counts,
    outline_mask,
    remove_isolated,
)


def test_isolated_pixel_removed_and_protect_respected():
    idx = np.zeros((7, 7), np.uint8)
    idx[3, 3] = 2
    assert remove_isolated(idx)[3, 3] == 0
    prot = np.zeros((7, 7), bool)
    prot[3, 3] = True
    assert remove_isolated(idx, protect=prot)[3, 3] == 2


def test_checker_and_lines_survive():
    yy, xx = np.mgrid[0:8, 0:8]
    checker = ((yy + xx) % 2).astype(np.uint8)
    assert np.array_equal(remove_isolated(checker), checker)
    line = np.zeros((8, 8), np.uint8)
    line[4, :] = 1
    assert np.array_equal(remove_isolated(line), line)


def test_neighbour_counts():
    idx = np.zeros((3, 3), np.uint8)
    idx[1, 1] = 1
    c = neighbour_counts(idx, 2)
    assert c[:, 1, 1].tolist() == [8, 0]
    assert c.sum(0).max() == 8


def test_outline_on_dark_side():
    img = np.zeros((10, 20, 3), np.float32)
    img[:, :10] = 50
    img[:, 10:] = 220
    m = outline_mask(img, threshold=30, thickness=1)
    assert m[:, 9].all() and not m[:, 10:].any() and not m[:, :8].any()
    m2 = outline_mask(img, threshold=30, thickness=2)
    assert m2.sum() > m.sum() and not m2[:, 10:].any()
    pal = np.array([[200, 200, 200], [10, 10, 10]], np.uint8)
    assert darkest_index(pal) == 1
    out = apply_outline(np.zeros((10, 20), np.uint8), m, 1)
    assert (out == 1).sum() == m.sum()
