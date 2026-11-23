import numpy as np
from PIL import Image

from retrocore.preview import PC98_PIXEL_ASPECT, aspect_preview, crt_preview, screen_pixel_aspect


def test_pixel_aspect():
    assert abs(PC98_PIXEL_ASPECT - 1.2) < 1e-9
    assert abs(screen_pixel_aspect(640, 200) - 2.4) < 1e-9


def test_aspect_preview_size():
    img = Image.new("P", (640, 400))
    out = aspect_preview(img, scale=1)
    assert out.size == (640, 480)          # 4:3
    assert aspect_preview(img, scale=2).size == (1280, 960)


def test_crt_preview_scanlines():
    img = Image.new("RGB", (16, 10), (200, 200, 200))
    out = crt_preview(img, pixel_aspect=1.0, scale=4, blur=0, bloom=0, scanline=0.5)
    a = np.asarray(out, np.float32)
    assert out.size == (64, 40)
    rows = a[:, 8, 0]
    assert rows[1] > rows[0] and rows[4 + 1] > rows[4]   # raster centre brighter than edge
    out = crt_preview(img)                                 # defaults run
    assert out.mode == "RGB"
