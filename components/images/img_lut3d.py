import numpy as np
from PIL import Image
import os
from ..tree import PRIMERE_ROOT


def _load_cube(path):
    size  = None
    table = []

    LUT_DIR   = os.path.join(PRIMERE_ROOT, 'components', 'images', 'luts')
    cube_path = os.path.join(LUT_DIR, path)

    with open(cube_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "LUT_3D_SIZE" in line:
                size = int(line.split()[-1])
                continue
            if any(k in line for k in ("DOMAIN_MIN", "DOMAIN_MAX", "TITLE", "LUT_1D_SIZE")):
                continue
            if line[0].isdigit() or line[0] == "-":
                table.append([float(x) for x in line.split()])

    table = np.array(table, dtype=np.float32)
    lut = table.reshape((size, size, size, 3))
    return lut, size

def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def _apply_lut_trilinear(arr, lut, size):
    sc  = np.clip(arr, 0.0, 1.0) * (size - 1)

    sr  = sc[..., 0]
    sg  = sc[..., 1]
    sb  = sc[..., 2]

    r0  = np.floor(sr).astype(np.int32)
    g0  = np.floor(sg).astype(np.int32)
    b0  = np.floor(sb).astype(np.int32)

    r1  = np.clip(r0 + 1, 0, size - 1)
    g1  = np.clip(g0 + 1, 0, size - 1)
    b1  = np.clip(b0 + 1, 0, size - 1)

    fr  = (sr - r0)[..., np.newaxis]
    fg  = (sg - g0)[..., np.newaxis]
    fb  = (sb - b0)[..., np.newaxis]

    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    c00  = c000 * (1 - fr) + c001 * fr
    c01  = c010 * (1 - fr) + c011 * fr
    c10  = c100 * (1 - fr) + c101 * fr
    c11  = c110 * (1 - fr) + c111 * fr

    c0   = c00  * (1 - fg) + c01  * fg
    c1   = c10  * (1 - fg) + c11  * fg

    return c0 * (1 - fb) + c1 * fb


def img_lut3d(
    image:         Image.Image,
    lut_path:      str,
    intensity:     float = 1.0,
    input_space:   str   = "sRGB",
    output_space:  str   = "sRGB",
) -> Image.Image:

    img      = image.convert("RGB")
    arr_srgb = np.array(img, dtype=np.float32) / 255.0  # keep original sRGB for blending
    arr = arr_srgb.copy()
    if input_space.lower() == "linear":
        arr = _srgb_to_linear(arr)
    lut, size = _load_cube(lut_path)

    mapped = _apply_lut_trilinear(arr, lut, size)
    if output_space.lower() == "linear":
        mapped = _linear_to_srgb(mapped)

    mapped = np.clip(mapped, 0.0, 1.0)
    out = arr_srgb * (1.0 - intensity) + mapped * intensity
    out = np.clip(out, 0.0, 1.0)

    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")
