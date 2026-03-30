import numpy as np
from PIL import Image
import os
from ..tree import PRIMERE_ROOT


def _load_cube(path):
    size = None
    table = []

    LUT_DIR = os.path.join(PRIMERE_ROOT, 'components', 'images', 'luts')
    cube_path = os.path.join(LUT_DIR, path)

    with open(cube_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "LUT_3D_SIZE" in line:
                size = int(line.split()[-1])
            elif line[0].isdigit() or line[0] == "-":
                table.append([float(x) for x in line.split()])

    table = np.array(table, dtype=np.float32)
    lut = table.reshape((size, size, size, 3))
    return lut, size

def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)

def _apply_lut_trilinear(arr, lut, size):
    arr = np.clip(arr, 0.0, 1.0) * (size - 1)
    i0 = np.floor(arr).astype(int)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = arr - i0
    c000 = lut[i0[..., 0], i0[..., 1], i0[..., 2]]
    c100 = lut[i1[..., 0], i0[..., 1], i0[..., 2]]
    c010 = lut[i0[..., 0], i1[..., 1], i0[..., 2]]
    c110 = lut[i1[..., 0], i1[..., 1], i0[..., 2]]
    c001 = lut[i0[..., 0], i0[..., 1], i1[..., 2]]
    c101 = lut[i1[..., 0], i0[..., 1], i1[..., 2]]
    c011 = lut[i0[..., 0], i1[..., 1], i1[..., 2]]
    c111 = lut[i1[..., 0], i1[..., 1], i1[..., 2]]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c00 = c000 * (1 - fx)[..., None] + c100 * fx[..., None]
    c01 = c001 * (1 - fx)[..., None] + c101 * fx[..., None]
    c10 = c010 * (1 - fx)[..., None] + c110 * fx[..., None]
    c11 = c011 * (1 - fx)[..., None] + c111 * fx[..., None]
    c0 = c00 * (1 - fy)[..., None] + c10 * fy[..., None]
    c1 = c01 * (1 - fy)[..., None] + c11 * fy[..., None]
    return c0 * (1 - fz)[..., None] + c1 * fz[..., None]

def img_lut3d(
    image: Image.Image,
    lut_path: str,
    intensity: float = 1.0,
    interpolation: str = "trilinear",
    input_space: str = "sRGB",
    output_space: str = "sRGB",
    precision: bool = True,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if not precision:
        arr = np.round(arr * 255.0) / 255.0
    else:
        arr = np.round(arr * 65535.0) / 65535.0
    if input_space.lower() == "linear":
        arr = _srgb_to_linear(arr)
    lut, size = _load_cube(lut_path)
    if interpolation == "trilinear":
        mapped = _apply_lut_trilinear(arr, lut, size)
    else:
        mapped = _apply_lut_trilinear(arr, lut, size)

    if output_space.lower() == "srgb":
        mapped = _linear_to_srgb(mapped)

    out = arr * (1.0 - intensity) + mapped * intensity
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")