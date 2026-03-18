import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def img_shade_level(image: Image.Image, shade_level: float = 0, radius: float = None, detail_mode: str = "medium",) -> Image.Image:
    if shade_level == 0:
        return image.convert("RGB")

    if not (-100 <= shade_level <= 100):
        raise ValueError(f"shade_level must be -100 … +100, got {shade_level}")

    valid_modes = {"fine", "medium", "broad"}
    if detail_mode not in valid_modes:
        raise ValueError(f"detail_mode must be one of {valid_modes}, got '{detail_mode}'")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    def rgb_to_lab(rgb):
        linear = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4
        )
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz   = linear @ M.T
        xyz_n = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        xyz_r = xyz / xyz_n
        def f(t):
            delta = 6.0 / 29.0
            return np.where(t > delta ** 3, np.cbrt(t),
                            t / (3 * delta ** 2) + 4.0 / 29.0)
        fx, fy, fz = f(xyz_r[...,0]), f(xyz_r[...,1]), f(xyz_r[...,2])
        return np.stack([116.0*fy - 16.0, 500.0*(fx-fy), 200.0*(fy-fz)], axis=-1)

    def lab_to_rgb(lab):
        L, a, b = lab[...,0], lab[...,1], lab[...,2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        def f_inv(t):
            delta = 6.0 / 29.0
            return np.where(t > delta, t ** 3,
                            3 * delta**2 * (t - 4.0/29.0))
        xyz_n = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        xyz   = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1) * xyz_n
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ], dtype=np.float32)
        linear = xyz @ M_inv.T
        srgb   = np.where(linear <= 0.0031308, linear * 12.92,
                          1.055 * np.power(np.clip(linear, 0, None), 1.0/2.4) - 0.055)
        return np.clip(srgb, 0.0, 1.0)

    lab = rgb_to_lab(arr)
    L   = lab[..., 0]
    H, W = L.shape

    if radius is None:
        short = min(H, W)
        radius = {"fine": max(1.0, short * 0.005),
                  "medium": max(1.0, short * 0.01),
                  "broad":  max(2.0, short * 0.03)}[detail_mode]

    L_blurred = gaussian_filter(L, sigma=radius)
    detail    = L - L_blurred
    strength  = (shade_level / 100.0) * (3.0 if shade_level > 0 else 1.0)

    lab_new        = lab.copy()
    lab_new[..., 0] = np.clip(L + strength * detail, 0.0, 100.0)

    return Image.fromarray((lab_to_rgb(lab_new) * 255).astype(np.uint8), mode="RGB")
