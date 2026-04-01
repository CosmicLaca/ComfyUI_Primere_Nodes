import numpy as np
from PIL import Image


PAPER_PRESETS = {
    "BN (ISO R 130, soft)": {"contrast": 0.88, "toe": 0.10, "shoulder": 0.04},
    "B (ISO R 110, medium-soft)": {"contrast": 0.96, "toe": 0.08, "shoulder": 0.05},
    "N (ISO R 90, normal)": {"contrast": 1.04, "toe": 0.06, "shoulder": 0.06},
    "H (ISO R 70, hard)": {"contrast": 1.18, "toe": 0.05, "shoulder": 0.08},
    "HD (ISO R 50, extra hard)": {"contrast": 1.32, "toe": 0.03, "shoulder": 0.10},
    "Grade 00 (very soft)": {"contrast": 0.84, "toe": 0.12, "shoulder": 0.04},
    "Grade 0 (soft)": {"contrast": 0.92, "toe": 0.09, "shoulder": 0.05},
    "Grade 2 (normal)": {"contrast": 1.00, "toe": 0.07, "shoulder": 0.06},
    "Grade 3 (normal-hard)": {"contrast": 1.12, "toe": 0.06, "shoulder": 0.07},
    "Grade 4 (hard)": {"contrast": 1.24, "toe": 0.05, "shoulder": 0.09},
    "Grade 5 (ultra hard)": {"contrast": 1.36, "toe": 0.03, "shoulder": 0.12},
}


def _apply_tone_curve(x, contrast, toe, shoulder):
    x = np.clip(0.5 + (x - 0.5) * contrast, 0.0, 1.0)
    if toe > 0:
        x = x * (1.0 - toe) + np.power(x, 1.25) * toe
    if shoulder > 0:
        inv = 1.0 - x
        x = 1.0 - (inv * (1.0 - shoulder) + np.power(inv, 1.25) * shoulder)
    return np.clip(x, 0.0, 1.0)


def _apply_paper_expiration(arr, years):
    out = arr.copy()

    fog = np.clip(years * 0.006, 0, 0.18)
    out = out + fog * (1.0 - out)

    blue_fade = np.clip(1.0 - (years * 0.012), 0.65, 1.0)
    out[..., 2] *= blue_fade

    contrast = np.clip(1.0 - (years * 0.01), 0.7, 1.0)
    out = np.clip((out - 0.5) * contrast + 0.5, 0.0, 1.0)

    return out


def img_photo_paper(
    image: Image.Image,
    paper_type: str = "N (ISO R 90, normal)",
    color_paper: bool = False,
    paper_base: str = "RC",
    paper_intensity: float = 100.0,
    expiration_years: float = 0.0,
) -> Image.Image:
    src = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    preset = PAPER_PRESETS.get(paper_type, PAPER_PRESETS["N (ISO R 90, normal)"])

    contrast = preset["contrast"]
    toe = preset["toe"]
    shoulder = preset["shoulder"]

    if paper_base == "FB":
        toe += 0.02
        shoulder += 0.01
        contrast *= 0.98
    elif paper_base == "RC":
        contrast *= 1.02

    intensity = float(np.clip(paper_intensity, 0.0, 200.0))
    strength = (intensity - 100.0) / 100.0
    contrast = np.clip(contrast * (1.0 + 0.25 * strength), 0.5, 2.5)
    toe = np.clip(toe * (1.0 + 0.35 * strength), 0.0, 0.35)
    shoulder = np.clip(shoulder * (1.0 + 0.35 * strength), 0.0, 0.35)

    if color_paper:
        out = _apply_tone_curve(src, contrast, toe, shoulder)
        sat = 1.02 if paper_base == "RC" else 0.98
        mean = out.mean(axis=2, keepdims=True)
        out = np.clip(mean + (out - mean) * sat, 0.0, 1.0)
    else:
        # In B&W mode, force grayscale source before blending to avoid color casts
        # at any intensity value (including > 100).
        luma = 0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]
        src = np.stack([luma, luma, luma], axis=-1)
        tone = _apply_tone_curve(luma, contrast, toe, shoulder)
        out = np.stack([tone, tone, tone], axis=-1)

    out = _apply_paper_expiration(out, expiration_years)

    if intensity <= 100.0:
        amount = (intensity / 100.0) * 0.90
    else:
        amount = 1.0 + ((intensity - 100.0) / 100.0) * 1.10
    amount = np.clip(amount, 0.0, 2.0)

    blended = np.clip(src + (out - src) * amount, 0.0, 1.0)
    return Image.fromarray((blended * 255.0).astype(np.uint8), mode="RGB")
