import os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def img_similarity(image_list: list) -> dict:
    _WEIGHTS     = {"phash": 0.35, "histogram": 0.30, "ssim": 0.35}
    _COMPARE_SIZE = (64, 64)

    def _load(path):
        return Image.open(path).convert("RGB")

    def _phash(img, size=16):
        arr = np.array(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
        return arr > arr.mean()

    def _histogram(img, bins=64):
        parts = []
        for ch in img.split():
            h, _ = np.histogram(np.array(ch), bins=bins, range=(0, 256))
            h = h.astype(np.float32)
            s = h.sum()
            parts.append(h / s if s > 0 else h)
        return np.concatenate(parts)

    def _ssim_arr(img):
        return np.array(img.convert("L").resize(_COMPARE_SIZE, Image.LANCZOS), dtype=np.float32) / 255.0

    def _features(img):
        return {"phash": _phash(img), "hist": _histogram(img), "ssim": _ssim_arr(img)}

    def _score(ref, target):
        ph  = 1.0 - np.count_nonzero(ref["phash"] != target["phash"]) / ref["phash"].size
        hi  = min(1.0, float(np.sum(np.sqrt(ref["hist"] * target["hist"]))))
        ss  = (ssim(ref["ssim"], target["ssim"], data_range=1.0) + 1.0) / 2.0
        raw = _WEIGHTS["phash"] * ph + _WEIGHTS["histogram"] * hi + _WEIGHTS["ssim"] * ss
        return float(round(min(1.0, max(0.0, raw)) ** 0.65 * 100, 1))

    # ── Main logic ────────────────────────────────────────────────────────────
    if not image_list:
        return {}

    ref_features = _features(_load(image_list[0]))
    result = {Path(str(os.path.basename(image_list[0]))).stem: float(100.0)}

    for path in image_list[1:]:
        name = Path(str(os.path.basename(path))).stem
        try:
            result[name] = _score(ref_features, _features(_load(path)))
        except Exception as e:
            print(f"[img_similarity] Could not process '{path}': {e}")
            result[name] = 0.0

    return result
