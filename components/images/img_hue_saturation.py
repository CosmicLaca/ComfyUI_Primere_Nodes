import numpy as np
from PIL import Image


def img_hue_saturation(
    image:           Image.Image,
    channels_data:   dict,
    channel_width:   float = 50,
    skin_protection: bool  = True,
) -> Image.Image:
    VALID_CHANNELS  = {'master', 'red', 'green', 'blue'}
    CHANNEL_CENTRES = {'red': 0.0, 'green': 120.0, 'blue': 240.0}

    if not (0 <= channel_width <= 100):
        raise ValueError(f"channel_width must be 0 … 100, got {channel_width}")

    def _is_zero(v):
        return v == 0 or v is None

    all_zero = all(
        _is_zero(p.get('hue', 0)) and
        _is_zero(p.get('saturation', 0)) and
        _is_zero(p.get('lightness', 0)) and
        _is_zero(p.get('vibrance', 0))
        for p in channels_data.values()
        if p  # skip empty dicts
    )
    if all_zero or not channels_data:
        return image.convert("RGB")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    Cmax  = np.maximum(np.maximum(R, G), B)
    Cmin  = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    with np.errstate(invalid='ignore', divide='ignore'):
        h = np.where(delta == 0, 0,
            np.where(Cmax == R, 60 * (((G - B) / delta) % 6),
            np.where(Cmax == G, 60 * (((B - R) / delta) + 2),
                                60 * (((R - G) / delta) + 4))))
    h = h % 360.0

    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(Cmax == 0, 0.0, delta / Cmax)
    v = Cmax

    t          = channel_width / 100.0
    hard_deg   = 20  + t * 55    #  20° …  75°
    feather_deg = 10 + t * 35    #  10° …  45°
    outer_deg  = hard_deg + feather_deg

    total_hue       = np.zeros_like(h)
    total_sat       = np.zeros_like(s)
    total_lightness = np.zeros_like(h)
    total_vibrance  = np.zeros_like(h)

    for ch, params in channels_data.items():
        ch = ch.strip().lower()
        if ch not in VALID_CHANNELS or not params:
            continue

        if ch == 'master':
            mask = np.ones(h.shape, dtype=np.float32)
        else:
            centre = CHANNEL_CENTRES[ch]
            diff   = np.abs(((h - centre + 180) % 360) - 180)
            mask   = np.where(
                diff <= hard_deg, 1.0,
                np.where(
                    diff <= outer_deg,
                    1.0 - (diff - hard_deg) / feather_deg,
                    0.0
                )
            ).astype(np.float32)

        total_hue       += mask * params.get('hue',        0)
        total_sat       += mask * (params.get('saturation', 0) / 100.0)
        total_lightness += mask * (params.get('lightness',  0) / 100.0)
        total_vibrance  += mask * (params.get('vibrance',   0) / 100.0)

    h_new = (h + total_hue) % 360.0
    s_new = np.where(total_sat >= 0, s + total_sat * (1.0 - s), s + total_sat * s)
    s_new = np.clip(s_new, 0.0, 1.0)

    if np.any(total_vibrance != 0):
        if skin_protection:
            skin_diff   = np.abs(((h_new - 25.0 + 180) % 360) - 180)
            skin_mask   = np.where(skin_diff <= 35.0, 1.0, np.where(skin_diff <= 55.0, 1.0 - (skin_diff - 35.0) / 20.0, 0.0))
            vib_mask = 1.0 - skin_mask   # 0 on skin, 1 elsewhere
        else:
            vib_mask = np.ones_like(h_new)

        s_new += np.where(total_vibrance >= 0, (1.0 - s_new) * vib_mask * total_vibrance, s_new * vib_mask * total_vibrance)
        s_new = np.clip(s_new, 0.0, 1.0)

    # ── HSV → RGB ─────────────────────────────────────────────────────────────
    h6 = h_new / 60.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s_new)
    q  = v * (1.0 - f * s_new)
    t_ = v * (1.0 - (1.0 - f) * s_new)

    rgb_sectors = np.stack([
        np.where(i==0, v, np.where(i==1, q, np.where(i==2, p,
         np.where(i==3, p, np.where(i==4, t_, v))))),
        np.where(i==0, t_, np.where(i==1, v, np.where(i==2, v,
         np.where(i==3, q, np.where(i==4, p, p))))),
        np.where(i==0, p, np.where(i==1, p, np.where(i==2, t_,
         np.where(i==3, v, np.where(i==4, v, q))))),
    ], axis=-1)

    # ── Apply lightness ───────────────────────────────────────────────────────
    if np.any(total_lightness != 0):
        L3 = total_lightness[:, :, np.newaxis]
        rgb_sectors = np.where(L3 > 0, rgb_sectors + L3 * (1.0 - rgb_sectors), rgb_sectors + L3 * rgb_sectors)

    result = np.clip(rgb_sectors, 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")