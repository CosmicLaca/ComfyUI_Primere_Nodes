import numpy as np
from PIL import Image


def img_hue_saturation(image: Image.Image, channels_data: dict) -> Image.Image:
    VALID_CHANNELS  = {'master', 'r', 'g', 'b'}
    CHANNEL_CENTRES = {'r': 0.0, 'g': 120.0, 'b': 240.0}

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

    total_hue      = np.zeros_like(h)
    total_sat      = np.zeros_like(s)
    total_lightness = np.zeros_like(h)
    total_vibrance  = np.zeros_like(h)

    for ch, params in channels_data.items():
        if ch not in VALID_CHANNELS:
            continue

        if ch == 'master':
            mask = np.ones(h.shape, dtype=np.float32)
        else:
            centre = CHANNEL_CENTRES[ch]
            diff   = np.abs(((h - centre + 180) % 360) - 180)
            mask   = np.where(diff <= 45, 1.0,
                     np.where(diff <= 75, 1.0 - (diff - 45) / 30.0, 0.0)).astype(np.float32)

        total_hue       += mask * params.get('hue', 0)
        total_sat       += mask * (params.get('saturation', 0) / 100.0)
        total_lightness += mask * (params.get('lightness', 0) / 100.0)
        total_vibrance  += mask * (params.get('vibrance', 0) / 100.0)

    h_new = (h + total_hue) % 360.0

    s_new = np.where(total_sat >= 0,
                     s + total_sat * (1.0 - s),
                     s + total_sat * s)
    s_new = np.clip(s_new, 0.0, 1.0)

    skin_diff       = np.abs(((h_new - 25.0 + 180) % 360) - 180)
    skin_mask       = np.where(skin_diff <= 35.0, 1.0,
                      np.where(skin_diff <= 55.0, 1.0 - (skin_diff - 35.0) / 20.0,
                               0.0)).astype(np.float32)
    skin_protection = 1.0 - skin_mask

    s_new += np.where(total_vibrance >= 0,
                      (1.0 - s_new) * skin_protection * total_vibrance,
                      s_new         * skin_protection * total_vibrance)
    s_new = np.clip(s_new, 0.0, 1.0)

    h6 = h_new / 60.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s_new)
    q  = v * (1.0 - f * s_new)
    t  = v * (1.0 - (1.0 - f) * s_new)

    rgb_sectors = np.stack([
        np.where(i==0, v, np.where(i==1, q, np.where(i==2, p,
         np.where(i==3, p, np.where(i==4, t, v))))),
        np.where(i==0, t, np.where(i==1, v, np.where(i==2, v,
         np.where(i==3, q, np.where(i==4, p, p))))),
        np.where(i==0, p, np.where(i==1, p, np.where(i==2, t,
         np.where(i==3, v, np.where(i==4, v, q))))),
    ], axis=-1)

    if np.any(total_lightness != 0):
        L3 = total_lightness[:, :, np.newaxis]
        rgb_sectors = np.where(L3 > 0,
                               rgb_sectors + L3 * (1.0 - rgb_sectors),
                               rgb_sectors + L3 * rgb_sectors)

    result = np.clip(rgb_sectors, 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")
