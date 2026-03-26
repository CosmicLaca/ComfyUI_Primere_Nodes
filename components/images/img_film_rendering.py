import numpy as np
from PIL import Image

FILM_PRESETS = {

    "fuji_astia_100_CF": {
        "desc": "Fuji Astia 100 — soft, low contrast, neutral skin tones, subtle colours",
        "iso": 100, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.00, 0.97),
        "rolloff": 0.80,
        "shadow_lift": (0.04, 0.04, 0.04),
        "hd": {
            "r": {"toe": 0.35, "gamma": 0.80, "shoulder": 0.40},
            "g": {"toe": 0.35, "gamma": 0.80, "shoulder": 0.40},
            "b": {"toe": 0.33, "gamma": 0.78, "shoulder": 0.38},
        },
    },

    "fuji_provia_100_CF": {
        "desc": "Fuji Provia 100F — standard/neutral, accurate colour, moderate contrast",
        "iso": 100, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.01, 1.02),
        "rolloff": 0.82,
        "shadow_lift": (0.02, 0.02, 0.03),
        "hd": {
            "r": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
            "g": {"toe": 0.40, "gamma": 0.92, "shoulder": 0.52},
            "b": {"toe": 0.42, "gamma": 0.92, "shoulder": 0.52},
        },
    },

    "fuji_velvia_100_CF": {
        "desc": "Fuji Velvia 100 — punchy, very saturated, high contrast, vivid greens and blues",
        "iso": 100, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.03, 1.06),
        "rolloff": 0.78,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {
            "r": {"toe": 0.55, "gamma": 1.15, "shoulder": 0.65},
            "g": {"toe": 0.58, "gamma": 1.20, "shoulder": 0.68},
            "b": {"toe": 0.60, "gamma": 1.25, "shoulder": 0.70},
        },
    },

    "fuji_superia_400_CF": {
        "desc": "Fuji Superia 400 — consumer negative, warm greens, slight grain character",
        "iso": 400, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.02, 1.03, 0.96),
        "rolloff": 0.78,
        "shadow_lift": (0.04, 0.04, 0.03),
        "hd": {
            "r": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
            "g": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
            "b": {"toe": 0.32, "gamma": 0.80, "shoulder": 0.40},
        },
    },

    "fuji_400h_CF": {
        "desc": "Fuji 400H — soft highlights, cool shadows, popular portrait film",
        "iso": 400, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (0.99, 1.01, 1.04),
        "rolloff": 0.72,
        "shadow_lift": (0.05, 0.05, 0.06),
        "hd": {
            "r": {"toe": 0.32, "gamma": 0.78, "shoulder": 0.35},
            "g": {"toe": 0.34, "gamma": 0.80, "shoulder": 0.37},
            "b": {"toe": 0.36, "gamma": 0.80, "shoulder": 0.38},
        },
    },

    "fuji_eterna_250d_CF": {
        "desc": "Fuji Eterna 250D — cinema film, daylight balanced, soft contrast, desaturated highlights",
        "iso": 250, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (0.99, 1.01, 1.03),
        "rolloff": 0.72,
        "shadow_lift": (0.05, 0.05, 0.05),
        "hd": {
            "r": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.36},
            "g": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.38},
            "b": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.38},
        },
    },

    "fuji_natura_1600_CF": {
        "desc": "Fujifilm Natura 1600 — High-speed nostalgia, magenta shadows, high grain",
        "iso": 1600, "grain_type": "organic", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.02, 0.97, 1.05),
        "rolloff": 0.65,
        "shadow_lift": (0.08, 0.04, 0.09),
        "hd": {
            "r": {"toe": 0.35, "gamma": 0.80, "shoulder": 0.40},
            "g": {"toe": 0.35, "gamma": 0.80, "shoulder": 0.40},
            "b": {"toe": 0.35, "gamma": 0.80, "shoulder": 0.40},
        },
    },

    "fuji_reala_100_CF": {
        "desc": "Fuji Reala 100 — accurate colour, natural skin, superb fine grain",
        "iso": 100, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.01, 1.01),
        "rolloff": 0.82,
        "shadow_lift": (0.02, 0.02, 0.02),
        "hd": {
            "r": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
            "g": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
            "b": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
        },
    },

    "fuji_pro_400h_soft_CF": {
        "desc": "Fuji Pro 400H Soft — pastel palette, cool shadows, portrait-friendly skin response",
        "iso": 400, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (0.99, 1.01, 1.05),
        "rolloff": 0.70,
        "shadow_lift": (0.06, 0.06, 0.07),
        "hd": {
            "r": {"toe": 0.30, "gamma": 0.76, "shoulder": 0.34},
            "g": {"toe": 0.32, "gamma": 0.78, "shoulder": 0.36},
            "b": {"toe": 0.34, "gamma": 0.80, "shoulder": 0.38},
        },
    },

    "kodak_vision3_500t_CF": {
        "desc": "Kodak Vision3 500T — cinema negative, tungsten balanced, warm shadows, teal highlights",
        "iso": 500, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.03, 0.99, 0.96),
        "rolloff": 0.74,
        "shadow_lift": (0.05, 0.04, 0.06),
        "hd": {
            "r": {"toe": 0.38, "gamma": 0.85, "shoulder": 0.45},
            "g": {"toe": 0.36, "gamma": 0.83, "shoulder": 0.43},
            "b": {"toe": 0.40, "gamma": 0.83, "shoulder": 0.44},
        },
    },

    "kodak_kodachrome_64_CF": {
        "desc": "Kodak Kodachrome 64 — iconic warm reds, deep blues, high contrast, rich shadows",
        "iso": 64, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.06, 0.98, 0.94),
        "rolloff": 0.80,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {
            "r": {"toe": 0.60, "gamma": 1.10, "shoulder": 0.72},
            "g": {"toe": 0.45, "gamma": 0.95, "shoulder": 0.58},
            "b": {"toe": 0.42, "gamma": 0.90, "shoulder": 0.55},
        },
    },

    "kodak_ektachrome_100vs_CF": {
        "desc": "Kodak Ektachrome 100VS — very saturated, cool blues, strong greens",
        "iso": 100, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (0.98, 1.02, 1.05),
        "rolloff": 0.79,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {
            "r": {"toe": 0.50, "gamma": 1.05, "shoulder": 0.62},
            "g": {"toe": 0.55, "gamma": 1.10, "shoulder": 0.68},
            "b": {"toe": 0.58, "gamma": 1.15, "shoulder": 0.72},
        },
    },

    "kodak_portra_160_CF": {
        "desc": "Kodak Portra 160 — warm skin tones, soft highlights, low contrast, fine grain",
        "iso": 160, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.04, 1.01, 0.95),
        "rolloff": 0.70,
        "shadow_lift": (0.04, 0.03, 0.03),
        "hd": {
            "r": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.38},
            "g": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.36},
            "b": {"toe": 0.30, "gamma": 0.76, "shoulder": 0.33},
        },
    },

    "kodak_portra_400_CF": {
        "desc": "Kodak Portra 400 — Warm skin tones, soft highlights, signature yellow-red bias",
        "iso": 400, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.06, 0.98, 0.92),
        "rolloff": 0.72,
        "shadow_lift": (0.05, 0.04, 0.04),
        "hd": {
            "r": {"toe": 0.32, "gamma": 0.85, "shoulder": 0.38},
            "g": {"toe": 0.35, "gamma": 0.88, "shoulder": 0.40},
            "b": {"toe": 0.40, "gamma": 0.95, "shoulder": 0.45},
        },
    },

    "kodak_portra_800_CF": {
        "desc": "Kodak Portra 800 — warm skin rendering, low-light latitude, soft highlight rolloff",
        "iso": 800, "grain_type": "organic", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.05, 1.00, 0.93),
        "rolloff": 0.70,
        "shadow_lift": (0.06, 0.05, 0.05),
        "hd": {
            "r": {"toe": 0.34, "gamma": 0.84, "shoulder": 0.40},
            "g": {"toe": 0.36, "gamma": 0.86, "shoulder": 0.42},
            "b": {"toe": 0.40, "gamma": 0.92, "shoulder": 0.46},
        },
    },

    "kodak_gold_200_CF": {
        "desc": "Kodak Gold 200 — consumer film, warm golden tone, boosted yellows and reds",
        "iso": 200, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.05, 1.02, 0.92),
        "rolloff": 0.76,
        "shadow_lift": (0.03, 0.03, 0.02),
        "hd": {
            "r": {"toe": 0.45, "gamma": 0.95, "shoulder": 0.55},
            "g": {"toe": 0.43, "gamma": 0.92, "shoulder": 0.52},
            "b": {"toe": 0.32, "gamma": 0.78, "shoulder": 0.38},
        },
    },

    "kodak_ultramax_400_CF": {
        "desc": "Kodak Ultramax 400 — vivid warm colours, punchy contrast, popular street film",
        "iso": 400, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.04, 1.01, 0.93),
        "rolloff": 0.77,
        "shadow_lift": (0.02, 0.02, 0.01),
        "hd": {
            "r": {"toe": 0.50, "gamma": 1.00, "shoulder": 0.62},
            "g": {"toe": 0.48, "gamma": 0.98, "shoulder": 0.60},
            "b": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.44},
        },
    },

    "kodak_tri_x_400_CF": {
        "desc": "Kodak Tri-X 400 — B&W look in colour, strong contrast, warm shadow tint",
        "iso": 400, "grain_type": "organic", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.02, 1.00, 0.97),
        "rolloff": 0.80,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {
            "r": {"toe": 0.58, "gamma": 1.10, "shoulder": 0.70},
            "g": {"toe": 0.58, "gamma": 1.10, "shoulder": 0.70},
            "b": {"toe": 0.55, "gamma": 1.08, "shoulder": 0.68},
        },
    },

    "kodak_ektar_100_CF": {
        "desc": "Kodak Ektar 100 — finest grain colour negative, ultra-vivid, strong reds",
        "iso": 100, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.06, 1.00, 0.93),
        "rolloff": 0.79,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {
            "r": {"toe": 0.55, "gamma": 1.10, "shoulder": 0.68},
            "g": {"toe": 0.48, "gamma": 1.00, "shoulder": 0.60},
            "b": {"toe": 0.38, "gamma": 0.85, "shoulder": 0.48},
        },
    },

    "kodak_colorplus_200_CF": {
        "desc": "Kodak ColorPlus 200 — entry-level warmth, slightly flat, classic holiday snap",
        "iso": 200, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.04, 1.02, 0.94),
        "rolloff": 0.76,
        "shadow_lift": (0.03, 0.03, 0.02),
        "hd": {
            "r": {"toe": 0.42, "gamma": 0.90, "shoulder": 0.50},
            "g": {"toe": 0.40, "gamma": 0.88, "shoulder": 0.48},
            "b": {"toe": 0.30, "gamma": 0.76, "shoulder": 0.36},
        },
    },

    "agfa_vista_200_CF": {
        "desc": "Agfa Vista 200 — cool shadows, slight blue-green tint, soft contrast",
        "iso": 200, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (0.97, 1.00, 1.04),
        "rolloff": 0.79,
        "shadow_lift": (0.04, 0.04, 0.05),
        "hd": {
            "r": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.42},
            "g": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.44},
            "b": {"toe": 0.37, "gamma": 0.84, "shoulder": 0.46},
        },
    },

    "lomography_lomo_100_CF": {
        "desc": "Lomography 100 — high contrast, cross-process look, boosted saturation",
        "iso": 100, "grain_type": "organic", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.05, 0.97, 1.02),
        "rolloff": 0.75,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {
            "r": {"toe": 0.62, "gamma": 1.20, "shoulder": 0.75},
            "g": {"toe": 0.52, "gamma": 1.05, "shoulder": 0.65},
            "b": {"toe": 0.58, "gamma": 1.15, "shoulder": 0.70},
        },
    },

    "ilford_xp2_400_CF": {
        "desc": "Ilford XP2 Super 400 — chromogenic B&W, neutral, clean shadows",
        "iso": 400, "grain_type": "gaussian", "grain_color": "monochrome",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.83,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {
            "r": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
            "g": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
            "b": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
        },
    },

    "cinestill_400d_CF": {
        "desc": "CineStill 400D — daylight-balanced cinema look, smooth rolloff, moderate halation",
        "iso": 400, "grain_type": "organic", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (1.01, 1.00, 0.98),
        "rolloff": 0.73,
        "shadow_lift": (0.04, 0.04, 0.04),
        "hd": {
            "r": {"toe": 0.36, "gamma": 0.84, "shoulder": 0.42},
            "g": {"toe": 0.36, "gamma": 0.84, "shoulder": 0.42},
            "b": {"toe": 0.34, "gamma": 0.82, "shoulder": 0.40},
        },
    },

    "cinestill_800t_CF": {
        "desc": "CineStill 800T (Cinema) — Tungsten balanced, cool shadows, halation-heavy look",
        "iso": 800, "grain_type": "organic", "grain_color": "color",
        "bw": False, "type": "CF",
        "bias": (0.90, 0.96, 1.15),
        "rolloff": 0.75,
        "shadow_lift": (0.02, 0.05, 0.08),
        "hd": {
            "r": {"toe": 0.45, "gamma": 1.10, "shoulder": 0.50},
            "g": {"toe": 0.40, "gamma": 0.95, "shoulder": 0.45},
            "b": {"toe": 0.35, "gamma": 0.85, "shoulder": 0.40},
        },
    },

    "ilford_hp5_400_BWF": {
        "desc": "Ilford HP5 Plus 400 — classic panchromatic, neutral grey, forgiving latitude",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.82,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.40, "gamma": 0.92, "shoulder": 0.52},
               "g": {"toe": 0.40, "gamma": 0.92, "shoulder": 0.52},
               "b": {"toe": 0.40, "gamma": 0.92, "shoulder": 0.52}},
    },

    "ilford_delta_100_BWF": {
        "desc": "Ilford Delta 100 — fine grain, cool neutral tone, excellent shadow detail",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.28, 0.60, 0.12),
        "tint": (0.98, 0.99, 1.01),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.84,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.38, "gamma": 0.90, "shoulder": 0.50},
               "g": {"toe": 0.38, "gamma": 0.90, "shoulder": 0.50},
               "b": {"toe": 0.38, "gamma": 0.90, "shoulder": 0.50}},
    },

    "ilford_delta_3200_BWF": {
        "desc": "Ilford Delta 3200 — very high ISO, lifted shadows, compressed highlights",
        "iso": 3200, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.59, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.65,
        "shadow_lift": (0.08, 0.08, 0.08),
        "hd": {"r": {"toe": 0.30, "gamma": 0.75, "shoulder": 0.35},
               "g": {"toe": 0.30, "gamma": 0.75, "shoulder": 0.35},
               "b": {"toe": 0.30, "gamma": 0.75, "shoulder": 0.35}},
    },

    "ilford_sfx_200_BWF": {
        "desc": "Ilford SFX 200 — extended red sensitivity, pseudo-infrared, dark skies, bright foliage",
        "iso": 200, "grain_type": "gaussian", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.60, 0.30, 0.10),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.80,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {"r": {"toe": 0.50, "gamma": 1.05, "shoulder": 0.65},
               "g": {"toe": 0.50, "gamma": 1.05, "shoulder": 0.65},
               "b": {"toe": 0.50, "gamma": 1.05, "shoulder": 0.65}},
    },

    "ilford_fp4_125_BWF": {
        "desc": "Ilford FP4 Plus 125 — classic medium contrast, smooth tonal transitions, fine grain",
        "iso": 125, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.58, 0.12),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.83,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
               "g": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
               "b": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50}},
    },

    "kentmere_400_BWF": {
        "desc": "Kentmere 400 — budget classic, punchier midtones, documentary street character",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.31, 0.58, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.79,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.50, "gamma": 1.03, "shoulder": 0.62},
               "g": {"toe": 0.50, "gamma": 1.03, "shoulder": 0.62},
               "b": {"toe": 0.50, "gamma": 1.03, "shoulder": 0.62}},
    },

    "kodak_technical_pan_BWF": {
        "desc": "Kodak Technical Pan — ultra-high contrast, extremely fine grain, scientific/forensic",
        "iso": 25, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.25, 0.65, 0.10),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.84,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {"r": {"toe": 0.70, "gamma": 1.30, "shoulder": 0.80},
               "g": {"toe": 0.70, "gamma": 1.30, "shoulder": 0.80},
               "b": {"toe": 0.70, "gamma": 1.30, "shoulder": 0.80}},
    },

    "kodak_tmax_100_BWF": {
        "desc": "Kodak T-Max 100 — ultra-fine grain, high contrast, deep clean blacks",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.27, 0.62, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.82,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {"r": {"toe": 0.58, "gamma": 1.15, "shoulder": 0.72},
               "g": {"toe": 0.58, "gamma": 1.15, "shoulder": 0.72},
               "b": {"toe": 0.58, "gamma": 1.15, "shoulder": 0.72}},
    },

    "kodak_tmax_400_BWF": {
        "desc": "Kodak T-Max 400 — fine grain for ISO 400, excellent tonal range",
        "iso": 400, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.28, 0.61, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.81,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.45, "gamma": 0.98, "shoulder": 0.58},
               "g": {"toe": 0.45, "gamma": 0.98, "shoulder": 0.58},
               "b": {"toe": 0.45, "gamma": 0.98, "shoulder": 0.58}},
    },

    "kodak_tri_x_400_BWF": {
        "desc": "Kodak Tri-X 400 B&W — iconic, punchy, deep blacks, photojournalism classic",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.32, 0.58, 0.10),
        "tint": (1.01, 1.00, 0.99),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.80,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {"r": {"toe": 0.60, "gamma": 1.12, "shoulder": 0.72},
               "g": {"toe": 0.60, "gamma": 1.12, "shoulder": 0.72},
               "b": {"toe": 0.60, "gamma": 1.12, "shoulder": 0.72}},
    },

    "agfa_apx_100_BWF": {
        "desc": "Agfa APX 100 — smooth midtones, slightly warm neutral, soft shadow gradation",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.59, 0.11),
        "tint": (1.01, 1.00, 0.99),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.80,
        "shadow_lift": (0.02, 0.02, 0.02),
        "hd": {"r": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
               "g": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48},
               "b": {"toe": 0.38, "gamma": 0.88, "shoulder": 0.48}},
    },

    "agfa_apx_400_BWF": {
        "desc": "Agfa APX 400 — medium grain, contrasty midtones, green-sensitive",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.29, 0.61, 0.10),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.79,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.45, "gamma": 0.98, "shoulder": 0.58},
               "g": {"toe": 0.45, "gamma": 0.98, "shoulder": 0.58},
               "b": {"toe": 0.45, "gamma": 0.98, "shoulder": 0.58}},
    },

    "rollei_rpx_400_BWF": {
        "desc": "Rollei RPX 400 — very deep blacks, punchy street photography look",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.59, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.80,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {"r": {"toe": 0.62, "gamma": 1.15, "shoulder": 0.75},
               "g": {"toe": 0.62, "gamma": 1.15, "shoulder": 0.75},
               "b": {"toe": 0.62, "gamma": 1.15, "shoulder": 0.75}},
    },

    "rollei_retro_80s_BWF": {
        "desc": "Rollei Retro 80S — high acutance look, deep skies, crisp edges and clean highlights",
        "iso": 80, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.36, 0.54, 0.10),
        "tint": (0.99, 1.00, 1.01),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.82,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {"r": {"toe": 0.56, "gamma": 1.10, "shoulder": 0.70},
               "g": {"toe": 0.56, "gamma": 1.10, "shoulder": 0.70},
               "b": {"toe": 0.56, "gamma": 1.10, "shoulder": 0.70}},
    },

    "fomapan_100_BWF": {
        "desc": "Fomapan 100 — orthochromatic character, blue-sensitive, soft contrast, vintage look",
        "iso": 100, "grain_type": "gaussian", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.22, 0.55, 0.23),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.80,
        "shadow_lift": (0.03, 0.03, 0.03),
        "hd": {"r": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.45},
               "g": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.45},
               "b": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.45}},
    },

    "selenium_tone_BWF": {
        "desc": "Selenium toning — cool blue-purple shadow tone, archival darkroom process",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (0.96, 0.97, 1.04),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.82,
        "shadow_lift": (0.01, 0.01, 0.01),
        "hd": {"r": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
               "g": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50},
               "b": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.50}},
    },

    "sepia_tone_BWF": {
        "desc": "Sepia toning — warm brown throughout, classic Victorian / vintage look",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.08, 1.00, 0.82),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.78,
        "shadow_lift": (0.04, 0.03, 0.02),
        "hd": {"r": {"toe": 0.38, "gamma": 0.86, "shoulder": 0.48},
               "g": {"toe": 0.38, "gamma": 0.86, "shoulder": 0.48},
               "b": {"toe": 0.38, "gamma": 0.86, "shoulder": 0.48}},
    },

    "gold_tone_BWF": {
        "desc": "Gold toning — warm golden highlights, cooler shadows, elegant darkroom effect",
        "iso": 400, "grain_type": "organic", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.05, 1.02, 0.88),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.80,
        "shadow_lift": (0.02, 0.02, 0.02),
        "hd": {"r": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.52},
               "g": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.52},
               "b": {"toe": 0.40, "gamma": 0.90, "shoulder": 0.52}},
    },

    "cyanotype_BWF": {
        "desc": "Cyanotype — deep cyan-blue alternative process print look",
        "iso": 400, "grain_type": "gaussian", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (0.78, 0.90, 1.15),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.81,
        "shadow_lift": (0.02, 0.02, 0.02),
        "hd": {"r": {"toe": 0.38, "gamma": 0.86, "shoulder": 0.48},
               "g": {"toe": 0.38, "gamma": 0.86, "shoulder": 0.48},
               "b": {"toe": 0.38, "gamma": 0.86, "shoulder": 0.48}},
    },

    "platinum_palladium_BWF": {
        "desc": "Platinum/Palladium print — long tonal scale, subtle warm neutral, rich shadow detail",
        "iso": 400, "grain_type": "fine", "grain_color": "monochrome",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.02, 1.01, 0.99),
        "bias": (1.00, 1.00, 1.00),
        "rolloff": 0.76,
        "shadow_lift": (0.03, 0.03, 0.03),
        "hd": {"r": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.42},
               "g": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.42},
               "b": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.42}},
    },

    "canon_5d_mark2_CCD": {
        "desc": "Canon 5D Mark II — warm romantic colour, gentle highlight rolloff, smooth skin tones",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.04, 1.00, 0.97),
        "matrix": [[1.06,-0.04,-0.02],[-0.03,1.04,-0.01],[-0.02,-0.06,1.08]],
        "shadow_lift": (0.008, 0.006, 0.005),
        "highlight_rolloff": 0.35,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.50),(0.75,0.75),(1.0,0.96)],
        },
    },

    "canon_5d_mark1_CCD": {
        "desc": "Canon 5D Mark I — original full-frame CCD, warm character, pleasing colour",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.05, 1.00, 0.95),
        "matrix": [[1.08,-0.05,-0.03],[-0.03,1.05,-0.02],[-0.03,-0.07,1.10]],
        "shadow_lift": (0.012, 0.009, 0.007),
        "highlight_rolloff": 0.45,
        "curves": {
            "r": [(0.0,0.02),(0.25,0.27),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.76),(1.0,0.96)],
            "b": [(0.0,0.01),(0.25,0.24),(0.5,0.50),(0.75,0.74),(1.0,0.95)],
        },
    },

    "canon_1dx_CCD": {
        "desc": "Canon 1Dx — professional sports/press, accurate neutral colour, punchy contrast",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.02, 1.00, 0.99),
        "matrix": [[1.04,-0.02,-0.02],[-0.02,1.03,-0.01],[-0.01,-0.04,1.05]],
        "shadow_lift": (0.006, 0.005, 0.005),
        "highlight_rolloff": 0.25,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "b": [(0.0,0.00),(0.25,0.24),(0.5,0.50),(0.75,0.77),(1.0,0.98)],
        },
    },

    "sony_a7iii_CCD": {
        "desc": "Sony A7 III — neutral accurate colour, cool shadow character, high dynamic range",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.00, 1.01, 1.02),
        "matrix": [[1.02,-0.01,-0.01],[-0.01,1.03,0.00],[0.00,-0.02,1.02]],
        "shadow_lift": (0.004, 0.004, 0.006),
        "highlight_rolloff": 0.20,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,1.00)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
        },
    },

    "sony_a7rii_CCD": {
        "desc": "Sony A7R II — very high resolution, neutral-cool, extremely detailed",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (0.99, 1.01, 1.03),
        "matrix": [[1.01,-0.01,0.00],[-0.01,1.03,0.00],[0.00,-0.02,1.02]],
        "shadow_lift": (0.003, 0.003, 0.005),
        "highlight_rolloff": 0.18,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,1.00)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.78),(1.0,1.00)],
        },
    },

    "nikon_d70_ccd_CCD": {
        "desc": "Nikon D70 CCD — classic early DSLR rendering, punchy colour and crisp micro-contrast",
        "iso": 200, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CCD",
        "bias": (1.03, 1.01, 0.98),
        "matrix": [[1.08, -0.04, -0.03], [-0.02, 1.06, -0.04], [-0.01, -0.03, 1.05]],
        "shadow_lift": (0.01, 0.01, 0.01),
        "highlight_rolloff": 0.28,
        "curves": {
            "r": [(0, 0.00), (0.25, 0.28), (0.5, 0.54), (0.75, 0.79), (1, 0.99)],
            "g": [(0, 0.00), (0.25, 0.27), (0.5, 0.53), (0.75, 0.78), (1, 0.99)],
            "b": [(0, 0.00), (0.25, 0.24), (0.5, 0.49), (0.75, 0.74), (1, 0.96)],
        },
    },

    "nikon_d800_CCD": {
        "desc": "Nikon D800 — neutral accurate, slightly cool shadows, excellent detail",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.00, 1.01, 1.01),
        "matrix": [[1.03,-0.02,-0.01],[-0.01,1.03,0.00],[0.00,-0.03,1.03]],
        "shadow_lift": (0.005, 0.005, 0.007),
        "highlight_rolloff": 0.22,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.77),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
        },
    },

    "nikon_d3_CCD": {
        "desc": "Nikon D3 — warm classic DSLR rendering, photojournalism standard",
        "iso": 200, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.03, 1.01, 0.97),
        "matrix": [[1.05,-0.03,-0.02],[-0.02,1.04,-0.01],[-0.01,-0.05,1.06]],
        "shadow_lift": (0.008, 0.007, 0.006),
        "highlight_rolloff": 0.30,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.50),(0.75,0.75),(1.0,0.96)],
        },
    },

    "fuji_xt3_CCD": {
        "desc": "Fuji X-T3 — film-simulation-inspired colour science, warm midtones, X-Trans",
        "iso": 160, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.03, 1.01, 0.98),
        "matrix": [[1.05,-0.03,-0.02],[-0.02,1.05,-0.02],[-0.01,-0.04,1.05]],
        "shadow_lift": (0.007, 0.006, 0.005),
        "highlight_rolloff": 0.40,
        "curves": {
            "r": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.75),(1.0,0.96)],
        },
    },

    "fuji_gfx_CCD": {
        "desc": "Fuji GFX 100 — medium format digital, very neutral, exceptional tonal gradation",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.01, 1.01, 1.00),
        "matrix": [[1.02,-0.01,-0.01],[-0.01,1.03,-0.01],[0.00,-0.02,1.02]],
        "shadow_lift": (0.005, 0.005, 0.004),
        "highlight_rolloff": 0.28,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
        },
    },

    "leica_m9_CCD": {
        "desc": "Leica M9 CCD — Iconic Kodak KAF-18500 sensor, deep reds, high micro-contrast",
        "iso": 160, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.06, 1.00, 0.98),
        "matrix": [[1.18, -0.12, -0.06], [-0.05, 1.10, -0.05], [-0.02, -0.10, 1.12]],
        "shadow_lift": (0.005, 0.005, 0.008),
        "highlight_rolloff": 0.45,
        "curves": {
            "r": [(0.0, 0.01), (0.25, 0.27), (0.5, 0.53), (0.75, 0.78), (1.0, 0.97)],
            "g": [(0.0, 0.01), (0.25, 0.26), (0.5, 0.52), (0.75, 0.77), (1.0, 0.97)],
            "b": [(0.0, 0.02), (0.25, 0.25), (0.5, 0.50), (0.75, 0.74), (1.0, 0.95)],
        },
    },

    "leica_m11_CCD": {
        "desc": "Leica M11 CMOS — modern Leica, very neutral and clinical, faithful colour science",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.01, 1.01, 1.00),
        "matrix": [[1.02,-0.01,-0.01],[-0.01,1.02,0.00],[0.00,-0.02,1.02]],
        "shadow_lift": (0.004, 0.004, 0.004),
        "highlight_rolloff": 0.22,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
        },
    },

    "hasselblad_x2d_CCD": {
        "desc": "Hasselblad X2D — 100MP medium format, clinical precision, very wide tonal range",
        "iso": 100, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.00, 1.01, 1.01),
        "matrix": [[1.01,0.00,-0.01],[0.00,1.02,0.00],[0.00,-0.01,1.01]],
        "shadow_lift": (0.003, 0.003, 0.003),
        "highlight_rolloff": 0.15,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "b": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
        },
    },

    "olympus_omd_CCD": {
        "desc": "Olympus OM-D E-M1 — punchy vivid colour, slightly cool, contrasty rendering",
        "iso": 200, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "CCD",
        "bias": (1.01, 1.02, 1.02),
        "matrix": [[1.03,-0.01,-0.02],[-0.01,1.04,-0.01],[-0.01,-0.02,1.03]],
        "shadow_lift": (0.005, 0.005, 0.006),
        "highlight_rolloff": 0.20,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.52),(0.75,0.78),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.79),(1.0,1.00)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.78),(1.0,0.99)],
        },
    },

    "epson_r_d1_ccd_CCD": {
        "desc": "Epson R-D1 CCD — rangefinder CCD signature, gentle rolloff and warm analogue character",
        "iso": 160, "grain_type": "fine", "grain_color": "color",
        "bw": False, "type": "CCD",
        "bias": (1.02, 1.00, 0.98),
        "matrix": [[1.06, -0.03, -0.03], [-0.01, 1.04, -0.03], [-0.01, -0.02, 1.04]],
        "shadow_lift": (0.02, 0.02, 0.02),
        "highlight_rolloff": 0.32,
        "curves": {
            "r": [(0, 0.01), (0.25, 0.27), (0.5, 0.52), (0.75, 0.77), (1, 0.98)],
            "g": [(0, 0.01), (0.25, 0.27), (0.5, 0.52), (0.75, 0.76), (1, 0.98)],
            "b": [(0, 0.01), (0.25, 0.25), (0.5, 0.50), (0.75, 0.74), (1, 0.96)],
        },
    },

    "iphone4_MOB": {
        "desc": "iPhone 4 MOB — Small sensor look, harsh digital clipping, high saturation",
        "iso": 80, "grain_type": "gaussian", "grain_color": "color",
        "bw": False, "type": "MOB",
        "bias": (1.02, 1.00, 1.05),
        "matrix": [[1.20,-0.10,-0.10],[-0.10,1.20,-0.10],[-0.10,-0.10,1.20]],
        "shadow_lift": (0.01, 0.01, 0.01),
        "highlight_rolloff": 0.05,
        "curves": {
            "r": [(0,0),(1,1)], "g": [(0,0),(1,1)], "b": [(0,0),(1,1)]
        },
    },

    "iphone6s_MOB": {
        "desc": "iPhone 6s MOB — classic Apple mobile colour, slightly warm skin and constrained DR look",
        "iso": 64, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (1.03, 1.01, 0.96),
        "matrix": [[1.07, -0.04, -0.03], [-0.02, 1.06, -0.04], [-0.01, -0.03, 1.05]],
        "shadow_lift": (0.05, 0.05, 0.06),
        "highlight_rolloff": 0.28,
        "curves": {
            "r": [(0, 0.02), (0.25, 0.28), (0.5, 0.54), (0.75, 0.79), (1, 0.97)],
            "g": [(0, 0.02), (0.25, 0.28), (0.5, 0.54), (0.75, 0.79), (1, 0.97)],
            "b": [(0, 0.02), (0.25, 0.26), (0.5, 0.50), (0.75, 0.74), (1, 0.95)],
        },
    },

    "iphone12_MOB": {
        "desc": "iPhone 12 MOB — Smart HDR, deep fusion, punchy contrast, warm skin bias",
        "iso": 32, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (1.03, 1.01, 0.97),
        "matrix": [[1.06,-0.03,-0.03],[-0.02,1.06,-0.04],[-0.01,-0.03,1.04]],
        "shadow_lift": (0.04, 0.04, 0.05),
        "highlight_rolloff": 0.35,
        "curves": {
            "r": [(0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.78),(1,0.97)],
            "g": [(0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1,0.97)],
            "b": [(0,0.02),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1,0.96)],
        },
    },

    "iphone15_pro_MOB": {
        "desc": "iPhone 15 Pro MOB — 48MP main, photogenic realism, cinematic colour science",
        "iso": 16, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (1.02, 1.01, 0.99),
        "matrix": [[1.04,-0.02,-0.02],[-0.01,1.04,-0.03],[0.00,-0.02,1.02]],
        "shadow_lift": (0.03, 0.03, 0.04),
        "highlight_rolloff": 0.40,
        "curves": {
            "r": [(0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.77),(1,0.98)],
            "g": [(0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1,0.98)],
            "b": [(0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1,0.97)],
        },
    },

    "samsung_s24_ultra_MOB": {
        "desc": "Samsung S24 Ultra MOB — 200MP, vivid punchy colour, boosted saturation",
        "iso": 25, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (1.04, 1.02, 0.96),
        "matrix": [[1.08,-0.04,-0.04],[-0.02,1.06,-0.04],[-0.01,-0.04,1.05]],
        "shadow_lift": (0.02, 0.02, 0.03),
        "highlight_rolloff": 0.25,
        "curves": {
            "r": [(0,0.00),(0.25,0.26),(0.5,0.53),(0.75,0.80),(1,1.00)],
            "g": [(0,0.00),(0.25,0.26),(0.5,0.53),(0.75,0.80),(1,1.00)],
            "b": [(0,0.01),(0.25,0.25),(0.5,0.50),(0.75,0.76),(1,0.97)],
        },
    },

    "google_pixel8_pro_MOB": {
        "desc": "Pixel 8 Pro MOB — Tensor G3, natural skin tones, excellent dynamic range",
        "iso": 20, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (1.00, 1.01, 1.01),
        "matrix": [[1.03,-0.01,-0.02],[0.00,1.04,-0.04],[0.00,-0.02,1.02]],
        "shadow_lift": (0.05, 0.05, 0.06),
        "highlight_rolloff": 0.38,
        "curves": {
            "r": [(0,0.02),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1,0.97)],
            "g": [(0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1,0.97)],
            "b": [(0,0.02),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1,0.97)],
        },
    },

    "google_pixel2_MOB": {
        "desc": "Google Pixel 2 MOB — HDR+ era look, neutral white balance and lifted detail in shadows",
        "iso": 50, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (0.99, 1.01, 1.02),
        "matrix": [[1.04, -0.02, -0.02], [0.00, 1.05, -0.05], [0.00, -0.02, 1.03]],
        "shadow_lift": (0.06, 0.06, 0.07),
        "highlight_rolloff": 0.34,
        "curves": {
            "r": [(0, 0.02), (0.25, 0.26), (0.5, 0.51), (0.75, 0.76), (1, 0.97)],
            "g": [(0, 0.02), (0.25, 0.27), (0.5, 0.52), (0.75, 0.77), (1, 0.97)],
            "b": [(0, 0.02), (0.25, 0.27), (0.5, 0.52), (0.75, 0.77), (1, 0.98)],
        },
    },

    "google_pixel3_MOB": {
        "desc": "Pixel 3 MOB — Computational HDR look, lifted shadows, cool clean bias",
        "iso": 50, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (0.98, 1.00, 1.02),
        "matrix": [[1.05, 0, 0], [0, 1.05, 0], [0, 0, 1.05]],
        "shadow_lift": (0.06, 0.06, 0.07),
        "highlight_rolloff": 0.30,
        "curves": {
            "r": [(0, 0.1), (0.5, 0.5), (1, 0.9)],
            "g": [(0, 0.1), (0.5, 0.5), (1, 0.9)],
            "b": [(0, 0.1), (0.5, 0.5), (1, 0.9)]
        },
    },

    "sony_zv1_MOB": {
        "desc": "Sony ZX1 MOB — 1-inch sensor vlog camera, cinematic colour, wide DR",
        "iso": 125, "grain_type": "fine", "grain_color": "monochrome",
        "bw": False, "type": "MOB",
        "bias": (1.01, 1.01, 1.00),
        "matrix": [[1.03,-0.01,-0.02],[-0.01,1.04,-0.03],[0.00,-0.02,1.02]],
        "shadow_lift": (0.006, 0.005, 0.006),
        "highlight_rolloff": 0.32,
        "curves": {
            "r": [(0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1,0.98)],
            "g": [(0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1,0.98)],
            "b": [(0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1,0.97)],
        },
    },
}

def img_film_rendering(
    image:     Image.Image,
    rendering: str   = "kodak_vision3_500t_CF",
    intensity: float = 100,
    add_grain: bool  = False,
    add_halation: bool = False,
    expiration_years: float = 0.0
) -> Image.Image:
    if intensity == 0:
        return image.convert("RGB")

    preset = FILM_PRESETS.get(rendering)
    if not preset:
        valid = ", ".join(sorted(FILM_PRESETS.keys()))
        raise ValueError(f"Unknown rendering '{rendering}'. Valid: {valid}")

    if not (0 <= intensity <= 200):
        raise ValueError(f"intensity must be 0-200, got {intensity}")

    preset_base = FILM_PRESETS[rendering]
    img         = image.convert("RGB")
    arr         = np.array(img, dtype=np.float32) / 255.0
    orig        = arr.copy()
    H, W        = arr.shape[:2]

    analysis = _analyse_image(arr)
    preset   = _adapt_preset(preset_base, analysis)

    active_type = preset.get("type", preset_base.get("type", "CF"))
    if active_type == "BWF":
        arr_out = _apply_bw(arr, preset)
    elif active_type in ["CCD", "MOB"]:
        arr_out = _apply_sensor(arr, preset)
    else:
        arr_out = _apply_colour(arr, preset)

    if active_type == "BWF" and intensity > 100:
        result = arr_out.copy()
        push = (intensity - 100) / 100.0
        grey = result[..., 0]
        contrast_factor = 1.0 + push * 0.8
        grey = np.clip((grey - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
        shadow_power = 1.0 + push * 0.6
        grey = np.power(np.clip(grey, 0.0, 1.0), shadow_power)
        hi_push = 0.85 - push * 0.10
        hi_mask = np.clip((grey - hi_push) / (1.0 - hi_push + 1e-6), 0.0, 1.0)
        grey = np.where(grey > hi_push,
                        hi_push + (1.0 - hi_push) * (1.0 - (1.0 - hi_mask) ** 2),
                        grey)
        grey = np.clip(grey, 0.0, 1.0)
        tint = np.array(preset.get("tint", (1.0, 1.0, 1.0)), dtype=np.float32)
        result = np.stack([grey * tint[0], grey * tint[1], grey * tint[2]], axis=-1)
        result = np.clip(result, 0.0, 1.0)
    else:
        blend  = intensity / 100.0
        result = orig + blend * (arr_out - orig)
        result = np.clip(result, 0.0, 1.0)

    if expiration_years > 0 and active_type not in ["CCD", "MOB"]:
        result = _apply_expiration(result, expiration_years)

    if add_halation:
        result = _apply_halation(result, preset, H, W)

    if add_grain:
        result = _apply_grain(result, preset, H, W)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")

def list_film_presets() -> dict:
    return {k: v["desc"] for k, v in FILM_PRESETS.items()}

def list_presets_by_type() -> dict:
    result = {"CF": [], "BWF": [], "CCD": [], "MOB": []}
    for k, v in FILM_PRESETS.items():
        t = v["type"]
        if t not in result:
            result[t] = []
        result[t].append(k)
    return result

def _analyse_image(arr: np.ndarray) -> dict:
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    lum_flat = lum.ravel()

    median_lum         = float(np.median(lum_flat))
    lum_std            = float(lum_flat.std())
    shadow_fraction    = float((lum_flat < 0.2).mean())
    highlight_fraction = float((lum_flat > 0.8).mean())
    midtone_fraction   = float(((lum_flat >= 0.2) & (lum_flat <= 0.8)).mean())
    p05, p95           = float(np.percentile(lum_flat, 5)), float(np.percentile(lum_flat, 95))
    dynamic_range      = p95 - p05

    ch_means = np.array([arr[..., c].mean() for c in range(3)], dtype=np.float32)
    grey_mean = float(ch_means.mean())
    dominant_cast = (ch_means - grey_mean) / (grey_mean + 1e-6)

    ch_max = arr.max(axis=-1)
    ch_min = arr.min(axis=-1)
    mean_saturation = float(((ch_max - ch_min) / (ch_max + 1e-6)).mean())

    return {
        "median_lum":         median_lum,
        "lum_std":            lum_std,
        "shadow_fraction":    shadow_fraction,
        "highlight_fraction": highlight_fraction,
        "midtone_fraction":   midtone_fraction,
        "mean_saturation":    mean_saturation,
        "dominant_cast":      dominant_cast,
        "dynamic_range":      dynamic_range,
        "is_lowkey":          median_lum < 0.42,
        "is_highkey":         median_lum > 0.58,
        "is_flat":            lum_std < 0.12,
        "is_desaturated":     mean_saturation < 0.08,
    }

def _adapt_preset(preset: dict, analysis: dict) -> dict:
    import copy
    p = copy.deepcopy(preset)

    median = analysis["median_lum"]
    std = analysis["lum_std"]
    hi_frac = analysis["highlight_fraction"]
    sh_frac = analysis["shadow_fraction"]
    cast = analysis["dominant_cast"]
    desat = analysis["is_desaturated"]
    flat = analysis["is_flat"]

    if "hd" in p:
        if flat:
            gamma_scale = 1.10
        elif std > 0.28:
            gamma_scale = 0.93
        else:
            gamma_scale = 1.0

        if gamma_scale != 1.0:
            for ch_key in p["hd"]:
                p["hd"][ch_key]["gamma"] = float(
                    np.clip(p["hd"][ch_key]["gamma"] * gamma_scale, 0.5, 1.8))

    if analysis["is_lowkey"] and "hd" in p:
        toe_scale = 0.88
        for ch_key in p["hd"]:
            p["hd"][ch_key]["toe"] = float(
                np.clip(p["hd"][ch_key]["toe"] * toe_scale, 0.2, 0.85))

    if analysis["is_highkey"] and "hd" in p:
        sh_scale = 1.08
        for ch_key in p["hd"]:
            p["hd"][ch_key]["shoulder"] = float(
                np.clip(p["hd"][ch_key]["shoulder"] * sh_scale, 0.25, 0.95))

    if "rolloff" in p:
        if hi_frac < 0.05:
            p["rolloff"] = float(max(p["rolloff"] - 0.06, 0.55))

    if "shadow_lift" in p:
        lift = np.array(p["shadow_lift"], dtype=np.float32)
        if sh_frac < 0.05:
            lift = np.clip(lift * 1.4, 0.0, 0.15)
        elif sh_frac > 0.50:
            lift = lift * 0.75
        p["shadow_lift"] = tuple(float(v) for v in lift)

    if "bias" in p and not p.get("bw", False):
        bias = np.array(p["bias"], dtype=np.float32)

        compensation = cast * 0.30
        bias = bias - compensation

        if median < 0.35:
            push = (0.35 - median) * 0.75
            bias += push
        elif median > 0.65:
            pull = (median - 0.65) * 0.75
            bias -= pull

        if desat:
            deviation = bias - 1.0
            bias = 1.0 + deviation * 1.5

        p["bias"] = tuple(float(v) for v in np.clip(bias, 0.7, 1.4))

    return p

_ISO_GRAIN = {
      25: (12,  0.55),
      50: (16,  0.60),
      64: (20,  0.65),
     100: (28,  0.70),
     160: (36,  0.80),
     200: (42,  0.90),
     250: (48,  0.95),
     400: (60,  1.10),
     500: (68,  1.20),
     800: (85,  1.50),
    1600: (110, 2.00),
    3200: (145, 2.80),
}
_ISO_REFERENCE_AREA = 1920 * 1280

def _make_grain_params(preset: dict, H: int, W: int) -> dict:
    iso        = preset.get("iso", 400)
    grain_type = preset.get("grain_type", "gaussian")
    grain_color = preset.get("grain_color", "monochrome")

    iso_keys   = sorted(_ISO_GRAIN.keys())
    nearest    = min(iso_keys, key=lambda k: abs(k - iso))
    intensity, base_size = _ISO_GRAIN[nearest]
    intensity = intensity / 1.5

    area_scale = np.sqrt((H * W) / _ISO_REFERENCE_AREA)
    grain_size = float(np.clip(base_size * area_scale, 0.5, 8.0))

    shadow_strength    = 1.3 if preset.get("bw", False) else 1.0
    highlight_strength = 0.2 if preset.get("bw", False) else 0.3

    color_tint = "neutral"
    if preset["type"] == "CCD":
        color_tint  = "cool"
        intensity   = max(3, intensity // 3)
        grain_size  = float(np.clip(grain_size * 0.5, 0.5, 3.0))

    return {
        "intensity":          float(intensity),
        "grain_size":         grain_size,
        "grain_type":         grain_type,
        "color_mode":         grain_color,
        "color_tint":         color_tint,
        "shadow_strength":    shadow_strength,
        "highlight_strength": highlight_strength,
        "midtone_peak":       0.4,
    }

def _apply_grain(arr: np.ndarray, preset: dict, H: int, W: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    params = _make_grain_params(preset, H, W)
    iso = preset.get("iso", 400)
    gt = params["grain_type"]
    cm = params["color_mode"]

    rng = np.random.default_rng(None)

    sigma = (params["intensity"] / 255.0 * 40.0) / 255.0
    gs = params["grain_size"]
    mp = params["midtone_peak"]

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    bell = np.exp(-0.5 * ((lum - mp) / 0.28) ** 2)
    shadow_mask = np.clip(1.0 - lum / (mp + 1e-6), 0, 1)
    highlight_mask = np.clip((lum - mp) / (1.0 - mp + 1e-6), 0, 1)
    lum_mask = bell * (1.0 + shadow_mask * (params["shadow_strength"] - 1.0) + highlight_mask * (params["highlight_strength"] - 1.0))
    lum_mask = np.clip(lum_mask, 0, None)

    def make_noise(shape):
        raw = rng.standard_normal(shape).astype(np.float32)
        if gt == "gaussian":
            if gs > 0.6:
                raw = gaussian_filter(raw, sigma=gs * 0.5)
        elif gt == "organic":
            coarse = gaussian_filter(
                rng.standard_normal(shape).astype(np.float32), sigma=gs * 2.0)
            fine = gaussian_filter(raw, sigma=gs * 0.3)
            raw = coarse * 0.6 + fine * 0.4
        elif gt == "fine":
            raw = gaussian_filter(raw, sigma=max(0.3, gs * 0.2))
        return raw

    if cm == "monochrome":
        base = make_noise((H, W))
        nr = ng = nb = base
    else:
        correlation = float(np.clip(1.0 - (iso / 1600.0), 0.3, 0.9))

        base_lum = make_noise((H, W))
        nr = base_lum * correlation + make_noise((H, W)) * (1.0 - correlation)
        ng = base_lum * correlation + make_noise((H, W)) * (1.0 - correlation)
        nb = base_lum * correlation + make_noise((H, W)) * (1.0 - correlation)

        norm_factor = 1.0 / np.sqrt(correlation ** 2 + (1.0 - correlation) ** 2)
        nr *= norm_factor
        ng *= norm_factor
        nb *= norm_factor

    if params["color_tint"] == "cool":
        tr, tg, tb = 0.80, 0.95, 1.25
    else:
        tr = tg = tb = 1.0

    out = arr.copy()
    out[..., 0] = np.clip(arr[..., 0] + nr * sigma * lum_mask * tr, 0, 1)
    out[..., 1] = np.clip(arr[..., 1] + ng * sigma * lum_mask * tg, 0, 1)
    out[..., 2] = np.clip(arr[..., 2] + nb * sigma * lum_mask * tb, 0, 1)
    return out

def _apply_expiration(arr: np.ndarray, years: float) -> np.ndarray:
    out = arr.copy()

    fog = np.clip(years * 0.006, 0, 0.18)
    out = out + fog * (1.0 - out)

    blue_fade = np.clip(1.0 - (years * 0.012), 0.65, 1.0)
    out[..., 2] *= blue_fade

    contrast = np.clip(1.0 - (years * 0.01), 0.7, 1.0)
    out = np.clip((out - 0.5) * contrast + 0.5, 0.0, 1.0)

    return out

def _apply_halation(arr: np.ndarray, preset: dict, H: int, W: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    area_scale = np.sqrt((H * W) / (1920 * 1280))
    base_radius = 8.0 * area_scale

    desc = preset.get("desc", "").lower()
    is_cinema = "cinema" in desc or "vision3" in desc
    is_bw = preset.get("bw", False)

    strength = 0.55 if is_cinema else 0.25

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    threshold = 0.80

    hi_mask = np.clip((lum - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)

    bright_spots = arr * (hi_mask ** 2)[..., None]

    out = arr.copy()

    if is_bw:
        blur = gaussian_filter(bright_spots[..., 0], sigma=base_radius)
        out[..., 0] = np.clip(out[..., 0] + blur * strength, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + blur * strength, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + blur * strength, 0.0, 1.0)
    else:
        r_blur = gaussian_filter(bright_spots[..., 0], sigma=base_radius)
        g_blur = gaussian_filter(bright_spots[..., 1], sigma=base_radius * 0.6)
        b_blur = gaussian_filter(bright_spots[..., 2], sigma=base_radius * 0.2)

        out[..., 0] = np.clip(out[..., 0] + r_blur * strength * 1.2, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + g_blur * strength * 0.3, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + b_blur * strength * 0.0, 0.0, 1.0)

    return out

def _hd_curve(x: np.ndarray, toe: float, gamma: float, shoulder: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)

    k_toe   = toe * 10.0
    toe_out = 1.0 / (1.0 + np.exp(-k_toe * (x - toe * 0.5)))
    toe_out = toe_out - (1.0 / (1.0 + np.exp(-k_toe * (0.0 - toe * 0.5))))
    toe_out = toe_out / (1.0 / (1.0 + np.exp(-k_toe * (1.0 - toe * 0.5))) -
                          1.0 / (1.0 + np.exp(-k_toe * (0.0 - toe * 0.5))) + 1e-8)

    straight = np.clip(0.5 + gamma * (x - 0.5), 0.0, 1.0)

    k_sh      = shoulder * 10.0
    sh_offset = 1.0 - shoulder * 0.5
    sh_out    = 1.0 / (1.0 + np.exp(-k_sh * (x - sh_offset)))
    sh_out    = sh_out / (1.0 / (1.0 + np.exp(-k_sh * (1.0 - sh_offset))) + 1e-8)

    w_toe  = np.clip(1.0 - x / 0.4, 0.0, 1.0) ** 2
    w_sh   = np.clip((x - 0.6) / 0.4, 0.0, 1.0) ** 2
    w_mid  = 1.0 - w_toe - w_sh

    result = w_toe * toe_out + w_mid * straight + w_sh * sh_out
    return np.clip(result, 0.0, 1.0)

def _apply_lut(channel: np.ndarray, pts: list) -> np.ndarray:
    pts_arr = np.array(pts, dtype=np.float64)
    lut     = np.interp(np.linspace(0, 1, 256), pts_arr[:,0], pts_arr[:,1]).astype(np.float32)
    indices = np.clip((channel * 255).astype(np.int32), 0, 255)
    return lut[indices]

def _apply_colour(arr: np.ndarray, preset: dict) -> np.ndarray:
    bias = np.array(preset["bias"], dtype=np.float32)
    arr  = np.clip(arr * bias, 0.0, 1.0)

    hd = preset["hd"]
    R  = _hd_curve(arr[..., 0], **hd["r"])
    G  = _hd_curve(arr[..., 1], **hd["g"])
    B  = _hd_curve(arr[..., 2], **hd["b"])
    arr = np.stack([R, G, B], axis=-1)

    rolloff  = preset.get("rolloff", 0.80)
    hi_start = rolloff
    if hi_start < 1.0:
        hi_mask       = np.clip((arr - hi_start) / (1.0 - hi_start), 0.0, 1.0)
        hi_compressed = hi_start + (1.0 - hi_start) * (1.0 - (1.0 - hi_mask) ** 2)
        arr           = np.where(arr > hi_start, hi_compressed, arr)

    lift = np.array(preset.get("shadow_lift", (0.0, 0.0, 0.0)), dtype=np.float32)
    if lift.any():
        arr = arr + lift * (1.0 - arr)

    return np.clip(arr, 0.0, 1.0)

def _apply_bw(arr: np.ndarray, preset: dict) -> np.ndarray:
    mix  = preset["mix"]
    grey = np.clip(arr[...,0]*mix[0] + arr[...,1]*mix[1] + arr[...,2]*mix[2], 0.0, 1.0)

    hd   = preset["hd"]
    grey = _hd_curve(grey, **hd["r"])

    rolloff  = preset.get("rolloff", 0.82)
    hi_start = rolloff
    if hi_start < 1.0:
        hi_mask       = np.clip((grey - hi_start) / (1.0 - hi_start), 0.0, 1.0)
        hi_compressed = hi_start + (1.0 - hi_start) * (1.0 - (1.0 - hi_mask) ** 2)
        grey          = np.where(grey > hi_start, hi_compressed, grey)

    lift = np.array(preset.get("shadow_lift", (0.0, 0.0, 0.0)), dtype=np.float32)
    grey_lifted = grey + lift.mean() * (1.0 - grey)

    tint = np.array(preset["tint"], dtype=np.float32)
    return np.clip(np.stack([
        grey_lifted * tint[0],
        grey_lifted * tint[1],
        grey_lifted * tint[2],
    ], axis=-1), 0.0, 1.0)

def _apply_sensor(arr: np.ndarray, preset: dict) -> np.ndarray:
    linear = np.where(arr <= 0.04045, arr/12.92, ((arr+0.055)/1.055)**2.4)
    bias   = np.array(preset["bias"], dtype=np.float32)
    linear = np.clip(linear * bias, 0, 1)
    M      = np.array(preset["matrix"], dtype=np.float32)
    H, W   = linear.shape[:2]
    flat   = np.clip(linear.reshape(-1,3) @ M.T, 0, 1)
    linear = flat.reshape(H, W, 3)
    rolloff = preset["highlight_rolloff"]
    if rolloff > 0:
        hi_start      = 1.0 - rolloff
        hi_mask       = np.clip((linear - hi_start) / rolloff, 0, 1)
        hi_compressed = hi_start + rolloff * (1.0 - (1.0 - hi_mask)**2)
        linear        = np.where(linear > hi_start, hi_compressed, linear)
    srgb = np.where(linear <= 0.0031308, linear*12.92,
                    1.055 * np.power(np.clip(linear, 0, None), 1.0/2.4) - 0.055)
    srgb = np.clip(srgb, 0, 1)
    curves = preset["curves"]
    srgb   = np.clip(np.stack([
        _apply_lut(srgb[...,0], curves["r"]),
        _apply_lut(srgb[...,1], curves["g"]),
        _apply_lut(srgb[...,2], curves["b"]),
    ], axis=-1), 0, 1)
    lift = np.array(preset["shadow_lift"], dtype=np.float32)
    return np.clip(srgb + lift * (1.0 - srgb), 0, 1)
