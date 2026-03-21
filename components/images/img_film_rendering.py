import numpy as np
from PIL import Image


# ── Film / Sensor preset library ─────────────────────────────────────────────
#
# All presets now include layered film physics parameters:
#
# Colour film (_CF) and B&W film (_BWF):
#   "bias"        : (R, G, B) multiplicative colour bias
#   "hd"          : Hurter-Driffield characteristic curve parameters per channel
#                   Each channel: {"toe": float, "gamma": float, "shoulder": float}
#                   toe      = steepness of shadow compression (0.3–0.8)
#                              higher = more shadow detail compression
#                   gamma    = midtone contrast / straight-line slope (0.7–1.4)
#                              higher = more contrast in midtones
#                   shoulder = steepness of highlight rolloff (0.3–0.9)
#                              higher = harder highlight rolloff (less blooming)
#   "rolloff"     : float 0–1, where highlight shoulder begins (luminance)
#                   lower = shoulder starts earlier (softer highlights overall)
#   "shadow_lift" : (R, G, B) black point colour cast — the colour of film base
#
# Digital sensor (_CCD): unchanged structure, gains layered rendering too
#
# The H&D sigmoid function applied per channel:
#   f(x) = shoulder_out / (1 + exp(-k_mid * (x - x0)))
# where toe/gamma/shoulder parameters set the shape of each zone.

FILM_PRESETS = {

    # ─────────────────────────────────────────────────────────────────────────
    # COLOUR FILMS (_CF)
    # ─────────────────────────────────────────────────────────────────────────

    "fuji_astia_100_CF": {
        "desc": "Fuji Astia 100 — soft, low contrast, neutral skin tones, subtle colours",
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

    "kodak_kodachrome_64_CF": {
        "desc": "Kodak Kodachrome 64 — iconic warm reds, deep blues, high contrast, rich shadows",
        "bw": False, "type": "CF",
        "bias": (1.06, 0.98, 0.94),
        "rolloff": 0.80,
        "shadow_lift": (0.00, 0.00, 0.00),
        "hd": {
            "r": {"toe": 0.60, "gamma": 1.10, "shoulder": 0.72},  # red: high contrast
            "g": {"toe": 0.45, "gamma": 0.95, "shoulder": 0.58},  # green: moderate
            "b": {"toe": 0.42, "gamma": 0.90, "shoulder": 0.55},  # blue: slightly less
        },
    },

    "kodak_ektachrome_100vs_CF": {
        "desc": "Kodak Ektachrome 100VS — very saturated, cool blues, strong greens",
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
        "bw": False, "type": "CF",
        "bias": (1.04, 1.01, 0.95),
        "rolloff": 0.70,
        "shadow_lift": (0.04, 0.03, 0.03),
        "hd": {
            "r": {"toe": 0.35, "gamma": 0.82, "shoulder": 0.38},  # red: very soft
            "g": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.36},
            "b": {"toe": 0.30, "gamma": 0.76, "shoulder": 0.33},  # blue: softest
        },
    },

    "kodak_portra_400_CF": {
        "desc": "Kodak Portra 400 — versatile portrait film, warm, slightly lifted shadows",
        "bw": False, "type": "CF",
        "bias": (1.03, 1.00, 0.96),
        "rolloff": 0.73,
        "shadow_lift": (0.05, 0.04, 0.04),
        "hd": {
            "r": {"toe": 0.37, "gamma": 0.85, "shoulder": 0.40},
            "g": {"toe": 0.35, "gamma": 0.83, "shoulder": 0.38},
            "b": {"toe": 0.33, "gamma": 0.80, "shoulder": 0.36},
        },
    },

    "kodak_gold_200_CF": {
        "desc": "Kodak Gold 200 — consumer film, warm golden tone, boosted yellows and reds",
        "bw": False, "type": "CF",
        "bias": (1.05, 1.02, 0.92),
        "rolloff": 0.76,
        "shadow_lift": (0.03, 0.03, 0.02),
        "hd": {
            "r": {"toe": 0.45, "gamma": 0.95, "shoulder": 0.55},
            "g": {"toe": 0.43, "gamma": 0.92, "shoulder": 0.52},
            "b": {"toe": 0.32, "gamma": 0.78, "shoulder": 0.38},  # blue: compressed
        },
    },

    "kodak_ultramax_400_CF": {
        "desc": "Kodak Ultramax 400 — vivid warm colours, punchy contrast, popular street film",
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

    "agfa_vista_200_CF": {
        "desc": "Agfa Vista 200 — cool shadows, slight blue-green tint, soft contrast",
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

    "kodak_vision3_500t_CF": {
        "desc": "Kodak Vision3 500T — cinema negative, tungsten balanced, warm shadows, teal highlights",
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

    "fuji_eterna_250d_CF": {
        "desc": "Fuji Eterna 250D — cinema film, daylight balanced, soft contrast, desaturated highlights",
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

    # ─────────────────────────────────────────────────────────────────────────
    # B&W FILMS (_BWF)
    # ─────────────────────────────────────────────────────────────────────────

    "ilford_hp5_400_BWF": {
        "desc": "Ilford HP5 Plus 400 — classic panchromatic, neutral grey, forgiving latitude",
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

    "kodak_tmax_100_BWF": {
        "desc": "Kodak T-Max 100 — ultra-fine grain, high contrast, deep clean blacks",
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

    "fomapan_100_BWF": {
        "desc": "Fomapan 100 — orthochromatic character, blue-sensitive, soft contrast, vintage look",
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

    # ─────────────────────────────────────────────────────────────────────────
    # DIGITAL SENSORS (_CCD) — unchanged structure
    # ─────────────────────────────────────────────────────────────────────────

    "canon_5d_mark2_CCD": {
        "desc": "Canon 5D Mark II — warm romantic colour, gentle highlight rolloff, smooth skin tones",
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

    "nikon_d800_CCD": {
        "desc": "Nikon D800 — neutral accurate, slightly cool shadows, excellent detail",
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
        "desc": "Leica M9 CCD — iconic true CCD sensor, warm romantic colour, beautiful highlight glow",
        "bw": False, "type": "CCD",
        "bias": (1.05, 1.01, 0.94),
        "matrix": [[1.07,-0.04,-0.03],[-0.03,1.05,-0.02],[-0.02,-0.08,1.10]],
        "shadow_lift": (0.010, 0.008, 0.007),
        "highlight_rolloff": 0.50,
        "curves": {
            "r": [(0.0,0.02),(0.25,0.27),(0.5,0.53),(0.75,0.78),(1.0,0.97)],
            "g": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.24),(0.5,0.50),(0.75,0.74),(1.0,0.95)],
        },
    },

    "leica_m11_CCD": {
        "desc": "Leica M11 CMOS — modern Leica, very neutral and clinical, faithful colour science",
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
}


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def img_film_rendering(
    image:     Image.Image,
    rendering: str   = "kodak_kodachrome_64_CF",
    intensity: float = 100,
) -> Image.Image:
    """
    Film stock and digital sensor rendering simulation.

    Colour film (_CF) and B&W film (_BWF) presets use a layered H&D pipeline:
      1. Colour bias (sensor/emulsion spectral response)
      2. Per-channel Hurter-Driffield sigmoid curve (toe / gamma / shoulder)
         with differential channel response — each channel has its own
         characteristic curve producing real colour separation across tones
      3. Smooth highlight rolloff shoulder (not hard clip)
      4. Shadow lift / film base colour cast

    Digital sensor (_CCD) presets use the existing matrix pipeline.

    Args:
        image     : PIL Image (RGB)
        rendering : Preset name (see FILM_PRESETS keys)
        intensity : 0 … 200.
                    0   = passthrough.
                    100 = full preset rendering.
                    101–200 = overdrive — extrapolates beyond the preset
                    for a more dramatic effect. 200 = double the difference
                    from original. User can dial back if too strong.
    Returns:
        PIL Image (RGB)
    """
    if intensity == 0:
        return image.convert("RGB")

    if rendering not in FILM_PRESETS:
        valid = ", ".join(sorted(FILM_PRESETS.keys()))
        raise ValueError(f"Unknown rendering '{rendering}'. Valid: {valid}")

    if not (0 <= intensity <= 200):
        raise ValueError(f"intensity must be 0–200, got {intensity}")

    preset = FILM_PRESETS[rendering]
    img    = image.convert("RGB")
    arr    = np.array(img, dtype=np.float32) / 255.0
    orig   = arr.copy()

    if preset["type"] == "BWF":
        arr_out = _apply_bw(arr, preset)
    elif preset["type"] == "CCD":
        arr_out = _apply_sensor(arr, preset)
    else:
        arr_out = _apply_colour(arr, preset)

    # ── Blend with overdrive support ──────────────────────────────────────────
    # intensity 0–100: linear blend  orig → arr_out
    # intensity 100–200: extrapolate beyond arr_out
    #   result = orig + blend * (arr_out - orig)
    # At blend=1.0 (intensity=100): result = arr_out  (same as before)
    # At blend=2.0 (intensity=200): result = 2*arr_out - orig (double push)
    blend  = intensity / 100.0
    result = orig + blend * (arr_out - orig)
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def list_film_presets() -> dict:
    """Returns {preset_name: description} for all presets."""
    return {k: v["desc"] for k, v in FILM_PRESETS.items()}


def list_presets_by_type() -> dict:
    """Returns presets grouped by type: {"CF": [...], "BWF": [...], "CCD": [...]}"""
    result = {"CF": [], "BWF": [], "CCD": []}
    for k, v in FILM_PRESETS.items():
        result[v["type"]].append(k)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# H&D characteristic curve
# ─────────────────────────────────────────────────────────────────────────────

def _hd_curve(x: np.ndarray, toe: float, gamma: float, shoulder: float) -> np.ndarray:
    """
    Hurter-Driffield sigmoid characteristic curve for one emulsion layer.

    Models the three zones of real film response:
      Toe      — shadow compression, low contrast, detail preservation
      Straight — midtone linear region, main contrast zone
      Shoulder — highlight compression, smooth rolloff, no hard clipping

    Implementation: piecewise sigmoid blend
      toe zone:      sigmoid centred at 0,         steepness = toe * 8
      straight zone: linear ramp with slope gamma
      shoulder zone: sigmoid centred at 1,         steepness = shoulder * 8

    Args:
        x        : input values 0–1
        toe      : shadow steepness / compression  (0.3 = soft, 0.7 = hard)
        gamma    : midtone contrast slope          (0.7 = low, 1.3 = high)
        shoulder : highlight compression steepness (0.3 = gentle, 0.8 = abrupt)
    Returns:
        output values 0–1
    """
    x = np.clip(x, 0.0, 1.0)

    # ── Toe sigmoid (shadow zone) ─────────────────────────────────────────────
    # Maps 0 → 0, pulls shadow tones upward gently
    k_toe   = toe * 10.0
    toe_out = 1.0 / (1.0 + np.exp(-k_toe * (x - toe * 0.5)))
    toe_out = toe_out - (1.0 / (1.0 + np.exp(-k_toe * (0.0 - toe * 0.5))))
    toe_out = toe_out / (1.0 / (1.0 + np.exp(-k_toe * (1.0 - toe * 0.5))) -
                          1.0 / (1.0 + np.exp(-k_toe * (0.0 - toe * 0.5))) + 1e-8)

    # ── Straight line (midtone zone) ──────────────────────────────────────────
    # Linear with slope gamma, pivoted at midpoint (0.5, 0.5)
    straight = np.clip(0.5 + gamma * (x - 0.5), 0.0, 1.0)

    # ── Shoulder sigmoid (highlight zone) ────────────────────────────────────
    # Maps 1 → 1, compresses highlights gently
    k_sh      = shoulder * 10.0
    sh_offset = 1.0 - shoulder * 0.5
    sh_out    = 1.0 / (1.0 + np.exp(-k_sh * (x - sh_offset)))
    sh_out    = sh_out / (1.0 / (1.0 + np.exp(-k_sh * (1.0 - sh_offset))) + 1e-8)

    # ── Blend zones by luminance ──────────────────────────────────────────────
    # Smooth blend weights: toe dominates in shadows, shoulder in highlights,
    # straight line in midtones
    w_toe  = np.clip(1.0 - x / 0.4, 0.0, 1.0) ** 2
    w_sh   = np.clip((x - 0.6) / 0.4, 0.0, 1.0) ** 2
    w_mid  = 1.0 - w_toe - w_sh

    result = w_toe * toe_out + w_mid * straight + w_sh * sh_out
    return np.clip(result, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _apply_lut(channel: np.ndarray, pts: list) -> np.ndarray:
    """Legacy LUT for CCD presets."""
    pts_arr = np.array(pts, dtype=np.float64)
    lut     = np.interp(np.linspace(0, 1, 256), pts_arr[:,0], pts_arr[:,1]).astype(np.float32)
    indices = np.clip((channel * 255).astype(np.int32), 0, 255)
    return lut[indices]


def _apply_colour(arr: np.ndarray, preset: dict) -> np.ndarray:
    """
    Colour film H&D pipeline:
      1. Colour bias
      2. Per-channel H&D sigmoid with differential toe/shoulder
         (channels respond differently across tonal zones → real colour separation)
      3. Highlight shoulder rolloff
      4. Shadow lift
    """
    # ── 1. Colour bias ────────────────────────────────────────────────────────
    bias = np.array(preset["bias"], dtype=np.float32)
    arr  = np.clip(arr * bias, 0.0, 1.0)

    # ── 2. Per-channel H&D curve ──────────────────────────────────────────────
    hd = preset["hd"]
    R  = _hd_curve(arr[..., 0], **hd["r"])
    G  = _hd_curve(arr[..., 1], **hd["g"])
    B  = _hd_curve(arr[..., 2], **hd["b"])
    arr = np.stack([R, G, B], axis=-1)

    # ── 3. Smooth highlight rolloff ───────────────────────────────────────────
    # Applied after the H&D curve, in the output domain.
    # Uses a smooth quadratic shoulder rather than hard clip.
    rolloff  = preset.get("rolloff", 0.80)
    hi_start = rolloff
    if hi_start < 1.0:
        hi_mask       = np.clip((arr - hi_start) / (1.0 - hi_start), 0.0, 1.0)
        hi_compressed = hi_start + (1.0 - hi_start) * (1.0 - (1.0 - hi_mask) ** 2)
        arr           = np.where(arr > hi_start, hi_compressed, arr)

    # ── 4. Shadow lift / film base colour ─────────────────────────────────────
    lift = np.array(preset.get("shadow_lift", (0.0, 0.0, 0.0)), dtype=np.float32)
    if lift.any():
        arr = arr + lift * (1.0 - arr)

    return np.clip(arr, 0.0, 1.0)


def _apply_bw(arr: np.ndarray, preset: dict) -> np.ndarray:
    """
    B&W film H&D pipeline:
      1. Spectral sensitivity channel mix
      2. H&D curve on the grey channel
      3. Highlight rolloff
      4. Shadow lift
      5. Chemical toning tint
    """
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
    """Digital sensor pipeline — unchanged from previous version."""
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
