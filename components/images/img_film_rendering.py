import numpy as np
from PIL import Image


# ── Film / Sensor preset library ─────────────────────────────────────────────
# Colour film (_CF):
#   "bw"     : False
#   "type"   : "CF"
#   "bias"   : (R, G, B) multiplicative gain
#   "curves" : {"r","g","b"} each list of (in, out) control points 0–1
#
# B&W film (_BWF):
#   "bw"     : True
#   "type"   : "BWF"
#   "mix"    : (R, G, B) spectral sensitivity weights, sum to 1.0
#   "tint"   : (R, G, B) chemical toning colour bias
#   "bias"   : (1,1,1) unused, kept for consistency
#   "curves" : {"r","g","b"} identical — applied to the grey channel
#
# Digital sensor (_CCD):
#   "bw"     : False
#   "type"   : "CCD"
#   "bias"   : (R, G, B) sensor colour filter bias
#   "matrix" : 3×3 colour science matrix (applied in linear light)
#   "curves" : {"r","g","b"} camera tone curve / picture profile
#   "shadow_lift" : (R, G, B) black point lift — sensor noise floor colour
#   "highlight_rolloff" : float 0–1, controls highlight compression strength

FILM_PRESETS = {

    # ─────────────────────────────────────────────────────────────────────────
    # COLOUR FILMS (_CF)
    # ─────────────────────────────────────────────────────────────────────────

    "fuji_astia_100_CF": {
        "desc": "Fuji Astia 100 — soft, low contrast, neutral skin tones, subtle colours",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.00, 0.97),
        "curves": {
            "r": [(0.0,0.05),(0.25,0.27),(0.5,0.52),(0.75,0.76),(1.0,0.97)],
            "g": [(0.0,0.04),(0.25,0.27),(0.5,0.52),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.04),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.96)],
        },
    },

    "fuji_provia_100_CF": {
        "desc": "Fuji Provia 100F — standard/neutral, accurate colour, moderate contrast",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.01, 1.02),
        "curves": {
            "r": [(0.0,0.02),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.03),(0.25,0.27),(0.5,0.53),(0.75,0.77),(1.0,0.99)],
        },
    },

    "fuji_velvia_100_CF": {
        "desc": "Fuji Velvia 100 — punchy, very saturated, high contrast, vivid greens and blues",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.03, 1.06),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.22),(0.5,0.50),(0.75,0.78),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.23),(0.5,0.52),(0.75,0.80),(1.0,1.00)],
            "b": [(0.0,0.00),(0.25,0.24),(0.5,0.54),(0.75,0.82),(1.0,1.00)],
        },
    },

    "fuji_superia_400_CF": {
        "desc": "Fuji Superia 400 — consumer negative, warm greens, slight grain character",
        "bw": False, "type": "CF",
        "bias": (1.02, 1.03, 0.96),
        "curves": {
            "r": [(0.0,0.04),(0.25,0.28),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.04),(0.25,0.29),(0.5,0.54),(0.75,0.78),(1.0,0.97)],
            "b": [(0.0,0.03),(0.25,0.25),(0.5,0.49),(0.75,0.73),(1.0,0.94)],
        },
    },

    "fuji_400h_CF": {
        "desc": "Fuji 400H — soft highlights, cool shadows, popular portrait film",
        "bw": False, "type": "CF",
        "bias": (0.99, 1.01, 1.04),
        "curves": {
            "r": [(0.0,0.05),(0.25,0.27),(0.5,0.51),(0.75,0.75),(1.0,0.96)],
            "g": [(0.0,0.05),(0.25,0.28),(0.5,0.52),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.06),(0.25,0.29),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
        },
    },

    "kodak_kodachrome_64_CF": {
        "desc": "Kodak Kodachrome 64 — iconic warm reds, deep blues, high contrast, rich shadows",
        "bw": False, "type": "CF",
        "bias": (1.06, 0.98, 0.94),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.24),(0.5,0.52),(0.75,0.80),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.22),(0.5,0.49),(0.75,0.76),(1.0,0.98)],
            "b": [(0.0,0.00),(0.25,0.21),(0.5,0.48),(0.75,0.75),(1.0,0.97)],
        },
    },

    "kodak_ektachrome_100vs_CF": {
        "desc": "Kodak Ektachrome 100VS — very saturated, cool blues, strong greens",
        "bw": False, "type": "CF",
        "bias": (0.98, 1.02, 1.05),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.23),(0.5,0.50),(0.75,0.78),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.24),(0.5,0.52),(0.75,0.80),(1.0,1.00)],
            "b": [(0.0,0.00),(0.25,0.25),(0.5,0.54),(0.75,0.82),(1.0,1.00)],
        },
    },

    "kodak_portra_160_CF": {
        "desc": "Kodak Portra 160 — warm skin tones, soft highlights, low contrast, fine grain",
        "bw": False, "type": "CF",
        "bias": (1.04, 1.01, 0.95),
        "curves": {
            "r": [(0.0,0.04),(0.25,0.28),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.03),(0.25,0.27),(0.5,0.52),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.03),(0.25,0.25),(0.5,0.49),(0.75,0.73),(1.0,0.94)],
        },
    },

    "kodak_portra_400_CF": {
        "desc": "Kodak Portra 400 — versatile portrait film, warm, slightly lifted shadows",
        "bw": False, "type": "CF",
        "bias": (1.03, 1.00, 0.96),
        "curves": {
            "r": [(0.0,0.05),(0.25,0.28),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.04),(0.25,0.27),(0.5,0.52),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.04),(0.25,0.25),(0.5,0.50),(0.75,0.74),(1.0,0.95)],
        },
    },

    "kodak_gold_200_CF": {
        "desc": "Kodak Gold 200 — consumer film, warm golden tone, boosted yellows and reds",
        "bw": False, "type": "CF",
        "bias": (1.05, 1.02, 0.92),
        "curves": {
            "r": [(0.0,0.03),(0.25,0.28),(0.5,0.54),(0.75,0.79),(1.0,0.98)],
            "g": [(0.0,0.03),(0.25,0.27),(0.5,0.53),(0.75,0.78),(1.0,0.97)],
            "b": [(0.0,0.02),(0.25,0.23),(0.5,0.47),(0.75,0.71),(1.0,0.92)],
        },
    },

    "kodak_ultramax_400_CF": {
        "desc": "Kodak Ultramax 400 — vivid warm colours, punchy contrast, popular street film",
        "bw": False, "type": "CF",
        "bias": (1.04, 1.01, 0.93),
        "curves": {
            "r": [(0.0,0.02),(0.25,0.26),(0.5,0.52),(0.75,0.79),(1.0,0.99)],
            "g": [(0.0,0.02),(0.25,0.26),(0.5,0.52),(0.75,0.78),(1.0,0.98)],
            "b": [(0.0,0.01),(0.25,0.22),(0.5,0.47),(0.75,0.73),(1.0,0.94)],
        },
    },

    "kodak_tri_x_400_CF": {
        "desc": "Kodak Tri-X 400 — B&W look in colour, strong contrast, warm shadow tint",
        "bw": False, "type": "CF",
        "bias": (1.02, 1.00, 0.97),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.21),(0.5,0.50),(0.75,0.79),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.21),(0.5,0.50),(0.75,0.79),(1.0,1.00)],
            "b": [(0.0,0.00),(0.25,0.20),(0.5,0.49),(0.75,0.78),(1.0,0.99)],
        },
    },

    "agfa_vista_200_CF": {
        "desc": "Agfa Vista 200 — cool shadows, slight blue-green tint, soft contrast",
        "bw": False, "type": "CF",
        "bias": (0.97, 1.00, 1.04),
        "curves": {
            "r": [(0.0,0.04),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.96)],
            "g": [(0.0,0.04),(0.25,0.27),(0.5,0.52),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.05),(0.25,0.28),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
        },
    },

    "lomography_lomo_100_CF": {
        "desc": "Lomography 100 — high contrast, cross-process look, boosted saturation",
        "bw": False, "type": "CF",
        "bias": (1.05, 0.97, 1.02),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.24),(0.5,0.53),(0.75,0.82),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.20),(0.5,0.48),(0.75,0.77),(1.0,0.98)],
            "b": [(0.0,0.00),(0.25,0.23),(0.5,0.52),(0.75,0.80),(1.0,1.00)],
        },
    },

    "ilford_xp2_400_CF": {
        "desc": "Ilford XP2 Super 400 — chromogenic B&W, neutral, clean shadows",
        "bw": False, "type": "CF",
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.01),(0.25,0.23),(0.5,0.50),(0.75,0.77),(1.0,0.99)],
            "g": [(0.0,0.01),(0.25,0.23),(0.5,0.50),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.23),(0.5,0.50),(0.75,0.77),(1.0,0.99)],
        },
    },

    "kodak_vision3_500t_CF": {
        "desc": "Kodak Vision3 500T — cinema negative, tungsten balanced, warm shadows, teal highlights",
        "bw": False, "type": "CF",
        "bias": (1.03, 0.99, 0.96),
        "curves": {
            "r": [(0.0,0.05),(0.25,0.28),(0.5,0.52),(0.75,0.76),(1.0,0.96)],
            "g": [(0.0,0.04),(0.25,0.27),(0.5,0.51),(0.75,0.75),(1.0,0.96)],
            "b": [(0.0,0.06),(0.25,0.27),(0.5,0.51),(0.75,0.76),(1.0,0.97)],
        },
    },

    "fuji_eterna_250d_CF": {
        "desc": "Fuji Eterna 250D — cinema film, daylight balanced, soft contrast, desaturated highlights",
        "bw": False, "type": "CF",
        "bias": (0.99, 1.01, 1.03),
        "curves": {
            "r": [(0.0,0.05),(0.25,0.27),(0.5,0.51),(0.75,0.75),(1.0,0.95)],
            "g": [(0.0,0.05),(0.25,0.28),(0.5,0.52),(0.75,0.76),(1.0,0.96)],
            "b": [(0.0,0.05),(0.25,0.28),(0.5,0.52),(0.75,0.76),(1.0,0.96)],
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
        "curves": {
            "r": [(0.0,0.01),(0.25,0.24),(0.5,0.51),(0.75,0.77),(1.0,0.99)],
            "g": [(0.0,0.01),(0.25,0.24),(0.5,0.51),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.24),(0.5,0.51),(0.75,0.77),(1.0,0.99)],
        },
    },

    "ilford_delta_100_BWF": {
        "desc": "Ilford Delta 100 — fine grain, cool neutral tone, excellent shadow detail",
        "bw": True, "type": "BWF",
        "mix": (0.28, 0.60, 0.12),
        "tint": (0.98, 0.99, 1.01),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
        },
    },

    "ilford_delta_3200_BWF": {
        "desc": "Ilford Delta 3200 — very high ISO, lifted shadows, compressed highlights",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.59, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.08),(0.25,0.28),(0.5,0.52),(0.75,0.75),(1.0,0.94)],
            "g": [(0.0,0.08),(0.25,0.28),(0.5,0.52),(0.75,0.75),(1.0,0.94)],
            "b": [(0.0,0.08),(0.25,0.28),(0.5,0.52),(0.75,0.75),(1.0,0.94)],
        },
    },

    "kodak_tmax_100_BWF": {
        "desc": "Kodak T-Max 100 — ultra-fine grain, high contrast, deep clean blacks",
        "bw": True, "type": "BWF",
        "mix": (0.27, 0.62, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.22),(0.5,0.51),(0.75,0.80),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.22),(0.5,0.51),(0.75,0.80),(1.0,1.00)],
            "b": [(0.0,0.00),(0.25,0.22),(0.5,0.51),(0.75,0.80),(1.0,1.00)],
        },
    },

    "kodak_tmax_400_BWF": {
        "desc": "Kodak T-Max 400 — fine grain for ISO 400, excellent tonal range",
        "bw": True, "type": "BWF",
        "mix": (0.28, 0.61, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.01),(0.25,0.23),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "g": [(0.0,0.01),(0.25,0.23),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.23),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
        },
    },

    "kodak_tri_x_400_BWF": {
        "desc": "Kodak Tri-X 400 B&W — iconic, punchy, deep blacks, photojournalism classic",
        "bw": True, "type": "BWF",
        "mix": (0.32, 0.58, 0.10),
        "tint": (1.01, 1.00, 0.99),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.21),(0.5,0.50),(0.75,0.80),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.21),(0.5,0.50),(0.75,0.80),(1.0,1.00)],
            "b": [(0.0,0.00),(0.25,0.21),(0.5,0.50),(0.75,0.80),(1.0,1.00)],
        },
    },

    "agfa_apx_100_BWF": {
        "desc": "Agfa APX 100 — smooth midtones, slightly warm neutral, soft shadow gradation",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.59, 0.11),
        "tint": (1.01, 1.00, 0.99),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.02),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
            "g": [(0.0,0.02),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
            "b": [(0.0,0.02),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
        },
    },

    "agfa_apx_400_BWF": {
        "desc": "Agfa APX 400 — medium grain, contrasty midtones, green-sensitive",
        "bw": True, "type": "BWF",
        "mix": (0.29, 0.61, 0.10),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.01),(0.25,0.23),(0.5,0.51),(0.75,0.79),(1.0,0.99)],
            "g": [(0.0,0.01),(0.25,0.23),(0.5,0.51),(0.75,0.79),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.23),(0.5,0.51),(0.75,0.79),(1.0,0.99)],
        },
    },

    "rollei_rpx_400_BWF": {
        "desc": "Rollei RPX 400 — very deep blacks, punchy street photography look",
        "bw": True, "type": "BWF",
        "mix": (0.30, 0.59, 0.11),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.00),(0.25,0.20),(0.5,0.50),(0.75,0.81),(1.0,1.00)],
            "g": [(0.0,0.00),(0.25,0.20),(0.5,0.50),(0.75,0.81),(1.0,1.00)],
            "b": [(0.0,0.00),(0.25,0.20),(0.5,0.50),(0.75,0.81),(1.0,1.00)],
        },
    },

    "fomapan_100_BWF": {
        "desc": "Fomapan 100 — orthochromatic character, blue-sensitive, soft contrast, vintage look",
        "bw": True, "type": "BWF",
        "mix": (0.22, 0.55, 0.23),
        "tint": (1.00, 1.00, 1.00),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.03),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.97)],
            "g": [(0.0,0.03),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.97)],
            "b": [(0.0,0.03),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.97)],
        },
    },

    "selenium_tone_BWF": {
        "desc": "Selenium toning — cool blue-purple shadow tone, archival darkroom process",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (0.96, 0.97, 1.04),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.01),(0.25,0.23),(0.5,0.50),(0.75,0.77),(1.0,0.99)],
            "g": [(0.0,0.01),(0.25,0.23),(0.5,0.50),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.23),(0.5,0.50),(0.75,0.77),(1.0,0.99)],
        },
    },

    "sepia_tone_BWF": {
        "desc": "Sepia toning — warm brown throughout, classic Victorian / vintage look",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.08, 1.00, 0.82),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.04),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.04),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "b": [(0.0,0.04),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
        },
    },

    "gold_tone_BWF": {
        "desc": "Gold toning — warm golden highlights, cooler shadows, elegant darkroom effect",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.05, 1.02, 0.88),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.02),(0.25,0.24),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "g": [(0.0,0.02),(0.25,0.24),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "b": [(0.0,0.02),(0.25,0.24),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
        },
    },

    "cyanotype_BWF": {
        "desc": "Cyanotype — deep cyan-blue alternative process print look",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (0.78, 0.90, 1.15),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.02),(0.25,0.23),(0.5,0.50),(0.75,0.76),(1.0,0.97)],
            "g": [(0.0,0.02),(0.25,0.23),(0.5,0.50),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.02),(0.25,0.23),(0.5,0.50),(0.75,0.76),(1.0,0.97)],
        },
    },

    "platinum_palladium_BWF": {
        "desc": "Platinum/Palladium print — long tonal scale, warm neutral, rich shadow detail",
        "bw": True, "type": "BWF",
        "mix": (0.299, 0.587, 0.114),
        "tint": (1.02, 1.01, 0.99),
        "bias": (1.00, 1.00, 1.00),
        "curves": {
            "r": [(0.0,0.03),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.97)],
            "g": [(0.0,0.03),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.97)],
            "b": [(0.0,0.03),(0.25,0.26),(0.5,0.51),(0.75,0.75),(1.0,0.97)],
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # DIGITAL CAMERA SENSORS (_CCD)
    #
    # "matrix" : 3×3 colour science matrix applied in linear light.
    #            Rows = output R, G, B. Columns = input R, G, B.
    #            Identity = [[1,0,0],[0,1,0],[0,0,1]]
    # "shadow_lift" : (R, G, B) black point offset — sensor noise floor colour
    # "highlight_rolloff" : 0–1, how aggressively highlights are compressed
    # ─────────────────────────────────────────────────────────────────────────

    "canon_5d_mark2_CCD": {
        "desc": "Canon 5D Mark II — warm romantic colour, gentle highlight rolloff, "
                "smooth skin tones, classic full-frame DSLR",
        "bw": False, "type": "CCD",
        "bias": (1.04, 1.00, 0.97),
        "matrix": [
            [ 1.06, -0.04, -0.02],
            [-0.03,  1.04, -0.01],
            [-0.02, -0.06,  1.08],
        ],
        "shadow_lift": (0.008, 0.006, 0.005),
        "highlight_rolloff": 0.35,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.50),(0.75,0.75),(1.0,0.96)],
        },
    },

    "canon_5d_mark1_CCD": {
        "desc": "Canon 5D Mark I — original full-frame, warm CCD character, "
                "lower dynamic range but very pleasing colour rendering",
        "bw": False, "type": "CCD",
        "bias": (1.05, 1.00, 0.95),
        "matrix": [
            [ 1.08, -0.05, -0.03],
            [-0.03,  1.05, -0.02],
            [-0.03, -0.07,  1.10],
        ],
        "shadow_lift": (0.012, 0.009, 0.007),
        "highlight_rolloff": 0.45,
        "curves": {
            "r": [(0.0,0.02),(0.25,0.27),(0.5,0.53),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.76),(1.0,0.96)],
            "b": [(0.0,0.01),(0.25,0.24),(0.5,0.50),(0.75,0.74),(1.0,0.95)],
        },
    },

    "canon_1dx_CCD": {
        "desc": "Canon 1Dx — professional sports/press, accurate neutral colour, "
                "punchy contrast, excellent highlight detail",
        "bw": False, "type": "CCD",
        "bias": (1.02, 1.00, 0.99),
        "matrix": [
            [ 1.04, -0.02, -0.02],
            [-0.02,  1.03, -0.01],
            [-0.01, -0.04,  1.05],
        ],
        "shadow_lift": (0.006, 0.005, 0.005),
        "highlight_rolloff": 0.25,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.78),(1.0,0.99)],
            "b": [(0.0,0.00),(0.25,0.24),(0.5,0.50),(0.75,0.77),(1.0,0.98)],
        },
    },

    "sony_a7iii_CCD": {
        "desc": "Sony A7 III — neutral accurate colour, cool shadow character, "
                "high dynamic range, clinical modern look",
        "bw": False, "type": "CCD",
        "bias": (1.00, 1.01, 1.02),
        "matrix": [
            [ 1.02, -0.01, -0.01],
            [-0.01,  1.03,  0.00],
            [ 0.00, -0.02,  1.02],
        ],
        "shadow_lift": (0.004, 0.004, 0.006),
        "highlight_rolloff": 0.20,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,1.00)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
        },
    },

    "sony_a7rii_CCD": {
        "desc": "Sony A7R II — very high resolution, neutral-cool, extremely detailed, "
                "wide colour gamut rendering",
        "bw": False, "type": "CCD",
        "bias": (0.99, 1.01, 1.03),
        "matrix": [
            [ 1.01, -0.01,  0.00],
            [-0.01,  1.03,  0.00],
            [ 0.00, -0.02,  1.02],
        ],
        "shadow_lift": (0.003, 0.003, 0.005),
        "highlight_rolloff": 0.18,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,1.00)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.78),(1.0,1.00)],
        },
    },

    "nikon_d800_CCD": {
        "desc": "Nikon D800 — neutral accurate, slightly cool shadows, excellent detail, "
                "professional landscape/studio camera",
        "bw": False, "type": "CCD",
        "bias": (1.00, 1.01, 1.01),
        "matrix": [
            [ 1.03, -0.02, -0.01],
            [-0.01,  1.03,  0.00],
            [ 0.00, -0.03,  1.03],
        ],
        "shadow_lift": (0.005, 0.005, 0.007),
        "highlight_rolloff": 0.22,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.77),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
        },
    },

    "nikon_d3_CCD": {
        "desc": "Nikon D3 — warm classic DSLR rendering, photojournalism standard, "
                "smooth pleasing colour, good shadow detail",
        "bw": False, "type": "CCD",
        "bias": (1.03, 1.01, 0.97),
        "matrix": [
            [ 1.05, -0.03, -0.02],
            [-0.02,  1.04, -0.01],
            [-0.01, -0.05,  1.06],
        ],
        "shadow_lift": (0.008, 0.007, 0.006),
        "highlight_rolloff": 0.30,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.50),(0.75,0.75),(1.0,0.96)],
        },
    },

    "fuji_xt3_CCD": {
        "desc": "Fuji X-T3 — film-simulation-inspired colour science, warm midtones, "
                "distinctive X-Trans rendering, vivid but natural",
        "bw": False, "type": "CCD",
        "bias": (1.03, 1.01, 0.98),
        "matrix": [
            [ 1.05, -0.03, -0.02],
            [-0.02,  1.05, -0.02],
            [-0.01, -0.04,  1.05],
        ],
        "shadow_lift": (0.007, 0.006, 0.005),
        "highlight_rolloff": 0.40,
        "curves": {
            "r": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "g": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.75),(1.0,0.96)],
        },
    },

    "fuji_gfx_CCD": {
        "desc": "Fuji GFX 100 — medium format digital, very neutral and clean, "
                "exceptional tonal gradation, subtle warm-neutral",
        "bw": False, "type": "CCD",
        "bias": (1.01, 1.01, 1.00),
        "matrix": [
            [ 1.02, -0.01, -0.01],
            [-0.01,  1.03, -0.01],
            [ 0.00, -0.02,  1.02],
        ],
        "shadow_lift": (0.005, 0.005, 0.004),
        "highlight_rolloff": 0.28,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.26),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
        },
    },

    "leica_m9_CCD": {
        "desc": "Leica M9 CCD — iconic true CCD sensor, warm romantic colour, "
                "beautiful highlight glow, beloved by photographers for its "
                "three-dimensional rendering",
        "bw": False, "type": "CCD",
        "bias": (1.05, 1.01, 0.94),
        "matrix": [
            [ 1.07, -0.04, -0.03],
            [-0.03,  1.05, -0.02],
            [-0.02, -0.08,  1.10],
        ],
        "shadow_lift": (0.010, 0.008, 0.007),
        "highlight_rolloff": 0.50,
        "curves": {
            "r": [(0.0,0.02),(0.25,0.27),(0.5,0.53),(0.75,0.78),(1.0,0.97)],
            "g": [(0.0,0.02),(0.25,0.27),(0.5,0.52),(0.75,0.77),(1.0,0.97)],
            "b": [(0.0,0.01),(0.25,0.24),(0.5,0.50),(0.75,0.74),(1.0,0.95)],
        },
    },

    "leica_m11_CCD": {
        "desc": "Leica M11 CMOS — modern Leica, very neutral and clinical, "
                "60MP resolution character, faithful colour science",
        "bw": False, "type": "CCD",
        "bias": (1.01, 1.01, 1.00),
        "matrix": [
            [ 1.02, -0.01, -0.01],
            [-0.01,  1.02,  0.00],
            [ 0.00, -0.02,  1.02],
        ],
        "shadow_lift": (0.004, 0.004, 0.004),
        "highlight_rolloff": 0.22,
        "curves": {
            "r": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
            "g": [(0.0,0.01),(0.25,0.26),(0.5,0.52),(0.75,0.77),(1.0,0.99)],
            "b": [(0.0,0.01),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.98)],
        },
    },

    "hasselblad_x2d_CCD": {
        "desc": "Hasselblad X2D — 100MP medium format, extremely neutral and clean, "
                "very wide tonal range, clinical precision colour",
        "bw": False, "type": "CCD",
        "bias": (1.00, 1.01, 1.01),
        "matrix": [
            [ 1.01,  0.00, -0.01],
            [ 0.00,  1.02,  0.00],
            [ 0.00, -0.01,  1.01],
        ],
        "shadow_lift": (0.003, 0.003, 0.003),
        "highlight_rolloff": 0.15,
        "curves": {
            "r": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "g": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
            "b": [(0.0,0.00),(0.25,0.25),(0.5,0.51),(0.75,0.76),(1.0,0.99)],
        },
    },

    "olympus_omd_CCD": {
        "desc": "Olympus OM-D E-M1 — punchy vivid colour, slightly cool, "
                "Micro Four Thirds sensor, sharp contrasty rendering",
        "bw": False, "type": "CCD",
        "bias": (1.01, 1.02, 1.02),
        "matrix": [
            [ 1.03, -0.01, -0.02],
            [-0.01,  1.04, -0.01],
            [-0.01, -0.02,  1.03],
        ],
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
    if intensity == 0:
        return image.convert("RGB")

    if rendering not in FILM_PRESETS:
        valid = ", ".join(sorted(FILM_PRESETS.keys()))
        raise ValueError(f"Unknown rendering '{rendering}'. Valid: {valid}")

    if not (0 <= intensity <= 100):
        raise ValueError(f"intensity must be 0–100, got {intensity}")

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

    blend  = intensity / 100.0
    result = orig * (1.0 - blend) + arr_out * blend
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8), mode="RGB")


def list_film_presets() -> dict:
    """Returns {preset_name: description} for all presets. Use for ComfyUI combo."""
    return {k: v["desc"] for k, v in FILM_PRESETS.items()}


def list_presets_by_type() -> dict:
    """Returns presets grouped by type: {"CF": [...], "BWF": [...], "CCD": [...]}"""
    result = {"CF": [], "BWF": [], "CCD": []}
    for k, v in FILM_PRESETS.items():
        result[v["type"]].append(k)
    return result


def _apply_lut(channel: np.ndarray, pts: list) -> np.ndarray:
    pts_arr = np.array(pts, dtype=np.float64)
    lut     = np.interp(np.linspace(0, 1, 256), pts_arr[:,0], pts_arr[:,1]).astype(np.float32)
    indices = np.clip((channel * 255).astype(np.int32), 0, 255)
    return lut[indices]


def _apply_colour(arr: np.ndarray, preset: dict) -> np.ndarray:
    bias    = np.array(preset["bias"], dtype=np.float32)
    arr     = np.clip(arr * bias, 0, 1)
    curves  = preset["curves"]
    return np.clip(np.stack([
        _apply_lut(arr[..., 0], curves["r"]),
        _apply_lut(arr[..., 1], curves["g"]),
        _apply_lut(arr[..., 2], curves["b"]),
    ], axis=-1), 0, 1)


def _apply_bw(arr: np.ndarray, preset: dict) -> np.ndarray:
    mix  = preset["mix"]
    grey = np.clip(arr[...,0]*mix[0] + arr[...,1]*mix[1] + arr[...,2]*mix[2], 0, 1)
    grey = _apply_lut(grey, preset["curves"]["r"])
    tint = np.array(preset["tint"], dtype=np.float32)
    return np.clip(np.stack([grey*tint[0], grey*tint[1], grey*tint[2]], axis=-1), 0, 1)


def _apply_sensor(arr: np.ndarray, preset: dict) -> np.ndarray:
    # ── 1. Linearise sRGB (remove gamma) ──────────────────────────────────────
    linear = np.where(arr <= 0.04045, arr/12.92,
                      ((arr + 0.055)/1.055)**2.4)

    # ── 2. Apply sensor colour bias ────────────────────────────────────────────
    bias   = np.array(preset["bias"], dtype=np.float32)
    linear = np.clip(linear * bias, 0, 1)

    # ── 3. Apply 3×3 colour science matrix ────────────────────────────────────
    M      = np.array(preset["matrix"], dtype=np.float32)
    H, W   = linear.shape[:2]
    flat   = linear.reshape(-1, 3)
    flat   = np.clip(flat @ M.T, 0, 1)
    linear = flat.reshape(H, W, 3)

    # ── 4. Apply highlight rolloff in linear light ─────────────────────────────
    # Compress values above rolloff_start toward 1.0 with a smooth curve
    rolloff = preset["highlight_rolloff"]
    if rolloff > 0:
        hi_start = 1.0 - rolloff
        hi_mask  = np.clip((linear - hi_start) / rolloff, 0, 1)
        # Smooth S-curve compression in the highlight zone
        hi_compressed = hi_start + rolloff * (1.0 - (1.0 - hi_mask)**2)
        linear = np.where(linear > hi_start, hi_compressed, linear)

    # ── 5. Re-apply sRGB gamma ─────────────────────────────────────────────────
    srgb = np.where(linear <= 0.0031308, linear*12.92,
                    1.055 * np.power(np.clip(linear, 0, None), 1.0/2.4) - 0.055)
    srgb = np.clip(srgb, 0, 1)

    # ── 6. Apply camera tone curve (picture profile) ───────────────────────────
    curves = preset["curves"]
    srgb   = np.clip(np.stack([
        _apply_lut(srgb[..., 0], curves["r"]),
        _apply_lut(srgb[..., 1], curves["g"]),
        _apply_lut(srgb[..., 2], curves["b"]),
    ], axis=-1), 0, 1)

    # ── 7. Apply shadow lift / sensor noise floor colour ──────────────────────
    lift = np.array(preset["shadow_lift"], dtype=np.float32)
    srgb = np.clip(srgb + lift * (1.0 - srgb), 0, 1)

    return srgb
