from ..components.tree import TREE_RASTERIX
from ..components.tree import PRIMERE_ROOT
import random
import folder_paths

from ..components.images import img_shade_level as img_shade_level
from ..components.images import img_brightness_contrast as img_brightness_contrast
from ..components.images import img_color_balance as img_color_balance
from ..components.images import img_hue_saturation as img_hue_saturation
from ..components.images import img_levels_auto as img_levels_auto
from ..components.images import isgen_detect_ext_full as isgen_detect_ext_full
from ..components.images import img_film_grain as img_film_grain
from ..components.images import img_blur as img_blur
from ..components.images import img_selective_tone as img_selective_tone
from ..components.images import img_smart_lighting as img_smart_lighting
from ..components.images import img_white_balance as img_white_balance
from ..components.images import img_film_rendering as img_film_rendering
from ..components.images.img_film_rendering import FILM_PRESETS
from ..components.images import img_lens_effects as img_lens_effects
from ..components.images import img_levels_compress as img_levels_compress
from ..components.images import img_dithering as img_dithering
from ..components.images import histogram as histogram
from ..components.images import img_posterize as img_posterize
from ..components.images import img_solarization_bw as img_solarization_bw
from ..components.images import img_clarity as img_clarity
from ..components.images import img_dehaze as img_dehaze
from ..components.images import img_local_laplacian as img_local_laplacian
from ..components.images import img_frequency_separation as img_frequency_separation
from ..components.images import img_filmic_curve as img_filmic_curve
from ..components.images import img_lut3d as img_lut3d
from ..components.images import img_edge_jitter as img_edge_jitter
from ..components.images import img_depth_blur as img_depth_blur
from ..components.images import img_photo_paper as img_photo_paper
from ..components.images.img_photo_paper import PAPER_PRESETS

from ..components import utility
from .Dashboard import PrimereModelConceptSelector as PrimereModelConceptSelector
import os
from server import PromptServer

FILM_PRESETS_BY_TYPE = img_film_rendering.list_presets_by_type()
FILM_TYPES = ["All"] + sorted(FILM_PRESETS_BY_TYPE.keys())

class PrimereRasterix:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_rasterix"
    CATEGORY = TREE_RASTERIX
    OUTPUT_NODE = True

    MODELLIST = PrimereModelConceptSelector.MODELLIST
    CONCEPT_LIST =  PrimereModelConceptSelector.CONCEPT_LIST
    FILM_TYPES = FILM_TYPES
    FILM_PRESETS_BY_TYPE = FILM_PRESETS_BY_TYPE

    LUT_DIR = os.path.join(PRIMERE_ROOT, 'components', 'images', 'luts')

    SECTION_TITLES = [
        {"before": "concepts", "name": "rasterix_main", "title": "🧭 Project Setup", "color": "#5C3D34", "text_color": "#EAF1F8", "label": "Choose model concept/model for save-load profiles and set precision for the full pipeline."},

        {"after": "precision", "name": "rasterix_auto_levels", "title": "🎚 Auto Levels & Gamma", "color": "#6A4A2A", "text_color": "#EAF1F8", "label": "Photoshop-style auto levels with threshold protection and optional target gamma alignment. Inspired by Adobe Photoshop."},
        {"after": "gamma_target", "name": "rasterix_white_balance", "title": "🔦 White Balance", "color": "#6A4A2A", "text_color": "#EAF1F8", "label": "Correct temperature and tint first to establish a neutral color baseline for later grading. Inspired by Adobe Camera Raw and DxO Photolab."},
        {"after": "wb_tint", "name": "rasterix_smart_lighting", "title": "💡 Smart Lighting", "color": "#6A4A2A", "text_color": "#EAF1F8", "label": "Adaptive light shaping to recover perceived depth and readability before local effects. Inspired by DxO Photolab"},

        {"after": "smart_lighting", "name": "rasterix_dehaze", "title": "🌫 Atmosphere: Dehaze", "color": "#3E5C4B", "text_color": "#EAF1F8", "label": "Reduce haze and veiling glow while preserving natural contrast and color balance. Inspired by Adobe Lightroom Dehaze."},
        {"after": "dehaze_contrast", "name": "rasterix_depth_blur", "title": "🌀 Atmosphere: Depth Blur", "color": "#3E5C4B", "text_color": "#EAF1F8", "label": "Depth-guided lens blur to separate subject and background with controllable focus falloff."},
        {"after": "depth_gamma", "name": "rasterix_blur", "title": "🫗 Atmosphere: Creative Blur", "color": "#3E5C4B", "text_color": "#EAF1F8", "label": "Apply additional blur styles for softness, abstraction, or cinematic diffusion."},

        {"after": "edge_threshold", "name": "rasterix_brightness_contrast", "title": "🧊 Tone: Brightness & Contrast", "color": "#405985", "text_color": "#EAF1F8", "label": "Global tone shaping for exposure feel and contrast punch after atmospheric corrections. Inspired by Adobe Photoshop."},
        {"after": "use_legacy", "name": "rasterix_portrait_retouch", "title": "🪒 Tone: Portrait Retouch", "color": "#405985", "text_color": "#EAF1F8", "label": "Frequency-based skin and texture workflow for gentle portrait cleanup and separation. Inspired by professional Photoshop retouch workflows."},
        {"after": "blend_mode", "name": "rasterix_local_laplacian", "title": "🧱 Tone: Edge-Aware Pyramid", "color": "#405985", "text_color": "#EAF1F8", "label": "Local Laplacian contrast/detail enhancement with strong edge preservation."},

        {"after": "levels", "name": "rasterix_analog_film", "title": "🎞 Creative: Analog Film / CCD", "color": "#3B5E68", "text_color": "#EAF1F8", "label": "Stylized film and sensor-era rendering for mood, palette, and texture character. Inspired by DxO."},
        {"after": "expiration_years", "name": "rasterix_photo_paper", "title": "🧪 Creative: Photo Paper Simulation", "color": "#3B5E68", "text_color": "#EAF1F8", "label": "Darkroom-inspired paper response with selectable grade, RC/FB base, color/B&W mode, and controlled print intensity."},
        {"after": "paper_intensity", "name": "rasterix_lut_reader", "title": "📷 Creative: LUT .cube Reader", "color": "#3B5E68", "text_color": "#EAF1F8", "label": "Load and blend LUT looks for fast creative direction and consistent show style. Inspired by Blackmagic DaVinci Resolve and DxO."},
        {"after": "color_space", "name": "rasterix_filmic_camera", "title": "🎥 Creative: Filmic Camera Curve", "color": "#3B5E68", "text_color": "#EAF1F8", "label": "Camera-like highlight roll-off and tonal response for cinematic dynamic range behavior. Inspired by Adobe Camera Raw."},

        {"after": "pivot", "name": "rasterix_selective_tone", "title": "🎛 Color: Selective Tone Zones", "color": "#6A5636", "text_color": "#EAF1F8", "label": "Zone-based tonal pushes for highlights, midtones, shadows, and blacks. Inspired by DxO Photolab"},
        {"after": "selective_tone_strength", "name": "rasterix_color_balance", "title": "⚖ Color: Balance Wheels", "color": "#6A5636", "text_color": "#EAF1F8", "label": "Color-balance style adjustments per tonal range with luminosity preservation options. Inspired by DaVinci Resolve and Photoshop color wheels."},
        {"after": "color_balance_separation", "name": "rasterix_hsl", "title": "🌈 Color: HSL Sculpting", "color": "#6A5636", "text_color": "#EAF1F8", "label": "Hue, saturation, lightness, and vibrance targeting by color channel. Inspired by Adobe Lightroom and Photoshop HSL panel."},

        {"after": "hsl_skin_protection", "name": "rasterix_shade_detailer", "title": "💎 Detail: Microcontrast", "color": "#554267", "text_color": "#EAF1F8", "label": "Fine local contrast shaping to emphasize texture and perceived detail. Inspired by DxO PhotoLab microcontrast tools."},
        {"after": "shade_strength", "name": "rasterix_clarity", "title": "🔍 Detail: Midtone Clarity", "color": "#554267", "text_color": "#EAF1F8", "label": "Midtone-focused clarity enhancement for crispness without excessive global contrast. Inspired by Adobe Lightroom Clarity."},

        {"after": "edge_preservation", "name": "rasterix_endpoints", "title": "🔛 Output: Black/White Endpoints", "color": "#5A603E", "text_color": "#EAF1F8", "label": "Set endpoint compression and clipping behavior for final output anchoring. Inspired by Adobe Photoshop Levels."},
        {"after": "skip_if_no_clip", "name": "rasterix_dithering", "title": "🧩 Output: Dithering & Diffusion", "color": "#5A603E", "text_color": "#EAF1F8", "label": "Reduce banding and smooth gradients using dither and error diffusion tools. Inspired by Floyd-Steinberg error diffusion."},

        {"after": "error_diffusion", "name": "rasterix_histogram", "title": "📊 Analysis: Histogram", "color": "#35586A", "text_color": "#EAF1F8", "label": "View channel histograms for fast clipping, balance, and tonal distribution checks. Inspired by Adobe Photoshop (and all other) Histogram."},
    ]

    @classmethod
    def _list_luts(cls):
        lut_entries = ["None"]
        if not os.path.exists(cls.LUT_DIR):
            return lut_entries

        for f in sorted(os.listdir(cls.LUT_DIR)):
            full_path = os.path.join(cls.LUT_DIR, f)
            if os.path.isfile(full_path) and f.lower().endswith(".cube"):
                lut_entries.append(f)

        for d in sorted(os.listdir(cls.LUT_DIR)):
            subdir = os.path.join(cls.LUT_DIR, d)
            if os.path.isdir(subdir):
                for f in sorted(os.listdir(subdir)):
                    if f.lower().endswith(".cube"):
                        lut_entries.append(f"{d}/{f}")

        return lut_entries

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concepts": (["Auto"] + cls.CONCEPT_LIST,),
                "models": (["Auto"] + cls.MODELLIST,),

                "image":                 ("IMAGE", {"forceInput": True}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),

                "auto_normalize":        ("BOOLEAN", {"default": False, "label_off": "No auto levels", "label_on": "Apply auto levels"}),
                "auto_levels_threshold": ("FLOAT",   {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "auto_gamma": ("BOOLEAN", {"default": False, "label_on": "Auto gamma: ON", "label_off": "Auto gamma:: OFF"}),
                "gamma_target": ("FLOAT", {"default": 128.0, "min": 0.0, "max": 255.0, "step": 0.1}),

                "use_white_balance": ("BOOLEAN", {"default": False, "label_off": "Ignore white balance", "label_on": "Apply white balance"}),
                "wb_temperature": ("FLOAT", {"default": 6500, "min": 2000, "max": 12000, "step": 100}),
                "wb_tint":        ("FLOAT", {"default": 0,    "min": -100, "max": 100,   "step": 1}),

                "use_smart_lighting": ("BOOLEAN", {"default": False, "label_off": "Ignore smart lightning", "label_on": "Apply smart lightning"}),
                "smart_lighting": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1}),

                "use_dehaze": ("BOOLEAN", {"default": False, "label_off": "Ignore dehaze", "label_on": "Apply dehaze"}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "dehaze_radius": ("INT", {"default": 15, "min": 3, "max": 100, "step": 1}),
                "omega": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "t0": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "dehaze_contrast": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01}),

                "use_depth_blur": ("BOOLEAN", {"default": False, "label_off": "Ignore depth blur", "label_on": "Apply depth blur"}),
                "auto_optimize": ("BOOLEAN", {"default": False, "label_off": "Use custom inputs", "label_on": "Optimize settings by focus"}),
                "use_DA_v3": ("BOOLEAN", {"default": False, "label_off": "Depth-anything V2", "label_on": "Depth-anything V3"}),
                "focus_depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_range": ("FLOAT", {"default": 0.200, "min": 0.001, "max": 1.000, "step": 0.001}),
                "max_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "depth_gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),

                "use_blur": ("BOOLEAN", {"default": False, "label_off": "Ignore blur", "label_on": "Apply blur"}),
                "blur_type":      (["gaussian", "box", "motion", "bilateral", "lens"], {"default": "bilateral"}),
                "blur_intensity": ("FLOAT",   {"default": 0.0, "min": 0.0, "max": 5.0,   "step": 0.1}),
                "blur_radius":    ("FLOAT",   {"default": 2.0, "min": 0.5, "max": 50.0,  "step": 0.5}),
                "angle":          ("FLOAT",   {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "bilateral_edge_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_edge_only": ("BOOLEAN", {"default": False, "label_off": "Full image blur", "label_on": "Flat areas only, edges protected"}),
                "edge_threshold": ("FLOAT",   {"default": 0.0, "min": 0.0, "max": 1.0,   "step": 0.01}),

                "use_brightness_contrast": ("BOOLEAN", {"default": False, "label_off": "Ignore brightness-contrast", "label_on": "Apply brightness-contrast"}),
                "brightness": ("FLOAT", {"default": 0, "min": -150, "max": 150, "step": 1}),
                "contrast":   ("FLOAT", {"default": 0, "min": -50,  "max": 100, "step": 1}),
                "use_legacy": ("BOOLEAN", {"default": False, "label_off": "Use non-linear shift", "label_on": "Use adaptive offset"}),

                "use_frequency_separation": ("BOOLEAN", {"default": False, "label_off": "Ignore frequency separation", "label_on": "Apply frequency separation"}),
                "radius": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.1}),
                "low_freq_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "high_freq_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "blend_mode": (["add", "multiply", "overlay"], {"default": "add"}),

                "use_local_laplacian": ("BOOLEAN", {"default": False, "label_off": "Ignore local laplacian", "label_on": "Apply local laplacian"}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "laplacian_contrast": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.01}),
                "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "levels": ("INT", {"default": 8, "min": 4, "max": 32, "step": 1}),

                "use_film_rendering": ("BOOLEAN", {"default": False, "label_off": "Ignore film rendering", "label_on": "Apply film rendering"}),
                "film_type": (cls.FILM_TYPES, {"default": "All"}),
                "film_rendering": (list(FILM_PRESETS.keys()), {"default": list(FILM_PRESETS.keys())[0]}),
                "film_rendering_intensity": ("FLOAT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "iso_grain": ("BOOLEAN", {"default": False, "label_off": "Ignore ISO grain", "label_on": "Add ISO grain"}),
                "halation": ("BOOLEAN", {"default": False, "label_off": "Ignore halation", "label_on": "Add halation"}),
                "expiration_years": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1}),

                "use_photo_paper": ("BOOLEAN", {"default": False, "label_off": "Ignore photo paper", "label_on": "Apply photo paper"}),
                "photo_paper": (list(PAPER_PRESETS.keys()), {"default": list(PAPER_PRESETS.keys())[0]}),
                "color_paper": ("BOOLEAN", {"default": False, "label_off": "B&W paper", "label_on": "Color paper"}),
                "paper_base": (["RC", "FB"], {"default": "RC"}),
                "paper_expiration_years": ("FLOAT", {"default": 0, "min": 0, "max": 30, "step": 0.1}),
                "paper_intensity": ("FLOAT", {"default": 100, "min": 0, "max": 200, "step": 1}),

                "use_lut": ("BOOLEAN", {"default": False, "label_off": "Ignore LUT", "label_on": "Apply LUT"}),
                "lut_file": (cls._list_luts(),),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_space": (["sRGB", "linear"], {"default": "sRGB"}),

                "use_filmic": ("BOOLEAN", {"default": False, "label_off": "Ignore filmic", "label_on": "Apply filmic"}),
                "curve_type": (["filmic", "log"], {"default": "filmic"}),
                "filmic_contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "highlight_rolloff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_lift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                "use_selective_tone": ("BOOLEAN", {"default": False, "label_off": "Ignore selective tone", "label_on": "Apply selective tone"}),
                "selective_tone_value":      ("FLOAT", {"default": 0,   "min": -100, "max": 100, "step": 1}),
                "selective_tone_zone":       (["highlights", "midtones", "shadows", "blacks"], {"default": "midtones"}),
                "selective_tone_separation": ("FLOAT", {"default": 50,  "min": 0,    "max": 100, "step": 1}),
                "selective_tone_strength":   ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0, "step": 0.01}),

                "use_color_balance": ("BOOLEAN", {"default": False, "label_off": "Ignore color balance", "label_on": "Apply color balance"}),
                "color_balance_cyan_red":           ("FLOAT",   {"default": 0,  "min": -100, "max": 100, "step": 1}),
                "color_balance_magenta_green":       ("FLOAT",   {"default": 0,  "min": -100, "max": 100, "step": 1}),
                "color_balance_yellow_blue":         ("FLOAT",   {"default": 0,  "min": -100, "max": 100, "step": 1}),
                "color_balance_tone":                (["highlights", "midtones", "shadows"], {"default": "midtones"}),
                "color_balance_preserve_luminosity": ("BOOLEAN", {"default": False, "label_off": "Modify luminosity", "label_on": "Restore original luminosity"}),
                "color_balance_separation":          ("FLOAT",   {"default": 50, "min": 0,    "max": 100, "step": 1}),

                "use_hsl": ("BOOLEAN", {"default": False, "label_off": "Ignore HSL", "label_on": "Apply HSL"}),
                "hsl_hue":           ("FLOAT",   {"default": 0,    "min": -180, "max": 180, "step": 1}),
                "hsl_saturation":    ("FLOAT",   {"default": 0,    "min": -100, "max": 100, "step": 1}),
                "hsl_lightness":     ("FLOAT",   {"default": 0,    "min": -100, "max": 100, "step": 1}),
                "hsl_vibrance":      ("FLOAT",   {"default": 0,    "min": -100, "max": 100, "step": 1}),
                "hsl_channel":       (["master", "red", "green", "blue"], {"default": "master"}),
                "hsl_channel_width": ("FLOAT",   {"default": 50,   "min": 0,    "max": 100, "step": 1}),
                "hsl_skin_protection": ("BOOLEAN", {"default": True, "label_off": "Vibrance affects skin tones", "label_on": "Skin tones protected from vibrance"}),

                "use_shade_detailer": ("BOOLEAN", {"default": False, "label_off": "Ignore shade detailer", "label_on": "Apply shade detailer"}),
                "shade_level":    ("FLOAT", {"default": 0,   "min": -100, "max": 100, "step": 1}),
                "shade_radius":   ("FLOAT", {"default": 0,   "min": 0,    "max": 50,  "step": 0.5}),
                "detail_mode":    (["fine", "medium", "broad"], {"default": "medium"}),
                "shade_strength": ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0, "step": 0.01}),

                "use_clarity": ("BOOLEAN", {"default": False, "label_off": "Ignore clarity", "label_on": "Apply clarity"}),
                "clarity_strength": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
                "clarity_radius": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "midtone_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "edge_preservation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),

                "use_level_endpoints": ("BOOLEAN", {"default": False, "label_off": "Ignore endpoint offset", "label_on": "Apply endpoint offset"}),
                "black_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "white_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "skip_if_no_clip": ("BOOLEAN", {"default": False, "label_off": "Offset all values", "label_on": "Skip if no clips"}),

                "normalize_gaps": ("BOOLEAN", {"default": False, "label_on": "Anti-comb filter: ON", "label_off": "Anti-comb filter: OFF"}),
                "normalize_midpeaks": ("BOOLEAN", {"default": False, "label_on": "Anti-spike filter: ON", "label_off": "Anti-spike filter: OFF"}),
                "peak_width": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "dither_quantization": ("BOOLEAN", {"default": False, "label_off": "Dither quantization OFF", "label_on": "Dither quantization ON"}),
                "adaptive_dither_strength": ("BOOLEAN", {"default": False, "label_off": "Keep dither strength", "label_on": "Increase dither strength"}),
                "error_diffusion": ("BOOLEAN", {"default": False, "label_off": "Error diffusion OFF", "label_on": "Error diffusion ON"}),

                "show_histogram": ("BOOLEAN", {"default": False, "label_off": "Ignore histogram", "label_on": "Create histogram"}),
                "histogram_source":        ("BOOLEAN", {"default": False, "label_off": "Show output histogram", "label_on": "Show input histogram"}),
                "histogram_channel":    (["RGB", "RED", "GREEN", "BLUE"], {"default": "RGB"}),
                "histogram_style":      (["bars", "lines", "waveform", "heatmap", "stacked", "luma", "parade", "gradient", "glow", "dots", "step", "log", "percentile", "inverse"], {"default": "bars"}),
            },
            "optional": {
                "model_concept": ("STRING", {"default": None, "forceInput": True}),
                "model_name": ("CHECKPOINT_NAME", {"default": None, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": utility.MAX_SEED, "forceInput": True}),
            },
            "hidden": {
                "id": "UNIQUE_ID",
            }
        }

    def primere_rasterix(self, **kwargs):
        concepts = kwargs.get('concepts', 'Auto')
        models = kwargs.get('models', 'Auto')
        model_concept = kwargs.get('model_concept', None)
        model_name = kwargs.get('model_name', None)
        active_concept = model_concept if concepts == "Auto" else concepts
        active_display = active_concept

        auto_runtime_mode = concepts == "Auto" and models == "Auto"

        if auto_runtime_mode:
            raw_model = model_name
            model_key = os.path.splitext(os.path.basename(raw_model))[0] if raw_model else None
            json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix_settings.json')
            concept_data = utility.json2tuple(json_path)
            if model_key and concept_data and model_key in concept_data:
                lookup_key = model_key
                active_display = model_key
            else:
                lookup_key = active_concept
                active_display = active_concept
            if not concept_data or lookup_key not in concept_data:
                PromptServer.instance.send_sync("primere.rasterix_setting", {"status": "missing", "concept": active_concept})
            else:
                saved = concept_data[lookup_key]
                for k, v in saved.items():
                    if k in kwargs:
                        kwargs[k] = v

        image = kwargs.get('image')
        precision = kwargs.get('precision', False)
        seed = kwargs.get('seed', 0)
        auto_normalize = kwargs.get('auto_normalize', False)
        auto_levels_threshold = kwargs.get('auto_levels_threshold', 0.2)
        normalize_midpeaks = kwargs.get('normalize_midpeaks', False)
        peak_width = kwargs.get('peak_width', 3)
        auto_gamma = kwargs.get('auto_gamma', False)
        gamma_target = kwargs.get('gamma_target', 128.0)
        use_white_balance = kwargs.get('use_white_balance', False)
        wb_temperature = kwargs.get('wb_temperature', 6500)
        wb_tint = kwargs.get('wb_tint', 0)
        use_depth_blur = kwargs.get('use_depth_blur', False)
        auto_optimize = kwargs.get('auto_optimize', False)
        use_DA_v3 = kwargs.get('use_DA_v3', False)
        focus_depth = kwargs.get('focus_depth', 0.5)
        depth_range = kwargs.get('depth_range', 0.200)
        max_blur = kwargs.get('bilateral_edge_sensitivity', 8.0)
        depth_gamma = kwargs.get('depth_gamma', 1.0)
        use_blur = kwargs.get('use_blur', False)
        blur_type = kwargs.get('blur_type', "bilateral")
        blur_intensity = kwargs.get('blur_intensity', 0.0)
        blur_radius = kwargs.get('blur_radius', 2.0)
        angle = kwargs.get('angle', 0.0)
        bilateral_edge_sensitivity = kwargs.get('bilateral_edge_sensitivity', 0.5)
        blur_edge_only = kwargs.get('blur_edge_only', False)
        edge_threshold = kwargs.get('edge_threshold', 0.0)
        use_smart_lighting = kwargs.get('use_smart_lighting', False)
        smart_lighting = kwargs.get('smart_lighting', 0)
        use_dehaze = kwargs.get('use_dehaze', False)
        strength = kwargs.get('strength', 0.7)
        dehaze_radius = kwargs.get('dehaze_radius', 15)
        omega = kwargs.get('omega', 0.95)
        t0 = kwargs.get('t0', 0.1)
        dehaze_contrast = kwargs.get('dehaze_contrast', 1.05)
        use_brightness_contrast = kwargs.get('use_brightness_contrast', False)
        brightness = kwargs.get('brightness', 0)
        contrast = kwargs.get('contrast', 0)
        use_legacy = kwargs.get('use_legacy', False)
        use_frequency_separation = kwargs.get('use_frequency_separation', False)
        radius = kwargs.get('radius', 3.0)
        low_freq_strength = kwargs.get('low_freq_strength', 1.0)
        high_freq_strength = kwargs.get('high_freq_strength', 1.0)
        blend_mode = kwargs.get('blend_mode', 'add')
        use_local_laplacian = kwargs.get('use_local_laplacian', False)
        sigma = kwargs.get('sigma', 1.0)
        laplacian_contrast = kwargs.get('laplacian_contrast', 1.2)
        detail = kwargs.get('detail', 1.0)
        levels = kwargs.get('levels', 8)
        use_film_rendering = kwargs.get('use_film_rendering', False)
        film_type = "All" if auto_runtime_mode else kwargs.get('film_type', "All")
        film_rendering = kwargs.get('film_rendering', list(FILM_PRESETS.keys())[0])
        film_rendering_intensity = kwargs.get('film_rendering_intensity', 100)
        iso_grain = kwargs.get('iso_grain', False)
        halation = kwargs.get('halation', False)
        expiration_years = kwargs.get('expiration_years', 0)
        use_photo_paper = kwargs.get('use_photo_paper', False)
        photo_paper = kwargs.get('photo_paper', "N (ISO R 90, normal)")
        color_paper = kwargs.get('color_paper', False)
        paper_base = kwargs.get('paper_base', "RC")
        paper_expiration_years = kwargs.get('paper_expiration_years', 0)
        paper_intensity = kwargs.get('paper_intensity', 100)
        use_filmic = kwargs.get('use_filmic', False)
        curve_type = kwargs.get('curve_type', "filmic")
        filmic_contrast = kwargs.get('filmic_contrast', 1.0)
        highlight_rolloff = kwargs.get('highlight_rolloff', 0.5)
        shadow_lift = kwargs.get('shadow_lift', 0.0)
        pivot = kwargs.get('pivot', 0.5)
        use_selective_tone = kwargs.get('use_selective_tone', False)
        selective_tone_separation = kwargs.get('selective_tone_separation', 50)
        selective_tone_strength = kwargs.get('selective_tone_strength', 0.5)
        use_color_balance = kwargs.get('use_color_balance', False)
        color_balance_preserve_luminosity = kwargs.get('color_balance_preserve_luminosity', False)
        color_balance_separation = kwargs.get('color_balance_separation', 50)
        use_lut = kwargs.get('use_lut', False)
        lut_file = kwargs.get('lut_file', "None")
        intensity = kwargs.get('intensity', 1.0)
        color_space = kwargs.get('color_space', "sRGB")
        use_hsl = kwargs.get('use_hsl', False)
        hsl_channel_width = kwargs.get('hsl_channel_width', 50)
        hsl_skin_protection = kwargs.get('hsl_skin_protection', True)
        use_shade_detailer = kwargs.get('use_shade_detailer', False)
        shade_strength = kwargs.get('shade_strength', 0.5)
        use_clarity = kwargs.get('use_clarity', False)
        clarity_strength = kwargs.get('clarity_strength', 0.5)
        clarity_radius = kwargs.get('clarity_radius', 2.0)
        midtone_range = kwargs.get('midtone_range', 0.5)
        edge_preservation = kwargs.get('edge_preservation', 0.8)
        use_level_endpoints = kwargs.get('use_level_endpoints', False)
        black_offset = kwargs.get('black_offset', 0.0)
        white_offset = kwargs.get('white_offset', 0.0)
        skip_if_no_clip = kwargs.get('skip_if_no_clip', False)
        normalize_gaps = kwargs.get('normalize_gaps', False)
        dither_quantization = kwargs.get('dither_quantization', False)
        adaptive_dither_strength = kwargs.get('adaptive_dither_strength', False)
        error_diffusion = kwargs.get('error_diffusion', False)
        show_histogram = kwargs.get('show_histogram', False)
        histogram_source = kwargs.get('histogram_source', False)
        histogram_channel = kwargs.get('histogram_channel', "RGB")
        histogram_style = kwargs.get('histogram_style', "bars")
        node_id = kwargs.get('id', None)

        pil_img = utility.tensor_to_image(image)
        pil_img_input = pil_img.copy()

        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}

        if auto_normalize:
            pil_img = img_levels_auto.img_levels_auto(image=pil_img, auto_normalize=auto_normalize, threshold=auto_levels_threshold, normalize_gaps=normalize_gaps, normalize_midpeaks=False, peak_width=peak_width, auto_gamma=auto_gamma, gamma_target=gamma_target, precision=precision, seed=seed)

        if use_white_balance and (wb_temperature != 6500 or wb_tint != 0):
            pil_img = img_white_balance.img_white_balance(image=pil_img, temperature=wb_temperature, tint=wb_tint)

        if use_smart_lighting and smart_lighting != 0:
            pil_img = img_smart_lighting.img_smart_lighting(image=pil_img, intensity=smart_lighting)

        if use_dehaze and strength > 0:
            pil_img = img_dehaze.img_dehaze(image=pil_img, strength=strength, radius=dehaze_radius, omega=omega, t0=t0, contrast=dehaze_contrast, precision=precision)

        if use_depth_blur and focus_depth > 0 and max_blur > 0:
            pil_img = img_depth_blur.img_depth_blur(image=pil_img, focus_depth=focus_depth, depth_range=depth_range, max_blur=max_blur, depth_gamma=depth_gamma, auto_optimize=auto_optimize, use_v3=use_DA_v3)

        if use_blur and blur_intensity != 0:
            pil_img = img_blur.img_blur(image=pil_img, blur_type=blur_type, intensity=blur_intensity, radius=blur_radius, angle=angle, edge_only=blur_edge_only, bilateral_edge_sensitivity=bilateral_edge_sensitivity, edge_threshold=edge_threshold)

        if use_brightness_contrast and (brightness != 0 or contrast != 0):
            pil_img = img_brightness_contrast.img_brightness_contrast(image=pil_img, brightness=brightness, contrast=contrast, use_legacy=use_legacy)

        if use_frequency_separation:
            pil_img = img_frequency_separation.img_frequency_separation(image=pil_img, radius=radius, low_freq_strength=low_freq_strength, high_freq_strength=high_freq_strength, blend_mode=blend_mode)

        if use_local_laplacian:
            pil_img = img_local_laplacian.img_local_laplacian(image=pil_img, sigma=sigma, contrast=laplacian_contrast, detail=detail, levels=levels)

        if film_type != "All":
            allowed_presets = self.FILM_PRESETS_BY_TYPE.get(film_type, [])
            if allowed_presets and film_rendering not in allowed_presets:
                film_rendering = allowed_presets[0]
        if use_film_rendering and film_rendering_intensity != 0:
            pil_img = img_film_rendering.img_film_rendering(image=pil_img, rendering=film_rendering, intensity=film_rendering_intensity, add_grain=iso_grain, add_halation=halation, expiration_years=expiration_years)

        if use_lut and lut_file != "None":
            lut_path = os.path.join(self.LUT_DIR, lut_file)
            pil_img = img_lut3d.img_lut3d(image=pil_img, lut_path=lut_path, intensity=intensity, input_space=color_space, output_space=color_space)

        if use_filmic:
            pil_img = img_filmic_curve.img_filmic_curve(image=pil_img, curve_type=curve_type, contrast=filmic_contrast, highlight_rolloff=highlight_rolloff, shadow_lift=shadow_lift, pivot=pivot)

        if use_photo_paper and paper_intensity != 0:
            pil_img = img_photo_paper.img_photo_paper(image=pil_img, paper_type=photo_paper, color_paper=color_paper, paper_base=paper_base, paper_intensity=paper_intensity, expiration_years=paper_expiration_years)

        st_data = rasterix_data.get('selective_tone', {})
        if use_selective_tone and st_data:
            pil_img = img_selective_tone.img_selective_tone(image=pil_img, channels_data=st_data, separation=selective_tone_separation, strength=selective_tone_strength)

        cb_data = rasterix_data.get('color_balance', {})
        if use_color_balance and cb_data:
            pil_img = img_color_balance.img_color_balance(image=pil_img, channels_data=cb_data, preserve_luminosity=color_balance_preserve_luminosity, separation=color_balance_separation)

        hs_data = rasterix_data.get('hue_saturation', {})
        if use_hsl and hs_data:
            pil_img = img_hue_saturation.img_hue_saturation(image=pil_img, channels_data=hs_data, channel_width=hsl_channel_width, skin_protection=hsl_skin_protection)

        shade_data = rasterix_data.get('shade', {})
        if use_shade_detailer and shade_data:
            for mode, vals in shade_data.items():
                lvl = vals.get('shade_level', 0)
                if lvl != 0:
                    rad = vals.get('shade_radius', 0)
                    pil_img = img_shade_level.img_shade_level(image=pil_img, shade_level=lvl, radius=rad, strength=shade_strength)

        if use_clarity and strength != 0:
            pil_img = img_clarity.img_clarity(image=pil_img, strength=clarity_strength, radius=clarity_radius, midtone_range=midtone_range, edge_preservation=edge_preservation, precision=precision)

        if use_level_endpoints and (black_offset != 0 or white_offset != 0):
            pil_img = img_levels_compress.img_levels_compress(image=pil_img, black_offset=black_offset, white_offset=white_offset, skip_if_no_clip=skip_if_no_clip, high_precision=precision)

        if dither_quantization or error_diffusion or normalize_midpeaks:
            pil_img = img_dithering.img_dithering(image=pil_img, dither_quantization=dither_quantization, adaptive_dither_strength=adaptive_dither_strength, error_diffusion=error_diffusion, normalize_midpeaks=normalize_midpeaks, peak_width=peak_width, high_precision=precision, seed=seed)

        histogram.rasterix_hist_cache_store(pil_img_input, pil_img, precision, node_id=node_id)
        if show_histogram:
            histogram.rasterix_hist_cache_store(pil_img_input, pil_img, precision, node_id=node_id)
            active_hist = histogram.rasterix_hist_render_selected(pil_img_input, pil_img, precision, histogram_source, histogram_channel, histogram_style, node_id=node_id)
            suffix      = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))
            temp_file   = f"rasterix_hist_{suffix}.png"
            active_hist.save(os.path.join(folder_paths.temp_directory, temp_file), compress_level=1)
            return {"ui": {"images": [{"filename": temp_file, "subfolder": "", "type": "temp"}], "active_concept": [active_display]}, "result": (utility.image_to_tensor(pil_img),), }
        else:
            INVALID_IMAGE_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'images')
            INVALID_IMAGE = os.path.join(INVALID_IMAGE_PATH, "No_histogram_08.jpg")
            images = utility.ImageLoaderFromPath(INVALID_IMAGE)
            r1 = random.randint(1000, 9999)
            temp_filename = f"Primere_ComfyUI_{r1}.png"
            os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
            TEMP_FILE = os.path.join(folder_paths.get_temp_directory(), temp_filename)
            utility.tensor_to_image(images[0]).save(TEMP_FILE)
            return {"ui": {"images": [{"filename": temp_filename, "subfolder": "", "type": "temp"}], "active_concept": [active_display]}, "result": (utility.image_to_tensor(pil_img),),}

class PrimereAutoNormalize:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_auto_normalize"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),
                "auto_normalize": ("BOOLEAN", {"default": False, "label_off": "No auto levels", "label_on": "Apply auto levels"}),
                "auto_levels_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "auto_gamma": ("BOOLEAN", {"default": False, "label_on": "Auto gamma: ON", "label_off": "Auto gamma:: OFF"}),
                "gamma_target": ("FLOAT", {"default": 128.0, "min": 0.0, "max": 255.0, "step": 0.1}),
                "normalize_gaps": ("BOOLEAN", {"default": False, "label_on": "Anti-comb filter: ON", "label_off": "Anti-comb filter: OFF"}),
                "normalize_midpeaks": ("BOOLEAN", {"default": False, "label_on": "Anti-spike filter: ON", "label_off": "Anti-spike filter: OFF"}),
                "peak_width": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": utility.MAX_SEED, "forceInput": True}),
            }
        }

    def primere_auto_normalize(self, image, precision, auto_normalize, auto_levels_threshold, auto_gamma, gamma_target, normalize_gaps, normalize_midpeaks, peak_width, seed = None):
        pil_img = utility.tensor_to_image(image)
        if auto_normalize:
            pil_img = img_levels_auto.img_levels_auto(image=pil_img, auto_normalize=auto_normalize, threshold=auto_levels_threshold, normalize_gaps=normalize_gaps, normalize_midpeaks=normalize_midpeaks, peak_width=peak_width, auto_gamma=auto_gamma, gamma_target=gamma_target, precision=precision, seed=seed)
        return (utility.image_to_tensor(pil_img),)


class PrimereWhiteBalance:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_white_balance"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_white_balance": ("BOOLEAN", {"default": False, "label_off": "Ignore white balance", "label_on": "Apply white balance"}),
                "wb_temperature": ("FLOAT", {"default": 6500, "min": 2000, "max": 12000, "step": 100}),
                "wb_tint": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    def primere_white_balance(self, image, use_white_balance, wb_temperature, wb_tint):
        pil_img = utility.tensor_to_image(image)
        if use_white_balance and (wb_temperature != 6500 or wb_tint != 0):
            pil_img = img_white_balance.img_white_balance(image=pil_img, temperature=wb_temperature, tint=wb_tint)
        return (utility.image_to_tensor(pil_img),)


class PrimereSmartLighting:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_smart_lighting"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_smart_lighting": ("BOOLEAN", {"default": False, "label_off": "Ignore smart lightning", "label_on": "Apply smart lightning"}),
                "smart_lighting": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    def primere_smart_lighting(self, image, use_smart_lighting, smart_lighting):
        pil_img = utility.tensor_to_image(image)
        if use_smart_lighting and smart_lighting != 0:
            pil_img = img_smart_lighting.img_smart_lighting(image=pil_img, intensity=smart_lighting)
        return (utility.image_to_tensor(pil_img),)


class PrimereBlur:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_blur"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_blur": ("BOOLEAN", {"default": False, "label_off": "Ignore blur", "label_on": "Apply blur"}),
                "blur_type": (["gaussian", "box", "motion", "bilateral", "lens"], {"default": "bilateral"}),
                "blur_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 50.0, "step": 0.5}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "bilateral_edge_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_edge_only": ("BOOLEAN", {"default": False, "label_off": "Full image blur", "label_on": "Flat areas only, edges protected"}),
                "edge_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def primere_blur(self, image, use_blur, blur_type, blur_intensity, blur_radius, angle, bilateral_edge_sensitivity, blur_edge_only, edge_threshold):
        pil_img = utility.tensor_to_image(image)
        if use_blur and blur_intensity != 0:
            pil_img = img_blur.img_blur(image=pil_img, blur_type=blur_type, intensity=blur_intensity, radius=blur_radius, angle=angle, edge_only=blur_edge_only, bilateral_edge_sensitivity=bilateral_edge_sensitivity, edge_threshold=edge_threshold)
        return (utility.image_to_tensor(pil_img),)


class PrimereBrightnessContrast:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_brightness_contrast"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_brightness_contrast": ("BOOLEAN", {"default": False, "label_off": "Ignore brightness-contrast", "label_on": "Apply brightness-contrast"}),
                "brightness": ("FLOAT", {"default": 0, "min": -150, "max": 150, "step": 1}),
                "contrast": ("FLOAT", {"default": 0, "min": -50, "max": 100, "step": 1}),
                "use_legacy": ("BOOLEAN", {"default": False, "label_off": "Use non-linear shift", "label_on": "Use adaptive offset"}),
            }
        }

    def primere_brightness_contrast(self, image, use_brightness_contrast, brightness, contrast, use_legacy):
        pil_img = utility.tensor_to_image(image)
        if use_brightness_contrast and (brightness != 0 or contrast != 0):
            pil_img = img_brightness_contrast.img_brightness_contrast(image=pil_img, brightness=brightness, contrast=contrast, use_legacy=use_legacy)
        return (utility.image_to_tensor(pil_img),)


class PrimereFilmRendering:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_film_rendering"
    CATEGORY = TREE_RASTERIX
    FILM_TYPES = ["All", "CF", "BWF", "CCD", "MOB"]
    FILM_PRESETS_BY_TYPE = img_film_rendering.list_presets_by_type()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_film_rendering": ("BOOLEAN", {"default": False, "label_off": "Ignore film rendering", "label_on": "Apply film rendering"}),
                "film_type": (cls.FILM_TYPES, {"default": "All"}),
                "film_rendering": (list(FILM_PRESETS.keys()), {"default": list(FILM_PRESETS.keys())[0]}),
                "film_rendering_intensity": ("FLOAT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "iso_grain": ("BOOLEAN", {"default": False, "label_off": "Ignore ISO grain", "label_on": "Add ISO grain"}),
                "halation": ("BOOLEAN", {"default": False, "label_off": "Ignore halation", "label_on": "Add halation"}),
                "expiration_years": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1}),
            }
        }

    def primere_film_rendering(self, image, film_type, use_film_rendering, film_rendering, film_rendering_intensity, iso_grain, halation, expiration_years):
        pil_img = utility.tensor_to_image(image)

        if film_type != "All":
            allowed_presets = self.FILM_PRESETS_BY_TYPE.get(film_type, [])
            if allowed_presets and film_rendering not in allowed_presets:
                film_rendering = allowed_presets[0]

        if use_film_rendering and film_rendering_intensity != 0:
            pil_img = img_film_rendering.img_film_rendering(image=pil_img, rendering=film_rendering, intensity=film_rendering_intensity, add_grain=iso_grain, add_halation=halation, expiration_years=expiration_years)

        return (utility.image_to_tensor(pil_img),)


class PrimereSelectiveTone:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_selective_tone"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_selective_tone": ("BOOLEAN", {"default": False, "label_off": "Ignore selective tone", "label_on": "Apply selective tone"}),
                "selective_tone_value": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "selective_tone_zone": (["highlights", "midtones", "shadows", "blacks"], {"default": "midtones"}),
                "selective_tone_separation": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "selective_tone_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def primere_selective_tone(self, image, use_selective_tone, selective_tone_value, selective_tone_zone, selective_tone_separation, selective_tone_strength):
        pil_img = utility.tensor_to_image(image)
        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}
        st_data = rasterix_data.get('selective_tone', {})
        if use_selective_tone and st_data:
            pil_img = img_selective_tone.img_selective_tone(image=pil_img, channels_data=st_data, separation=selective_tone_separation, strength=selective_tone_strength)
        return (utility.image_to_tensor(pil_img),)


class PrimereColorBalance:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_color_balance"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_color_balance": ("BOOLEAN", {"default": False, "label_off": "Ignore color balance", "label_on": "Apply color balance"}),
                "color_balance_cyan_red": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "color_balance_magenta_green": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "color_balance_yellow_blue": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "color_balance_tone": (["highlights", "midtones", "shadows"], {"default": "midtones"}),
                "color_balance_preserve_luminosity": ("BOOLEAN", {"default": False, "label_off": "Modify luminosity", "label_on": "Restore original luminosity"}),
                "color_balance_separation": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 1}),
            }
        }

    def primere_color_balance(self, image, use_color_balance, color_balance_cyan_red, color_balance_magenta_green, color_balance_yellow_blue, color_balance_tone, color_balance_preserve_luminosity, color_balance_separation):
        pil_img = utility.tensor_to_image(image)
        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}
        cb_data = rasterix_data.get('color_balance', {})
        if use_color_balance and cb_data:
            pil_img = img_color_balance.img_color_balance(image=pil_img, channels_data=cb_data, preserve_luminosity=color_balance_preserve_luminosity, separation=color_balance_separation)
        return (utility.image_to_tensor(pil_img),)


class PrimereHSL:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_hsl"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_hsl": ("BOOLEAN", {"default": False, "label_off": "Ignore HSL", "label_on": "Apply HSL"}),
                "hsl_hue": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "hsl_saturation": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "hsl_lightness": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "hsl_vibrance": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "hsl_channel": (["master", "red", "green", "blue"], {"default": "master"}),
                "hsl_channel_width": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "hsl_skin_protection": ("BOOLEAN", {"default": True, "label_off": "Vibrance affects skin tones", "label_on": "Skin tones protected from vibrance"}),
            }
        }

    def primere_hsl(self, image, use_hsl, hsl_hue, hsl_saturation, hsl_lightness, hsl_vibrance, hsl_channel, hsl_channel_width, hsl_skin_protection):
        pil_img = utility.tensor_to_image(image)
        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}
        hs_data = rasterix_data.get('hue_saturation', {})
        if use_hsl and hs_data:
            pil_img = img_hue_saturation.img_hue_saturation(image=pil_img, channels_data=hs_data, channel_width=hsl_channel_width, skin_protection=hsl_skin_protection)
        return (utility.image_to_tensor(pil_img),)


class PrimereShadeDetailer:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_shade_detailer"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_shade_detailer": ("BOOLEAN", {"default": False, "label_off": "Ignore shade detailer", "label_on": "Apply shade detailer"}),
                "shade_level": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shade_radius": ("FLOAT", {"default": 0, "min": 0, "max": 50, "step": 0.5}),
                "detail_mode": (["fine", "medium", "broad"], {"default": "medium"}),
                "shade_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def primere_shade_detailer(self, image, use_shade_detailer, shade_level, shade_radius, detail_mode, shade_strength):
        pil_img = utility.tensor_to_image(image)
        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}
        shade_data = rasterix_data.get('shade', {})
        if use_shade_detailer and shade_data:
            for mode, vals in shade_data.items():
                lvl = vals.get('shade_level', 0)
                if lvl != 0:
                    rad = vals.get('shade_radius', 0)
                    pil_img = img_shade_level.img_shade_level(image=pil_img, shade_level=lvl, radius=rad, strength=shade_strength)

        return (utility.image_to_tensor(pil_img),)


class PrimereLevelEndpoints:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_level_endpoints"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),
                "use_level_endpoints": ("BOOLEAN", {"default": False, "label_off": "Ignore endpoint offset", "label_on": "Apply endpoint offset"}),
                "black_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "white_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 25.0, "step": 0.1}),
                "skip_if_no_clip": ("BOOLEAN", {"default": False, "label_off": "Offset all values", "label_on": "Skip if no clips"}),
            }
        }

    def primere_level_endpoints(self, image, precision, use_level_endpoints, black_offset, white_offset, skip_if_no_clip):
        pil_img = utility.tensor_to_image(image)
        if use_level_endpoints and (black_offset != 0 or white_offset != 0):
            pil_img = img_levels_compress.img_levels_compress(image=pil_img, black_offset=black_offset, white_offset=white_offset, skip_if_no_clip=skip_if_no_clip, high_precision=precision)

        return (utility.image_to_tensor(pil_img),)

class PrimerePosterize:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_posterize"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_posterize": ("BOOLEAN", {"default": False, "label_off": "Ignore posterize", "label_on": "Apply posterize"}),
                "shades": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
                "channels": (["Red", "Green", "Blue"], {"default": "Red"}),
            }
        }

    def primere_posterize(self, image, use_posterize, shades, channels):
        pil_img = utility.tensor_to_image(image)
        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}
        poster_data = rasterix_data.get('posterize', {})
        if use_posterize and poster_data:
            pil_img = img_posterize.img_posterize(image=pil_img, channels_data=poster_data)
        return (utility.image_to_tensor(pil_img),)

class PrimereDithering:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_dithering"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),
                "normalize_midpeaks": ("BOOLEAN", {"default": False, "label_on": "Anti-spike filter: ON", "label_off": "Anti-spike filter: OFF"}),
                "peak_width": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "dither_quantization": ("BOOLEAN", {"default": False, "label_off": "Dither quantization OFF", "label_on": "Dither quantization ON"}),
                "adaptive_dither_strength": ("BOOLEAN", {"default": False, "label_off": "Keep dither strength", "label_on": "Increase dither strength"}),
                "error_diffusion": ("BOOLEAN", {"default": False, "label_off": "Error diffusion OFF", "label_on": "Error diffusion ON"}),
            }
        }

    def primere_dithering(self, image, precision, normalize_midpeaks, peak_width, dither_quantization, adaptive_dither_strength, error_diffusion):
        pil_img = utility.tensor_to_image(image)
        if dither_quantization or error_diffusion or normalize_midpeaks:
            pil_img = img_dithering.img_dithering(image=pil_img, dither_quantization=dither_quantization, adaptive_dither_strength=adaptive_dither_strength, error_diffusion=error_diffusion, normalize_midpeaks=normalize_midpeaks, peak_width=peak_width, high_precision=precision)

        return (utility.image_to_tensor(pil_img),)


class PrimereAIDetectionBypasser:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_ai_detection_bypasser"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_ai_detection_bypasser": ("BOOLEAN", {"default": False, "label_off": "AI detection bypass off", "label_on": "AI detection bypass on"}),
                "adb_freq_strength": ("FLOAT", {"default": 0.019, "min": 0.0, "max": 0.1, "step": 0.001}),
                "adb_variance_strength": ("FLOAT", {"default": 0.32, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adb_unsharp_percent": ("INT", {"default": 38, "min": 0, "max": 150, "step": 1}),
                "adb_jpeg_cycles": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1}),
            }
        }

    def primere_ai_detection_bypasser(self, image, use_ai_detection_bypasser, adb_freq_strength, adb_variance_strength, adb_unsharp_percent, adb_jpeg_cycles):
        pil_img = utility.tensor_to_image(image)
        if use_ai_detection_bypasser:
            pil_img = isgen_detect_ext_full.bypass_ai_detector(image=pil_img, freq_strength=adb_freq_strength, variance_strength=adb_variance_strength, unsharp_percent=adb_unsharp_percent, jpeg_cycles=adb_jpeg_cycles)

        return (utility.image_to_tensor(pil_img),)

class PrimereRasterixGrain:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_rasterix_grain"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE", {"forceInput": True}),

                "use_grain": ("BOOLEAN", {"default": False, "label_off": "Ignore grain", "label_on": "Apply grain"}),
                "intensity":          ("FLOAT", {"default": 20.0, "min": 0.0,  "max": 100.0, "step": 0.5}),
                "grain_size":         ("FLOAT", {"default": 1.0,  "min": 0.5,  "max": 8.0,   "step": 0.1}),
                "grain_type":         (["gaussian", "organic", "salt_pepper", "fine"], {"default": "gaussian"}),
                "color_mode":         (["color", "monochrome"], {"default": "color"}),
                "color_tint":         (["neutral", "warm", "cool", "green", "custom"], {"default": "neutral"}),
                "color_tint_r":       ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "color_tint_g":       ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "color_tint_b":       ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "shadow_strength":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "highlight_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 3.0, "step": 0.05}),
                "midtone_peak":       ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "vignette_boost":     ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": utility.MAX_SEED, "forceInput": True}),
            }
        }

    def primere_rasterix_grain(self, image, use_grain, intensity, grain_size, grain_type, color_mode, color_tint, color_tint_r, color_tint_g, color_tint_b, shadow_strength, highlight_strength, midtone_peak, vignette_boost, seed=None):
        if intensity == 0 or use_grain == False:
            return (image,)

        pil_img = utility.tensor_to_image(image)
        pil_img = img_film_grain.img_film_grain(
            image=pil_img,
            intensity=intensity,
            grain_size=grain_size,
            grain_type=grain_type,
            color_mode=color_mode,
            color_tint=color_tint,
            color_tint_rgb=(color_tint_r, color_tint_g, color_tint_b),
            shadow_strength=shadow_strength,
            highlight_strength=highlight_strength,
            midtone_peak=midtone_peak,
            vignette_boost=vignette_boost,
            seed=seed,
        )
        return (utility.image_to_tensor(pil_img),)


class PrimereRasterixLens:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_rasterix_lens"
    CATEGORY = TREE_RASTERIX

    SECTION_TITLES = [
        {"before": "use_vignette", "name": "lensfx_vignette", "title": "🌑 Vignette Control", "color": "#4C5E70", "text_color": "#EAF1F8", "label": "Master toggle and parameters for vignette: darken corners/edges with strength, radius, feather, and shape. Inspired by Adobe Lightroom."},
        {"before": "use_chroma", "name": "lensfx_chromatic", "title": "🌈 Chromatic Aberration", "color": "#6A4C70", "text_color": "#EAF1F8", "label": "Toggle and control chromatic aberration: intensity, falloff, and fringe color for realistic color fringing. Inspired by Adobe Camera Raw."},
        {"before": "use_bokeh", "name": "lensfx_bokeh", "title": "✨ Bokeh Effect", "color": "#6A5636", "text_color": "#EAF1F8", "label": "Enable bokeh simulation: radius, blade count, highlight boost, and cat-eye shaping for dreamy out-of-focus highlights. Inspired by Adobe Photoshop."},
        {"before": "use_distortion", "name": "lensfx_distortion", "title": "🌀 Lens Distortion", "color": "#3E5C4B", "text_color": "#EAF1F8", "label": "Toggle barrel/pincushion/zoom distortion: simulate classic lens imperfections with fine-grained controls. Inspired by DxO PhotoLab."},
        {"before": "use_flare", "name": "lensfx_flare", "title": "☀️ Lens Flare", "color": "#705C4C", "text_color": "#EAF1F8", "label": "Activate realistic lens flare: intensity, position, streak/ghost count, length, and color tinting. Inspired by Adobe Photoshop."},
        {"before": "use_halation", "name": "lensfx_halation", "title": "🌟 Halation Glow", "color": "#4C705E", "text_color": "#EAF1F8", "label": "Toggle halation: glow around bright areas with intensity, radius, threshold, and warmth adjustment. Inspired by Blackmagic DaVinci Resolve."},
        {"before": "use_focus", "name": "lensfx_focus", "title": "🔍 Focus Falloff", "color": "#5C4C70", "text_color": "#EAF1F8", "label": "Enable selective focus blur: radius, mode (horizontal/vertical/radial/oval), position, width, and feather. Inspired by Adobe Photoshop."},
        {"before": "use_spherical", "name": "lensfx_spherical", "title": "🌐 Spherical Aberration", "color": "#704C5E", "text_color": "#EAF1F8", "label": "Toggle spherical aberration: intensity, radius, and zone (centre/edge/global) for soft-focus effects."},
        {"before": "use_anamorphic", "name": "lensfx_anamorphic", "title": "📽️ Anamorphic Lens", "color": "#4C706A", "text_color": "#EAF1F8", "label": "Apply anamorphic characteristics: intensity, streak color/length, oval bokeh, and blue bias for cinematic look. Inspired by Blackmagic DaVinci Resolve."},
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),

                "use_vignette":      ("BOOLEAN", {"default": False, "label_off": "Ignore vignette", "label_on": "Apply vignette"}),
                "vignette_strength": ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 0.01}),
                "vignette_radius":   ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "vignette_feather":  ("FLOAT", {"default": 0.4,  "min": 0.0, "max": 1.0,  "step": 0.01}),
                "vignette_shape":    (["circular", "oval", "corner"], {"default": "circular"}),

                "use_chroma":          ("BOOLEAN", {"default": False, "label_off": "Ignore chromatic aberration", "label_on": "Apply chromatic aberration"}),
                "chroma_intensity":    ("FLOAT", {"default": 2.0, "min": 0.0,  "max": 10.0, "step": 0.1}),
                "chroma_falloff":      ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "chroma_fringe_color": (["red_blue", "green_magenta", "yellow_purple"], {"default": "red_blue"}),

                "use_bokeh":             ("BOOLEAN", {"default": False, "label_off": "Ignore bokeh", "label_on": "Apply bokeh"}),
                "bokeh_radius":          ("FLOAT", {"default": 8.0, "min": 0.0, "max": 40.0, "step": 0.5}),
                "bokeh_blades":          ("INT",   {"default": 0,   "min": 0,   "max": 12,   "step": 1}),
                "bokeh_highlight_boost": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "bokeh_cat_eye":         ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.01}),

                "use_distortion":        ("BOOLEAN", {"default": False, "label_off": "Ignore lens distortion", "label_on": "Apply lens distortion"}),
                "distortion_barrel":     ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distortion_pincushion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distortion_zoom":       ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),

                "use_flare":          ("BOOLEAN", {"default": False, "label_off": "Ignore lens flare", "label_on": "Apply lens flare"}),
                "flare_intensity":    ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "flare_pos_x":        ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "flare_pos_y":        ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "flare_streak_count": ("INT",   {"default": 6,   "min": 2,   "max": 12,   "step": 1}),
                "flare_streak_length":("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0,  "step": 0.01}),
                "flare_ghost_count":  ("INT",   {"default": 4,   "min": 0,   "max": 8,    "step": 1}),
                "flare_color":        (["warm", "cool", "neutral", "rainbow"], {"default": "warm"}),

                "use_halation":       ("BOOLEAN", {"default": False, "label_off": "Ignore halation", "label_on": "Apply halation"}),
                "halation_intensity": ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 0.01}),
                "halation_radius":    ("FLOAT", {"default": 15.0, "min": 2.0, "max": 50.0, "step": 0.5}),
                "halation_threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "halation_warmth":    ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0,  "step": 0.01}),

                "use_focus":         ("BOOLEAN", {"default": False, "label_off": "Ignore focus falloff", "label_on": "Apply focus falloff"}),
                "focus_blur_radius": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "focus_mode":        (["horizontal", "vertical", "radial", "oval"], {"default": "horizontal"}),
                "focus_pos":         ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "focus_width":       ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "focus_feather":     ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0,  "step": 0.01}),

                "use_spherical":       ("BOOLEAN", {"default": False, "label_off": "Ignore spherical aberration", "label_on": "Apply spherical aberration"}),
                "spherical_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "spherical_radius":    ("FLOAT", {"default": 3.0, "min": 0.5, "max": 15.0, "step": 0.5}),
                "spherical_zone":      (["centre", "edge", "global"], {"default": "centre"}),

                "use_anamorphic":           ("BOOLEAN", {"default": False, "label_off": "Ignore anamorphic", "label_on": "Apply anamorphic"}),
                "anamorphic_intensity":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anamorphic_streak_color":  (["blue", "warm", "white"], {"default": "blue"}),
                "anamorphic_streak_length": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anamorphic_oval_bokeh":    ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anamorphic_blue_bias":     ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def primere_rasterix_lens(self, image, use_vignette, vignette_strength, vignette_radius, vignette_feather, vignette_shape, use_chroma, chroma_intensity, chroma_falloff, chroma_fringe_color, use_bokeh, bokeh_radius, bokeh_blades, bokeh_highlight_boost, bokeh_cat_eye, use_distortion, distortion_barrel, distortion_pincushion, distortion_zoom, use_flare, flare_intensity, flare_pos_x, flare_pos_y, flare_streak_count, flare_streak_length, flare_ghost_count, flare_color, use_halation, halation_intensity, halation_radius, halation_threshold, halation_warmth, use_focus, focus_blur_radius, focus_mode, focus_pos, focus_width, focus_feather, use_spherical, spherical_intensity, spherical_radius, spherical_zone, use_anamorphic, anamorphic_intensity, anamorphic_streak_color, anamorphic_streak_length, anamorphic_oval_bokeh, anamorphic_blue_bias):
        pil_img = utility.tensor_to_image(image)
        pil_img = img_lens_effects.img_lens_effect(
            image=pil_img,
            vignette_strength=vignette_strength if use_vignette else 0,
            vignette_radius=vignette_radius,
            vignette_feather=vignette_feather,
            vignette_shape=vignette_shape,
            chroma_intensity=chroma_intensity if use_chroma else 0,
            chroma_falloff=chroma_falloff,
            chroma_fringe_color=chroma_fringe_color,
            bokeh_radius=bokeh_radius if use_bokeh else 0,
            bokeh_blades=bokeh_blades,
            bokeh_highlight_boost=bokeh_highlight_boost,
            bokeh_cat_eye=bokeh_cat_eye,
            distortion_barrel=distortion_barrel if use_distortion else 0,
            distortion_pincushion=distortion_pincushion if use_distortion else 0,
            distortion_zoom=distortion_zoom,
            flare_intensity=flare_intensity if use_flare else 0,
            flare_pos_x=flare_pos_x,
            flare_pos_y=flare_pos_y,
            flare_streak_count=flare_streak_count,
            flare_streak_length=flare_streak_length,
            flare_ghost_count=flare_ghost_count,
            flare_color=flare_color,
            halation_intensity=halation_intensity if use_halation else 0,
            halation_radius=halation_radius,
            halation_threshold=halation_threshold,
            halation_warmth=halation_warmth,
            focus_blur_radius=focus_blur_radius if use_focus else 0,
            focus_mode=focus_mode,
            focus_pos=focus_pos,
            focus_width=focus_width,
            focus_feather=focus_feather,
            spherical_intensity=spherical_intensity if use_spherical else 0,
            spherical_radius=spherical_radius,
            spherical_zone=spherical_zone,
            anamorphic_intensity=anamorphic_intensity if use_anamorphic else 0,
            anamorphic_streak_color=anamorphic_streak_color,
            anamorphic_streak_length=anamorphic_streak_length,
            anamorphic_oval_bokeh=anamorphic_oval_bokeh,
            anamorphic_blue_bias=anamorphic_blue_bias,
        )
        return (utility.image_to_tensor(pil_img),)

class PrimereHistogram:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_histogram"
    CATEGORY = TREE_RASTERIX
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),
                "show_histogram": ("BOOLEAN", {"default": False, "label_off": "Ignore histogram", "label_on": "Create histogram"}),
                "histogram_channel": (["RGB", "RED", "GREEN", "BLUE"], {"default": "RGB"}),
                "histogram_style": (["bars", "lines", "waveform", "heatmap", "stacked", "luma", "parade", "gradient", "glow", "dots", "step", "log", "percentile", "inverse"], {"default": "bars"}),
            },
            "hidden": {
                "id": "UNIQUE_ID",
            }
        }

    def primere_histogram(self, image, precision, show_histogram=False, histogram_channel="RGB", histogram_style="bars", id=None):
        pil_img = utility.tensor_to_image(image)
        pil_img_input = pil_img.copy()

        histogram.rasterix_hist_cache_store(pil_img_input, pil_img, precision, node_id=id)

        if show_histogram:
            histogram.rasterix_hist_cache_store(pil_img_input, pil_img, precision, node_id=id)
            active_hist = histogram.rasterix_hist_render_selected(pil_img_input, pil_img, precision, True, histogram_channel, histogram_style, node_id=id)
            suffix = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))
            temp_file = f"rasterix_hist_{suffix}.png"
            active_hist.save(os.path.join(folder_paths.temp_directory, temp_file), compress_level=1)
            return {"ui": {"images": [{"filename": temp_file, "subfolder": "", "type": "temp"}]}, "result": (utility.image_to_tensor(pil_img),), }
        else:
            INVALID_IMAGE_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'images')
            INVALID_IMAGE = os.path.join(INVALID_IMAGE_PATH, "No_histogram_08.jpg")
            images = utility.ImageLoaderFromPath(INVALID_IMAGE)
            r1 = random.randint(1000, 9999)
            temp_filename = f"Primere_ComfyUI_{r1}.png"
            os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
            TEMP_FILE = os.path.join(folder_paths.get_temp_directory(), temp_filename)
            utility.tensor_to_image(images[0]).save(TEMP_FILE)
            return {"ui": {"images": [{"filename": temp_filename, "subfolder": "", "type": "temp"}]}, "result": (utility.image_to_tensor(pil_img),),}

class PrimereSolarizationBW:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_solarization_bw"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_solarization": ("BOOLEAN", {"default": False, "label_off": "Ignore solarization", "label_on": "Apply solarization"}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),

                "color_mode": ("BOOLEAN", {"default": False, "label_off": "Keep unchanged", "label_on": "Force B&W"}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 0.5, "step": 0.01}),
                "edge_boost": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05}),
                "edge_radius": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01}),
                "hard_paper": ("BOOLEAN", {"default": False, "label_off": "Soft paper", "label_on": "Hard paper"}),
                "grain_modulation": ("BOOLEAN", {"default": False, "label_off": "No grain modulation", "label_on": "Grain-modulated inversion"}),
                "grain_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_scale": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.1}),

            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": utility.MAX_SEED, "forceInput": True}),
            }
        }

    def primere_solarization_bw(self, image, color_mode, use_solarization, precision, strength, pivot, sigma, edge_boost, edge_radius, contrast, hard_paper, grain_modulation, grain_strength, grain_scale, seed = None):
        pil_img = utility.tensor_to_image(image)
        if use_solarization:
            pil_img = img_solarization_bw.img_solarization_bw(image=pil_img, color_mode=color_mode, strength=strength, pivot=pivot, sigma=sigma, edge_boost=edge_boost, edge_radius=edge_radius, contrast=contrast, precision=precision, hard_paper=hard_paper, grain_modulation=grain_modulation, grain_strength=grain_strength, grain_scale=grain_scale, seed=seed)

        return (utility.image_to_tensor(pil_img),)

class PrimereClarity:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_clarity"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_clarity": ("BOOLEAN", {"default": False, "label_off": "Ignore clarity", "label_on": "Apply clarity"}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),

                "strength": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "midtone_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "edge_preservation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def primere_clarity(self, image, use_clarity, precision, strength, radius, midtone_range, edge_preservation):
        pil_img = utility.tensor_to_image(image)
        if use_clarity and strength != 0:
            pil_img = img_clarity.img_clarity(image=pil_img, strength=strength, radius=radius, midtone_range=midtone_range, edge_preservation=edge_preservation, precision=precision)

        return (utility.image_to_tensor(pil_img),)

class PrimereDehaze:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_dehaze"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_dehaze": ("BOOLEAN", {"default": False, "label_off": "Ignore dehaze", "label_on": "Apply dehaze"}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),

                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "radius": ("INT", {"default": 15, "min": 3, "max": 100, "step": 1}),
                "omega": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "t0": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01}),
            }
        }

    def primere_dehaze(self, image, use_dehaze, precision, strength, radius, omega, t0, contrast):
        pil_img = utility.tensor_to_image(image)
        if use_dehaze and strength > 0:
            pil_img = img_dehaze.img_dehaze(image=pil_img, strength=strength, radius=radius, omega=omega, t0=t0, contrast=contrast, precision=precision)

        return (utility.image_to_tensor(pil_img),)

class PrimereLocalLaplacian:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_local_laplacian"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_local_laplacian": ("BOOLEAN", {"default": False, "label_off": "Ignore local laplacian", "label_on": "Apply local laplacian"}),

                "sigma": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.01}),
                "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "levels": ("INT", {"default": 8, "min": 4, "max": 32, "step": 1}),
            }
        }

    def primere_local_laplacian(self, image, use_local_laplacian, sigma, contrast, detail, levels):
        pil_img = utility.tensor_to_image(image)
        if use_local_laplacian:
            pil_img = img_local_laplacian.img_local_laplacian(image=pil_img, sigma=sigma, contrast=contrast, detail=detail, levels=levels)

        return (utility.image_to_tensor(pil_img),)

class PrimereFrequencySeparation:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_frequency_separation"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_frequency_separation": ("BOOLEAN", {"default": False, "label_off": "Ignore frequency separation", "label_on": "Apply frequency separation"}),

                "radius": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.1}),
                "low_freq_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "high_freq_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "blend_mode": (["add", "multiply", "overlay"], {"default": "add"}),
            }
        }

    def primere_frequency_separation(self, image, use_frequency_separation, radius, low_freq_strength, high_freq_strength, blend_mode):
        pil_img = utility.tensor_to_image(image)
        if use_frequency_separation:
            pil_img = img_frequency_separation.img_frequency_separation(image=pil_img, radius=radius, low_freq_strength=low_freq_strength, high_freq_strength=high_freq_strength, blend_mode=blend_mode)

        return (utility.image_to_tensor(pil_img),)

class PrimereFilmicCurve:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_filmic_curve"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_filmic": ("BOOLEAN", {"default": False, "label_off": "Ignore filmic", "label_on": "Apply filmic"}),

                "curve_type": (["filmic", "log"], {"default": "filmic"}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "highlight_rolloff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_lift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def primere_filmic_curve(self, image, use_filmic, curve_type, contrast, highlight_rolloff, shadow_lift, pivot):
        pil_img = utility.tensor_to_image(image)
        if use_filmic:
            pil_img = img_filmic_curve.img_filmic_curve(image=pil_img, curve_type=curve_type, contrast=contrast, highlight_rolloff=highlight_rolloff, shadow_lift=shadow_lift, pivot=pivot)

        return (utility.image_to_tensor(pil_img),)

class PrimereLUT3D:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_lut3d"
    CATEGORY = TREE_RASTERIX

    LUT_DIR = os.path.join(PRIMERE_ROOT, 'components', 'images', 'luts')

    @classmethod
    def _list_luts(cls):
        lut_entries = ["None"]
        if not os.path.exists(cls.LUT_DIR):
            return lut_entries

        for f in sorted(os.listdir(cls.LUT_DIR)):
            full_path = os.path.join(cls.LUT_DIR, f)
            if os.path.isfile(full_path) and f.lower().endswith(".cube"):
                lut_entries.append(f)

        for d in sorted(os.listdir(cls.LUT_DIR)):
            subdir = os.path.join(cls.LUT_DIR, d)
            if os.path.isdir(subdir):
                for f in sorted(os.listdir(subdir)):
                    if f.lower().endswith(".cube"):
                        lut_entries.append(f"{d}/{f}")

        return lut_entries

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_lut": ("BOOLEAN", {"default": False, "label_off": "Ignore LUT", "label_on": "Apply LUT"}),

                "lut_file": (cls._list_luts(),),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_space": (["sRGB", "linear"], {"default": "sRGB"}),
            }
        }

    def primere_lut3d(self, image, use_lut, lut_file, intensity, color_space):
        pil_img = utility.tensor_to_image(image)
        if use_lut and lut_file != "None":
            lut_path = os.path.join(self.LUT_DIR, lut_file)
            pil_img = img_lut3d.img_lut3d(image=pil_img, lut_path=lut_path, intensity=intensity, input_space=color_space, output_space=color_space)

        return (utility.image_to_tensor(pil_img),)

class PrimereEdgeJitter:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_edge_jitter"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_edge_jitter": ("BOOLEAN", {"default": False, "label_off": "Ignore edge jitter", "label_on": "Apply edge jitter"}),
                "precision": ("BOOLEAN", {"default": False, "label_off": "8 bit", "label_on": "16 bit"}),

                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1}),
                "edge_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "randomness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": utility.MAX_SEED, "forceInput": True}),
            }
        }

    def primere_edge_jitter(self, image, use_edge_jitter, precision, strength, radius, edge_threshold, randomness, seed=None):
        pil_img = utility.tensor_to_image(image)
        if use_edge_jitter and strength > 0:
            pil_img = img_edge_jitter.img_edge_jitter(image=pil_img, strength=strength, radius=radius, edge_threshold=edge_threshold, randomness=randomness, seed=seed, precision=precision)

        return (utility.image_to_tensor(pil_img),)

class PrimereDepthBlur:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_depth_blur"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_depth_blur": ("BOOLEAN", {"default": False, "label_off": "Ignore depth blur", "label_on": "Apply depth blur"}),
                "auto_optimize": ("BOOLEAN", {"default": False, "label_off": "Use custom inputs", "label_on": "Optimize settings by focus"}),
                "use_DA_v3": ("BOOLEAN", {"default": False, "label_off": "Depth-anything V2", "label_on": "Depth-anything V3"}),
                "focus_depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_range": ("FLOAT", {"default": 0.200, "min": 0.001, "max": 1.000, "step": 0.001}),
                "max_blur": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "depth_gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),
            }
        }

    def primere_depth_blur(self, image, use_depth_blur, auto_optimize, use_DA_v3, focus_depth, depth_range, max_blur, depth_gamma):
        pil_img = utility.tensor_to_image(image)
        if use_depth_blur:
            pil_img = img_depth_blur.img_depth_blur(image=pil_img, focus_depth=focus_depth, depth_range=depth_range, max_blur=max_blur, depth_gamma=depth_gamma, auto_optimize=auto_optimize, use_v3=use_DA_v3)

        return (utility.image_to_tensor(pil_img),)

class PrimerePhotoPaper:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_photo_paper"
    CATEGORY = TREE_RASTERIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_photo_paper": ("BOOLEAN", {"default": False, "label_off": "Ignore photo paper", "label_on": "Apply photo paper"}),
                "photo_paper": (list(PAPER_PRESETS.keys()), {"default": list(PAPER_PRESETS.keys())[0]}),
                "color_paper": ("BOOLEAN", {"default": False, "label_off": "B&W paper", "label_on": "Color paper"}),
                "paper_base": (["RC", "FB"], {"default": "RC"}),
                "paper_expiration_years": ("FLOAT", {"default": 0, "min": 0, "max": 30, "step": 0.1}),
                "paper_intensity": ("FLOAT", {"default": 100, "min": 0, "max": 200, "step": 1}),
            }
        }

    def primere_photo_paper(self, image, use_photo_paper, photo_paper, color_paper, paper_base, paper_intensity, paper_expiration_years):
        pil_img = utility.tensor_to_image(image)
        if use_photo_paper and paper_intensity != 0:
            pil_img = img_photo_paper.img_photo_paper(image=pil_img, paper_type=photo_paper, color_paper=color_paper, paper_base=paper_base, paper_intensity=paper_intensity, expiration_years=paper_expiration_years)

        return (utility.image_to_tensor(pil_img),)