from ..components.tree import TREE_RASTERIX
from ..components.tree import PRIMERE_ROOT
import random
import folder_paths
import comfy.utils
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
from ..components import utility
from .Dashboard import PrimereModelConceptSelector as PrimereModelConceptSelector
import os

class PrimereRasterix:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "primere_rasterix"
    CATEGORY = TREE_RASTERIX
    OUTPUT_NODE = True

    MODELLIST = PrimereModelConceptSelector.MODELLIST
    CONCEPT_LIST =  PrimereModelConceptSelector.CONCEPT_LIST

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

                "use_film_rendering": ("BOOLEAN", {"default": False, "label_off": "Ignore film rendering", "label_on": "Apply film rendering"}),
                "film_rendering": (list(FILM_PRESETS.keys()), {"default": list(FILM_PRESETS.keys())[0]}),
                "film_rendering_intensity": ("FLOAT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "iso_grain": ("BOOLEAN", {"default": False, "label_off": "Ignore ISO grain", "label_on": "Add ISO grain"}),
                "halation": ("BOOLEAN", {"default": False, "label_off": "Ignore halation", "label_on": "Add halation"}),
                "expiration_years": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1}),

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

                "use_ai_detection_bypasser": ("BOOLEAN", {"default": False, "label_off": "AI detection bypass off", "label_on": "AI detection bypass on"}),
                "adb_freq_strength":     ("FLOAT", {"default": 0.019, "min": 0.0, "max": 0.1,  "step": 0.001}),
                "adb_variance_strength": ("FLOAT", {"default": 0.32,  "min": 0.0, "max": 1.0,  "step": 0.01}),
                "adb_unsharp_percent":   ("INT",   {"default": 38,    "min": 0,   "max": 150,  "step": 1}),
                "adb_jpeg_cycles":       ("INT",   {"default": 4,     "min": 0,   "max": 6,    "step": 1}),

                "show_histogram": ("BOOLEAN", {"default": False, "label_off": "Ignore histogram", "label_on": "Create histogram"}),
                "histogram_source":        ("BOOLEAN", {"default": False, "label_off": "Show output histogram", "label_on": "Show input histogram"}),
                "histogram_channel":    (["RGB", "RED", "GREEN", "BLUE"], {"default": "RGB"}),
                "histogram_style":      (["bars", "lines", "waveform", "heatmap", "stacked", "luma", "parade"], {"default": "bars"}),
            },
            "optional": {
                "model_concept": ("STRING", {"default": None, "forceInput": True}),
                "model_name": ("CHECKPOINT_NAME", {"default": None, "forceInput": True}),
            }
        }

    def primere_rasterix(self, concepts, models, image, precision, auto_normalize, auto_levels_threshold, normalize_midpeaks, peak_width, auto_gamma, gamma_target, use_white_balance, wb_temperature, wb_tint, use_blur, blur_type, blur_intensity, blur_radius, angle, bilateral_edge_sensitivity, blur_edge_only, edge_threshold, use_smart_lighting, smart_lighting, use_brightness_contrast, brightness, contrast, use_legacy, use_film_rendering, film_rendering, film_rendering_intensity, iso_grain, halation, expiration_years, use_selective_tone, selective_tone_value, selective_tone_zone, selective_tone_separation, selective_tone_strength, use_color_balance, color_balance_cyan_red, color_balance_magenta_green, color_balance_yellow_blue, color_balance_tone, color_balance_preserve_luminosity, color_balance_separation, use_hsl, hsl_hue, hsl_saturation, hsl_lightness, hsl_vibrance, hsl_channel, hsl_channel_width, hsl_skin_protection, use_shade_detailer, shade_level, shade_radius, detail_mode, shade_strength, use_ai_detection_bypasser, adb_freq_strength, adb_variance_strength, adb_unsharp_percent, adb_jpeg_cycles, use_level_endpoints,  black_offset, white_offset, skip_if_no_clip, normalize_gaps, dither_quantization, adaptive_dither_strength, error_diffusion, show_histogram=False, histogram_source=False, histogram_channel="RGB", histogram_style="gradient", model_concept=None, model_name=None):
        pil_img = utility.tensor_to_image(image)
        pil_img_input = pil_img.copy()

        rasterix_json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
        rasterix_data = utility.json2tuple(rasterix_json_path) or {}

        if auto_normalize:
            pil_img = img_levels_auto.img_levels_auto(image=pil_img, auto_normalize=auto_normalize, threshold=auto_levels_threshold, normalize_gaps=normalize_gaps, normalize_midpeaks=False, peak_width=peak_width, auto_gamma=auto_gamma, gamma_target=gamma_target, precision=precision)

        if use_white_balance and (wb_temperature != 6500 or wb_tint != 0):
            pil_img = img_white_balance.img_white_balance(image=pil_img, temperature=wb_temperature, tint=wb_tint)

        if use_blur and blur_intensity != 0:
            pil_img = img_blur.img_blur(image=pil_img, blur_type=blur_type, intensity=blur_intensity, radius=blur_radius, angle=angle, edge_only=blur_edge_only, bilateral_edge_sensitivity=bilateral_edge_sensitivity, edge_threshold=edge_threshold)

        if use_smart_lighting and smart_lighting != 0:
            pil_img = img_smart_lighting.img_smart_lighting(image=pil_img, intensity=smart_lighting)

        if use_brightness_contrast and (brightness != 0 or contrast != 0):
            pil_img = img_brightness_contrast.img_brightness_contrast(image=pil_img, brightness=brightness, contrast=contrast, use_legacy=use_legacy)

        if use_film_rendering and film_rendering_intensity != 0:
            pil_img = img_film_rendering.img_film_rendering(image=pil_img, rendering=film_rendering, intensity=film_rendering_intensity, add_grain=iso_grain, add_halation=halation, expiration_years=expiration_years)

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

        if use_level_endpoints and (black_offset != 0 or white_offset != 0):
            pil_img = img_levels_compress.img_levels_compress(image=pil_img, black_offset=black_offset, white_offset=white_offset, skip_if_no_clip=skip_if_no_clip, high_precision=precision)

        if dither_quantization or error_diffusion or normalize_midpeaks:
            pil_img = img_dithering.img_dithering(image=pil_img, dither_quantization=dither_quantization, adaptive_dither_strength=adaptive_dither_strength, error_diffusion=error_diffusion, normalize_midpeaks=normalize_midpeaks, peak_width=peak_width, high_precision=precision)

        if use_ai_detection_bypasser:
            pil_img = isgen_detect_ext_full.bypass_ai_detector(image=pil_img, freq_strength=adb_freq_strength, variance_strength=adb_variance_strength, unsharp_percent=adb_unsharp_percent, jpeg_cycles=adb_jpeg_cycles)

        if show_histogram:
            hist_dir  = os.path.join(PRIMERE_ROOT, 'front_end', 'images')
            rendered  = {}
            hstyle = ["bars", "lines", "waveform", "heatmap", "stacked", "luma", "parade"]
            hchannels = ["RGB", "RED", "GREEN", "BLUE"]
            pbar = comfy.utils.ProgressBar(len(hstyle) * len(hchannels))
            for st in hstyle:
                for ch in hchannels:
                    rendered[("in",  ch, st)] = histogram.rasterix_histogram_render(pil_img_input, ch, st, precision)
                    rendered[("out", ch, st)] = histogram.rasterix_histogram_render(pil_img,       ch, st, precision)
                    rendered[("in",  ch, st)].save(os.path.join(hist_dir, f'input_histogram_{ch.lower()}_{st}.jpg'),  quality=90)
                    rendered[("out", ch, st)].save(os.path.join(hist_dir, f'output_histogram_{ch.lower()}_{st}.jpg'), quality=90)
                    pbar.update(1)

            active_hist = rendered[("in" if histogram_source else "out", histogram_channel, histogram_style)]
            suffix      = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))
            temp_file   = f"rasterix_hist_{suffix}.png"
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
            }
        }

    def primere_auto_normalize(self, image, precision, auto_normalize, auto_levels_threshold, auto_gamma, gamma_target, normalize_gaps, normalize_midpeaks, peak_width):
        pil_img = utility.tensor_to_image(image)
        if auto_normalize:
            pil_img = img_levels_auto.img_levels_auto(image=pil_img, auto_normalize=auto_normalize, threshold=auto_levels_threshold, normalize_gaps=normalize_gaps, normalize_midpeaks=normalize_midpeaks, peak_width=peak_width, auto_gamma=auto_gamma, gamma_target=gamma_target, precision=precision)
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "use_film_rendering": ("BOOLEAN", {"default": False, "label_off": "Ignore film rendering", "label_on": "Apply film rendering"}),
                "film_rendering": (list(FILM_PRESETS.keys()), {"default": list(FILM_PRESETS.keys())[0]}),
                "film_rendering_intensity": ("FLOAT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "iso_grain": ("BOOLEAN", {"default": False, "label_off": "Ignore ISO grain", "label_on": "Add ISO grain"}),
                "halation": ("BOOLEAN", {"default": False, "label_off": "Ignore halation", "label_on": "Add halation"}),
                "expiration_years": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1}),
            }
        }

    def primere_film_rendering(self, image, use_film_rendering, film_rendering, film_rendering_intensity, iso_grain, halation, expiration_years):
        pil_img = utility.tensor_to_image(image)
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