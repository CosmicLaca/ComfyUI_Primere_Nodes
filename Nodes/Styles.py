from ..components.tree import TREE_STYLES
from ..components.tree import PRIMERE_ROOT
import os
import tomli

class PrimereStylePile:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "styleple"
    CATEGORY = TREE_STYLES

    @staticmethod
    def get_all_styles(toml_path: str):
        with open(toml_path, "rb") as f:
            style_def_neg = tomli.load(f)
        return style_def_neg

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE = cls.get_all_styles(os.path.join(DEF_TOML_DIR, "stylepile.toml"))

        cls.ART_TYPES = STYLE['art-type']
        cls.CONCEPTS = ['None'] + sorted(STYLE['concepts']['concepts'])
        cls.ARTISTS = ['None'] + sorted(STYLE['artists']['artists'])
        cls.ART_MOVEMENTS = ['None'] + sorted(STYLE['art-movements']['art-movements'])
        cls.COLORS = ['None'] + sorted(STYLE['colors']['colors'])
        cls.DIRECTIONS = ['None'] + sorted(STYLE['directions']['directions'])
        cls.MOODS = ['None'] + sorted(STYLE['moods']['moods'])
        cls.MJSTYLES = ['None'] + sorted(STYLE['mjstyles']['mjstyles'])

        return {
            "required": {
                "art_type": (list(cls.ART_TYPES.keys()),),
                "art_type_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "color": (cls.COLORS,),
                "color_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mood": (cls.MOODS,),
                "mood_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "direction": (cls.DIRECTIONS,),
                "direction_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "concept": (cls.CONCEPTS,),
                "concept_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "artist": (cls.ARTISTS,),
                "artist_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "movement": (cls.ART_MOVEMENTS,),
                "movement_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "midjourney_styles": (cls.MJSTYLES,),
                "midjourney_styles_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

    def styleple(self, concept, concept_strength, art_type, art_type_strength, artist, artist_strength, movement, movement_strength, midjourney_styles, midjourney_styles_strength, color, color_strength, mood, mood_strength, direction, direction_strength, opt_pos_style = '', opt_neg_style = ''):
        opt_pos_style = f'({opt_pos_style})' if opt_pos_style is not None and opt_pos_style.strip(' ,;') != '' else ''
        opt_neg_style = f'({opt_neg_style})' if opt_neg_style is not None and opt_neg_style.strip(' ,;') != '' else ''

        art_type_positive = self.ART_TYPES[art_type]['positive']
        art_type_negative = self.ART_TYPES[art_type]['negative']
        art_type_pos_str = f'({art_type_positive}:{art_type_strength:.2f})' if art_type_positive is not None and art_type_positive != 'None' and art_type_positive.strip(' ,;') != '' else ''
        art_type_neg_str = f'({art_type_negative}:{art_type_strength:.2f})' if art_type_negative is not None and art_type_negative != 'None' and art_type_negative.strip(' ,;') != '' else ''

        color = f'({color}:{color_strength:.2f})' if color is not None and color != 'None' and color.strip(' ,;') != '' else ''
        mood = f'({mood}:{mood_strength:.2f})' if mood is not None and mood != 'None' and mood.strip(' ,;') != '' else ''
        direction = f'({direction}:{direction_strength:.2f})' if direction is not None and direction != 'None' and direction.strip(' ,;') != '' else ''
        concept = f'({concept}:{concept_strength:.2f})' if concept is not None and concept != 'None' and concept.strip(' ,;') != '' else ''
        artist = f'({artist}:{artist_strength:.2f})' if artist is not None and artist != 'None' and artist.strip(' ,;') != '' else ''
        movement = f'({movement}:{movement_strength:.2f})' if movement is not None and movement != 'None' and movement.strip(' ,;') != '' else ''
        midjourney_styles = f'({midjourney_styles}:{midjourney_styles_strength:.2f})' if midjourney_styles is not None and midjourney_styles != 'None' and midjourney_styles.strip(' ,;') != '' else ''

        positive_text = f'{opt_pos_style}, {art_type_pos_str}, {color}, {mood}, {direction}, {concept}, {artist}, {movement}, {midjourney_styles}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(", , ", ", ")
        negative_text = f'{opt_neg_style}, {art_type_neg_str}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ")

        return (positive_text, negative_text,)
