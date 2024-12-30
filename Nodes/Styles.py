from ..components.tree import TREE_STYLES
from ..components.tree import PRIMERE_ROOT
import os
from ..components import utility
from ..components import stylehandler

class PrimereStylePile:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "stylepile"
    CATEGORY = TREE_STYLES

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE_FILE = os.path.join(DEF_TOML_DIR, "stylepile.toml")
        STYLE_RESULT = stylehandler.toml2node(STYLE_FILE)
        INPUT_DICT_FINAL = {'required': STYLE_RESULT[0]}
        cls.STYLE_PROMPTS_POS = STYLE_RESULT[1]
        cls.STYLE_PROMPTS_NEG = STYLE_RESULT[2]

        INPUT_DICT_OPTIONAL = {
            'optional': {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

        cls.INPUT_DICT_RESULT = utility.merge_dict(INPUT_DICT_FINAL, INPUT_DICT_OPTIONAL)
        return cls.INPUT_DICT_RESULT

    def stylepile(self, opt_pos_style = None, opt_neg_style = None, **kwargs):
        input_data = kwargs
        original = self
        style_text_result = StyleParser(opt_pos_style, opt_neg_style, input_data, original)

        return (style_text_result[0], style_text_result[1],)

class PrimereMidjourneyStyles:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "mjstyles"
    CATEGORY = TREE_STYLES

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE_FILE = os.path.join(DEF_TOML_DIR, "mj_styles.toml")
        STYLE_RESULT = stylehandler.toml2node(STYLE_FILE)
        INPUT_DICT_FINAL = {'required': STYLE_RESULT[0]}
        cls.STYLE_PROMPTS_POS = STYLE_RESULT[1]
        cls.STYLE_PROMPTS_NEG = STYLE_RESULT[2]

        INPUT_DICT_OPTIONAL = {
            'optional': {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

        cls.INPUT_DICT_RESULT = utility.merge_dict(INPUT_DICT_FINAL, INPUT_DICT_OPTIONAL)
        return cls.INPUT_DICT_RESULT

    def mjstyles(self, opt_pos_style = None, opt_neg_style = None, **kwargs):
        input_data = kwargs
        original = self
        style_text_result = StyleParser(opt_pos_style, opt_neg_style, input_data, original)

        return (style_text_result[0], style_text_result[1],)

class PrimereLensStyles:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "lensstyles"
    CATEGORY = TREE_STYLES

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE_FILE = os.path.join(DEF_TOML_DIR, "lens_styles.toml")
        STYLE_RESULT = stylehandler.toml2node(STYLE_FILE)
        INPUT_DICT_FINAL = {'required': STYLE_RESULT[0]}
        cls.STYLE_PROMPTS_POS = STYLE_RESULT[1]
        cls.STYLE_PROMPTS_NEG = STYLE_RESULT[2]

        INPUT_DICT_OPTIONAL = {
            'optional': {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

        cls.INPUT_DICT_RESULT = utility.merge_dict(INPUT_DICT_FINAL, INPUT_DICT_OPTIONAL)
        return cls.INPUT_DICT_RESULT

    def lensstyles(self, opt_pos_style = None, opt_neg_style = None, **kwargs):
        input_data = kwargs
        original = self
        style_text_result = StyleParser(opt_pos_style, opt_neg_style, input_data, original)

        return (style_text_result[0], style_text_result[1],)

class PrimereEmotionsStyles:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "mjstyles"
    CATEGORY = TREE_STYLES

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE_FILE = os.path.join(DEF_TOML_DIR, "emotions_style.toml")
        STYLE_RESULT = stylehandler.toml2node(STYLE_FILE)
        INPUT_DICT_FINAL = {'required': STYLE_RESULT[0]}
        cls.STYLE_PROMPTS_POS = STYLE_RESULT[1]
        cls.STYLE_PROMPTS_NEG = STYLE_RESULT[2]

        INPUT_DICT_OPTIONAL = {
            'optional': {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

        cls.INPUT_DICT_RESULT = utility.merge_dict(INPUT_DICT_FINAL, INPUT_DICT_OPTIONAL)
        return cls.INPUT_DICT_RESULT

    def mjstyles(self, opt_pos_style = None, opt_neg_style = None, **kwargs):
        input_data = kwargs
        original = self
        style_text_result = StyleParser(opt_pos_style, opt_neg_style, input_data, original)

        return (style_text_result[0], style_text_result[1],)

def StyleParser(opt_pos_style, opt_neg_style, input_data, original):
    opt_pos_style = f'{opt_pos_style}' if opt_pos_style is not None and opt_pos_style.strip(' ,;') != '' else ''
    opt_neg_style = f'{opt_neg_style}' if opt_neg_style is not None and opt_neg_style.strip(' ,;') != '' else ''

    final_style_string_pos = ''
    final_style_string_neg = ''
    for inputKey, inputValue in input_data.items():
        if inputKey.endswith("_strength") == False:
            if (inputValue != 'None' and inputValue in original.STYLE_PROMPTS_POS):
                strength_key = inputKey + '_strength'
                style_strength = input_data.get(strength_key)
                style_prompt_pos = original.STYLE_PROMPTS_POS[inputValue]
            else:
                strength_key = inputKey + '_strength'
                style_strength = input_data.get(strength_key)
                style_prompt_pos = input_data.get(inputKey)

            if style_strength != 1 and style_strength is not None:
                style_string_pos = f'({style_prompt_pos}:{style_strength:.2f})' if style_prompt_pos is not None and style_prompt_pos != 'None' and style_prompt_pos.strip(' ,;') != '' else ''
            else:
                style_string_pos = f'{style_prompt_pos}' if style_prompt_pos is not None and style_prompt_pos != 'None' and style_prompt_pos.strip(' ,;') != '' else ''
            final_style_string_pos += style_string_pos + ', '

            if (inputValue != 'None' and inputValue in original.STYLE_PROMPTS_NEG):
                strength_key = inputKey + '_strength'
                style_strength = input_data.get(strength_key)
                style_prompt_neg = original.STYLE_PROMPTS_NEG[inputValue]
            else:
                style_prompt_neg = None

            if style_strength != 1 and style_strength is not None:
                style_string_neg = f'({style_prompt_neg}:{style_strength:.2f})' if style_prompt_neg is not None and style_prompt_neg != 'None' and style_prompt_neg.strip(' ,;') != '' else ''
            else:
                style_string_neg = f'{style_prompt_neg}' if style_prompt_neg is not None and style_prompt_neg != 'None' and style_prompt_neg.strip(' ,;') != '' else ''
            final_style_string_neg += style_string_neg + ', '

    positive_text = f'{opt_pos_style}, {final_style_string_pos}'.strip(' ,;').replace(", , ", ", ").replace(", , ",", ").replace(", , ", ", ").replace(", , ", ", ")
    negative_text = f'{opt_neg_style}, {final_style_string_neg}'.strip(' ,;').replace(", , ", ", ").replace(", , ",", ").replace(", , ", ", ").replace(", , ", ", ")

    return (positive_text, negative_text)

