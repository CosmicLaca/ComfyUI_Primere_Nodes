from ..components.tree import TREE_STYLES
from ..components.tree import PRIMERE_ROOT
import os
from ..components import utility
from ..components import stylehandler

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

class PrimereCustomStyles:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "customstyles"
    CATEGORY = TREE_STYLES

    @classmethod
    def INPUT_TYPES(cls):
        style_dir = os.path.join(PRIMERE_ROOT, 'Toml', 'Styles')
        style_files = []
        if os.path.isdir(style_dir):
            style_files = sorted([f for f in os.listdir(style_dir) if f.lower().endswith('.toml')], key=str.casefold)
        if len(style_files) == 0:
            style_files = ["None"]

        return {
            "required": {
                "style_source": (style_files,),
            },
            "optional": {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt_extra": "PROMPT",
            }
        }

    def customstyles(self, style_source, opt_pos_style = None, opt_neg_style = None, extra_pnginfo = None, prompt_extra = None, **kwargs):
        style_dir = os.path.join(PRIMERE_ROOT, 'Toml', 'Styles')
        style_source = str(style_source) if style_source is not None else ""
        style_file = os.path.basename(style_source)
        style_path = os.path.join(style_dir, style_file)

        if not os.path.isfile(style_path):
            return (f'{opt_pos_style}'.strip(' ,;') if opt_pos_style is not None else "", f'{opt_neg_style}'.strip(' ,;') if opt_neg_style is not None else "",)

        style_result = stylehandler.toml2node(style_path)
        style_input_dict = style_result[0]
        self.STYLE_PROMPTS_POS = style_result[1]
        self.STYLE_PROMPTS_NEG = style_result[2]

        input_data = kwargs.copy()
        if extra_pnginfo is not None and prompt_extra is not None and 'workflow' in extra_pnginfo:
            workflow_data = extra_pnginfo['workflow']['nodes']
            custom_values = utility.getInputsFromWorkflowByNode(workflow_data, 'PrimereCustomStyles', prompt_extra)
            if isinstance(custom_values, dict):
                for key, value in custom_values.items():
                    if key != 'style_source':
                        input_data[key] = value

        valid_keys = set(style_input_dict.keys())
        filtered_input_data = {k: v for k, v in input_data.items() if k in valid_keys}
        style_text_result = StyleParser(opt_pos_style, opt_neg_style, filtered_input_data, self)

        return (style_text_result[0], style_text_result[1],)