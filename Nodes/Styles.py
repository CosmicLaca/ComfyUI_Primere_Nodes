from ..components.tree import TREE_STYLES
from ..components.tree import PRIMERE_ROOT
import os
import tomli

class PrimereStylePile:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "stylepile"
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
            },
            "optional": {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

    def stylepile(self, concept, concept_strength, art_type, art_type_strength, artist, artist_strength, movement, movement_strength, color, color_strength, mood, mood_strength, direction, direction_strength, opt_pos_style = '', opt_neg_style = ''):
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

        positive_text = f'{opt_pos_style}, {art_type_pos_str}, {color}, {mood}, {direction}, {concept}, {artist}, {movement}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(", , ", ", ")
        negative_text = f'{opt_neg_style}, {art_type_neg_str}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ")

        return (positive_text, negative_text,)

class PrimereMidjourneyStyles:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STYLE+", "STYLE-")
    FUNCTION = "mjstyles"
    CATEGORY = TREE_STYLES

    @staticmethod
    def get_all_styles(toml_path: str):
        with open(toml_path, "rb") as f:
            style_def_neg = tomli.load(f)
        return style_def_neg

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE = cls.get_all_styles(os.path.join(DEF_TOML_DIR, "mj_styles.toml"))

        ABSTRACTKeyList = list(STYLE['ABSTRACT'].keys())
        ABSTRACT_LIST = []
        ABSTRACT_LIST_DICT = {}
        for ABSTRACTKey in ABSTRACTKeyList:
            LISTVALUE = STYLE['ABSTRACT'][ABSTRACTKey]['MainStyle'] + '::' + STYLE['ABSTRACT'][ABSTRACTKey]['SubStyle'] + '::' +  STYLE['ABSTRACT'][ABSTRACTKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['ABSTRACT'][ABSTRACTKey]['MjPrompt']
            ABSTRACT_LIST.append(LISTVALUE)
            ABSTRACT_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_ABSTRACT = ['None'] + ABSTRACT_LIST
        cls.ABSTRACT_LIST_DICT = ABSTRACT_LIST_DICT

        ARTKeyList = list(STYLE['ART'].keys())
        ART_LIST = []
        ART_LIST_DICT = {}
        for ARTKey in ARTKeyList:
            LISTVALUE = STYLE['ART'][ARTKey]['MainStyle'] + '::' + STYLE['ART'][ARTKey]['SubStyle'] + '::' +  STYLE['ART'][ARTKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['ART'][ARTKey]['MjPrompt']
            ART_LIST.append(LISTVALUE)
            ART_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_ART = ['None'] + ART_LIST
        cls.ART_LIST_DICT = ART_LIST_DICT

        CUBISMKeyList = list(STYLE['CUBISM'].keys())
        CUBISM_LIST = []
        CUBISM_LIST_DICT = {}
        for CUBISMKey in CUBISMKeyList:
            LISTVALUE = STYLE['CUBISM'][CUBISMKey]['MainStyle'] + '::' + STYLE['CUBISM'][CUBISMKey]['SubStyle'] + '::' +  STYLE['CUBISM'][CUBISMKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['CUBISM'][CUBISMKey]['MjPrompt']
            CUBISM_LIST.append(LISTVALUE)
            CUBISM_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_CUBISM = ['None'] + CUBISM_LIST
        cls.CUBISM_LIST_DICT = CUBISM_LIST_DICT

        NEOKeyList = list(STYLE['NEO'].keys())
        NEO_LIST = []
        NEO_LIST_DICT = {}
        for NEOKey in NEOKeyList:
            LISTVALUE = STYLE['NEO'][NEOKey]['MainStyle'] + '::' + STYLE['NEO'][NEOKey]['SubStyle'] + '::' +  STYLE['NEO'][NEOKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['NEO'][NEOKey]['MjPrompt']
            NEO_LIST.append(LISTVALUE)
            NEO_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_NEO = ['None'] + NEO_LIST
        cls.NEO_LIST_DICT = NEO_LIST_DICT

        PRECISIONISTKeyList = list(STYLE['PRECISIONIST'].keys())
        PRECISIONIST_LIST = []
        PRECISIONIST_LIST_DICT = {}
        for PRECISIONISTKey in PRECISIONISTKeyList:
            LISTVALUE = STYLE['PRECISIONIST'][PRECISIONISTKey]['MainStyle'] + '::' + STYLE['PRECISIONIST'][PRECISIONISTKey]['SubStyle'] + '::' +  STYLE['PRECISIONIST'][PRECISIONISTKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['PRECISIONIST'][PRECISIONISTKey]['MjPrompt']
            PRECISIONIST_LIST.append(LISTVALUE)
            PRECISIONIST_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_PRECISIONIST = ['None'] + PRECISIONIST_LIST
        cls.PRECISIONIST_LIST_DICT = PRECISIONIST_LIST_DICT

        REALISMKeyList = list(STYLE['REALISM'].keys())
        REALISM_LIST = []
        REALISM_LIST_DICT = {}
        for REALISMKey in REALISMKeyList:
            LISTVALUE = STYLE['REALISM'][REALISMKey]['MainStyle'] + '::' + STYLE['REALISM'][REALISMKey]['SubStyle'] + '::' +  STYLE['REALISM'][REALISMKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['REALISM'][REALISMKey]['MjPrompt']
            REALISM_LIST.append(LISTVALUE)
            REALISM_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_REALISM = ['None'] + REALISM_LIST
        cls.REALISM_LIST_DICT = REALISM_LIST_DICT

        RENAISSANCEKeyList = list(STYLE['RENAISSANCE'].keys())
        RENAISSANCE_LIST = []
        RENAISSANCE_LIST_DICT = {}
        for RENAISSANCEKey in RENAISSANCEKeyList:
            LISTVALUE = STYLE['RENAISSANCE'][RENAISSANCEKey]['MainStyle'] + '::' + STYLE['RENAISSANCE'][RENAISSANCEKey]['SubStyle'] + '::' +  STYLE['RENAISSANCE'][RENAISSANCEKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['RENAISSANCE'][RENAISSANCEKey]['MjPrompt']
            RENAISSANCE_LIST.append(LISTVALUE)
            RENAISSANCE_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_RENAISSANCE = ['None'] + RENAISSANCE_LIST
        cls.RENAISSANCE_LIST_DICT = RENAISSANCE_LIST_DICT

        ROMANTICISMKeyList = list(STYLE['ROMANTICISM'].keys())
        ROMANTICISM_LIST = []
        ROMANTICISM_LIST_DICT = {}
        for ROMANTICISMKey in ROMANTICISMKeyList:
            LISTVALUE = STYLE['ROMANTICISM'][ROMANTICISMKey]['MainStyle'] + '::' + STYLE['ROMANTICISM'][ROMANTICISMKey]['SubStyle'] + '::' +  STYLE['ROMANTICISM'][ROMANTICISMKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['ROMANTICISM'][ROMANTICISMKey]['MjPrompt']
            ROMANTICISM_LIST.append(LISTVALUE)
            ROMANTICISM_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_ROMANTICISM = ['None'] + ROMANTICISM_LIST
        cls.ROMANTICISM_LIST_DICT = ROMANTICISM_LIST_DICT

        SYMBOLISMKeyList = list(STYLE['SYMBOLISM'].keys())
        SYMBOLISM_LIST = []
        SYMBOLISM_LIST_DICT = {}
        for SYMBOLISMKey in SYMBOLISMKeyList:
            LISTVALUE = STYLE['SYMBOLISM'][SYMBOLISMKey]['MainStyle'] + '::' + STYLE['SYMBOLISM'][SYMBOLISMKey]['SubStyle'] + '::' +  STYLE['SYMBOLISM'][SYMBOLISMKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['SYMBOLISM'][SYMBOLISMKey]['MjPrompt']
            SYMBOLISM_LIST.append(LISTVALUE)
            SYMBOLISM_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_SYMBOLISM = ['None'] + SYMBOLISM_LIST
        cls.SYMBOLISM_LIST_DICT = SYMBOLISM_LIST_DICT

        MJ_ARTISTS_STYLESKeyList = list(STYLE['MJ_ARTISTS_STYLES'].keys())
        MJ_ARTISTS_STYLES_LIST = []
        MJ_ARTISTS_STYLES_LIST_DICT = {}
        for MJ_ARTISTS_STYLESKey in MJ_ARTISTS_STYLESKeyList:
            LISTVALUE = STYLE['MJ_ARTISTS_STYLES'][MJ_ARTISTS_STYLESKey]['MainStyle'] + '::' + STYLE['MJ_ARTISTS_STYLES'][MJ_ARTISTS_STYLESKey]['SubStyle'] + '::' +  STYLE['MJ_ARTISTS_STYLES'][MJ_ARTISTS_STYLESKey]['ArtistName']
            LISTVALUE = LISTVALUE.replace('::::', '::')
            PROMPT = STYLE['MJ_ARTISTS_STYLES'][MJ_ARTISTS_STYLESKey]['MjPrompt']
            MJ_ARTISTS_STYLES_LIST.append(LISTVALUE)
            MJ_ARTISTS_STYLES_LIST_DICT[LISTVALUE] = PROMPT
        cls.MJSTYLES_MJ_ARTISTS_STYLES = ['None'] + MJ_ARTISTS_STYLES_LIST
        cls.MJ_ARTISTS_STYLES_LIST_DICT = MJ_ARTISTS_STYLES_LIST_DICT

        cls.MJSTYLESKEYS = ['None'] + sorted(STYLE['MJSTYLES']['MJSTYLES'])

        return {
            "required": {
                "abstract": (cls.MJSTYLES_ABSTRACT,),
                "abstract_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "art": (cls.MJSTYLES_ART,),
                "art_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "cubism": (cls.MJSTYLES_CUBISM,),
                "cubism_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "neo": (cls.MJSTYLES_NEO,),
                "neo_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "precisionist": (cls.MJSTYLES_PRECISIONIST,),
                "precisionist_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "realism": (cls.MJSTYLES_REALISM,),
                "realism_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "renaissance": (cls.MJSTYLES_RENAISSANCE,),
                "renaissance_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "romanticism": (cls.MJSTYLES_ROMANTICISM,),
                "romanticism_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "symbolism": (cls.MJSTYLES_SYMBOLISM,),
                "symbolism_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "artists_styles": (cls.MJSTYLES_MJ_ARTISTS_STYLES,),
                "artists_styles_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "midjourney_stylekeys": (cls.MJSTYLESKEYS,),
                "midjourney_stylekeys_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "opt_pos_style": ("STRING", {"forceInput": True}),
                "opt_neg_style": ("STRING", {"forceInput": True}),
            }
        }

    def mjstyles(self, abstract, abstract_strength, midjourney_stylekeys, art, art_strength, cubism, cubism_strength, neo, neo_strength, precisionist, precisionist_strength, realism, realism_strength, renaissance, renaissance_strength, romanticism, romanticism_strength, symbolism, symbolism_strength, artists_styles, artists_styles_strength, midjourney_stylekeys_strength, opt_pos_style = '', opt_neg_style = ''):
        opt_pos_style = f'({opt_pos_style})' if opt_pos_style is not None and opt_pos_style.strip(' ,;') != '' else ''
        opt_neg_style = f'({opt_neg_style})' if opt_neg_style is not None and opt_neg_style.strip(' ,;') != '' else ''

        abstract_prompt = None
        if (abstract != 'None' and abstract in self.ABSTRACT_LIST_DICT):
            abstract_prompt = self.ABSTRACT_LIST_DICT[abstract]
        abstract = f'({abstract_prompt}:{abstract_strength:.2f})' if abstract_prompt is not None and abstract_prompt != 'None' and abstract_prompt.strip(' ,;') != '' else ''

        art_prompt = None
        if (art != 'None' and art in self.ART_LIST_DICT):
            art_prompt = self.ART_LIST_DICT[art]
        art = f'({art_prompt}:{art_strength:.2f})' if art_prompt is not None and art_prompt != 'None' and art_prompt.strip(' ,;') != '' else ''

        cubism_prompt = None
        if (cubism != 'None' and cubism in self.CUBISM_LIST_DICT):
            cubism_prompt = self.CUBISM_LIST_DICT[cubism]
        cubism = f'({cubism_prompt}:{cubism_strength:.2f})' if cubism_prompt is not None and cubism_prompt != 'None' and cubism_prompt.strip(' ,;') != '' else ''

        neo_prompt = None
        if (neo != 'None' and neo in self.NEO_LIST_DICT):
            neo_prompt = self.NEO_LIST_DICT[neo]
        neo = f'({neo_prompt}:{neo_strength:.2f})' if neo_prompt is not None and neo_prompt != 'None' and neo_prompt.strip(' ,;') != '' else ''

        precisionist_prompt = None
        if (precisionist != 'None' and precisionist in self.PRECISIONIST_LIST_DICT):
            precisionist_prompt = self.PRECISIONIST_LIST_DICT[precisionist]
        precisionist = f'({precisionist_prompt}:{precisionist_strength:.2f})' if precisionist_prompt is not None and precisionist_prompt != 'None' and precisionist_prompt.strip(' ,;') != '' else ''

        realism_prompt = None
        if (realism != 'None' and realism in self.REALISM_LIST_DICT):
            realism_prompt = self.REALISM_LIST_DICT[realism]
        realism = f'({realism_prompt}:{realism_strength:.2f})' if realism_prompt is not None and realism_prompt != 'None' and realism_prompt.strip(' ,;') != '' else ''

        renaissance_prompt = None
        if (renaissance != 'None' and renaissance in self.RENAISSANCE_LIST_DICT):
            renaissance_prompt = self.RENAISSANCE_LIST_DICT[renaissance]
        renaissance = f'({renaissance_prompt}:{renaissance_strength:.2f})' if renaissance_prompt is not None and renaissance_prompt != 'None' and renaissance_prompt.strip(' ,;') != '' else ''

        romanticism_prompt = None
        if (romanticism != 'None' and romanticism in self.ROMANTICISM_LIST_DICT):
            romanticism_prompt = self.ROMANTICISM_LIST_DICT[romanticism]
        romanticism = f'({romanticism_prompt}:{romanticism_strength:.2f})' if romanticism_prompt is not None and romanticism_prompt != 'None' and romanticism_prompt.strip(' ,;') != '' else ''

        symbolism_prompt = None
        if (symbolism != 'None' and symbolism in self.SYMBOLISM_LIST_DICT):
            symbolism_prompt = self.SYMBOLISM_LIST_DICT[symbolism]
        symbolism = f'({symbolism_prompt}:{symbolism_strength:.2f})' if symbolism_prompt is not None and symbolism_prompt != 'None' and symbolism_prompt.strip(' ,;') != '' else ''

        artists_styles_prompt = None
        if (artists_styles != 'None' and artists_styles in self.MJ_ARTISTS_STYLES_LIST_DICT):
            artists_styles_prompt = self.MJ_ARTISTS_STYLES_LIST_DICT[artists_styles]
        artists_styles = f'({artists_styles_prompt}:{artists_styles_strength:.2f})' if artists_styles_prompt is not None and artists_styles_prompt != 'None' and artists_styles_prompt.strip(' ,;') != '' else ''

        midjourney_styleskeys = f'({midjourney_stylekeys}:{midjourney_stylekeys_strength:.2f})' if midjourney_stylekeys is not None and midjourney_stylekeys != 'None' and midjourney_stylekeys.strip(' ,;') != '' else ''

        positive_text = f'{opt_pos_style}, {abstract}, {art}, {cubism}, {neo}, {precisionist}, {realism}, {renaissance}, {romanticism}, {symbolism}, {artists_styles}, {midjourney_styleskeys}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(", , ", ", ")
        negative_text = f'{opt_neg_style}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ")

        return (positive_text, negative_text,)

