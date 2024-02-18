import tomli

def get_all_styles(toml_path: str):
    with open(toml_path, "rb") as f:
        style_def_neg = tomli.load(f)
    return style_def_neg

def toml2node(tomlpath):
    STYLE = get_all_styles(tomlpath)
    STYLES = ['None']
    INPUT_DICT = {}
    STRENGHT = "FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}
    LIST_DICT_POS = {}
    LIST_DICT_NEG = {}

    for STYLE_ONE in STYLE:
        KeyList = list(STYLE[STYLE_ONE].keys())
        LIST = []
        STYLE_KEY = STYLE_ONE.lower()

        for Key in KeyList:
            if Key in STYLE[STYLE_ONE]:
                if len(list(STYLE[STYLE_ONE].keys())) > 0 and isinstance(STYLE[STYLE_ONE][Key], dict) == True:
                    if len(list(STYLE[STYLE_ONE][Key].keys())) > 0:
                        LAST_KEY = list(STYLE[STYLE_ONE][Key].keys())[-1]
                        SECONT_TO_LAST_KEY = list(STYLE[STYLE_ONE][Key].keys())[-2]
                        LISTVALUE = ""
                        PROMPT_POS = ""
                        PROMPT_NEG = ""
                        for StyleKey in STYLE[STYLE_ONE][Key].keys():
                            if StyleKey != LAST_KEY and StyleKey.lower() != 'positive' and StyleKey.lower() != 'negative':
                                LISTVALUE = LISTVALUE + STYLE[STYLE_ONE][Key][StyleKey] + '::'
                                LISTVALUE = LISTVALUE.replace('::::', '::')
                            else:
                                if LAST_KEY.lower() == 'positive' and StyleKey.lower() == 'positive':
                                    PROMPT_POS = STYLE[STYLE_ONE][Key][StyleKey]
                                if LAST_KEY.lower() == 'negative' and StyleKey.lower() == 'negative':
                                    PROMPT_NEG = STYLE[STYLE_ONE][Key][StyleKey]

                                if SECONT_TO_LAST_KEY.lower() == 'positive' and StyleKey.lower() == 'positive':
                                    PROMPT_POS = STYLE[STYLE_ONE][Key][StyleKey]
                                if SECONT_TO_LAST_KEY.lower() == 'negative' and StyleKey.lower() == 'negative':
                                    PROMPT_NEG = STYLE[STYLE_ONE][Key][StyleKey]

                        LISTVALUE = LISTVALUE.strip(':')
                        LIST.append(LISTVALUE)
                        LIST_DICT_POS[LISTVALUE] = PROMPT_POS
                        LIST_DICT_NEG[LISTVALUE] = PROMPT_NEG
                        STYLES = (['None'] + LIST,)
                else:
                    STYLES = (['None'] + sorted(STYLE[Key][Key]),)

            INPUT_DICT[STYLE_KEY] = STYLES
            INPUT_DICT[STYLE_KEY + '_strength'] = STRENGHT

    return (INPUT_DICT, LIST_DICT_POS, LIST_DICT_NEG,)
