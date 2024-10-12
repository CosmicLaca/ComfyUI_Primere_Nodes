import tomli

def get_all_styles(toml_path: str):
    with open(toml_path, "rb") as f:
        style_def_neg = tomli.load(f)
    return style_def_neg

def toml2node(tomlpath, add_stength = True, exclude_names = []):
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
                if len(list(STYLE[STYLE_ONE].keys())) > 0 and isinstance(STYLE[STYLE_ONE][Key], dict):
                    if len(list(STYLE[STYLE_ONE][Key].keys())) > 0:
                        LAST_KEY = list(STYLE[STYLE_ONE][Key].keys())[-1]
                        SECONT_TO_LAST_KEY = list(STYLE[STYLE_ONE][Key].keys())[-2]
                        LISTVALUE = ""
                        PROMPT_POS = ""
                        PROMPT_NEG = ""
                        for StyleKey in STYLE[STYLE_ONE][Key].keys():
                            if StyleKey not in exclude_names:
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
            if add_stength == True:
                INPUT_DICT[STYLE_KEY + '_strength'] = STRENGHT

    return (INPUT_DICT, LIST_DICT_POS, LIST_DICT_NEG, STYLE,)

def csv2node(styles_csv, exclude_names = []):
    INPUT_DICT = {}

    subpathList = styles_csv['preferred_subpath']
    prompt_subpaths = list(set(subpathList))
    prompt_subpaths_sorted = sorted(prompt_subpaths, key=lambda x: 'nan' if (x != x) else x)

    for prompt_subpath in prompt_subpaths_sorted:
        if str(prompt_subpath) == "nan":
            prompt_subpath = 'Others'
            resultsBySubpath = styles_csv[styles_csv['preferred_subpath'].isnull()]
        else:
            resultsBySubpath = styles_csv[styles_csv['preferred_subpath'] == prompt_subpath]

        resultsByNames = list(resultsBySubpath['name'])
        resultsByNames.sort()
        INPUT_DICT[prompt_subpath] = (['None'] + resultsByNames,)

    return INPUT_DICT
