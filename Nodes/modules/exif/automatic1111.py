from ..exif.base_format import BaseFormat
import re

class Automatic1111(BaseFormat):
    def __init__(self, info: dict = None, raw: str = ""):
        super().__init__(info, raw)
        if not self._raw:
            self._raw = self._info.get("parameters")
        self.ProcessExif()

    def ProcessExif(self):
        exif_string = self._raw
        EXIF_LABELS = {
            "positive":'Positive prompt',
            "negative":'Negative prompt',
            "steps":'Steps',
            "sampler":'Sampler',
            "seed":'Seed',
            "variation_seed":'Variation seed',
            "variation_seed_strength":'Variation seed strength',
            "size_string":'Size',
            "model_hash":'Model hash',
            'model_name':'Model',
            "vae_hash":'VAE hash',
            "vae":'VAE',
            "lora_hashes":'Lora hashes',
            "cfg_scale":'CFG scale',
            "cfg_rescale":'CFG Rescale φ',
            "cfg_rescale_phi":'CFG Rescale phi',
            "rp_active":'RP Active',
            "rp_divide_mode":'RP Divide mode',
            "rp_matrix_submode":'RP Matrix submode',
            "rp_mask_submode":'RP Mask submode',
            "rp_prompt_submode":'RP Prompt submode',
            "rp_calc_mode":'RP Calc Mode',
            "rp_ratios":'RP Ratios',
            "rp_base_ratios":'RP Base Ratios',
            "rp_use_base":'RP Use Base',
            "rp_use_common":'RP Use Common',
            "rp_use_ncommon":'RP Use Ncommon',
            "rp_change_and":'RP Change AND',
            "rp_lora_neg_te_ratios":'RP LoRA Neg Te Ratios',
            "rp_lora_neg_u_ratios":'RP LoRA Neg U Ratios',
            "rp_threshold":'RP threshold',
            "npw_weight":'NPW_weight',
            "antiburn":'AntiBurn',
            "version":'Version',
            "template":'Template',
            "negative_template":'Negative Template',
            "face_restoration":'Face restoration',
            "postprocess_upscaler":'Postprocess upscaler',
            "postprocess_upscale_by":'Postprocess upscale by'
        }

        LABEL_END = ['\n', ',']
        STRIP_FROM_VALUE = ' ";\n'
        FORCE_STRING = ['model_hash', 'vae_hash', 'lora_hashes']
        FORCE_FLOAT = ['cfg_scale', 'cfg_rescale', 'cfg_rescale_phi', 'npw_weight']

        # FIRST_ROW = exif_string.split('\n', 1)[0]
        exif_string = 'Positive prompt: ' + exif_string

        SORTED_BY_STRING = dict(sorted(EXIF_LABELS.items(), key=lambda pos: exif_string.find(pos[1] + ':')))
        SORTED_KEYLIST = list(SORTED_BY_STRING.keys())
        FINAL_DICT = {}
        FLOAT_PATTERN = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'

        for LABEL_KEY, LABEL in SORTED_BY_STRING.items():
            NextValue = '\n'
            RealLabel = LABEL + ':'
            CurrentKeyIndex = (SORTED_KEYLIST.index(LABEL_KEY))
            NextKeyIndex = CurrentKeyIndex + 1

            if len(SORTED_KEYLIST) > NextKeyIndex:
                NextKey = SORTED_KEYLIST[NextKeyIndex]
                NextValue = SORTED_BY_STRING[NextKey] + ':'

            if RealLabel in exif_string:
                LabelStart = exif_string.find(RealLabel)
                NextLabelStart = exif_string.find(NextValue)
                LabelLength = len(RealLabel)
                ValueStart = exif_string.find(exif_string[(LabelStart + LabelLength):NextLabelStart])
                ValueLength = len(exif_string[(LabelStart + LabelLength):NextLabelStart])
                ValueRaw = exif_string[(LabelStart + LabelLength):NextLabelStart]
                FirstMatch = next((x for x in LABEL_END if x in exif_string[(ValueStart + ValueLength - 2):(ValueStart + ValueLength + 1)]), False)

                if CurrentKeyIndex >= 2 and FirstMatch == ',':
                    isUnknownValue = all(x in ValueRaw for x in [':', ','])
                    if isUnknownValue:
                        FirstMatchOfFaliled = ValueRaw.find(FirstMatch)
                        NextLabelStart = ValueStart

                if FirstMatch:
                    LabelEnd = exif_string.find(FirstMatch, NextLabelStart - 2)
                    LabelValue = exif_string[(LabelStart + LabelLength):LabelEnd]
                else:
                    LabelEnd = None
                    if CurrentKeyIndex >= 2 and FirstMatch == '\n' or FirstMatch == False:
                        badValue = exif_string[(LabelStart + LabelLength):LabelEnd]
                        isUnknownValue = all(x in badValue for x in [':', '\n'])
                        if isUnknownValue:
                            FirstMatchOfFaliled = badValue.find('\n')
                            LabelEnd = exif_string.find('\n', LabelStart + LabelLength + FirstMatchOfFaliled + 2)

                    LabelValue = exif_string[(LabelStart + LabelLength):LabelEnd]

                LabelValue = LabelValue.replace('Count=', '').strip(STRIP_FROM_VALUE)
                if not LabelValue:
                    LabelValue = None
                elif LabelValue == 'False':
                    LabelValue = False
                elif LabelValue.isdigit():
                    LabelValue = int(LabelValue)
                elif bool(re.match(FLOAT_PATTERN, LabelValue)):
                    LabelValue = float(LabelValue)

                if LABEL_KEY in FORCE_STRING:
                    LabelValue = str(LabelValue)

                if LABEL_KEY in FORCE_FLOAT:
                    LabelValue = float(LabelValue)

                if LABEL_KEY == 'size_string':
                    width, height = LabelValue.split("x")
                    FINAL_DICT['width'] = int(width)
                    FINAL_DICT['height'] = int(height)

                FINAL_DICT[LABEL_KEY] = LabelValue

        self._parameter = FINAL_DICT
