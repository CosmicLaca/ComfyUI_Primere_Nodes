import os
from pathlib import Path
from .utils import comfy_dir
from .utils import here

from .Nodes import Dashboard
from .Nodes import Inputs
from .Nodes import Styles
from .Nodes import Outputs
from .Nodes import Visuals
from .Nodes import Networks
from .Nodes import Segments
from .components import primereserver

import shutil
from datetime import datetime
from .components import utility

__version__ = "0.5.0"

comfy_frontend = os.path.join(comfy_dir, 'web', 'extensions')
frontend_target = os.path.join(comfy_frontend, 'Primere')
frontend_source = os.path.join(here, 'front_end')
is_frontend_symlinked = False

ClientTime = datetime.now()
UpdateRequired = '2024-03-08 20:00:00'
# IsDev = utility.get_value_from_cache('setup', 'is_dev')

if os.path.isdir(frontend_target) == True:
    try:
        is_link = os.readlink(frontend_target)
        is_frontend_symlinked = True
    except OSError:
        is_frontend_symlinked = False

    if is_frontend_symlinked == True:
        try:
            os.unlink(frontend_target)
            shutil.copytree(frontend_source, frontend_target)
            print('Primere front-end changed from symlink to real directory.')
        except Exception:
            print('[ERROR] - Cannnot copy Primere front-end folder to right path. Please delete symlink: ' + frontend_target + ' and copy files here manually from: ' + frontend_source)
    else:
        LastFrontend = utility.get_value_from_cache('dates', 'frontend_update')
        if LastFrontend == None:
            try:
                shutil.rmtree(frontend_target)
                shutil.copytree(frontend_source, frontend_target)
                print('[Primere front-end update] - Primere front-end files updated to latest version.')
                utility.add_value_to_cache('dates', 'frontend_update', str(ClientTime))
            except Exception:
                print('[ERROR] - Cannnot update Primere front-end folder to right path. Please delete directory: ' + frontend_target + ' and copy files here manually from: ' + frontend_source)
        else:
            newupdate = datetime.strptime(UpdateRequired, '%Y-%m-%d %H:%M:%S')
            lastupdate = datetime.strptime(LastFrontend, '%Y-%m-%d %H:%M:%S.%f')
            if lastupdate < newupdate:
                try:
                    shutil.rmtree(frontend_target)
                    shutil.copytree(frontend_source, frontend_target)
                    print('[Primere front-end update] - Primere front-end files updated to latest version.')
                    updated = utility.update_value_in_cache('dates', 'frontend_update', str(ClientTime))
                except Exception:
                    print('[ERROR] - Cannnot update Primere front-end folder to right path. Please delete directory: ' + frontend_target + ' and copy files here manually from: ' + frontend_source)

else:
    try:
        shutil.copytree(frontend_source, frontend_target)
        print('Primere front-end copied to target directory.')
    except Exception:
        print('[ERROR] - Cannnot copy Primere front-end folder to right path. Please delete directory: ' + frontend_target + ' and copy files here manually from: ' + frontend_source)

NODE_CLASS_MAPPINGS = {
    "PrimereSamplers": Dashboard.PrimereSamplers,
    "PrimereStepsCfg": Dashboard.PrimereStepsCfg,
    "PrimereSamplersSteps": Dashboard.PrimereSamplersSteps,
    "PrimereVAE": Dashboard.PrimereVAE,
    "PrimereCKPT": Dashboard.PrimereCKPT,
    "PrimereVAELoader": Dashboard.PrimereVAELoader,
    "PrimereCKPTLoader": Dashboard.PrimereCKPTLoader,
    "PrimerePromptSwitch": Dashboard.PrimerePromptSwitch,
    "PrimereSeed": Dashboard.PrimereSeed,
    "PrimereLatentNoise": Dashboard.PrimereFractalLatent,
    "PrimereCLIPEncoder": Dashboard.PrimereCLIP,
    "PrimereResolution": Dashboard.PrimereResolution,
    "PrimereClearPrompt": Dashboard.PrimereClearPrompt,
    "PrimereLCMSelector": Dashboard.PrimereLCMSelector,
    "PrimereModelConceptSelector": Dashboard.PrimereModelConceptSelector,
    "PrimereResolutionMultiplier": Dashboard.PrimereResolutionMultiplier,
    "PrimereResolutionMultiplierMPX": Dashboard.PrimereResolutionMultiplierMPX,
    "PrimereNetworkTagLoader": Dashboard.PrimereNetworkTagLoader,
    "PrimereModelKeyword": Dashboard.PrimereModelKeyword,
    "PrimereConceptDataTuple": Dashboard.PrimereConceptDataTuple,

    "PrimerePrompt": Inputs.PrimereDoublePrompt,
    "PrimereStyleLoader": Inputs.PrimereStyleLoader,
    "PrimereDynamicParser": Inputs.PrimereDynParser,
    "PrimereVAESelector": Inputs.PrimereVAESelector,
    "PrimereMetaRead": Inputs.PrimereMetaRead,
    "PrimereEmbeddingHandler": Inputs.PrimereEmbeddingHandler,
    "PrimereLoraStackMerger": Inputs.PrimereLoraStackMerger,
    "PrimereLoraKeywordMerger": Inputs.PrimereLoraKeywordMerger,
    "PrimereEmbeddingKeywordMerger": Inputs.PrimereEmbeddingKeywordMerger,
    "PrimereLycorisStackMerger": Inputs.PrimereLycorisStackMerger,
    "PrimereLycorisKeywordMerger": Inputs.PrimereLycorisKeywordMerger,
    "PrimereRefinerPrompt": Inputs.PrimereRefinerPrompt,
    "PrimerePromptOrganizer": Inputs.PrimerePromptOrganizer,
    "PrimereMetaHandler": Inputs.PrimereMetaHandler,
    "PrimereMetaDistributor": Inputs.PrimereMetaDistributor,

    "PrimereMetaSave": Outputs.PrimereMetaSave,
    "PrimereAnyOutput": Outputs.PrimereAnyOutput,
    "PrimereTextOutput": Outputs.PrimereTextOutput,
    "PrimereMetaCollector": Outputs.PrimereMetaCollector,
    "PrimereKSampler": Outputs.PrimereKSampler,
    "PrimerePreviewImage": Outputs.PrimerePreviewImage,

    "PrimereStylePile": Styles.PrimereStylePile,
    "PrimereMidjourneyStyles": Styles.PrimereMidjourneyStyles,
    "PrimereEmotionsStyles": Styles.PrimereEmotionsStyles,

    "PrimereVisualCKPT": Visuals.PrimereVisualCKPT,
    "PrimereVisualLORA": Visuals.PrimereVisualLORA,
    "PrimereVisualEmbedding": Visuals.PrimereVisualEmbedding,
    "PrimereVisualHypernetwork": Visuals.PrimereVisualHypernetwork,
    "PrimereVisualStyle": Visuals.PrimereVisualStyle,
    "PrimereVisualLYCORIS": Visuals.PrimereVisualLYCORIS,

    "PrimereLORA": Networks.PrimereLORA,
    "PrimereEmbedding": Networks.PrimereEmbedding,
    "PrimereHypernetwork": Networks.PrimereHypernetwork,
    "PrimereLYCORIS": Networks.PrimereLYCORIS,

    "PrimereImageSegments": Segments.PrimereImageSegments,
    "PrimereAnyDetailer": Segments.PrimereAnyDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimereSamplers": "Primere Sampler Selector",
    "PrimereStepsCfg": "Primere Steps & Cfg",
    "PrimereSamplersSteps": "Primere Samplers & Steps & Cfg",
    "PrimereVAE": "Primere VAE Selector",
    "PrimereCKPT": "Primere CKPT Selector",
    "PrimereVAELoader": "Primere VAE Loader",
    "PrimereCKPTLoader": "Primere CKPT Loader",
    "PrimerePromptSwitch": "Primere Prompt Switch",
    "PrimereSeed": 'Primere Seed',
    "PrimereLatentNoise": "Primere Noise Latent",
    "PrimereCLIPEncoder": "Primere Prompt Encoder",
    "PrimereResolution": "Primere Resolution",
    "PrimereClearPrompt": "Primere Prompt Cleaner",
    "PrimereLCMSelector": "Primere LCM selector",
    "PrimereModelConceptSelector": "Primere Model Concept Selector",
    "PrimereResolutionMultiplier": "Primere Resolution Multiplier",
    "PrimereResolutionMultiplierMPX": "Primere Resolution MPX",
    "PrimereNetworkTagLoader": 'Primere Network Tag Loader',
    "PrimereModelKeyword": "Primere Model Keyword",
    "PrimereConceptDataTuple": "Primere Concept Tuple",

    "PrimerePrompt": "Primere Prompt",
    "PrimereStyleLoader": "Primere Styles",
    "PrimereDynamicParser": "Primere Dynamic",
    "PrimereVAESelector": "Primere VAE Version Selector",
    "PrimereMetaRead": "Primere Exif Reader",
    "PrimereEmbeddingHandler": "Primere Embedding Handler",
    "PrimereLoraStackMerger": "Primere Lora Stack Merger",
    "PrimereLoraKeywordMerger": 'Primere Lora Keyword Merger',
    "PrimereEmbeddingKeywordMerger": "Primere Embedding Keyword Merger",
    "PrimereLycorisStackMerger": 'Primere Lycoris Stack Merger',
    "PrimereLycorisKeywordMerger": 'Primere Lycoris Keyword Merger',
    "PrimereRefinerPrompt": "Primere Refiner Prompt",
    "PrimerePromptOrganizer": "Primere Prompt Organizer",
    "PrimereMetaHandler": "Primere Image Recycler",
    "PrimereMetaDistributor": "Primere Meta Distributor",

    "PrimereMetaSave": "Primere Image Meta Saver",
    "PrimereAnyOutput": "Primere Any Debug",
    "PrimereTextOutput": "Primere Text Ouput",
    "PrimereMetaCollector": "Primere Meta Collector",
    "PrimereKSampler": "Primere KSampler",
    "PrimerePreviewImage": "Primere Preview Image",

    "PrimereStylePile": "Primere Style Pile",
    "PrimereMidjourneyStyles": "Primere Midjourney Styles",
    "PrimereEmotionsStyles": "Primere Emotions Styles",

    "PrimereVisualCKPT": "Primere Visual CKPT Selector",
    "PrimereVisualLORA": "Primere Visual LORA Selector",
    "PrimereVisualEmbedding": 'Primere Visual Embedding Selector',
    "PrimereVisualHypernetwork": 'Primere Visual Hypernetwork Selector',
    "PrimereVisualStyle": 'Primere Visual Style Selector',
    "PrimereVisualLYCORIS": 'Primere Visual LYCORIS Selector',

    "PrimereLORA": 'Primere LORA',
    "PrimereEmbedding": 'Primere Embedding',
    "PrimereHypernetwork": 'Primere Hypernetwork',
    "PrimereLYCORIS": 'Primere LYCORIS',

    "PrimereImageSegments": 'Primere Image Segments',
    "PrimereAnyDetailer": 'Primere Any Detailer',
}