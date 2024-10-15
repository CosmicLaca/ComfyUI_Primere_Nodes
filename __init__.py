import os
from pathlib import Path
from .components import primereserver
from .utils import comfy_dir
from .utils import here

from .Nodes import Dashboard
from .Nodes import Inputs
from .Nodes import Styles
from .Nodes import Outputs
from .Nodes import Visuals
from .Nodes import Networks
from .Nodes import Segments

import shutil

__version__ = "1.0.0"

comfy_frontend = os.path.join(comfy_dir, 'web', 'extensions')
frontend_target = os.path.join(comfy_frontend, 'Primere')
frontend_preview_target = os.path.join(comfy_frontend, 'PrimerePreviews')
frontend_source = os.path.join(here, 'front_end')
is_frontend_symlinked = False

WEB_DIRECTORY = "./front_end"
__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']

if os.path.isdir(frontend_target) == True:
    try:
        is_link = os.readlink(frontend_target)
        is_frontend_symlinked = True
    except OSError:
        is_frontend_symlinked = False

    if is_frontend_symlinked == True:
        try:
            os.unlink(frontend_target)
            print('Primere front-end symlinks deleted.')
        except Exception:
            print('[ERROR] - Cannnot unlink Primere front-end folder. Please delete symlink: ' + frontend_target + ' manially from: ' + frontend_source)

if os.path.exists(frontend_target):
    try:
        shutil.rmtree(frontend_target)
        print('Primere front-end folder deleted.')
    except Exception:
        print('[ERROR] - Cannnot delete Primere front-end folder. Please delete manually: ' + frontend_target + ' from: ' + frontend_source)

nodes = []
IGNORE_FRONTEND = ['fonts', 'images', 'keywords', 'jquery', 'vendor']
mainDirs = list(os.walk(frontend_source))[0][1]
valid_FElist = [s for s in mainDirs if s not in IGNORE_FRONTEND] + [frontend_source]

for subdirs in valid_FElist:
    scanPath = os.path.join(frontend_source, subdirs)
    scanFiles = list(Path(scanPath).glob('*.js')) + list(Path(scanPath).glob('*.css'))
    for regFile in scanFiles:
        nodes.append(regFile)

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
    "PrimereResolutionCoordinatorMPX": Dashboard.PrimereResolutionCoordinatorMPX,
    "PrimereNetworkTagLoader": Dashboard.PrimereNetworkTagLoader,
    "PrimereModelKeyword": Dashboard.PrimereModelKeyword,
    "PrimereConceptDataTuple": Dashboard.PrimereConceptDataTuple,
    "PrimereUpscaleModel": Dashboard.PrimereUpscaleModel,

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
    "PrimerePromptOrganizerCSV": Inputs.PrimerePromptOrganizerCSV,
    "PrimereMetaHandler": Inputs.PrimereMetaHandler,
    "PrimereMetaDistributor": Inputs.PrimereMetaDistributor,
    "PrimereMetaDistributorStage2": Inputs.PrimereMetaDistributorStage2,
    "PrimereNetworkDataCollector": Inputs.PrimereNetworkDataCollector,
    "PrimereMetaTupleCollector": Inputs.PrimereMetaTupleCollector,

    "PrimereMetaSave": Outputs.PrimereMetaSave,
    "PrimereAnyOutput": Outputs.PrimereAnyOutput,
    "PrimereTextOutput": Outputs.PrimereTextOutput,
    "PrimereMetaCollector": Outputs.PrimereMetaCollector,
    "PrimereKSampler": Outputs.PrimereKSampler,
    "PrimerePreviewImage": Outputs.PrimerePreviewImage,
    "PrimereAestheticCKPTScorer": Outputs.PrimereAestheticCKPTScorer,

    "PrimereStylePile": Styles.PrimereStylePile,
    "PrimereMidjourneyStyles": Styles.PrimereMidjourneyStyles,
    "PrimereEmotionsStyles": Styles.PrimereEmotionsStyles,

    "PrimereVisualCKPT": Visuals.PrimereVisualCKPT,
    "PrimereVisualLORA": Visuals.PrimereVisualLORA,
    "PrimereVisualEmbedding": Visuals.PrimereVisualEmbedding,
    "PrimereVisualHypernetwork": Visuals.PrimereVisualHypernetwork,
    "PrimereVisualStyle": Visuals.PrimereVisualStyle,
    "PrimereVisualLYCORIS": Visuals.PrimereVisualLYCORIS,
    "PrimereVisualPromptOrganizerCSV": Visuals.PrimereVisualPromptOrganizerCSV,

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
    "PrimereResolutionCoordinatorMPX": "Primere Resolution Coordinator",
    "PrimereNetworkTagLoader": 'Primere Network Tag Loader',
    "PrimereModelKeyword": "Primere Model Keyword",
    "PrimereConceptDataTuple": "Primere Concept Tuple",
    "PrimereUpscaleModel": "Primere Upscale Models",

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
    "PrimerePromptOrganizer": "Primere Prompt Organizer - TOML",
    "PrimerePromptOrganizerCSV": "Primere Prompt Organizer - CSV",
    "PrimereMetaHandler": "Primere Image Recycler",
    "PrimereMetaDistributor": "Primere Meta Distributor",
    "PrimereMetaDistributorStage2": "Primere Meta Distributor Stage 2",
    "PrimereNetworkDataCollector": "Primere Network Data Collector",
    "PrimereMetaTupleCollector": "Primere Meta Tuple Collector",

    "PrimereMetaSave": "Primere Image Meta Saver",
    "PrimereAnyOutput": "Primere Any Debug",
    "PrimereTextOutput": "Primere Text Ouput",
    "PrimereMetaCollector": "Primere Meta Collector",
    "PrimereKSampler": "Primere KSampler",
    "PrimerePreviewImage": "Primere Image Preview and Save as...",
    "PrimereAestheticCKPTScorer": "Primere Aesthetic Scorer",

    "PrimereStylePile": "Primere Style Pile",
    "PrimereMidjourneyStyles": "Primere Midjourney Styles",
    "PrimereEmotionsStyles": "Primere Emotions Styles",

    "PrimereVisualCKPT": "Primere Visual CKPT Selector",
    "PrimereVisualLORA": "Primere Visual LORA Selector",
    "PrimereVisualEmbedding": 'Primere Visual Embedding Selector',
    "PrimereVisualHypernetwork": 'Primere Visual Hypernetwork Selector',
    "PrimereVisualStyle": 'Primere Visual Style Selector',
    "PrimereVisualLYCORIS": 'Primere Visual LYCORIS Selector',
    "PrimereVisualPromptOrganizerCSV": 'Primere Visual Prompt CSV',

    "PrimereLORA": 'Primere LORA',
    "PrimereEmbedding": 'Primere Embedding',
    "PrimereHypernetwork": 'Primere Hypernetwork',
    "PrimereLYCORIS": 'Primere LYCORIS',

    "PrimereImageSegments": 'Primere Image Segments',
    "PrimereAnyDetailer": 'Primere Any Detailer',
}