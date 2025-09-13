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

__version__ = "1.6.0"

comfy_frontend = os.path.join(comfy_dir, 'web', 'extensions')
frontend_target = os.path.join(comfy_frontend, 'Primere')
frontend_preview_target = os.path.join(comfy_frontend, 'PrimerePreviews', 'images')
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

if os.path.isdir(frontend_preview_target) == True:
    preview_images = os.path.join(frontend_source, 'images')
    try:
        shutil.copytree(frontend_preview_target, preview_images, dirs_exist_ok=True, symlinks=False, ignore=None)
        print('Primere previews copied back to the original node directory.')
        original_prv_path = os.path.join(comfy_frontend, 'PrimerePreviews')
        shutil.rmtree(original_prv_path)
        print('Primere previews removed from Comfy web path.')
    except Exception:
        print('[ERROR] - Cannnot copy Primere previews to right path. Please copy manually from: ' + frontend_preview_target + ' to: ' + preview_images)

nodes = []
IGNORE_FRONTEND = ['fonts', 'keywords', 'jquery', 'vendor']
mainDirs = list(os.walk(frontend_source))[0][1]
valid_FElist = [s for s in mainDirs if s not in IGNORE_FRONTEND] + [frontend_source]

for subdirs in valid_FElist:
    scanPath = os.path.join(frontend_source, subdirs)
    scanFiles = list(Path(scanPath).glob('*.js')) + list(Path(scanPath).glob('*.map')) + list(Path(scanPath).glob('*.css')) + list(Path(scanPath).glob('*.jpg'))
    for regFile in scanFiles:
        nodes.append(regFile)

NODE_CLASS_MAPPINGS = {
    "PrimereSamplersSteps": Dashboard.PrimereSamplersSteps,
    "PrimereVAE": Dashboard.PrimereVAE,
    "PrimereCKPT": Dashboard.PrimereCKPT,
    "PrimereVAELoader": Dashboard.PrimereVAELoader,
    "PrimereCKPTLoader": Dashboard.PrimereCKPTLoader,
    "PrimerePromptSwitch": Dashboard.PrimerePromptSwitch,
    "PrimereSeed": Dashboard.PrimereSeed,
    "PrimereFastSeed": Dashboard.PrimereFastSeed,
    "PrimereLatentNoise": Dashboard.PrimereFractalLatent,
    "PrimereCLIPEncoder": Dashboard.PrimereCLIP,
    "PrimereResolution": Dashboard.PrimereResolution,
    "PrimereClearNetworkTagsPrompt": Dashboard.PrimereClearNetworkTagsPrompt,
    "PrimereDiTPurifyPrompt": Dashboard.PrimereDiTPurifyPrompt,
    "PrimereModelConceptSelector": Dashboard.PrimereModelConceptSelector,
    "PrimereConceptDataTuple": Dashboard.PrimereConceptDataTuple,
    "PrimereResolutionMultiplierMPX": Dashboard.PrimereResolutionMultiplierMPX,
    "PrimereResolutionCoordinatorMPX": Dashboard.PrimereResolutionCoordinatorMPX,
    "PrimereNetworkTagLoader": Dashboard.PrimereNetworkTagLoader,
    "PrimereModelKeyword": Dashboard.PrimereModelKeyword,
    "PrimereUpscaleModel": Dashboard.PrimereUpscaleModel,

    "PrimerePrompt": Inputs.PrimereDoublePrompt,
    "PrimereStyleLoader": Inputs.PrimereStyleLoader,
    "PrimereDynamicParser": Inputs.PrimereDynParser,
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
    "PrimereLLMEnhancer": Inputs.PrimereLLMEnhancer,
    "PrimereLLMEnhancerOptions": Inputs.PrimereLLMEnhancerOptions,
    "PrimereImgToPrompt": Inputs.PrimereImgToPrompt,

    "PrimereMetaSave": Outputs.PrimereMetaSave,
    "PrimereAnyOutput": Outputs.PrimereAnyOutput,
    "PrimereTextOutput": Outputs.PrimereTextOutput,
    "PrimereMetaCollector": Outputs.PrimereMetaCollector,
    "PrimereKSampler": Outputs.PrimereKSampler,
    "PrimerePreviewImage": Outputs.PrimerePreviewImage,
    "PrimereAestheticCKPTScorer": Outputs.PrimereAestheticCKPTScorer,
    "DebugToFile": Outputs.DebugToFile,

    "PrimereStylePile": Styles.PrimereStylePile,
    "PrimereMidjourneyStyles": Styles.PrimereMidjourneyStyles,
    "PrimereEmotionsStyles": Styles.PrimereEmotionsStyles,
    "PrimereLensStyles": Styles.PrimereLensStyles,

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
    "PrimereFaceAnalyzer": Segments.PrimereFaceAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimereSamplersSteps": "Primere Samplers & Steps & Cfg",
    "PrimereVAE": "Primere VAE Selector",
    "PrimereCKPT": "Primere CKPT Selector",
    "PrimereVAELoader": "Primere VAE Loader",
    "PrimereCKPTLoader": "Primere CKPT Loader",
    "PrimerePromptSwitch": "Primere Prompt Switch",
    "PrimereSeed": 'Primere Seed',
    "PrimereFastSeed": "Primere Fast Seed",
    "PrimereLatentNoise": "Primere Noise Latent",
    "PrimereCLIPEncoder": "Primere Prompt Encoder",
    "PrimereResolution": "Primere Resolution",
    "PrimereClearNetworkTagsPrompt": "Primere Network Tag Cleaner",
    "PrimereDiTPurifyPrompt": "Primere DiT Purify Prompt",
    "PrimereModelConceptSelector": "Primere Model Concept Selector",
    "PrimereResolutionMultiplierMPX": "Primere Resolution MPX",
    "PrimereResolutionCoordinatorMPX": "Primere Resolution Coordinator",
    "PrimereNetworkTagLoader": 'Primere Network Tag Loader',
    "PrimereModelKeyword": "Primere Model Keyword",
    "PrimereConceptDataTuple": "Primere Concept Tuple",
    "PrimereUpscaleModel": "Primere Upscale Models",

    "PrimerePrompt": "Primere Prompt",
    "PrimereStyleLoader": "Primere Styles",
    "PrimereDynamicParser": "Primere Dynamic",
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
    "PrimereLLMEnhancer": "Primere LLM Enhancer",
    "PrimereLLMEnhancerOptions": "Primere LLM Options",
    "PrimereImgToPrompt": "Primere Img2Prompt",

    "PrimereMetaSave": "Primere Image Meta Saver",
    "PrimereAnyOutput": "Primere Any Debug",
    "PrimereTextOutput": "Primere Text Ouput",
    "PrimereMetaCollector": "Primere Meta Collector",
    "PrimereKSampler": "Primere KSampler",
    "PrimerePreviewImage": "Primere Image Preview and Save as...",
    "PrimereAestheticCKPTScorer": "Primere Aesthetic Scorer",
    "DebugToFile": "Primere Debug To File",

    "PrimereStylePile": "Primere Style Pile",
    "PrimereMidjourneyStyles": "Primere Midjourney Styles",
    "PrimereEmotionsStyles": "Primere Emotions Styles",
    "PrimereLensStyles": "Primere Lens Styles",

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
    "PrimereFaceAnalyzer": "Primere Face Analyzer",
}