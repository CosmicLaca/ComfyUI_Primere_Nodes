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

__version__ = "0.2.0"

comfy_frontend = os.path.join(comfy_dir, 'web', 'extensions')
frontend_target = os.path.join(comfy_frontend, 'Primere')


if os.path.exists(frontend_target) == False:
    frontend_source = os.path.join(here, 'front_end')
    src = Path(frontend_source).as_posix()
    dst = Path(frontend_target).as_posix()

    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(src, dst)
        else:
            os.symlink(Path(frontend_source).as_posix(), Path(frontend_target).as_posix())
        print(f"Primere front-end folder symlinked to {frontend_target}")

    except OSError:
        print(f"Failed to create frint-end symlink to {frontend_target}, trying to copy it")
        try:
            import shutil
            shutil.copytree(frontend_source, frontend_target)
            print(f"Successfully copied {frontend_source} to {frontend_target}")
        except Exception as e:
            print(f"Failed to symlink and copy {frontend_source} to {frontend_target}. Please copy the folder manually.")
    except Exception as e:
        print(f"Failed to create symlink to {frontend_target}. Please copy the folder manually.")

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

    "PrimereMetaSave": Outputs.PrimereMetaSave,
    "PrimereAnyOutput": Outputs.PrimereAnyOutput,
    "PrimereTextOutput": Outputs.PrimereTextOutput,
    "PrimereMetaCollector": Outputs.PrimereMetaCollector,
    "PrimereKSampler": Outputs.PrimereKSampler,

    "PrimereStylePile": Styles.PrimereStylePile,
    "PrimereMidjourneyStyles": Styles.PrimereMidjourneyStyles,

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

    "PrimerePrompt": "Primere Prompt",
    "PrimereStyleLoader": "Primere Styles",
    "PrimereDynamicParser": "Primere Dynamic",
    "PrimereVAESelector": "Primere VAE Selector",
    "PrimereMetaRead": "Primere Exif Reader",
    "PrimereEmbeddingHandler": "Primere Embedding Handler",
    "PrimereLoraStackMerger": "Primere Lora Stack Merger",
    "PrimereLoraKeywordMerger": 'Primere Lora Keyword Merger',
    "PrimereEmbeddingKeywordMerger": "Primere Embedding Keyword Merger",
    "PrimereLycorisStackMerger": 'Primere Lycoris Stack Merger',
    "PrimereLycorisKeywordMerger": 'Primere Lycoris Keyword Merger',
    "PrimereRefinerPrompt": "Primere Refiner Prompt",

    "PrimereMetaSave": "Primere Image Meta Saver",
    "PrimereAnyOutput": "Primere Any Debug",
    "PrimereTextOutput": "Primere Text Ouput",
    "PrimereMetaCollector": "Primere Meta Collector",
    "PrimereKSampler": "Primere KSampler",

    "PrimereStylePile": "Primere Style Pile",
    "PrimereMidjourneyStyles": "Primere Midjourney Styles",

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