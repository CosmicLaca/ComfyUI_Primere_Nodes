# Primere nodes for ComfyUI

**Git link:** https://github.com/CosmicLaca/ComfyUI_Primere_Nodes

**Primere Youtube channel:** https://www.youtube.com/@PrimereComfydev/videos

**Install required 3rd party nodepack dependency:** https://github.com/city96/ComfyUI_ExtraModels

<hr>

Nodepack under development. Manual written by AI, please open issue if something wrong or missing. If you upgrade, check attached new workflows or use git to downgrade to previous version if something failed.

**ComfyUI core functions requirement:** This nodepack uses new ComfyUI core functions. Update your ComfyUI installation before using. Use `git pull` on your ComfyUI folder.

**All example workflows in the `Workflow` folder of the node root.**
- Within `Workflow` folder you will find subfolders:
  - **NewFE:** workflows compatible with current ComfyUI front-end
  - **Development:** workflows for node testing only

<hr>

> [!TIP]
> ## Something different: Universal API node 🚀
> Explore the Uniapi quick manual: **[PrimereApiProcessor (Uniapi) guide](Workflow/Manual/nodes/uniapi.md)**.
> 
> <img src="./Workflow/Manual/nodes/uniapi.jpg" width="500px">

<hr>

## Minimal workflow

<img src="./Workflow/Manual/wf_minimal.jpg" width="800px">

### <ins>[Detailed manual for included nodes](Workflow/Manual/nodes/minimal_workflow.md)</ins>

Core generation pipeline. Single prompt input with full model concept support.

**Supported model concepts:** SD1, SD2, SDXL, Illustrious, SD3, StableCascade, Chroma, Z-Image, Turbo, Flux, Nunchaku, QwenGen, QwenEdit, WanImg, KwaiKolors, Hunyuan, Playground, Pony, LCM, Lightning, Hyper, PixartSigma, SANA1024, SANA512, AuraFlow. Future support: HiDream, Mochi, WanT2V, WanI2V, Cosmos, Flux2, SSD, SegmindVega, KOALA, StableZero, SV3D, SD09, StableAudio, LTXV.

### Minimal workflow features:

- Central model concept selector node controls sampler, VAE, CLIP settings per model type
- Automatic model keyword insertion to prompt
- Prompt selector to any prompt sources
- Prompt can be saved to `CSV` file directly from the prompt input nodes
- `CSV` and `TOML` file readers for saved prompts, automatically organized, saved prompt selection by preview image (if preview created)
- Randomized latent noise for variations
- Prompt encoder with selectable custom clip model, long-clip mode with custom models, advanced encoding, injectable internal styles, last-layer options
- Sampler with `variation extender` and `Align Your Steps` features
- A1111 style network injection supported by text prompt (LoRA, LyCoris, Hypernetwork, Embedding)
- Automatized and manual image saver with optional **preview saver** for checkpoint selectors and saved .csv prompts
- Aesthetic scorer for final image quality assessment
- Upscaler (selectable Ultimate SD and hiresFix)
- Dynamic prompt support
- Auto clean incompatible network tags from prompt by model architecture

<hr>

## Basic workflow

<img src="./Workflow/Manual/wf_basic.jpg" width="800px">

### <ins>[Detailed manual for included nodes](Workflow/Manual/nodes/basic_workflow.md)</ins>

Professional prompt development workflow. Extended prompt management for testing and iteration.

**Same model support as Minimal workflow.**

### Basic workflow features:

#### Same as Minimal workflow plus:

- 12 prompt inputs with 1-click selector for prompt switching
- Efficient workflow for prompt developers testing multiple variations

<hr>

## Basic Production workflow

<img src="./Workflow/Manual/wf_basic_prod.jpg" width="800px">

### <ins>[Detailed manual for included nodes](Workflow/Manual/nodes/basic_production_workflow.md)</ins>

Full production pipeline with styling, refinement, and selective output.

**Same model support as Minimal workflow.**

### Basic Production workflow features:

#### Same as Basic workflow plus:

- 19 prompt sources for special cases (Daily Challenges, Articles, etc.)
- Style block: Style Pile, Midjourney, Emotions, Camera Lens (prompt injection styling)
- 4 separated refiner detailer blocks: Face, Eye, Mouth, Hands
- Selective image saver: saves only images exceeding user-defined aesthetic score threshold (portfolio filtering)

<hr>

## Workflow Comparison

| Feature | Minimal         | Basic | Basic Production |
|---------|-----------------|-------|------------------|
| **Primary Use** | Core generation | Prompt development | Production pipeline |
| **Prompt Inputs** | 4               | 12 | 19 |
| **Prompt Selector** | 1-click         | 1-click | 1-click |
| **Model Concepts** | 25+             | 25+ | 25+ |
| **Model Keyword Insertion** | ✓               | ✓ | ✓ |
| **CSV/TOML Readers** | ✓               | ✓ | ✓ |
| **Network Injection** | ✓               | ✓ | ✓ |
| **Variation Extender** | ✓               | ✓ | ✓ |
| **Image Saver** | ✓               | ✓ | ✓ |
| **Aesthetic Scorer** | ✓               | ✓ | ✓ |
| **Style Block** | —               | — | ✓ |
| **Refiner Blocks** | —               | — | ✓ (Face, Eye, Mouth, Hands) |
| **Selective Saver** | —               | — | ✓ |

<hr>
