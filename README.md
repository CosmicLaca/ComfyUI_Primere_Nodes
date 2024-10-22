# Primere nodes for ComfyUI

Git link: https://github.com/CosmicLaca/ComfyUI_Primere_Nodes

<hr>

Nodepack under development. Manual will be ready later. If you ugrade just check attached workflows or use git to downgrade to previous version.

<hr>

All workflows in the `Workflow` folder of the node root.

## Minimal workflow

<img src="./Workflow/primere-minimal.png" width="800px">

### Included features:
 
- Prompt selector to any prompt sources
- `CSV` and `TOML` file source reader for prompts, automatically organized, saved prompt selection by preview image
- Randomized latent noise
- Prompt encoder with selectable custom clip model, long-clip mode with custom models, advanced encoding, injectable styles, internal styles, last-layer options
- Sampler with `variation extender` and `Align Your Step`
- A1111 style network injection supported by text prompt (Lora, Lycorys, Hypernetwork, Embedding)
- Automatized and manual image saver. Manual image saver with optional preview saver for checkpoint selector and saved CSV prompts
- Upscaler (Ultimate SD and hiresFix)

**Examples:**

Visual checkpopint selection, automatized filtering by subdirectories (first row of buttons) and versions (second row of buttons):

<img src="./Workflow/Manual/visual_checkpoint.jpg" width="600px">

Visual checkpopint selection `(csv source)`, automatized filtering by categories:

<img src="./Workflow/Manual/visual_csv.jpg" width="600px">

<hr>

## Basic workflow

<img src="./Workflow/primere-basic.png" width="800px">

### Included features:

#### Same as Minimal workflow plus:

- Half-automatic concept selector
  - Supported concepts: "SD1", "SD2", "SDXL", "SD3", "StableCascade", "Turbo", "Flux", "KwaiKolors", "Hunyuan", "Playground", "Pony", "LCM", "Lightning", "Hyper"
  - Custom sampler settings for all supported concepts
  - Auto detection of selected model type
  - Auto download and apply speed loras at first usage from here: https://huggingface.co/ByteDance/Hyper-SD/tree/main **check your SSD space before**

- Terminal helper to detect and store model version:
  - Open `cmd` terminal window 
  - Activate your Comfy `venv`
  - Change to `[Your_comfy_folder]\custom_nodes\ComfyUI_Primere_Nodes\terminal_helpers\`
  - 