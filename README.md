# Primere nodes for ComfyUI

Git link: https://github.com/CosmicLaca/ComfyUI_Primere_Nodes

<hr>

Nodepack under development. Manual written by AI, please open issue if something wrong or missing. If you upgrade, just check attached new workflows or use git to downgrade to previous version if something failed.
All required workflows in the `Workflow` folder of the node root.

<hr>

## Minimal workflow: 

<img src="./Workflow/Manual/wf_minimal.jpg" width="800px">

### [Detailed manual for included nodes](Workflow/Manual/nodes/minimal_workflow.md)

### Minimal workflow features:
 
- Prompt selector to any prompt sources
- `CSV` and `TOML` file source readers for saved prompts, automatically organized, saved prompt selection by preview image (if saved)
- Randomized latent noise for variations
- Prompt encoder with selectable custom clip model, long-clip mode with custom models, advanced encoding, injectable internal styles, last-layer options
- Sampler with `variation extender` and `Align Your Step` features
- A1111 style network injection supported by text prompt (Lora, Lycorys, Hypernetwork, Embedding)
- Automatized and manual image saver. Manual image saver with optional **preview saver** for checkpoint selector and saved .csv prompts
- Upscaler (selectable Ultimate SD and hiresFix)

<hr>

## Basic workflow:

The big difference between **minimal** and **basic** workflows, that **basic** workflow automatically detect and support several model concepts. The main difference between workflows the large node group `Concept sampler group` with lot of samplers (1 sampler / concept) and the main `Model concept selector` node. 

<img src="./Workflow/Manual/wf_basic.jpg" width="800px">

### [Detailed manual for included nodes](Workflow/Manual/nodes/basic_workflow.md)

### Basic workflow features:

#### Same as Minimal workflow plus:

- **Half-automatic model concept selector:**
  - **Supported concepts:** SD1, SD2, SDXL, SD3, StableCascade, Turbo, Flux, KwaiKolors, Hunyuan, Playground, Pony, LCM, Lightning, Hyper, PixartSigma
  - Custom (and different) sampler settings for all concepts. The main ide that set sampler nodes only one time (`sampler`, `scheduler`, `step`, `cfg`) then just select model only what will use right sampler setting by `Model concept selector`. 
  - Auto detection of selected model type (if data already stored on external .json file, see longer [manual](Workflow/Manual/nodes/basic_workflow.md))
  - Auto **download** and apply Hyper, Lightning, and Turbo speed loras at first usage from here: https://huggingface.co/ByteDance/Hyper-SD/tree/main **check your SSD space before!**

**On the `Concept selector` node you will see `None` on all required fields, for example on Cascade and Flux files. You must install all required additional files manually to right path, then select correct model/clip/vae on lists.**