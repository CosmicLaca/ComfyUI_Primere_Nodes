# Primere nodes for ComfyUI

Git link: https://github.com/CosmicLaca/ComfyUI_Primere_Nodes

<hr>

Nodepack under development. Manual will be ready later. If you ugrade just check attached workflows or use git to downgrade to previous version.

<hr>

All workflows in the `Workflow` folder of the node root.

## Minimal workflow ready

<img src="./Workflow/primere-minimal.png" width="800px">

### Included features:
 
- Prompt selector
- `CSV` and `TOML` prompt source reader, automatically organized, saved prompt selection by preview image
- Randomized latent noise
- Prompt encoder with selectable custom clip model, long-clip mode, advanced encoding, injectable styles, internal styles, last-layer options
- Sampler with `variation extender` and `Align Your Step`
- A1111 style network injection supported by prompt (Lora, Lycorys, Hypernetwork, Embedding)
- Automatized and manual image saver. Manual image saver with optional preview saver for checkpoint selector and saved CSV prompts

**Examples:**

Visual checkpopint selection, automatized filtering by subdirectories (first row of buttons) and versions (second row of buttons):

<img src="./Workflow/Manual/visual_checkpoint.jpg" width="600px">

Visual checkpopint selection `(csv source)`, automatized filtering by categories:

<img src="./Workflow/Manual/visual_csv.jpg" width="600px">
