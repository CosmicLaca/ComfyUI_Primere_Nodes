# Minimal Workflow - Node Groups Manual

<hr>

## Dashboard Group

The Dashboard consolidates model loading, seed control, and resolution settings in a unified interface. This group handles all baseline generation parameters before prompt encoding.

<img src="./Dashboard.jpg" width="800px">

<hr>

### Visual Checkpoint Selector:

**Purpose:** Select and load AI model checkpoints with visual preview interface, automatic directory filtering, and aesthetic quality scoring.

#### Primary Settings:

| Setting | Purpose |
|---------|---------|
| `base_model` | Select checkpoint from dropdown or visual gallery |
| `show_modal` | Toggle visual gallery preview mode ON/OFF |
| `preview_path` | Choose preview image source: "Primiere legacy" or "Model path" |
| `show_hidden` | Show/hide hidden files and folders (dot-prefixed) |
| `random_model` | Automatically select random checkpoint from current folder |

#### Aesthetic Scoring Display on visual previews:

| Setting | Purpose |
|---------|---------|
| `aescore_percent_min` | Lower bound for quality scaling (maps to 0%) |
| `aescore_percent_max` | Upper bound for quality scaling (maps to 100%) |

### The visual preview modal:

<img src="./visual_checkpoint.jpg" width="600px">

#### Interface Layout:

**Row 1 - Directory Structure:** Filter checkpoints by folder organization (e.g., Root, Flux, SD1, SDXL, Photo, Design, Character, Style, etc.). Buttons represent your checkpoint folder hierarchy for quick categorization.

**Row 2 - Model Types:** Filter by supported model concept `(SD1, SD2, SDXL, Flux, Hunyuan, LCM, Lightning, Playground, Pony, SD3, Turbo, StableCascade, KwaiKolors, PixartSigma, SANA, Illustrious, Nunchaku, OwnGen, Wan2V, Wan1Edit, AuraFlow, Chroma, Waning, HiDream, Mochi, Cosmos, Flux2, LTXV, Z-Image)`.

**Row 3 - Controls:** Filter text search, clear filter, sort options `(aScore, Name, Version, Path, Date, Symlink, STime)`, and sort direction.

#### Preview badges show:

<img src="./previews.jpg" width="600px">

- **Top left:** Model concept (Flux, SD1, SDXL, etc.)
- **Top right:** Symlink type if applicable
- **Bottom:** Average aesthetic score as percentage (based on min/max range)

**Note:** Aesthetic scores require pre-computed data from aesthetic scorer node stored with checkpoint.

---

### Checkpoint Loader

Automatically loads selected checkpoint, VAE, and CLIP model based on Visual Checkpoint Selector choice and automatically deteted model concept. This node ccontolled by the data tupple on the external `control_data` input. Outputs:
- `loaded_model`: Model tensor
- `loaded_clip`: CLIP encoder
- `loaded_vae`: VAE decoder
- `control_data`: Model metadata and version info

**No manual configuration required** — uses cached settings of saved model concept json.

---

### Fast Seed Control

`Why fast? Because no fron-tend for seed generation, only for result display. For large queue settings much faster than anything else.` 

| Setting | Purpose |
|---------|---------|
| `seed_setup` | "Random" = new seed each run, "Custom" = use fixed value |
| `custom_seed` | Fixed seed value when `seed_setup` = "Custom" |
| `random_seed` | Read-only output showing last generated seed |

---

### Resolution Selector

**Purpose:** Define output image dimensions with preset aspect ratios or custom settings.

#### Basic Selection:

| Setting | Options                                        | Purpose |
|---------|------------------------------------------------|---------|
| `ratio` | Photo, Portrait, Square, HD, HD+, Old TV, etc. | Predefined aspect ratios |
| `resolution` | Auto / Manual                                  | "Auto" = model-based defaults, "Manual" = custom base resolution |

#### Model-Based Auto Resolution:

When `resolution` = "Auto":
- resolution automatically set by the internal settings for actual selected model concept.

#### Manual Resolution:

When `resolution` = "Manual", manually specify base dimensions per model:
- set the target resolution in megapixels on `manual_res` combo. This will overwrite the automatic value.

#### Orientation Control:

| Setting | Purpose |
|---------|---------|
| `orientation` | Horizontal or Vertical composition |
| `rnd_orientation` | Randomly alternate between orientations (useful for batch generation) |
| `round_to_standard` | Force dimensions to model-native standards |

#### Custom Aspect Ratios:

Enable `calculate_by_custom` to define custom ratios (e.g., 1.6:2.8):
- `custom_side_a`: First ratio component (e.g., 1.60)
- `custom_side_b`: Second ratio component (e.g., 2.80)

**Note:** Preset ratios editable in `Toml/resolution_ratios.toml`

<hr>

## Prompt Input Group

The Prompt Input Group handles manual prompt entry, saving, and visual selection of pre-built or saved prompts. This group provides flexible workflows for both real-time prompt writing and reusable prompt library management.

<img src="./Prompt_group.jpg" width="1000px">

<hr>

### Primiere Prompt (Manual Input)

**Purpose:** Write positive and negative prompts directly with organization metadata, then save to external CSV file for later reuse.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `PROMPT+` | Positive prompt text area (multiline) |
| `PROMPT-` | Negative prompt text area (multiline) |
| `SUBPATH` | Folder organization category for saved images |
| `MODEL` | Preferred checkpoint for this prompt |
| `ORIENTATION` | Preferred image orientation (None/Horizontal/Vertical/Random) |

#### Outputs:

| Output | Purpose                                                 |
|--------|---------------------------------------------------------|
| `PROMPT+` | Positive prompt text                                    |
| `PROMPT-` | Negative prompt text                                    |
| `SUBPATH` | Image save category                                     |
| `MODEL` | Preferred model                                         |
| `ORIENTATION` | Preferred orientation                                   |
| `PREFERRED` | Aggregation of model, subpath + orientation preferences |

#### Save to External CSV:

Click "Save prompt to file..." button to open the prompt saver dialog. This saves your manually crafted prompt to an external CSV file for reuse via Visual Style Selector or Visual Prompt CSV nodes.

<img src="./primere_prompt_save.jpg" width="600px">

**Dialog Fields:**
- **Prompt name:** Create new or overwrite existing saved prompt from list
- **Prompt category (subpath):** Assign category/folder (e.g., "SeasonBackground", "Architecture", "Nature")
- **Positive prompt:** Auto-populated from input, editable
- **Negative prompt:** Auto-populated from input, editable
- **Preferred Model:** Read-only, set from Primiere Prompt input node
- **Preferred Orientation:** Read-only, set from Primiere Prompt input node

**Workflow Benefit:** Save a well-crafted prompt once, then load it instantly across projects via Visual Style Selector or Visual Prompt CSV nodes without retyping.

**Backup Note:** Keep backup of `custom_nodes/ComfyUI_Primere_Nodes/stylecsv/styles.csv` before bulk changes.

---

### Visual Style Selector

**Purpose:** Select and load pre-saved prompts with visual gallery preview and category filtering.

<img src="./visual_csv.jpg" width="800px">

#### Settings:

| Setting | Options | Purpose |
|---------|---------|---------|
| `styles` | Dropdown | Choose saved prompt from list |
| `show_modal` | ON/OFF | Toggle between dropdown list and visual gallery preview |
| `show_hidden` | ON/OFF | Show/hide items with "nsfw" in name or path |
| `use_subpath` | ON/OFF | Apply the saved prompt's folder category to output |
| `use_model` | ON/OFF | Apply the saved prompt's preferred checkpoint |
| `use_orientation` | ON/OFF | Apply the saved prompt's preferred orientation |
| `random_prompt` | ON/OFF | Randomly select from same category as currently selected prompt |
| `aescore_percent_min` | Integer | Lower quality threshold (maps to 0%) for preview sorting |
| `aescore_percent_max` | Integer | Upper quality threshold (maps to 100%) for preview sorting |

#### Modal Interface:

**Row 1 - Category Buttons:** Filter by saved prompt categories `(Architecture, Art, Character, Design, Edit, Horror, Influencer, Nature, Photography, Sci-Fi, Vehicles, Others)`

**Row 2 - Controls:** Text filter, clear filter, sort options `(aScore, Name, Path)` and sort direction `(Acending, Descending)`.

**Gallery:** Preview images of each saved prompt with average value of aesthetic scores calculated from historic generations.

**CSV Source:** Rename `stylecsv/styles.example.csv` to `stylecsv/styles.csv` and populate with your own prompts. Format: `columns for name, positive, negative, category (subpath), preferred model, preferred orientation`.

---

### Visual Prompt CSV (Auto-Organized)

**Purpose:** Load saved prompts with automatic category organization and visual preview. Same functionality as Visual Style Selector but with pre-organized category dropdowns.

#### Settings:

Same as Visual Style Selector `(show_modal, show_hidden, use_subpath, use_model, use_orientation, random_prompt, aescore thresholds)`.

#### Workflow:

1. Select category from dropdown (e.g., "Architecture", "Nature")
2. Choose specific prompt from that category
3. All metadata `(prompt text, model, orientation)` loads automatically

**Primary Difference from Style Selector:** Categories automatically organized from CSV folder structure, useful for large prompt libraries.

---

### Primiere Prompt Switch

**Purpose:** Dynamically select between multiple prompt sources in a single node.

#### Functionality:

- Connect multiple prompt source nodes `(Primiere Prompt, Visual Style Selector, Visual Prompt CSV, or compatible nodes)`
- Use `select` input to choose active prompt (1, 2, 3, etc.)
- Only the selected prompt flows downstream; others inactive

**Use Case:** Test multiple prompt variants without reconnecting nodes. Connect 3-5 different prompt sets and toggle between them with a single numerical selector.

#### Outputs:

| Output | Purpose |
|--------|---------|
| `PROMPT+` | Active positive prompt from selected source |
| `PROMPT-` | Active negative prompt from selected source |
| `SUBPATH` | Active subpath from selected source |
| `MODEL` | Active model from selected source |
| `ORIENTATION` | Active orientation from selected source |
| `PREFERRED` | Aggregated preferences from selected source |

---

### Primiere Dynamic

**Purpose:** Support dynamic prompt syntax with random selection.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `dyn_prompt` | Prompt string with dynamic template syntax |
| `seed` | Seed for reproducible randomization |

**Syntax:** Use bracket notation for random selection: `{option1|option2|option3}`

**Output:** `PROMPT` — resolved prompt with one random option selected per bracket.

**Manual:** [https://github.com/adieyal/sd-dynamic-prompts/blob/main/docs/SYNTAX.md#basic-syntax](https://github.com/adieyal/sd-dynamic-prompts/blob/main/docs/SYNTAX.md#basic-syntax)

---

### Primiere Embedding Handler

**Purpose:** Auto-convert Automatic1111-style embedding syntax to ComfyUI format.

#### Conversion:

- **Input:** `Autumn` (plain word matching embedding filename, case-sensitive)
- **Output:** `embedding:Autumn` (ComfyUI-compatible format)

<img src="./prompt_embedding.jpg" width="600px">

**Requirement:** Embedding file must exist with exact name match.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `positive_prompt` | Positive prompt with potential embedding keywords |
| `negative_prompt` | Negative prompt with potential embedding keywords |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `positive_prompt` | Converted positive prompt with embedding: prefixes |
| `negative_prompt` | Converted negative prompt with embedding: prefixes |

---

<hr>

## Encoder, Sampler, Decoder Group

The Encoder, Sampler, Decoder group handles prompt encoding to latent space, noise generation, sampling/diffusion, latent-to-image conversion, and quality scoring. This is the core generation pipeline.

<img src="./Sampler_group.jpg" width="1200px">

<hr>

### Primiere Noise Latent

**Purpose:** Generate random latent noise tensor as initialization for diffusion process.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `width` | Image width in pixels |
| `height` | Image height in pixels |
| `noise_seed` | Seed for reproducible noise generation |
| `optional_vae` | VAE for noise scaling (optional) |

#### Settings:

| Setting | Purpose                                            |
|---------|----------------------------------------------------|
| `rand_noise_type` | Randomize noise type per generation (ON/OFF)       |
| `noise_type` | Noise distribution: white, gaussian, etc.          |
| `rand_alpha_exponent` | Randomize alpha exponent per generation (ON/OFF)   |
| `alpha_exponent` | Alpha blending exponent (default 1.000)            |
| `alpha_exp_rand_min` | Minimum random alpha exponent (default 0.500)      |
| `alpha_exp_rand_max` | Maximum random alpha exponent (default 1.500)      |
| `rand_modulator` | Randomize modulation per generation (ON/OFF)       |
| `modulator` | Noise modulation factor (default 1.00)             |
| `modulator_rand_min` | Minimum random modulation (default 0.80)           |
| `modulator_rand_max` | Maximum random modulation (default 1.40)           |
| `rand_device` | Randomize device (GPU/CPU) per generation (ON/OFF) |
| `device` | Compute device: cpu, cuda                          |
| `expand_random_limits` | Expand limits beyond defaults (ON/OFF)             |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `LATENTS` | Generated latent noise tensor |
| `PREVIEWS` | Preview images of noise pattern |
| `CONTROL_DATA` | Metadata and generation settings |

---

### Primiere Prompt Encoder

**Purpose:** Encode positive and negative prompts into CLIP embeddings for guidance during sampling.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `clip` | CLIP model from Checkpoint Loader |
| `positive_prompt` | Positive prompt text to encode |
| `negative_prompt` | Negative prompt text to encode |
| `enhanced_prompt` | Optional enhanced/refined positive prompt |
| `edit_image_list` | Optional image list for editing mode |
| `edit_vae` | Optional VAE for image editing |

#### Settings:

| Setting | Purpose |
|---------|---------|
| `negative_strength` | Strength multiplier for negative guidance (default 1.20) |
| `use_int_style` | Use internal style preprocessing (ON/OFF) |
| `int_style_pos` | Positive internal style application (None, custom, etc.) |
| `int_style_pos_strength` | Strength of positive style (default 1.00) |
| `int_style_neg` | Negative internal style (None, custom, etc.) |
| `int_style_neg_strength` | Strength of negative style (default 1.00) |
| `adv_encode` | Advanced encoding mode (ON/OFF) |
| `token_normalization` | Token normalization method: mean, max, etc. |
| `weight_interpretation` | Weight interpretation: comfy++, A1111, etc. |
| `enhanced_prompt_usage` | How to use enhanced prompt: T5-XXL, append, etc. |
| `enhanced_prompt_strength` | Strength of enhanced prompt (default 1.00) |
| `opt_pos_strength` | Optimization strength for positive (default 1.00) |
| `opt_neg_strength` | Optimization strength for negative (default 1.00) |
| `style_handling` | How to handle style injection: Style-prompt merge, etc. |
| `style_position` | Where to inject style: Style to end of prompt, etc. |
| `style_swap` | Swap style to T5/L vs prompt to default clip (ON/OFF) |
| `style_pos_strength` | Style positive strength (default 1.00) |
| `style_neg_strength` | Style negative strength (default 1.00) |
| `L_strength` | L-strength modifier (default 1.00) |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `COND+` | Positive CLIP conditioning |
| `COND-` | Negative CLIP conditioning |
| `PROMPT+` | Positive prompt text (pass-through) |
| `PROMPT-` | Negative prompt text (pass-through) |
| `T5XXL_PROMPT` | T5-XXL specific encoding |
| `PROMPT_L` | L-layer specific encoding |
| `PROMPT_L` | Additional L-layer variants (model-dependent) |
| `CONTROL_DATA` | Encoding metadata |

---

### Primiere KSampler

**Purpose:** Execute diffusion sampling process to generate latent representations from noise.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `model` | Diffusion model from Checkpoint Loader |
| `seed` | Seed for sampling reproducibility |
| `steps` | Number of sampling steps (higher = more quality, slower) |
| `cfg` | Classifier-free guidance scale (higher = stronger prompt adherence) |
| `sampler_name` | Sampling algorithm: euler, dpmpp, etc. |
| `scheduler_name` | Noise schedule: karras, exponential, etc. |
| `positive` | Positive CLIP conditioning from Encoder |
| `negative` | Negative CLIP conditioning from Encoder |
| `latent_image` | Starting latent tensor from Noise Latent |

#### Settings:

| Setting | Purpose |
|---------|---------|
| `denoise` | Denoising strength: 0.0-1.0 (1.0 = full generation, 0.0 = no change) |
| `variation_extender` | Enable variation/consistency extender (default 0.00) |
| `variation_batch_step` | Batch step for variation (default 0.00) |
| `variation_level` | Variation quality level: Maximize, Normal, etc. |
| `device` | Compute device: DEFAULT (uses model device) |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `LATENT` | Generated latent tensor |
| `CONTROL_DATA` | Sampling metadata |

---

### Primiere Aesthetic Scorer

**Purpose:** Calculate aesthetic quality score for generated images and optionally store for checkpoint/prompt metadata.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `image` | Image tensor to score |
| `control_data` | Optional metadata tuple from workflow |

#### Settings:

| Setting | Purpose |
|---------|---------|
| `get_aesthetic_score` | Calculate and return score (ON/OFF) |
| `add_to_checkpoint` | Store score with checkpoint metadata (ON/OFF) |
| `add_to_saved_prompt` | Store score with saved prompt (ON/OFF) |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `SCORE` | Aesthetic score value (0.0-10.0 or normalized) |

**Benefit:** Pre-compute quality metrics during generation for later filtering and sorting in Visual Checkpoint Selector and Visual Style Selector modals.

---

<hr>

## Network Tag Handler Group

The Network Tag Handler group manages network injection (LoRA, LyCoris, Hypernetwork, Embedding) from A1111-style prompt syntax. This group cleans incompatible tags and injects appropriate networks based on model architecture.

<img src="./Network_gropup.jpg" width="600px">

<hr>

### Primiere Network Tag Cleaner

**Purpose:** Remove incompatible network tags from prompts based on model architecture to prevent errors and ensure workflow compatibility.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `model_version` | Current model concept (auto-detected from checkpoint) |
| `positive_prompt` | Positive prompt text potentially containing network tags |
| `negative_prompt` | Negative prompt text potentially containing network tags |

#### Cleaning Modes:

**Auto Cleaner Mode:**
- `auto_remover`: Enable automatic removal based on detected model concept
- When ON: Automatically removes all incompatible network tags for the selected model concept
- Only keeps tags compatible with current checkpoint architecture

**Manual Cleaner Mode:**
- `auto_remover`: OFF
- Manually enable/disable per network type: embedding, LoRA, LyCoris, Hypernetwork
- Architecture-specific toggles for each model concept:

| Setting | Purpose |
|---------|---------|
| `remove_embedding` | Remove embedding tags (ON/OFF) |
| `remove_lora` | Remove LoRA tags (ON/OFF) |
| `remove_lycoris` | Remove LyCoris tags (ON/OFF) |
| `remove_hypernetwork` | Remove Hypernetwork tags (ON/OFF) |
| `remove_from_sd1` | Remove SD1-specific tags (ON/OFF) |
| `remove_from_sd2` | Remove SD2-specific tags (ON/OFF) |
| `remove_from_sdxl` | Remove SDXL-specific tags (ON/OFF) |
| `remove_from_illustrious` | Remove Illustrious-specific tags (ON/OFF) |
| `remove_from_sd3` | Remove SD3-specific tags (ON/OFF) |
| `remove_from_stablecascade` | Remove Stable Cascade-specific tags (ON/OFF) |
| `remove_from_chroma` | Remove Chroma-specific tags (ON/OFF) |
| `remove_from_z_image` | Remove Z-Image-specific tags (ON/OFF) |
| `remove_from_turbo` | Remove Turbo-specific tags (ON/OFF) |
| `remove_from_flux` | Remove Flux-specific tags (ON/OFF) |
| `remove_from_nunchaku` | Remove Nunchaku-specific tags (ON/OFF) |
| `remove_from_qwengen` | Remove QwenGen-specific tags (ON/OFF) |
| `remove_from_qwenedit` | Remove QwenEdit-specific tags (ON/OFF) |
| `remove_from_wanimg` | Remove WanImg-specific tags (ON/OFF) |
| `remove_from_kwaikolors` | Remove KwaiKolors-specific tags (ON/OFF) |
| `remove_from_hunyuan` | Remove Hunyuan-specific tags (ON/OFF) |
| `remove_from_playground` | Remove Playground-specific tags (ON/OFF) |
| `remove_from_pony` | Remove Pony-specific tags (ON/OFF) |
| `remove_from_lcm` | Remove LCM-specific tags (ON/OFF) |
| `remove_from_lightning` | Remove Lightning-specific tags (ON/OFF) |
| `remove_from_hyper` | Remove Hyper-specific tags (ON/OFF) |
| `remove_from_pixartsigma` | Remove PixArt-Sigma-specific tags (ON/OFF) |
| `remove_from_sana1024` | Remove SANA1024-specific tags (ON/OFF) |
| `remove_from_sana512` | Remove SANA512-specific tags (ON/OFF) |
| `remove_from_auraflow` | Remove AuraFlow-specific tags (ON/OFF) |
| `remove_from_hidream` | Remove HiDream-specific tags (ON/OFF) |

**Note:** Architecture-specific toggles automatically update when supported model concepts change system-wide.

#### Outputs:

| Output | Purpose |
|--------|---------|
| `PROMPT+` | Cleaned positive prompt |
| `PROMPT-` | Cleaned negative prompt |

#### Use Cases:

**Auto Mode (Recommended for general use):**
- Workflow automatically adapts to selected checkpoint
- Prevents errors from incompatible network types
- No manual intervention needed

**Manual Mode (For specific needs):**
- Keep SDXL LoRAs while removing others
- Selectively remove certain network types
- Fine-grain control over tag cleanup

---

### Primiere Network Tag Loader

**Purpose:** Process A1111-style network tags from prompt and inject appropriate networks `(LoRA, LyCoris, Hypernetwork)` into model and `CLIP` with customizable weights and keywords.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `model` | Diffusion model from Checkpoint Loader |
| `clip` | CLIP model from Checkpoint Loader |
| `positive_prompt` | Positive prompt containing network tags in A1111 format (e.g., `<lora:filename:weight>`) |
| `control_data` | Optional metadata tuple from workflow |

#### Network Processing Settings:

| Setting | Purpose |
|---------|---------|
| `process_lora` | Enable LoRA injection from prompt (ON/OFF) |
| `process_lycoris` | Enable LyCoris injection from prompt (ON/OFF) |
| `process_hypernetwork` | Enable Hypernetwork injection from prompt (ON/OFF) |
| `hypernetwork_safe_load` | Safe loading mode for hypernetworks to prevent conflicts (ON/OFF) |

#### Weight Configuration:

| Setting | Purpose |
|---------|---------|
| `copy_weight_to_clip` | Apply model weight to CLIP model as well (ON/OFF) |
| `lora_clip_custom_weight` | Custom weight multiplier for LoRA on CLIP (default 1.00) |
| `lycoris_clip_custom_weight` | Custom weight multiplier for LyCoris on CLIP (default 1.00) |

#### LoRA Keyword Injection:

| Setting | Purpose |
|---------|---------|
| `use_lora_keyword` | Extract and inject LoRA-specific keywords into prompt (ON/OFF) |
| `lora_keyword_placement` | Where to place LoRA keyword: First, Last, etc. |
| `lora_keyword_selection` | Selection method: Select in order, Random, etc. |
| `lora_keywords_num` | Number of LoRA keywords to include (default 1) |
| `lora_keyword_weight` | Weight multiplier for LoRA keywords (default 1.0) |

#### LyCoris Keyword Injection:

| Setting | Purpose |
|---------|---------|
| `use_lycoris_keyword` | Extract and inject LyCoris-specific keywords into prompt (ON/OFF) |
| `lycoris_keyword_placement` | Where to place LyCoris keyword: First, Last, etc. |
| `lycoris_keyword_selection` | Selection method: Select in order, Random, etc. |
| `lycoris_keywords_num` | Number of LyCoris keywords to include (default 1) |
| `lycoris_keyword_weight` | Weight multiplier for LyCoris keywords (default 1.0) |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `MODEL` | Model with injected networks |
| `CLIP` | CLIP with injected networks (if applicable) |
| `LORA_STACK` | Stack of applied LoRAs |
| `LYCORIS_STACK` | Stack of applied LyCoris |
| `HYPERNETWORK_STACK` | Stack of applied hypernetworks |
| `LORA_KEYWORD` | Extracted LoRA keywords used |
| `LYCORIS_KEYWORD` | Extracted LyCoris keywords used |

#### A1111 Syntax Support:

Network tags in prompts use A1111 format:
- `<lora:filename:weight>` — applies LoRA with specified weight
- `<lycoris:filename:weight>` — applies LyCoris with specified weight
- `<hypernet:filename:weight>` — applies Hypernetwork with specified weight
- `embedding:name` or just `name` — textual inversion embedding (converted by Embedding Handler)

Example prompt: `beautiful portrait, <lora:detail-enhancer:0.8>, <lycoris:style-modifier:0.6>, professional lighting`

---

### Primiere Model Keyword

**Purpose:** Automatically inject model-specific keywords into prompt based on selected checkpoint to optimize generation quality for that model.

#### Inputs:

| Input | Purpose |
|-------|---------|
| `model_name` | Checkpoint name (auto-detected from checkpoint loader) |

#### Settings:

| Setting | Purpose |
|---------|---------|
| `use_model_keyword` | Enable automatic model keyword injection (ON/OFF) |
| `model_keyword_placement` | Where to inject keyword: First, Last. |
| `model_keywords_num` | Number of model keywords to include (default 1) |
| `model_keyword_weight` | Weight multiplier for model keywords (default 1.4) |
| `select_keyword` | Manually choose specific keyword if multiple available (None to auto-select) |

#### Outputs:

| Output | Purpose |
|--------|---------|
| `MODEL_KEYWORD` | Extracted keyword(s) for this model |

#### Benefits:

- Automatically tunes prompts for specific model architectures
- Improves generation quality without manual keyword research
- Weight adjustment allows fine-tuning keyword influence

---

<hr>
