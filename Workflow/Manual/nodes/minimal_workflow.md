# <ins>Visual Checkpoint Selector:</ins>

<img src="visual_ckpt_selector.jpg" width="250px">

The Visual Checkpoint Selector helps you choose and manage AI models (checkpoints) in your workflow with a visual (or legacy list) interface.

### Basic Usage:
Select your AI model from the `base_model` dropdown list, which shows all installed models on your system.

### Visual Selection Mode:
Toggle `show_modal` to ON to open a visual gallery of preview images for each model. This makes it easier to choose the right model by seeing example outputs.

### Preview Settings:
- Select `preview_path`:
  - `Primere legacy` - uses custom path for preview images
  - `Model path` - uses preview images from your original model folder

- `show_hidden` controls visibility of hidden files/folders (those starting with a dot)

Visual `checkpoint` selection, automatized filtering by subdirectories (first row of buttons) and versions (second row of buttons):

<img src="visual_checkpoint.jpg" width="600px">

### Random Model Feature:
Turn on `random_model` to automatically select random models from the current folder. For example, if you've selected checkpoint from "Photo" folder, it will randomly pick from any model in that folder. This is useful for batch processing too.

### Aesthetic Score percent display:
- `aescore_percent_min`: Because the preview images show aesthetic scores if saved and measured, this value or less mean 0% 
- `aescore_percent_max`: Because the preview images show aesthetic scores if saved and measured, this value or more mean 100%

These scores help sorting models based on their quality ratings.

**When all data available, thse badges will visible in the preview:**

<img src="previews.jpg" width="600px">

- **Top left:** model concept (Flux, SD1, SD2, SDXL, etc...)
- **Top right:** if symlinked, what type of diffuser linked
- **Botom:** the average aesthetic score. Have to use aesthetic scorer node before store this data for checkpoints and saved prompts. The number is the average, but the percent depending on the checkpoint selector settings, where the `aescore_percent_min` and lower value mean 0%, `aescore_percent_max` and higher mean 100%.

<hr>

# <ins>Resolution Selector:</ins>

<img src="primere_resolution.jpg" width="250px">

This node helps you set the perfect image dimensions for your generations with preset ratios or custom settings.

### Basic Resolution Selection:
- Choose from predefined aspect ratios in the `ratio` dropdown (Photo, Portrait, Old TV, HD, HD+, Square, etc.)
- Use `resolution` mode:
 * "Auto" - automatically sets base resolution based on your selected model. If the model version available this settings useful.
 * "Manual" - lets you input custom dimensions bases using:
   - `sd1_res`: Base resolution for SD1 models (768 default)
   - `sdxl_res`: Base resolution for SDXL models (1024 default)
   - `turbo_res`: Base resolution for Turbo models (512 default)

### Image Orientation:
- Set `orientation` to Horizontal or Vertical
- Enable `rnd_orientation` to randomly switch between orientations. This function useful for batch generation.
- `round_to_standard` re-count dimensions to "standard" of selected AI models

### Custom Ratio Settings:
Enable `calculate_by_custom` to use your own aspect ratios (example: 1.6:2.8):
- `custom_side_a`: First side ratio (e.g., 1.60)
- `custom_side_b`: Second side ratio (e.g., 2.80)

Note: Aspect ratios can be customized by editing the external .toml configuration file from path: `Toml/resolution_ratios.toml`

<hr>

# <ins>Primere Prompt:</ins>

<img src="primere_prompt.jpg" width="250px">

The Prompt node provides advanced prompt control with organization features and special settings.

### Prompt Inputs:
- `positive_prompt`: Enter your main prompt describing what you want to create
- `negative_prompt`: Enter elements you want to avoid in the generation

### Organization Features:
- `subpath`: Save your generated images into themed folders (e.g., "CutePets", "Sci-Fi, etc...")
- `model`: Choose a specialized model for specific types of images (e.g., interior, exterior, etc...). This is standard model list.

### Orientation Control:
Choose how to handle image orientation:
- `None`: Use default orientation from `Resolution Selector` node
- `Random`: Randomly switch between **horizontal** and **vertical**. This function useful for batch generation.
- `Horizontal`: Force **horizontal** composition
- `Vertical`: Force **vertical** composition

The orientation setting helps compose your image properly for your selected subject matter.

<hr>

# <ins>Prompt Switch:</ins>

<img src="prompt_switch.jpg" width="250px">

A control node that lets you quickly switch between different prompt sources in your workflow, including `Style Selector` nodes.

### How It Works:
- Connect multiple prompt sources to this node (any different sources), but the connection and unconnection queue is important. If failed, just reload the browser.
- Use the `select` input to choose which prompt source to use (1-any)
- The selected prompt source becomes active, while others remain inactive

### Usage Example:
If you have different prompts for:
- Portrait shots
- Landscape scenes
- Character designs

You can connect all of them to the Prompt Switch and easily toggle between them using the selector, without needing to reconnect nodes or modify your workflow.

<hr>

# <ins>Visual Prompts (style) Selector:</ins>

<img src="primere_styles.jpg" width="250px">

A powerful tool that lets you save and load complete prompt configurations using a visual interface or simple list.

### Basic Usage:
- Choose from saved styles using the `styles` dropdown
- Toggle `show_modal` to switch between:
 * List view: Simple dropdown of style names
 * Visual view: Preview images of each style's output

Visual `saved prompt` selection `(csv source)`, automatized filtering by categories:

<img src="visual_csv.jpg" width="600px">

### Style Components:
Each saved style can include:
- Name of saved prompt
- Positive and negative prompts
- Specific orientation
- Custom save folder (subpath)
- Model preference

### Control Options:
- `use_subpath`: Apply the style's saved folder path
- `use_model`: Use the style's recommended/preferred model
- `use_orientation`: Apply the style's preferred orientation
- `show_hidden`: Show/hide styles names or path contains word `nsfw`
- `random_prompt`: Randomly select prompt from available styles from same subpath as selected

### Quality sorting:
- `aescore_percent_min`: Because the preview images show aesthetic scores if saved and measured, this value or less mean 0% 
- `aescore_percent_max`: Because the preview images show aesthetic scores if saved and measured, this value or more mean 100% 

Note: Styles are stored in an external .csv file that can be easily edited and shared. Rename `stylecsv/styles.example.csv` to `stylecsv/styles.csv` and use your own prompt collection.

<hr>

# <ins>Visual Prompts - auto organized:</ins>

<img src="visual_prompt_csv.jpg" width="250px">

This node organizes your saved prompts into categories for easier management, especially useful when you have a large saved collection of prompts.

### Category Organization:
- Prompts are automatically sorted into categories like:
  - Architecture
  - Art
  - Character
  - ...and more dependign your source .csv.

### How to Use:
1. Select a category from node
2. Choose a specific prompt from that category
3. All related settings (prompt, model, path) load automatically from .csv file: Rename `stylecsv/styles.example.csv` to `stylecsv/styles.csv` and use your own prompt collection.

## Control Options (same as than the Visual Prompts (style) Selector node)
- `show_modal`: Switch between list and visual preview mode
- `show_hidden`: Include/ignore hidden category items if name or path contains word `nsfw`
- `use_subpath`: Use saved folder paths (usually as prompt category)
- `use_model`: Apply recommended models
- `use_orientation`: Use saved orientation settings
- `random_prompt`: Pick random prompt from selected category

## Benefits
- Organized browsing instead of one long list
- Quick access to themed prompts
- Easy to find related styles
- Categories are created automatically from your saved paths

Note: Categories are created from the folder structure in your styles.csv file, making organization automatic and maintenance-free.

<hr>

# <ins>Dynamic Prompt Handler:</ins>

<img src="dynamic_prompt.jpg" width="250px">

This node processes dynamic text prompts with random variations and maintains seed control for consistent results. The node has two inputs: a dynamic prompt string and a seed value.

### Input Fields:
- `dyn_prompt`: Takes your prompt text containing dynamic sections
- `seed`: Controls randomization for consistent results

### Usage:
This node helps create varied prompts while maintaining reproducibility. When you input a prompt with dynamic sections (like {red|blue|green}), the node will randomly select one option based on the seed value. Perfect for batch processing or exploring variations while keeping track of successful combinations.

### Benefits:
- Consistent randomization with seed control
- Streamlines prompt variation workflow
- Integrates seamlessly with other prompt processing nodes
- Reduces manual prompt editing time
- Perfect for batch generation with controlled variations

Read manual of dynamic prompt syntax: https://github.com/adieyal/sd-dynamic-prompts/blob/main/docs/SYNTAX.md

<hr>

# <ins>Primere KSampler with Variations:</ins>

<img src="sampler.jpg" width="250px">

This KSampler node extends the standard sampling functionality with fine-tuned variation controls and performance options.

#### Variation System:
- `variation_extender`: Fine-tunes noise injection from 0.0 to 1.0, allowing subtle variations while maintaining the original image's core elements
- `variation_batch_step`: Enables progressive variation in batch processing by incrementing noise injection per step (e.g., 0.1 increment over 10 steps creates a gradual transformation sequence)
- `variation_level`: When set to "Maximize", randomizes the noise injection values for more diverse outputs

#### Performance Options
- `device`: Select processing hardware (CPU/GPU/Default)
- `align_your_steps`: Implements NVIDIA's AlignYourSteps technology, which helps maintain temporal consistency and reduces unwanted artifacts during the sampling process. Read details from here: https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/

### Additional Settings:
- `denoise`: Controls the denoising strength
- `model_sampling`: Adjusts the model's sampling parameters (usually for SD3 models only)

<hr>

# <ins>Primere Prompt Encoder:</ins>

<img src="clipping.jpg" width="250px">

This sophisticated prompt encoder node offers extensive control over prompt processing with multiple CLIP and LONG-CLIP models and advanced encoding options.

### CLIP Processing:
- `clip_mode`: Toggle between standard CLIP and Long-CLIP processing
  - `clip_model/longclip_model`: Model selection based on clip_mode switch
- `last_layer`: Fine-tune CLIP encoding by selecting specific negative layers for feature extraction
- `negative_strength`: Adjusts the intensity of negative prompt influence

### Style System:
- `use_int_style`: Enables internal style system loaded from external .toml configurations from the path: `Toml/default_neg.toml` and `Toml/default_pos.toml`
  - `int_style_pos/neg`: Select predefined styles by name
  - `int_style_pos/neg_strength`: Control strength of applied styles

### Advanced Encoding Options:
- `adv_encode`: Enables alternative (advanced) CLIP encoding methodology
- `token_normalization`: Controls how token weights are normalized (mean/none/length/length+mean)
- `weight_interpretation`: Defines how prompt weights are processed (comfy++/A1111/comfy)

### Enhanced Prompt System:
- `enhanced_prompt_usage`: Controls enhanced prompt processing
  - `None`: Ignores enhanced prompt
  - `Add`: Appends to end of positive prompt
  - `Replace`: Replace positive prompt (very strong modification)
  - `T5-XXL`: Uses enhanced prompt input for T5-XXL encoding if concept support T5 clipping
- `enhanced_prompt_strength`: Controls enhanced prompt influence if added to original positive prompt

### Additional Controls:
- `style_position`: Placement of additional style prompts (Top/Bottom)
  - `opt_pos/neg_strength`: Fine-tune optional prompt strengths
- `copy_prompt_to_l`: Enables SDXL first-pass prompt copying
  - `sdxl_l_strength`: Controls SDXL first-pass prompt intensity

<hr>

# <ins>Aesthetic Scorer:</ins>

<img src="ascorer.jpg" width="250px">

This node evaluates the aesthetic quality of generated images and saving scoring data to display in visual previews. It provides numerical scoring and statistical tracking for your generations.

### Key Features:
- `get_aesthetic_score`: Enables/disables image quality scoring
  - `add_to_checkpoint`: Save scores to checkpoint data, allowing you to sort image quality of different model checkpoints
  - `add_to_saved_prompt`: Save scores to saved prompts, helping identify (and sort) consistently high-performing prompts

### Benefits:
- Track generation quality automatically
- Compare performance across different checkpoints and prompts
- Identify your most successful prompts through statistical tracking
- View average scores in visual selectors
- Make data-driven decisions about your workflow settings

The scoring system helps optimize your workflow by providing objective (or subjective?) feedback on image quality and maintaining statistics for both checkpoints and prompts.

<hr>

# Primere Image Saver:

<img src="manual_img_saver.jpg" width="250px">

This versatile node provides comprehensive image saving functionality with **visual preview management** capabilities for your workflow.

### Save Modes:
#### Preview Save Mode: `Save as preview`
- Saves images as visual previews for checkpoints, LoRAs, Lycoris, Hypernetworks, Embedding, and saved prompt selections
- **Overwrite**: Replace existing or create new preview
- **Keep**: Preserve existing, only create if missing
- **Join horizontal**: Combine horizontally with existing preview or create new
- **Join vertical**: Combine vertically with existing preview or create new
- **Target selection**: Select only one target if more than one available in the process, for example Loras or Embeddings.

#### The bonus hidden feature, that one click very close under the save button, the previously saved preview visible if exist.

<img src="preview_secret.jpg" width="400px">

#### Local Storage Mode: `Save as any`
- `Format Options`: PNG, JPEG, WebP
- `Size Control`: Resize by specifying maximum dimension while preserving aspect ratio. 0 mean no resize
- `Quality Settings`: Adjust compression for JPEG/WebP formats

### Benefits:
- Create visual reference libraries for models, additional networks and prompts
- Flexible preview management for workflow organization
- Custom export settings for different use cases
- Space-efficient preview combinations
- Maintain organized model and prompt libraries with visual references