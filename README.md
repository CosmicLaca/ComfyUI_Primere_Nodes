# Primere nodes for ComfyUI

## Do it before first run, or the workflow will be failed in your environment:
1; Install missing Python libraries if not start for first try. Activate Comfy venv and use 'pip install -r requirements.txt' at the root folder of Primere nodes (or check error messages and install missing libs manually)

2; If started, use the last workflow on the 'Workflow' folder for first try, all nodes visible under the 'Primere Nodes' submenu if you need custom workflow later. If some other nodes missing and red in workflow, download or delete unloaded nodes. 

3; Set the right path for image saving in the node 'Primere Image Meta Saver' on 'output_path' input

4; Rename 'styles.example.csv' on the 'stylecsv' folder to 'syles.csv' or copy here your own A1111 style .csv file if you want to use 'Primere Styles' node. If you keep the renamed 'styles.example.csv', you will see image previews for 4 example prompts included.

5; **Set all selectors from your own environment.** Checkpoint, Lora, Embedding and Hypernetwork selectors will be failed if not choose right values from your own environment.

6; Choose image from your machine to the 'Primere Exif Reader'.

7; If the workflow failed, read the message in terminal.

8; Update your Comfy to lates version, I always do it before development, so my nodes compatible with lates Comfy version.

9; I develop my nodes and workflow continously, so do git pull once a week.

## Special features:
- Automatically detect if SD or SDXL checkpoint loaded, and control the whole process (e.g. resolution) by the result
- No need to switch nodes or workflow between SD and SDXL mode
- You can select model, subpath and orientation under the prompt input overwrite the system settings, same settings under the Styles loader node
- You can randomize the orientation if using batch mode
- One button LCM mode (see example workflow)
- Save .json and/or .txt file with process details, but these details saved to image as EXIF
- Read original A1111 style.csv file, handle dynamic prompts, example csv included
- Random noise generator for latent image
- Important and easy editable styles included in the text encoder as list
- Resolution selector by side ratios only, editable ratio source in external file, and auto detect checkpoint version for right final size
- Image size can be convert to "standard" values, fully customizable side ratios at the bottom of the resolution selector node
- Original image size multiplied to upscaler by two several ratios, one for SD and another one for SDXL models
- Remove previously included networks from prompts (Embedding, Lora, and Hypernetwork), use it if the used model incompatible with them, or if you want to try your prompt without included additional networks, or different networks 
- Embedding handler for A1111 compatible prompts (or .csv styles), this node convert A1111 Embeddings to ComfyUI
- Use more than one prompt or style inputs for testing, and select any by 'Prompt Switch' node
- Special image meta/EXIF reader, which handle model name and samplers from A1111 .png or .jpg, never was easier to recycle your older A1111 or Comfy images using same or several settings, with switches you can change the original seed/model/size/etc... to workflow settings
- Check/debug generation details
- (As I see, Comfy doesn't handle SD2.x checkpoints, always geting black image, but this is not my feature :-) )

# Nodes in the pack by submenus:

## Inputs:
### Primere Prompt: 
2 input fileds within one node for positive and negative prompts. 3 additional fields appear under the text inputs:
- **Subpath**: the prefered subpath for final image saving. This can be use for example the subject of the generated image, like 'sci-fi' 'art' or 'interior'.
- **Use model**: the prefered checkpoint for image rendering. If your prompt need special checkpoint, for example because product design or architechture, here you can force apply this model to the prompt rendering process.
- **Use orientation**: if you prefer vertical or horizontal orientation depending on your prompt, your rendering process will be use this setting instead of global setting from 'Primere Resolution' node. Useful for example for portraits, what usually better in vertical orientation. Random settings available here, use with batch mode if you need several orientations for same prompt.

If you set these fields, (where 'None' mean not set and use system settings) the workflow will use all of these settings for rendering your prompt instead of settings in 'Dashboard' group.

### Primere Styles:
Style (.csv) file reader, compatible with A1111 syle.csv, but little more than the original concept. The file must be copied/symlinked to the 'stylecsv' folder. Rename included 'style.example.csv' to 'style.csv' for first working example, and later edit this file manually.
- **A1111 compatible CSV headers required for this file**: 'name,prompt,negative_prompt'. But this version have more 3 required headers: 'prefered_subpath,prefered_model,prefered_orientation'. These new headers working like additional fields in the simple prompt input. 
- If you fill these 3 optional columns in the style.csv, the rendering process will use them. **These last 3 fields are optional**, if you leave empty the style will be rendering with system 'dashboard' settings, if fill and enable to use at the bottom switches of node, dashboard settings will be overwritten.
- You can enable/disable these additional settings by switches if already entered to csv but want to use system settings instead, no need to delete if you failed or want to try with dashboard settings.

### Primere Dynamic:
- This node render A1111 compatible dynamic prompts, including external wildcard files of A1111 dynamic prompt plugin. External files must be copied/symlinked to the 'wildcards' folder and use the '__filepath/of/file__' keyword within your prompt. Use this to decode all style.csv and double prompt inputs, because the output of prompt/style nodes not resolved by other comfy dynamic decoder/resolver.
- Check the included workflow how to use this node.

### Primere exif reader:
- This node read prompt-exif (called meta) from loaded image. Compatible with A1111 jpg and png, and usually with ComfyUI, but not with all workflows.
- This is very important (the most important) node in the example workflow, it has a central settings distribution role, not just read the exif data.
- The reader is tested with A1111 'jpg' and 'png' and Comfy 'jpg' and 'png'. Another exif parsers will be included soon, but if you send me AI generated image contains metadata what failed to read, I will do parser/debug for that.

This node output sending lot of data to the workflow from exif/meta or pnginfo if it's included to selected image, like model name, vae and sampler. Use this node to distribute settings, and simple off the 'use_exif' switch if you don't want to render image by this node, then you can use your own prompts and dashboard settings.

**Use several settings of switches what exif/meta data you want/don't want to use for image rendering.** If switch off something, dashboard settings (this is why must be connected this node input) will be used instead of image exif/meta.
#### For this node inputs connect all of your dashboard settings, like in the example workflow. If you switch off the exif reader with 'use_exif' switch, or ignore specified data for example the model, the input values will be used instead of image meta. The example workflow help to analize how to use this node.

### Primere Embedding Handler:
This node convert A1111 embeddings to Comfy embeddings. Use after dynamically decoded prompts (booth text and style). **No need to modify manually styles.csv from A1111 if you use this node.**

### Primere Lora Stack Merger:
This node merge two different Lora stacks, SD and SDXL. The output is useful to store Lora settings to the image meta.

### Primere Lora Keyword Merger:
With Lora stackers you can read model keywords. This node merge all selected Lora keywords to one string, and send to prompt encoder.

### Primere Embedding Keyword Merger:
This node merge positive and negative SD and SDXL embedding tags, to send them to the prompt encoder.

## Dashboard:
### Primere Sampler Selector:
Select sampler and scheduler in separated node, and wire outputs to the sampler (through exif reader input in the example workflow). This is very useful to separate from other non-setting nodes, and for LCM mode you need two several sampler settings. (see the example workflow, and try to undestand LCM setting)

### Primere Steps & Cfg:
Use this separated node for sampler/meta reader inputs. If you use LCM mode, you need 2 settings of this node. See and test the attached example workflow.

### Primere LCM Selector:
Use this node to switch on/off LCM mode in whole rendering process. Wire two sampler and cfg/steps setting to the inputs (one of them must be compatible with LCM settings), and connect this node output to the sampler/exif reader, like in the example workflow. The 'IS_LCM' output important for CKPT loader and the Exif reader for correct rendering.

### Primere VAE Selector:
This node is a simple VAE selector. Use 2 nodes in workflow, 1 for SD, 1 for SDXL compatible VAE for autimatized selection. The checkpoint selector and loader get the loaded checkpoint version.

### Primere CKPT Selector:
Simple checkpoint selector, but with extras:
- This node automatically detect if the selected model SD or SDXL. Use this output for automatic VAE or size selection and for prompt encoding, see example workflow for details. In Comfy SD2.x checkpoints not working well, use only SD1.x and SDXL.
- Check the "visual" version of this node, if you have previews for checkpoints, easier to select the best for your prompt. How to create preview for visual selection, read more this file.

### Primere VAE loader:
Use this node to convert VAE name to VAE.

### Primere CKPT Loader:
Use this node to convert checkpoint name to 'MODEL', 'CLIP' and 'VAE'. Use 'is_lcm' input for detect LCM mode, see the example workflow.
If you have downloaded .yaml file, and copied to the checkpoint directory with same filename, set use_yaml to true, and the loader will use your config file. No need to swithc off if .yaml file missing. If you find some problem or error, simply set it to false.
Play with 'strength_lcm_model' and 'strength_lcm_clip' values if set LCM mode on. 

### Primere Prompt Switch:
Use this node if you have more than one prompt input (for example several half-ready test prompts). Connect prompt/style node outputs to this node inputs and set the right index at the bottom. To connect 'subpath', 'model', and 'orientation' inputs are optional, only the positive and negative prompt required.

**Very important:** don't remove the connected node from the middle or from the top of inputs. Connect nodes in right queue, and disconnect them only from the last to first. If you getting js error becuase disconnected inputs in wrong gueue, just reload your browser and use 'reload node' menu with right click on node. 

### Primere Seed:
Use only one seed input for all. A1111 look node, connect this one node to all other seed inputs. 

### Primere Noise Latent
This node generate 'empty' latent image, but with several noise settings. **You can randomize these setting between min. and max. values using switches**, this cause small difference between generated images for same seed and settings, but you can freeze your noise and image if you disable variations of random noise generation.

### Primere Prompt Encoder:
- This node compatible booth SD and SDXL models, important to use 'model_version' input for correct working. Try several settings, you will get several results. 
- Use included positive and negative styles, and check the best result in prompt and image outputs. 
- If you getting error if use SD basemodel, you must update (git pull) your ComfyUI.
- The style source of this node is external file at 'Toml/default_neg.toml' and 'Toml/default_pos.toml' files, what you can edit if you need changes.
- Comfy internal encoders not compatible with SD2.x version, you will get black image if select this SD2.x checkpoint version from model selector.

### Primere Resolution:
- Select image size by side ratios only, and use 'model_version' input for correct SD or SDXL size on the output.  
- You can calculate image size by really custom ratios at the bottom float inputs (and set switch on), or just edit the ratio source file.
- The ratios of this node stored in external file at 'Toml/resolution_ratios.toml', what you can edit if you need changes.
- Use 'round_to_standard' switch if you want to modify the exactly calculated size to the 'officially' recommended SD / SDXL values. This is usually very small modification.
- Not sure what orientation the best for your prompt and want to test in batch image generation? Just set batch value on the Comfy menu and switch 'rnd_orientation' to randomize vertical and horizontal images.
- Set the base model (SD1.x not SDXL) resolution to 512, 768, 1024, or 1280. The official setting is 512, but I like 768 instead.

### Primere Resolution Multiplier:
Multiply the base image size for upscaling. Important to use 'model_version' if you want to use several multipliers for SD and SDXL models. Just switch off 'use_multiplier' on this node if you don't need to resize original image.

### Primere Prompt Cleaner:
This node remove Lora, Hypernetwork and Embedding (booth A1111 and Comfy) from the prompt and style inputs. Use switches what netowok(s) you want to remove or keep in the final prompt. Use 'remove_only_if_sdxl' if you want keep all of these networks for all SD models, and remove only if SDXL checkpoint selected.
**Important notice:** for loras and hypernetworks you don't need original tags in the prompt (for example: \<lora:your_lora_name>). If you keep original lora and hypernetwork tags you cant sure your image result use the lora only, or use the tag string in the prompt. I recommend always to remove lora and hypernetwork tags, but you can try what happan if keep.
The another thing, that you must remove original tags after 'Primere Network Tag Loader', because after prompt cleaner non tags for tag loader.

### Primere Network Tag Loader
This node loads addtional networks (Lora and Hypernetwork) to the CLIP and MODEL. You can read and use Lora keywords to send to prompt encoder or the keyword merger like in the example workflow. 

### Primere Model Keyword
This node loads model keyword. You can read and use model keywords to send directly to prompt encoder like in the example workflow.

## Outputs:
### Primere Meta Saver:
This node save the image, but with/without metadata, and save meta to .json file if you want. Wire metadata from the Exif reader node only, and use optional 'prefered_subpath' input if you want to overwrite the node settings by several prompt input nodes. Set 'output_path' input correctly, depending your system.

### Primere Any Debug:
Use this node to display 'any' output values of several nodes like prompts or metadata (**metadata is formatted**). See the example workflow for details.

### Primere Text Output
Use this node to diaplay text. 

## Styles:
### Primere Style Pile:
Style collection for generated images. Set and connect this node to the 'Prompt Encoder'. No forget to set and play with style strenght. The source of this node is external file at 'Toml/stylepile.toml', what you can edit if you need changes.

## Networks:
### Primere LORA
Lora stack for 6 loras. Important to use 'stack_version' list. Here you can select how you want to load selected Lora-s, for SD models only, for SDXL models only, or for booth (Any) what not recommended. Use 2 separated Lora stack for SD/SDXL checkpoints, and wire 'model_version' input for correct use.
- You can switch on/off loras, no need to choose 'None' from the list.
- If you use 'use_only_model_weight', the model_weight input values will be copied to clip_weight.
- If you switch off 'use_only_model_weight', you can set model_weight and clip_weight to several values.
- You can load and send to Prompt Encoder the Lora keyword if available. This is similar but not exactly same function that "Model Keyword" plugin in the A1111
- You can choose Lora keyword placement, which and how many keywords select if more than one available, how many keyword use of more than one available, and select in queue or random, and set the keyword weight in the prompt.
- Lora keyword is much better than to keep lora tag in the prompt.

### Primere Embedding
Select textual inversion called Embedding for your prompt. You have to use 2 several versions of this one, one for SD, and another one for SDXL checkpoints. Important to use 'model_version' input and 'stack_version' list, working similar than in the Lora stack. 
You can choose embedding placement in the prompt.

### Primere Hypernetwork
Use hypernetwork if you already have by this node. **Hypernetwork is harmful, because can run any code on your computer, so ignore/delete this node or download them from reliable source only**
Hypernetworks don't need seperated SD and SDXL sources, use only one stack for all, and set 'stack_version' to 'Any'. 

## Visuals:
Here are same functions like upper, but the selection (for example checkpoints, loras, embeddings, styles from style.csv and hypernetworks) **possible by image previews on modal**. Very similar than in several themes of A1111.
You must save images as preview to the right path and name, deails later. Preview can be **only .jpg** format with only .jpg extension.

### Primere Visual CKPT selector:
**Visual selector for checkpoints**. You must copy your original checkpoint subdirs to ComfyUI\custom_nodes\ComfyUI_Primere_Nodes\front_end\images\checkpoints\ path but only the preview images needed, same name as the checkpoint but with .jpg only extension.
As extra features you can enable/disable modal with 'show_modal' switch, and exclude files and paths from modal starts with . (point) character if show_hidden switch is off.

### Primere Visual Lora selector:
Same as than the 'Primere LORA' node, but with preview images of selection modal.  
You must copy your original lora subdirs to ComfyUI\custom_nodes\ComfyUI_Primere_Nodes\front_end\images\loras\ path but only the preview images needed, same name as the checkpoint but with .jpg only extension.

### Primere Visual Embedding selector:
Same as than the 'Primere Embedding' node, but with preview images of selection modal.  
You must copy your original embedding subdirs to ComfyUI\custom_nodes\ComfyUI_Primere_Nodes\front_end\images\embeddings\ path but only the preview images needed, same name as the embedding file but with .jpg only extension.

### Primere Visual Hypernetwork selector:
Same as than the 'Primere Hypernetwork' node, but with preview images of selection modal.  
You must copy your original hypernetwork subdirs to ComfyUI\custom_nodes\ComfyUI_Primere_Nodes\front_end\images\hypernetworks\ path but only the preview images needed, same name as the hypernetwork file but with .jpg only extension.

### Primere Visual Style selector:
Same as than the 'Primere Styles' node, but with preview images of selection modal.  
You must create .jpg images as preview with same name as the style name in the list, but **space characters must be changed to _.** For example if your style in the list is 'Architechture Exterior', you must save Architechture_Exterior.jpg to the path: ComfyUI\custom_nodes\ComfyUI_Primere_Nodes\front_end\images\styles\
Example style.csv included, if rename you will see 4 example previews.

# Contact:
#### Discord name: primere -> ask email if you need

# Licence:
#### Use these nodes for your own risk