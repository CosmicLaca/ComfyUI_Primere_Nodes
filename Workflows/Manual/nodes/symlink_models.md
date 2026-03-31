# <ins>Advanced Model Loading with Symlinks:</ins>

## Overview:
This node extends standard checkpoint loading capabilities by supporting symlinked model files. This feature is especially crucial for specialized models like Flux, Cascade, and KwaiKolors, which require unique UNet and diffuser architectures.

## Benefits of Symlink Support:
- Load non-standard model architectures through standard checkpoint selector
- Use multiple model variants without duplicating large files
- Organize models in multiple directories while maintaining compatibility
- Save disk space by linking to single source files
- Support for any custom model architecture through symlinks within one model loader

## <ins>Windows Symlink Setup:</ins>

### Creating Symlinks in Windows:
Open Command Prompt as Administrator and use:
```batch
# Basic syntax
mklink "checkpoint_full_path\any_model_name.safetensors" "source_full_path\original_file.gguf"

# Example for Flux model
mklink "[your_comfy_path]\models\checkpoints\Flux\unet_flux_v2.safetensors" "[your_comfy_path]\models\unet\Flux\flux_v2QN4.safetensors"

# Example for Cascade
mklink /H "[your_comfy_path]\models\checkpoints\Cascade\cascade_custom_stage_c.safetensors" "[your_comfy_path]\models\unet\Stable-Cascade\altcascade_v20.safetensors"

# Example for Flux GGUF model
mklink "[your_comfy_path]\models\checkpoints\Flux\gguf_flux_v2.safetensors" "[your_comfy_path]\models\diffusion_models\FLUX1\original_gguf_model.gguf"

# Example for Kolors model
mklink "[your_comfy_path]\models\checkpoints\Kolors\anyKolors.safetensors" "[your_comfy_path]\models\diffusers\OpenKolors\unet\diffusion_pytorch_model.fp16.safetensors"

```

for Kolors always symlink only diffusion_pytorch_model.fp16.safetensors file to Comfy's checkpoint folder, if use more than one model separate several directories, but just symlink the text_encoder folder between models, no need to copy

<hr>

### Windows Command Options:
- `/H` - Hard link (default, not important)
- `/J` - Directory junction
- `/D` - Directory symbolic link

## <ins>Linux Symlink Setup:</ins>

### Creating Symlinks in Linux:
```bash
# Basic syntax
ln -s /path/to/original/file.gguf /path/to/link/file.safetensors

# Example for Flux model
ln -s ~/models/Flux/flux_v2QN4.gguf ~/models/checkpoints/flux_v2.safetensors

# Example for Cascade
ln -s ~/models/StableCascade/stage_a.bin ~/models/checkpoints/cascade_stage1.safetensors
```

## Recommended Structure:

```
models/
├── checkpoints/
│   ├── flux_v2.safetensors -> ../models/unet/Flux/flux_v2QN4.safetensors
│   ├── cascade_s1.safetensors -> ../models/unet/Stable-Cascade/stage_a.safetensors
│   └── custom_kolors_v1.safetensors -> ../models/diffusers/OpenKolors/unet/diffusion_pytorch_model.fp16.safetensors
```

## Important Guidelines:
1. **File Extension Requirements**
   - Symlinked file must have `.safetensors`  (or `.ckpt` -> but why?) extension
   - Original file can have any extension (`.bin`, `.ckpt`, `gguf`, `bin`, `safetensors`, etc)
   - Node will access original file through symlink, but display simlynked filename on list

2. **Naming Conventions**
   - Symlink name can differ from original
   - Use descriptive names for better organization
   - Original filename preserved in node operations

3. **Directory Structure**
   - Keep original files in concept-specific folders
   - Create symlinks in standard checkpoint directory
   - Maintain organized hierarchy for easy management

<hr>

## Troubleshooting:

1. **Permission Issues**
   - Windows: Run Command Prompt as Administrator
   - Linux: Ensure proper file permissions (chmod)

2. **File Access Problems**
   - Verify original file paths
   - Check read permissions
   - Ensure no file locks

3. **Loading Issues**
   - Confirm `.safetensors` extension on symlink
   - Verify original file integrity
   - Check symlink path validity

## Best Practices:
1. **Organization**
   - Group similar models in dedicated folders
   - Use consistent naming conventions
   - Document symlink relationships

2. **Maintenance**
   - Regularly verify symlink integrity
   - Update symlinks when moving files
   - Keep backup of symlink structure

3. **Performance**
   - Use hard links when possible
   - Maintain files on same physical drive
   - Monitor disk space usage

Remember: Symlink support enables flexible model organization while maintaining compatibility with standard ComfyUI checkpoint selection.