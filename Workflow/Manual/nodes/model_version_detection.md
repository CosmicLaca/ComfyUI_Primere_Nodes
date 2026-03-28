# Model Version Detection & Caching Guide

## Overview

Primiere Model Control automatically adapts all generation settings (sampler, CFG, steps, VAE, CLIP, encoders, LoRAs, refiners) based on **model type/concept** (SD1, SDXL, Flux, Hunyuan, etc.). This automation requires the system to know which model concept each checkpoint belongs to.

This guide explains how the model type detection system works, how to set it up, and how to manually fix detection errors.

---

## How Model Type Detection Works

When you select a checkpoint in the workflow, the system determines its model concept through a **3-step hierarchy**:

| Step | Method | Example                                              |
|------|--------|------------------------------------------------------|
| **1. Metadata Detection** | Reads model concept from checkpoint file metadata | Flux checkpoint has `model_type: "flux"` in metadata |
| **2. Directory Name** | Infers model concept from parent folder name | Model in `models/checkpoints/Flux/` → Flux concept   |
| **3. Cache File** | Looks up pre-computed model-to-concept mapping | `.cache.json`: `"my-model": "Flux"`                  |

**If all three fail:** Model marked as "UNKNOWN" — Manual setup required.

---

## Auto-Detection (First Run)

On first workflow execution with a new checkpoint, the system automatically:

1. **Scans checkpoint metadata** for model type info
2. **Checks parent directory name** against supported concepts
3. **Stores result** in `Nodes/.cache/.cache.json` cache file
4. **Reuses cached result** on subsequent runs

**Benefit:** Works immediately for most models without manual setup, if metadata is correct or folder is named properly.

**Limitation:** Some checkpoints have incorrect/missing metadata, requiring manual verification.

---

## Terminal Helper: Batch Model Detection

For users with many checkpoints (50+), the terminal helper **pre-processes** all models in one pass instead of on-demand during workflow execution.

### When to Use Terminal Helper

✅ **Use if:**
- You have 50+ checkpoints
- You want to verify all models before running workflows
- You have symlinked models from other directories
- You want one-time processing instead of per-run detection

❌ **Not necessary if:**
- Few models (<20) with correct metadata or folder structure
- Lazy loading acceptable (auto-detect on first workflow use)

### Setup & Execution

**Step 1: Locate the Helper**

Terminal helper file: `ComfyUI_Primere_Nodes/terminal_helpers/model_version_cache.py`

**Step 2: Activate ComfyUI Virtual Environment**

```bash
# Windows (venv)
cd your-comfyui-folder
.\venv\Scripts\activate

# Windows (simple terminal)
.\venv\Scripts\activate.bat

# Windows (conda)
conda activate comfyui

# Linux/macOS
source venv/bin/activate
```

**Step 3: Run the Helper**

```bash
cd path/to/ComfyUI_Primere_Nodes
python terminal_helpers/model_version_cache.py
```

**Step 4: Review Console Output**

The helper prints a report showing all detected models and their concept assignments:

```
------------------- START -------------------------
145 models in system
--------------- CACHED MODELS INFO ---------------------
Model [1] / 145 cached from metadata: photon_v1 -> SD1
Model [2] / 145 cached from directory: my-flux-model -> Flux
Model [3] / 145 cached from directory: realistic_sdxl -> SDXL
Model [4] / 145 UNKNOWN | path: models/checkpoints/mystery_model_v2
...
```

**Look for "UNKNOWN" entries** — these need manual investigation.

---

## Cache File Format

The terminal helper generates: `Nodes/.cache/.cache.json`

This file is a key-value dictionary mapping checkpoint names → model concepts.

### Example Cache File

```json
{
    "model_version": {
        "model_01": "SD1",
        "model_02": "SDXL",
        "model_03": "SD1",
        "model_04": "SD1",
        "model_05": "KwaiKolors",
        "model_06": "Hunyuan",
        "model_07": "Hyper",
        "model_08": "SD3",
        "model_09-GGUF": "Flux",
        "model_10": "LCM",
        "model_11L-ightning": "Lightning",
        "model_12": "Pony",
        "qwenImage2512": "QwenGen",
        "model_14_v10": "Z-Image"
    }
}
```

### Key Points

- **Key:** Checkpoint filename WITHOUT extension (e.g., `photon_v1` not `photon_v1.safetensors`)
- **Value:** Model concept from supported list (see Supported Concepts below)
- **Auto-generated:** Terminal helper creates this on first run
- **User-editable:** Manually add/correct entries as needed

---

## Supported Model Concepts

The system recognizes these model concepts:

```
SD1, SD2, SDXL, Illustrious, SD3, StableCascade, Chroma, Z-Image, 
Turbo, Flux, Nunchaku, QwenGen, QwenEdit, WanImg, KwaiKolors, 
Hunyuan, Playground, Pony, LCM, Lightning, Hyper, PixartSigma, 
SANA1024, SANA512, AuraFlow

Future: HiDream, Mochi, WanT2V, WanI2V, Cosmos, Flux2, SSD, 
SegmindVega, KOALA, StableZero, SV3D, SD09, StableAudio, LTXV
```

---

## Manual Setup: Fixing Detection Errors

If terminal helper output shows "UNKNOWN" or incorrect concept assignment, **manually edit the cache file**.

### Method 1: Identify Model Type, Then Edit Cache

**Step 1: Identify Model Concept**

- Download page / model repo should list model type
- Search HuggingFace or CivitAI for model architecture info
- Look at checkpoint metadata (HuggingFace model cards usually show architecture)
- Check if model trained on SD1.5, SDXL, Flux base

**Step 2: Edit Cache File**

Location: `Nodes/.cache/.cache.json`

Example: You identified `mystery_model_v2` is actually Flux-based.

Original:
```json
"model_version": {
    "mystery_model_v2": "UNKNOWN | path: models/checkpoints/mystery_model_v2"
}
```

After fix:
```json
"model_version": {
    "mystery_model_v2": "Flux"
}
```

Save file and reload ComfyUI.

### Method 2: Organize by Directory

Instead of editing cache, organize checkpoints into concept-named folders. Terminal helper will auto-detect from folder structure.

**Example structure:**

```
models/
  checkpoints/
    SD1/
      photon_v1.safetensors
      old_sd1_model.safetensors
    SDXL/
      the_sdxl_model01.safetensors
      the_sdxl_model02.safetensors
    Flux/
      the_hunyuan_model-GGUF.safetensors
      the_hunyuan_model-dev.safetensors
    Hunyuan/
      the_hunyuan_model.safetensors
```

Terminal helper will read folder names and auto-populate cache.

**Advantages:** Future-proof, self-documenting, easy bulk organization.

---

## Symlinked Models

Primiere supports symlinked models from other directories:

- `models/unet/` (raw UNet checkpoints)
- `models/diffusers/` (Hugging Face diffusers format)
- `models/diffusion_models/` (other sources)

If you symlink these into `models/checkpoints/`, the system will:

1. **Detect symlink** in cache output: `(symlink from: original/path)`
2. **Resolve symlink** to original location
3. **Auto-detect concept** from original path or metadata
4. **Load correctly** via Checkpoint Loader

### Example

You have: `models/diffusers/flux-dev` (Flux model in diffusers format)

Create symlink: `models/checkpoints/flux-dev` → `../diffusers/flux-dev`

Terminal helper output:
```
Model [50] / 145 cached from directory: flux-dev -> Flux (symlink from: /path/to/models/diffusers/flux-dev)
```

Result: Model loads and routes through Model Control automation with Flux concept settings.

---

## Creating Symlinks to Model Files

Primiere supports loading model files from other directories via symlinks. Create a symlink in `models/checkpoints/` pointing to your actual model file location.

### Windows: Using `mklink` Command

Open terminal (Command Prompt or PowerShell) **as Administrator** and run:
```bash
mklink "path\to\ComfyUI\models\checkpoints\LinkName.safetensors" "path\to\original\model\model.safetensors"
```

**Example:**
```bash
mklink "C:\ComfyUI\models\checkpoints\my-flux-model.safetensors" "C:\ComfyUI\models\diffusers\flux-dev\model.safetensors"
```

Result: `models/checkpoints/my-flux-model.safetensors` → points to actual model file

**Note:** No `/D` or `/J` flag for file symlinks. Requires **Administrator privileges**.

---

### Windows: GUI Method (LinkShellExtension)

For users preferring GUI over terminal:

1. Download **LinkShellExtension**: https://schinagl.priv.at/nt/hardlinkshellext/linkshellextension.html
2. Install and restart Explorer
3. Right-click model file → **Pick Link Source**
4. Navigate to `models/checkpoints/` → Right-click → **Drop As** → **Symbolic Link**

Result: Symlink created without terminal commands.

---

### Linux / macOS: Using `ln` Command

Open terminal and run:
```bash
ln -s "/path/to/original/model/model.safetensors" "/path/to/ComfyUI/models/checkpoints/LinkName.safetensors"
```

**Example:**
```bash
ln -s "/home/user/models/diffusers/flux-dev/model.safetensors" "/home/user/ComfyUI/models/checkpoints/my-flux-model.safetensors"
```

Result: `models/checkpoints/my-flux-model.safetensors` → points to actual model file

**Note:** `-s` creates symbolic link. Use absolute paths for reliability.

---

### Verifying Symlinks

After creating symlink, terminal helper will detect it:
```
Model [50] / 145 cached from directory: my-flux-model -> Flux (symlink from: /path/to/models/diffusers/flux-dev/model.safetensors)
```

The system resolves symlinks and loads the original model file correctly.

---

## Workflow: Auto-Detect vs. Terminal Helper

### Scenario 1: Few Models, Good Metadata (5-20 checkpoints)

**Just use auto-detect:**

1. Run workflow normally
2. System detects model concept on first use
3. Settings cached automatically
4. Done — no manual setup needed

### Scenario 2: Many Models, Mixed Metadata (50+ checkpoints)

**Use terminal helper:**

1. Run terminal helper: `python terminal_helpers/model_version_cache.py`
2. Review output for "UNKNOWN" entries
3. Manually identify those models and edit cache file
4. Re-run terminal helper to verify (optional)
5. All models pre-processed, workflow runs instantly

### Scenario 3: Symlinked Models

**Terminal helper required:**

1. Create symlinks from other dirs → `models/checkpoints/`
2. Run terminal helper
3. Helper resolves symlinks and detects concepts
4. All symlinked models cached and ready

---

## Troubleshooting

**Problem: Model marked "UNKNOWN" after terminal helper**

- Check if model metadata is accessible (some encrypted models fail)
- Look up model on HuggingFace/CivitAI and identify concept
- Manually edit cache file with correct concept
- Verify model folder name doesn't match any concept name (typos confuse detector)

**Problem: Wrong concept assigned**

- Model metadata has incorrect `model_type` field (use metadata from download page instead)
- Parent folder name doesn't match actual concept
- Solution: Manually override in cache file or move to correctly-named folder

**Problem: Cache file not updating**

- ComfyUI cache or file lock issue
- Solution: Restart ComfyUI, verify `Nodes/.cache/` folder exists and is writable
- Delete `.cache.json` and re-run terminal helper

**Problem: Symlinked models not detected**

- Helper requires full symlink resolution permissions
- On Windows, run terminal AS ADMIN
- On Linux/macOS, verify symlink targets are readable

---

## Integration with Primiere Model Control

Once cache is populated:

1. **Select checkpoint** in Visual Checkpoint Selector (workflow)
2. **Model Control reads** checkpoint name
3. **Looks up concept** in cache file
4. **Auto-loads saved settings** for that concept
5. **All parameters adapt** (sampler, CFG, VAE, encoders, LoRAs, refiners)

**Result:** One click changes everything. No manual parameter tweaking.

---

## Cache File Location

**File:** `ComfyUI_Primere_Nodes/Nodes/.cache/.cache.json`

**Scope:** Local to this nodepack installation — different installs have separate caches

**Persistence:** Survives nodepack updates (cached in git-ignore)

**Backup:** Consider backing up `.cache.json` if you invest time manually correcting models

---

## Summary

| Aspect | Details |
|--------|---------|
| **Auto-Detect** | Runs on first workflow use, requires correct metadata or folder structure |
| **Terminal Helper** | Batch pre-processes all models, generates cache file, recommended for 50+ models |
| **Cache File** | JSON key-value map stored in `Nodes/.cache/.cache.json`, user-editable |
| **Manual Setup** | Edit cache file directly for incorrect/unknown models |
| **Symlinks** | Supported and auto-resolved by helper and loader |
| **Model Control** | Uses cached concept to auto-configure all generation settings |

**Bottom line:** For small collections, let auto-detect work. For large collections, run terminal helper once and manually fix any "UNKNOWN" entries. Then forget about model types — Primiere handles everything.

---
