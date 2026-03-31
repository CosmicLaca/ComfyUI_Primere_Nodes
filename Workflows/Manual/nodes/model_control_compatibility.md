# PrimereModelControl — Parameter × Model Compatibility

**Symbols:** ✅ Supported &nbsp; ❌ Not applicable &nbsp; ⚠️ Conditional (see notes) &nbsp; ? Unknown / future

---

## Table 1 — Core Sampling & Loading

| Model | sampler_name | scheduler_name | steps | override_steps | cfg | rescale_cfg | vae | vae_selection | clip_selection | last_layer | weight_dtype | precision |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SD1 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| SD2 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| SDXL | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Illustrious | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Pony | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Turbo | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| LCM | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Lightning | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Hyper | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Playground | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| SD3 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Flux | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Flux2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Nunchaku | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Chroma | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| StableCascade | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| AuraFlow | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| PixartSigma | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SANA1024 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| SANA512 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| KwaiKolors | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Hunyuan | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Z-Image | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| QwenGen | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| QwenEdit | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| SSD | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| SegmindVega | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| KOALA | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| StableZero | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| SD09 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| HiDream | ? | ? | ✅ | ✅ | ? | ❌ | ? | ❌ | ❌ | ❌ | ✅ | ❌ |
| Cosmos | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ? | ❌ | ❌ | ❌ | ? | ❌ |
| WanImg | ? | ? | ✅ | ✅ | ? | ❌ | ? | ❌ | ❌ | ❌ | ? | ❌ |
| WanT2V | ? | ? | ✅ | ✅ | ? | ❌ | ? | ❌ | ❌ | ❌ | ? | ❌ |
| WanI2V | ? | ? | ✅ | ✅ | ? | ❌ | ? | ❌ | ❌ | ❌ | ? | ❌ |
| Mochi | ? | ? | ✅ | ✅ | ? | ❌ | ? | ❌ | ❌ | ❌ | ? | ❌ |
| SV3D | ? | ? | ✅ | ✅ | ? | ❌ | ? | ❌ | ❌ | ❌ | ? | ❌ |
| StableAudio | ? | ? | ✅ | ✅ | ? | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ❌ |

**Notes — Table 1:**
- `rescale_cfg`: applied via `RescaleCFG.patch` only in Flux and Chroma loaders.
- `clip_selection`: SD3 can switch between baked and custom triple-CLIP loader.
- `last_layer`: applies CLIP stop layer; only relevant when CLIP is loaded from the checkpoint.
- `weight_dtype`: used by UNET/diffusion model loaders that accept a dtype argument (Flux family, SANA, Kolors, etc.).
- `precision`: maps to `ModelComputeDtype` (fp16/fp32 only); skipped for quant types.
- PixartSigma VAE: baked into main checkpoint; overridden by refiner checkpoint VAE when refiner is active.
- SSD, SegmindVega, KOALA, StableZero, SD09: treated as SD1/SDXL-family checkpoints.

---

## Table 2 — Encoders & Attention

| Model | encoder_1 | encoder_2 | encoder_3 | attn_preset | attn_self (q/k/v/out) | attn_cross (q/k/v/out) | sampler_type | align_your_steps |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SD1 | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| SD2 | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| SDXL | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| Illustrious | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| Pony | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| Turbo | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| LCM | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| Lightning | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| Hyper | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| Playground | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| SD3 | ✅ T5-XXL | ✅ CLIP-G | ✅ CLIP-L | ✅ CLIP | ❌ | ❌ | ksampler | ❌ |
| Flux | ✅ T5-XXL | ✅ CLIP-L | ❌ | ✅ CLIP | ❌ | ❌ | both | ❌ |
| Flux2 | ✅ T5-XXL | ✅ CLIP-L | ❌ | ✅ CLIP | ❌ | ❌ | both | ❌ |
| Nunchaku | ✅ T5-XXL | ✅ CLIP-L | ❌ | ✅ CLIP | ❌ | ❌ | both | ❌ |
| Chroma | ✅ T5-XXL | ❌ | ❌ | ❌ | ❌ | ❌ | ksampler | ❌ |
| StableCascade | ✅ Stage-B | ❌ | ✅ CLIP | ✅ CLIP | ❌ | ❌ | ksampler | ❌ |
| AuraFlow | ✅ | ❌ | ❌ | ✅ CLIP | ❌ | ❌ | ksampler | ❌ |
| PixartSigma | ✅ T5 | ❌ | ❌ | ❌ | ❌ | ❌ | ksampler | ❌ |
| SANA1024 | ✅ | ❌ | ❌ | ✅ CLIP | ❌ | ❌ | ksampler | ❌ |
| SANA512 | ✅ | ❌ | ❌ | ✅ CLIP | ❌ | ❌ | ksampler | ❌ |
| KwaiKolors | ✅ ChatGLM | ❌ | ❌ | ❌ | ❌ | ❌ | ksampler | ❌ |
| Hunyuan | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ksampler | ❌ |
| Z-Image | ✅ | ❌ | ❌ | ✅ CLIP | ❌ | ❌ | ksampler | ❌ |
| QwenGen | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ksampler | ❌ |
| QwenEdit | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ksampler | ❌ |
| SSD | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| SegmindVega | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| KOALA | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| StableZero | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| SD09 | ❌ | ❌ | ❌ | ✅ CLIP | ✅ | ✅ | ksampler | ✅ |
| HiDream | ? | ? | ? | ? | ❌ | ❌ | ? | ❌ |
| Cosmos | ? | ❌ | ❌ | ? | ❌ | ❌ | ksampler | ❌ |
| WanImg | ? | ❌ | ❌ | ? | ❌ | ❌ | ? | ❌ |
| WanT2V | ? | ❌ | ❌ | ? | ❌ | ❌ | ? | ❌ |
| WanI2V | ? | ❌ | ❌ | ? | ❌ | ❌ | ? | ❌ |
| Mochi | ? | ❌ | ❌ | ? | ❌ | ❌ | ? | ❌ |
| SV3D | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ❌ |
| StableAudio | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ❌ |

**Notes — Table 2:**
- `attn_self`: shared input values with CLIP attention (`attn_query/key/value/output`). Applies UNet `attn1` only for UNET_CONCEPTS models.
- `attn_cross`: applies UNet `attn2` only for UNET_CONCEPTS = {SD1, SD2, SDXL, Illustrious, Turbo, Pony, Hyper, Lightning, Playground, LCM}.
- `attn_preset`: CLIP-side attention (`CLIPAttentionMultiply`) applies to any model with a CLIP text encoder. UNet side only for UNET_CONCEPTS.
- `sampler_type = both`: Flux can use `custom_advanced` (guidance-based) or `ksampler` depending on workflow.
- SD1/SDXL family: encoder is baked into the checkpoint; `encoder_1/2/3` are ignored unless loading a UNET-only model via symlink.

---

## Table 3 — Advanced Model Sampling

| Model | model_sampling | edm_sampling | discrete_sampling | discrete_zsnr | sigma_max | sigma_min | flux_max_shift | flux_base_shift | beta_alpha | beta_beta | guidance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SD1 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| SD2 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| SDXL | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Illustrious | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Pony | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Turbo | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| LCM | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Lightning | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Hyper | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Playground | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| SD3 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Flux | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Flux2 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Nunchaku | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Chroma | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| StableCascade | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| AuraFlow | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| PixartSigma | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SANA1024 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SANA512 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| KwaiKolors | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hunyuan | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Z-Image | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| QwenGen | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| QwenEdit | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SSD | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| SegmindVega | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| KOALA | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| StableZero | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| SD09 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| HiDream | ? | ? | ❌ | ❌ | ❌ | ❌ | ? | ? | ? | ? | ? |
| Cosmos | ❌ | ✅ cosmos_rflow | ❌ | ❌ | ? | ? | ❌ | ❌ | ? | ? | ❌ |
| WanImg | ? | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ? | ❌ |
| WanT2V | ? | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ? | ❌ |
| WanI2V | ? | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ? | ❌ |
| Mochi | ? | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ? | ? | ❌ |
| SV3D | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| StableAudio | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Notes — Table 3:**
- `model_sampling`: patches `ModelSamplingSD3` shift at sampling time. Effective for SD3 (shift=3.0 typical), AuraFlow, KwaiKolors. Set to 0 to disable.
- `edm_sampling`: patches `ModelSamplingContinuousEDM` at model load time. Options: `edm_playground_v2.5` (Playground), `cosmos_rflow` (Cosmos). Other options (`edm`, `v_prediction`, `eps`) are generic aliases.
- `discrete_sampling`: patches `ModelSamplingDiscrete` at model load time. Only active for DISCRETE_CONCEPTS: SD1, SD2, SDXL, Illustrious, Turbo, Pony, Hyper, Lightning.
- `sigma_max / sigma_min`: paired with `edm_sampling` for `ContinuousEDM`. Only meaningful for Playground (and future Cosmos).
- `flux_max_shift / flux_base_shift`: patches `ModelSamplingFlux` at model load time. Flux, Flux2, Nunchaku (all Flux-architecture).
- `beta_alpha / beta_beta`: used when `scheduler_name = beta`. Applies to any model routed through PKSampler.
- `guidance`: Flux uses `FluxGuidance` conditioning. Chroma uses `CFGGuider`. Value is ignored for all other models.

---

## Quick Reference — Parameter Groups

| Group | Parameters | Key Models |
|---|---|---|
| Core | sampler_name, scheduler_name, steps, cfg | All |
| override_steps | override_steps | All |
| rescale_cfg | rescale_cfg | Flux, Chroma |
| VAE | vae, vae_selection | SD family, Flux, SD3, most others |
| CLIP layer | clip_selection, last_layer | SD family (last_layer), SD3 (clip_selection) |
| Load dtype | weight_dtype | Flux family, SANA, Kolors, Hunyuan, Qwen, Chroma |
| Precision | precision | SD1/SDXL family (fp16/fp32 only) |
| Encoders | encoder_1/2/3 | SD3, Flux, StableCascade, PixartSigma, others with external encoders |
| Attention CLIP | attn_preset, attn_query/key/value/output | All models with CLIP encoders |
| Attention UNet self | attn_query/key/value/output | UNET_CONCEPTS only (SD1/SDXL family + Playground/LCM) |
| Attention UNet cross | attn_cross_query/key/value/output | UNET_CONCEPTS only |
| Sampler type | sampler (custom_advanced/ksampler) | custom_advanced for Flux; ksampler for all others |
| Align Your Steps | align_your_steps | SD1/SDXL family (PKSampler path) |
| SD3 shift | model_sampling | SD3, AuraFlow, KwaiKolors |
| EDM | edm_sampling, sigma_max, sigma_min | Playground, Cosmos |
| Discrete | discrete_sampling, discrete_zsnr | SD1, SD2, SDXL, Illustrious, Turbo, Pony, Hyper, Lightning |
| Flux shift | flux_max_shift, flux_base_shift | Flux, Flux2, Nunchaku |
| Beta scheduler | beta_alpha, beta_beta | Any model via PKSampler when scheduler=beta |
| Guidance | guidance | Flux family, Chroma |
