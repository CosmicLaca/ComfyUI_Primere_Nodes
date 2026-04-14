# Rasterix Nodes Manual (Modular Version)

> This manual covers the **modular Rasterix nodes** from `Nodes/Rasterix.py`.
> 
> **Not covered here:** `PrimereRasterix` (all-in-one merged node).

### Important note about the merged node (PrimereRasterix)

Even though this manual is for modular nodes, there is one major benefit in the merged `PrimereRasterix` node:

- You can save **all Rasterix settings** as a profile by:
  - **Model concept** (for example: SDXL, Flux, Qwen, Z-Image, etc.), or
  - **Exact model name**.
- Later, when the same concept/model appears in your workflow, `PrimereRasterix` can auto-apply the saved profile.

So the merged node acts like a reusable “smart preset loader” tied to your model workflow context.

---

## Who this is for

This guide is for creative users (photographers, concept artists, stylizers) who want simple, practical image finishing controls without deep technical setup.

---

## How to use Rasterix nodes in a workflow

1. Start with your source image.
2. Add only the modules you need (you do **not** need all nodes).
3. Turn each module on with its `use_*` switch.
4. Keep settings subtle first, then increase gradually.
5. Typical order:
   - Normalize / White Balance / Smart Lighting
   - Atmosphere (dehaze, blur, depth blur)
   - Tone and color
   - Detail and finishing
   - Output cleanup + histogram check

---

## Node-by-node guide

---

### 1) PrimereAutoNormalize
**What it does:** Auto-levels and optional gamma balancing. Great as first cleanup.

**Best for:** Flat renders, low-contrast outputs, uneven exposure starts.

**Key controls:**
- `auto_normalize`
- `auto_levels_threshold`
- `auto_gamma`, `gamma_target`
- anti-comb / anti-spike filters (`normalize_gaps`, `normalize_midpeaks`)

**Benefit:** Fast tonal baseline before creative grading.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereAutoNormalize -->`

---

### 2) PrimereWhiteBalance
**What it does:** Corrects temperature + tint cast.

**Best for:** Blue/orange/green color cast cleanup.

**Key controls:** `wb_temperature`, `wb_tint`.

**Benefit:** Neutral base makes all later color edits easier.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereWhiteBalance -->`

---

### 3) PrimereSmartLighting
**What it does:** Adaptive light shaping.

**Best for:** Recovering depth/readability in difficult lighting.

**Key control:** `smart_lighting` intensity.

**Benefit:** More balanced perceived lighting without full relight.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereSmartLighting -->`

---

### 4) PrimereBlur
**What it does:** Creative/global blur (gaussian, motion, bilateral, lens).

**Best for:** Softening, diffusion, dreamy style, motion look.

**Key controls:** `blur_type`, `blur_intensity`, `blur_radius`, `angle`, edge protection.

**Benefit:** Direct mood control with one module.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereBlur -->`

---

### 5) PrimereBrightnessContrast
**What it does:** Global brightness/contrast shaping.

**Best for:** Quick punch or flattening.

**Key controls:** `brightness`, `contrast`, `use_legacy` mode.

**Benefit:** Fast global tone adjustment.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereBrightnessContrast -->`

---

### 6) PrimereFilmRendering
**What it does:** Film/sensor style presets (including grain/halation options).

**Best for:** Analog mood, cinematic palette, retro rendering.

**Key controls:** `film_type`, `film_rendering`, intensity, ISO grain, halation, expiration.

**Benefit:** One-node stylization with strong character.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereFilmRendering -->`

---

### 7) PrimerePhotoPaper
**What it does:** Darkroom paper response simulation.

**Best for:** Print-like finish and paper character.

**Key controls:** `photo_paper`, `paper_base` (RC/FB), `color_paper`, intensity, expiration.

**Benefit:** Final output gets physical/print personality.

**Screenshot placeholder:** `<!-- Add screenshot: PrimerePhotoPaper -->`

---

### 8) PrimereSelectiveTone
**What it does:** Zone-based tonal adjustment (highlights/midtones/shadows/blacks).

**Best for:** Targeted tone shaping without flattening full image.

**Key controls:** zone, value, separation, strength.

**Benefit:** Better local tone feel than simple global contrast.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereSelectiveTone -->`

---

### 9) PrimereColorBalance
**What it does:** Color balance wheels logic for tonal ranges.

**Best for:** Corrective grading and creative color direction.

**Key controls:** cyan-red, magenta-green, yellow-blue, tone range, preserve luminosity.

**Benefit:** Controlled color mood without destroying luminance.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereColorBalance -->`

---

### 10) PrimereHSL
**What it does:** Hue/Saturation/Lightness/Vibrance by channel.

**Best for:** Color isolation edits and skin-safe vibrance tweaks.

**Key controls:** hue/sat/lightness/vibrance + channel settings.

**Benefit:** Precise color sculpting.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereHSL -->`

---

### 11) PrimereShadeDetailer
**What it does:** Microcontrast-like detail shaping.

**Best for:** Texture pop and local depth.

**Key controls:** `shade_level`, `shade_radius`, detail mode, strength.

**Benefit:** Adds perceived detail without pure sharpening look.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereShadeDetailer -->`

---

### 12) PrimereClarity
**What it does:** Midtone clarity enhancement.

**Best for:** Crispness in portraits/landscapes.

**Key controls:** strength, radius, midtone range, edge preservation.

**Benefit:** Cleaner punch than raw sharpening.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereClarity -->`

---

### 13) PrimereLocalLaplacian
**What it does:** Edge-aware local contrast/detail enhancement.

**Best for:** Rich detail with protected edges.

**Key controls:** sigma, contrast, detail, levels.

**Benefit:** Strong detail enhancement with fewer halos.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereLocalLaplacian -->`

---

### 14) PrimereFrequencySeparation
**What it does:** Low/high frequency split workflow.

**Best for:** Portrait cleanup and texture/tone separation.

**Key controls:** radius, low/high strength, blend mode.

**Benefit:** Professional retouch-style control.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereFrequencySeparation -->`

---

### 15) PrimereDehaze
**What it does:** Removes haze/veil and restores contrast.

**Best for:** Foggy, milky, low-contrast atmospheres.

**Key controls:** strength, radius, omega, t0, contrast.

**Benefit:** Better atmospheric readability.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereDehaze -->`

---

### 16) PrimereDepthBlur
**What it does:** Depth-guided blur (focus separation).

**Best for:** Subject isolation, cinematic depth feel.

**Key controls:** focus depth, range, max blur, gamma, depth model/auto optimize.

**Benefit:** More natural focus falloff than flat blur.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereDepthBlur -->`

---

### 17) PrimereRasterixLens
**What it does:** Complete lens-effect toolbox (vignette, CA, bokeh, distortion, flare, halation, focus falloff, anamorphic, advanced optical/sensor effects).

**Best for:** Camera/lens realism and optical styling.

**Key idea:** Enable only needed blocks (`use_vignette`, `use_chroma`, etc.) and stack subtly.

**Benefit:** High-end lens personality from one module.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereRasterixLens -->`

---

### 18) PrimereRasterixGrain
**What it does:** Film grain with type, tint, tonal response controls.

**Best for:** Texture realism, analog finish, anti-plastic digital look.

**Key controls:** intensity, size, grain type, color mode/tint, shadow/highlight behavior.

**Benefit:** Organic texture finish.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereRasterixGrain -->`

---

### 19) PrimereFilmicCurve
**What it does:** Filmic/log tone curve and rolloff behavior.

**Best for:** Cinematic highlight/shadow response.

**Key controls:** curve type, contrast, highlight rolloff, shadow lift, pivot.

**Benefit:** Smoother dynamic-range feel.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereFilmicCurve -->`

---

### 20) PrimereLUT3D
**What it does:** Loads and blends `.cube` LUTs.

**Best for:** Fast look development and show consistency.

**Key controls:** LUT file, intensity, color space.

**Benefit:** Reusable color signature in one step.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereLUT3D -->`

---

### 21) PrimereLevelEndpoints
**What it does:** Black/white endpoint compression/offset.

**Best for:** Output anchoring, clip tuning.

**Key controls:** black offset, white offset, skip-if-no-clip.

**Benefit:** Cleaner final histogram edges.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereLevelEndpoints -->`

---

### 22) PrimerePosterize
**What it does:** Reduces tonal levels per channels.

**Best for:** Stylized flat-color look, graphic art.

**Key controls:** shades + channel setup.

**Benefit:** Strong abstraction effect.

**Screenshot placeholder:** `<!-- Add screenshot: PrimerePosterize -->`

---

### 23) PrimereDithering
**What it does:** Dither quantization and error diffusion, plus anti-spike smoothing.

**Best for:** Gradient banding reduction and stylized low-bit looks.

**Key controls:** dither quantization, adaptive strength, error diffusion.

**Benefit:** Better gradients after compression/posterize.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereDithering -->`

---

### 24) PrimereSolarizationBW
**What it does:** Solarization process (optional B&W conversion).

**Best for:** Experimental darkroom-style visuals.

**Key controls:** strength, pivot, sigma, edge boost, grain modulation.

**Benefit:** Signature artistic inversion style.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereSolarizationBW -->`

---

### 25) PrimereEdgeJitter
**What it does:** Controlled edge instability/organic jitter.

**Best for:** Handmade/analog or anti-clean digital feel.

**Key controls:** strength, radius, threshold, randomness (+seed).

**Benefit:** Adds natural imperfection.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereEdgeJitter -->`

---

### 26) PrimereAIDetectionBypasser
**What it does:** Applies perturbation pipeline aimed at detector resistance.

**Best for:** Experimental export strategy where detection robustness is needed.

**Key controls:** frequency/variance strength, unsharp amount, jpeg cycles.

**Benefit:** Alternative defensive post-process option.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereAIDetectionBypasser -->`

---

### 27) PrimereHistogram
**What it does:** Shows histogram visualization modes (RGB, channels, waveform-style variants).

**Best for:** Final quality check before save.

**Key controls:** channel + histogram style.

**Benefit:** Fast visual validation of clipping and tonal distribution.

**Screenshot placeholder:** `<!-- Add screenshot: PrimereHistogram -->`

---

## Practical starter recipes

### Quick natural cleanup
1. AutoNormalize
2. WhiteBalance
3. SmartLighting
4. Clarity (small amount)
5. Histogram

### Cinematic look
1. Dehaze (light)
2. FilmicCurve
3. LUT3D
4. FilmRendering / Lens
5. Grain

### Portrait polish
1. WhiteBalance
2. FrequencySeparation
3. Clarity (small)
4. HSL (skin-safe vibrance)
5. PhotoPaper (optional finish)

---

## Customization tips (creative-first)

- Use **one strong module + one subtle module** first.
- If image breaks, reduce strengths before adding new nodes.
- Save reusable mini-chains as your own “look presets”.
- Prefer Histogram at the end to prevent hidden clipping.

---

## Final note

If you need everything in one place, use the all-in-one `PrimereRasterix` node.

If you want flexibility and easier learning, use these modular nodes one-by-one.

---

## Extra: why PrimereRasterix profile save/load is powerful

The merged node is not only “all controls in one UI”, it is also a **workflow memory system**.

### What it can remember
- Full Rasterix setup:
  - tone controls
  - color controls
  - detail controls
  - film/LUT/lens style decisions
  - finishing/output settings

### How profile targeting works
You can save settings to:
- a **model concept profile** (example: “Flux look”, “SDXL look”), or
- a **specific model profile** (exact model name).

### What happens during generation
If your workflow contains model concept/model name metadata and it matches a saved profile:
- `PrimereRasterix` can load and apply that profile automatically.

### Why this helps creatives
- **Consistency at scale**: same visual language across many generations.
- **Faster production**: no need to rebuild grading chain for every run.
- **Safer experimentation**: keep trusted defaults while testing new ideas.
- **Universal workflow reuse**: one workflow can adapt styling by model context.

In short: modular nodes are best for learning and custom chains, while merged `PrimereRasterix` is ideal for profile-based, repeatable production pipelines.
