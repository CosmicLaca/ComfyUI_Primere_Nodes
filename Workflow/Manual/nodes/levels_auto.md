## Primere Rasterix (The ToneLab)
---
### Auto Levels Node

Per-channel automatic levels normalization with edge protection, anti-comb/anti-spike dithering and optional auto gamma.

### Inputs

| Input                    | Type    | Default     | Description |
|--------------------------|---------|-------------|-----------|
| `auto_normalize`         | BOOLEAN | True        | When OFF the node passes the image unchanged. When ON the full auto levels pipeline is applied. |
| `threshold`              | FLOAT   | 2.0         | Percent of pixels to clip at each end for black/white point detection (0.0–100.0). |
| `normalize_gaps`         | BOOLEAN | True        | Anti-comb filter. Applies TPDF dithering to fill quantization gaps. |
| `normalize_midpeaks`     | BOOLEAN | False       | Anti-spike filter. Smooths histogram peaks near gaps. |
| `peak_width`             | INT     | 3           | Only used when `normalize_midpeaks=True`. Distance from a gap that qualifies a bin for smoothing (1–10). |
| `auto_gamma`             | BOOLEAN | True        | Applies automatic per-channel gamma correction after stretch. |
| `gamma_target`           | FLOAT   | 128.0       | Target mean brightness after auto gamma (0–255 scale). |
| `precision`              | BOOLEAN | False       | False = 8-bit pipeline. True = internal 16-bit processing with final 8-bit output. |

---
### Histogram Analysis Details

The core black/white point detection is handled by `levels_detect_points()` and runs independently on each RGB channel.

#### Process per channel:

1. Build full histogram (`n_bins = max_val + 1`).
2. Compute cumulative sum.
3. Calculate cutoff = total_pixels × (threshold / 100.0).
4. **Black point**: scan from left until cumulative count ≥ cutoff.
5. **White point**: scan from right until pixels above current bin ≥ cutoff.
6. Compute `scale = max_val / (white_point - black_point)`.

If white_point ≤ black_point, it is forced to black_point + 1.

#### Threshold Effect Table

| Threshold | Clipped pixels (per end) | Typical result                  | Recommended for                     |
|-----------|---------------------------|----------------------------------|-------------------------------------|
| 0.5       | ~0.5%                     | Very gentle stretch              | Clean, high-quality images          |
| 1.0–3.0   | 1–3%                      | Balanced (default)               | Most AI-generated / photographic images |
| 5.0       | 5%                        | Moderate stretch                 | Flat or low-contrast images         |
| 10+       | 10%+                      | Aggressive stretch               | Very low-contrast / foggy images    |

**Note**: Analysis is purely percentile-based. It does **not** use mean, median or Otsu’s method.

---
### Edge Spread Mechanism

`levels_edge_spread()` is **always applied**, even if threshold = 0.

- Instead of hard clipping, clipped shadow and highlight pixels are redistributed using **rank ordering**.
- At 8-bit: spreads ~8 levels at the bottom and top.
- At 16-bit: spreads proportionally more (~2056 levels).
- Shadows below black_point are softly spread into [0 … edge_spread].
- Highlights above white_point are softly spread into [max_val - edge_spread … max_val].
- This prevents solid black/white blocks and preserves micro-detail in extremes.

---
### TPDF Dithering Methods

The node implements two independent TPDF (Triangular Probability Density Function) dithering stages.

#### 1. Gap Dithering (`levels_normalize_gaps`)

- **Purpose**: Fills quantization gaps (combing) caused by non-integer stretch scale factors.
- **Method**: Classic TPDF dither using two independent uniform random distributions:
  ```python
  noise = uniform(-half, half) + uniform(-half, half)
  
- **Amplitude**: Dynamically calculated as
  `amplitude = max(1.0, (scale / 1.275) ** 2.2) * (max_val / 255.0)`
- **Recommended**: Keep ON for almost all images.

#### 2. Midpeak Smoothing (`levels_normalize_midpeaks`)

- **Purpose**: Removes artificial histogram spikes near zero bins (common in AI-generated images).
- **Method**: Targeted TPDF dither applied only to pixels whose integer bin is near a gap.
- **Amplitude**: `amp = (peak_width / 2.0) * (max_val / 255.0)`
- **Recommended**: Enable on images showing posterization or visible banding after stretch.

Both dither stages use per-channel independent random generators (`rng_gap` and `rng_spike`) for reproducibility across runs.

---
### Full Pipeline

When `auto_normalize=True`, each RGB channel follows this exact order:

1. `levels_detect_points` — Histogram-based black/white point detection
2. `levels_stretch` — Linear stretch: black_point → 0, white_point → max_val
3. `levels_edge_spread` — Always-on rank-based edge protection
4. `levels_normalize_midpeaks` — Optional anti-spike TPDF smoothing (if enabled)
5. `levels_normalize_gaps` — Optional anti-comb TPDF dithering (if enabled)
6. `levels_auto_gamma` — Optional per-channel gamma correction (if enabled)

Final output is always RGB uint8.

### Parameter Details

#### auto_levels_threshold
Controls strength of contrast stretch. See table above.

#### normalize_gaps (Anti-comb filter
Fills quantization gaps. Uses full-image TPDF dither. Recommended: ON.

#### normalize_midpeaks (Anti-spike filter)
Targets histogram spikes near gaps. Uses targeted TPDF dither. Use with `peak_width`.

#### peak_width
Defines how wide the smoothing zone around gaps is (default 3 is balanced).

#### auto_gamma
Computes gamma per channel to reach `gamma_target` while anchoring black and white points. Clamped 0.25–4.0.

#### gamma_target
`0–255` scale.  
• `128` = neutral (default)  
• `110–120` = darker / cinematic  
• `140–155` = brighter / airy

#### precision
Enables full 16-bit internal processing (65536 bins) for cleaner results on generated content, then scales back to 8-bit output.

---

### Node Screenshot

*(Insert node screenshot here)*

### Example Images

**Original vs Auto Levels**  
*(Insert before/after comparison images here)*

**Different Threshold Values**  
*(Insert examples with threshold 0.5 / 2.0 / 5.0 here)*

**Edge Spread vs Hard Clip**  
*(Insert comparison showing edge spread protection here)*

**TPDF Dithering Effect (Gaps vs Midpeaks)**  
*(Insert examples demonstrating dithering removal of banding here)*

**With and Without Auto Gamma**  
*(Insert side-by-side examples here)*

**16-bit Precision Mode**  
*(Insert examples showing precision=True vs False here)*

---

### Usage Tips

- Start with defaults (`threshold=2.0`, `normalize_gaps=True`, `auto_gamma=True`).
- Increase `threshold` on flat or low-contrast inputs.
- Enable `normalize_midpeaks=True` when posterization or spikes are visible.
- Use `precision=True` after latent decoding or heavy upscaling.
- Lower `gamma_target` for moody/cinematic look, raise it for bright results.
- Apply Auto Levels **before** film rendering node for best combined analog look.
- Threshold > 8–10 often destroys shadow/highlight detail — use sparingly.

---

### Benefits

- Fully automatic per-channel levels without manual black/white point picking
- Built-in rank-based edge spread protection against hard clipping
- High-quality TPDF dithering (gaps + midpeaks) to eliminate posterization and combing
- Automatic gamma that preserves black and white points
- Optional 16-bit internal precision for AI-generated content
- Consistent, repeatable behaviour across different inputs
- Lightweight — only NumPy and PIL required