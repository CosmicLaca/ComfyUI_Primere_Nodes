## <ins>Cache File Structure:</ins>

The `.cache.json` file uses a simple key-value structure:
```json
{
    "model_filename": "concept_name",
    "another_model_filename": "concept_name"
}
```
<hr>

## <ins>Manual Corrections</ins>

### Common Detection Issues:

1. **Lightning Models**
   - Often detected as "SDXL"
   - Should be corrected to "Lightning"
   - Example correction:
   ```json
   "modelXL_Lightning4Steps": "Lightning",
   "realisticXL_Lightning6Steps": "Lightning"
   ```

2. **Hyper Models**
   - May be detected as "SDXL"
   - Should be corrected to "Hyper"
   - Example correction:
   ```json
   "hyperSDXL_v1": "Hyper",
   "hyperSuperModel_v2": "Hyper"
   ```

3. **Flux Models**
   - Some variants may be undetected, because symlinked
   - Should be labeled as "Flux"
   - Example correction:
   ```json
   "fluxV3_quality": "Flux",
   "fluxV2_speed": "Flux"
   ```

### Editing Process:
1. Open `.cache.json` in any text editor or IDE
2. Locate the incorrect model mapping by keywords `symlink` and `unknown` to found undetected / failed models
3. Change the concept value to the correct type (case sensitive, you can selec only these: `SD1`, `SD2`, `SDXL`, `SD3`, `StableCascade`, `Turbo`, `Flux`, `KwaiKolors`, `Hunyuan`, `Playground`, `Pony`, `LCM`, `Lightning`, `Hyper`, `PixartSigma`)
4. Save the file
5. Restart ComfyUI to apply changes

## Supported Concept Values:
Use these exact strings for concept values:
- "SD1"
- "SD2"
- "SDXL"
- "SD3"
- "StableCascade"
- "Turbo"
- "Flux"
- "KwaiKolors"
- "Hunyuan"
- "Playground"
- "Pony"
- "LCM"
- "Lightning"
- "Hyper"
- "PixartSigma"

## Best Practices:
1. **Backup Original File**
   - Keep a backup of the original `.cache.json` if already exist
   - Useful for reference or recovery backed up

2. **Naming Patterns**
   - Lightning models often include "Lightning" or "xSteps" in filename
   - Hyper models typically include "Hyper" in name
   - Flux models usually contain "Flux" in name

3. **Validation**
   - Ensure correct JSON formatting
   - Verify quotation marks and commas
   - Test workflow after changes

4. **Documentation**
   - Keep notes of manual corrections
   - Document special cases for future reference

## Troubleshooting:
- If the workflow fails to load after editing, check JSON syntax
- Verify concept names match exactly (case-sensitive)
- Ensure no trailing commas in JSON
- Restart ComfyUI after any changes

Remember: Correct concept detection is crucial for optimal model performance and appropriate sampler settings.