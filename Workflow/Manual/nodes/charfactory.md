# PrimereCharacterFactory — Compositing Character Builder

Build consistent characters by assembling them from modular attribute sections — age, body type, face shape, eyes, hair, makeup, outfit, footwear, accessories, pose, mood, lighting, and more.

---

## How it works

The node loads attribute categories from a structured TOML definition file. Each category is shown as a dropdown widget with curated options. You can:

- **Select specific values** per category (e.g. "almond" for eye shape, "chunky sneakers" for footwear)
- **Set to Random** to let the node pick from that category
- **Set to None** to skip that category entirely
- **Save full character presets** with a single click for reuse
- **Preview** what each value looks like before generating

The final output is a composed text prompt that chains all selected attributes into a coherent character description.

---

## Saving characters as reusable presets

1. Select values across the attribute sections to build your character
2. Click the **Save prompt** button (red, pinned at the top)
3. Give the preset a name and confirm
4. The preset appears in the **saved_character** dropdown — select it to instantly restore all values

Saved presets store every attribute, plus gender and content rating, so you can switch between characters without re-selecting each section manually.

---

## Preview modal — seeing options visually

Click on any section's dropdown or the saved_character widget to open a visual preview modal. This shows all available values for that section as image cards.

- **None** and **Random** cards are always pinned at the top
- Use the **filter** input to search by name
- Click the **Name** sort button to sort alphabetically, toggle between ascending/descending
- Click a card to select that value and close the modal

### Saving your own preview images

These are **manual previews** — you create them yourself after generating.

1. Run a generation with a single section selected (set everything else to None)
2. In the **PrimereOutput** node, use the image saver's preview target dropdown and select **Character Factory**
3. The preview is saved and linked to that specific value — next time you open the modal, your image appears

The preview filename is generated automatically from the section name and value. There is no automatic capture; you decide which outputs to save as previews.

---

## Tips for creative use

- **Single-section preview mode:** Select just one attribute (e.g. only "eye color") and set everything else to None. The output wraps with clean framing and detail prompts — useful for isolating and previewing individual features before assembling a full character
- **Random seed:** Connect a noise seed for reproducible randomization. Set to 0 for true randomness on each run
- **Gender and content rating** control which sections and options appear — feminine sections show for "woman", masculine for "man"; content rating filters appropriate material
- **Workflow metadata:** Character values are recovered from workflow metadata on graph load, so dynamic section widgets are restored even though they are created by the frontend

---

## Quick reference

| Section | What you control |
|---|---|
| saved_character | Load a previously saved full character preset |
| gender / content_rating | Filter available sections and values |
| noise_seed | Seed for Random selections |
| All other sections | Specific attribute pickers (age, body, face, outfit, etc.) |
