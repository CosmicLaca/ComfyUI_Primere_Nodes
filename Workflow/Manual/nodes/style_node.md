# PrimereCustomStyles (Style Node)

`PrimereCustomStyles` is a universal style loader node.

It reads `.toml` style files from:

- `Toml/Styles`

Then it automatically creates selectable style inputs (with strength controls) from the selected file.

---

## What the node does

- You select a style source file (`style_source`).
- The node reads the style definitions from that TOML.
- You select one or more style entries and set strength values.
- Node output returns only style text:
  - `STYLE+`
  - `STYLE-` (if available in the style file)

This output can be merged with your main prompt in your existing workflow.

---

## Why this is useful

- **Universal**: one node can use many style packs.
- **Customizable**: users can add their own TOML files without code changes.
- **Scalable**: easy to organize style packs by topic.
- **Model-agnostic**: usable for SD/Flux/Qwen/Z-Image and other text-to-image pipelines.

---

## Create your own style file

Put your file into:

- `Toml/Styles/your_style_name.toml`

### Internal rules (important)

Follow this structure:

```toml
[MainTopic]
   [MainTopic.N_0]
      MainStyle = "Main topic label"
      SubStyle = "Sub style label"
      Positive = "Detailed style injection text for positive prompt"
      Negative = "Optional negative style injection text"

   [MainTopic.N_1]
      MainStyle = "Main topic label"
      SubStyle = "Another sub style"
      Positive = "Another detailed positive style injection"
```

### Naming rules

- First table name and nested table prefix must match (example: `MainTopic` and `MainTopic.N_0`).
- Use unique indexes: `N_0`, `N_1`, `N_2`, ...
- Required keys per entry:
  - `MainStyle`
  - `SubStyle`
  - `Positive`
- Optional key:
  - `Negative`

---

## About the provided TOML files

`Toml/Styles` already contains multiple packs (for example atmosphere, structure, time, light, human traces, etc.).

Use them as practical templates for:

- section naming
- substyle naming
- prompt sentence style

You can duplicate one, rename it, then edit content for your own domain.

---

## Quick usage tips

- Keep `Positive` texts descriptive and specific (sentence style is recommended).
- Use strength controls to softly blend styles instead of hard replacement.
- Split large topics into multiple TOML files for easier maintenance.
