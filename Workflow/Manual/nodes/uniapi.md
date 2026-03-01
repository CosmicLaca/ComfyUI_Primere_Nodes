# PrimereApiProcessor (Uniapi) — Operator Guide

## 1) Mandatory provider config (do this first)

Before using Uniapi, rename:

- `json/apiconfig.example.json` → `json/apiconfig.json`

Then fill only providers you actually use (recommended for security and clarity), e.g.:

```json
{
  "ProviderName": {
    "Name": "ProviderName",
    "APIKEY": "your-secure-api-key"
  }
}
```

> ### IMPORTANT RULE:
> #### Critical provider-name matching rule
> 
> Provider key + `Name` value in `apiconfig.json` must match provider naming in `front_end/api_schemas.json`:
> 
> - `apiconfig.json`:
>   - top-level key: `"ProviderName"`
>   - `"Name": "ProviderName"`
> - `api_schemas.json`:
>   - top-level key: `"ProviderName"`
>   - service entry contains `"provider": "ProviderName"`
> 
> #### If these names are not aligned exactly, provider/service resolution will fail.

---

## 2) Operating model (user perspective)

`PrimereApiProcessor` is driven by a registry schema (`front_end/api_schemas.json`) generated from a request snippet (`terminal_helpers/snippet.py`) via the helper script (`terminal_helpers/api_snippet_to_json.py`).

Practical flow:
1. Capture/prepare one provider-service API call snippet in `terminal_helpers/snippet.py`.
2. Generate/update schema JSON with `api_snippet_to_json.py`.
3. Optionally edit generated service schema (especially `possible_parameters`, placeholders, and optional `response_handler`).
4. Run the node using `api_provider` + `api_service` matching the registry entry.
5. Response parsing is delegated to `components/API/responses/<handler>.py`.

---

## 3) Parameterizing the snippet-to-json helper

Run from `terminal_helpers/`:

```bash
python api_snippet_to_json.py --provider <ProviderName> --service <ServiceName>
```

### Modes

- **Upsert mode (default)**: merges/updates only the specified provider/service into `result.json`.
- **Replace mode**:

```bash
python api_snippet_to_json.py --provider <ProviderName> --service <ServiceName> --replace
```

This rewrites `result.json` with only the generated entry.

### What the helper extracts

- `request.endpoint` from the call target.
- `request.sdk_call.args/kwargs` from snippet args/kwargs.
- Placeholders (`{{...}}`) for variable-like values.
- `possible_parameters` from detected placeholders, excluding internal/common keys.

### Important conversion rule

Fixed constants remain fixed and **do not** become `possible_parameters`:

- `"output_format": "png"` → fixed in `request`, excluded from `possible_parameters`
- `"numeric_context": 5` → fixed in `request`, excluded from `possible_parameters`

Variable-like values become placeholders and can appear in `possible_parameters`:

- `"image": reference_images`
- `"mask": mask_images`
- `"safety_tolerance": safety_tolerance`

Type-marker strings are treated as variable placeholders:

- `"guidance": "FLOAT"`
- `"steps": "INT"`
- `"prompt_upsampling": "STRING"`

---

## 4) Editing `api_schemas.json` (advanced usage)

You can edit generated entries directly. Minimal service shape:

```json
{
  "Provider": {
    "Service": {
      "provider": "Provider",
      "service": "Service",
      "response_handler": "Provider_Service.py",
      "possible_parameters": {
        "model": ["model-a", "model-b"],
        "quality": ["low", "high"]
      },
      "request": {
        "method": "SDK",
        "endpoint": "client.images.generate",
        "sdk_call": {
          "args": [],
          "kwargs": {
            "model": "{{model}}",
            "quality": "{{quality}}",
            "prompt": "{{prompt}}"
          }
        }
      }
    }
  }
}
```

### `possible_parameters` significance

`possible_parameters` is the editable parameter registry for that service.

- Keys represent service-level knobs expected in templates/placeholders.
- Values are option lists presented/consumed as selectable presets/defaults.
- Omit keys that are fixed in request body (constants).
- Keep keys for values you want configurable at runtime.

### `import_modules` required for SDK dependencies

`import_modules` is a **service-level list of Python import lines** loaded before SDK request execution.

This allows each provider/service to define its own runtime dependencies without hardcoding imports in `Uniapi.py`.

Example for Google Gemini services:

```json
"import_modules": [
  "from google import genai",
  "from google.genai import types"
]
```

Supported line formats:

- `import module`
- `import module as alias`
- `from package import symbol`
- `from package import symbol as alias`

How it works at runtime:

1. Node selects the schema by `api_provider` + `api_service`.
2. Uniapi reads `schema["import_modules"]`.
3. Import lines are loaded into SDK execution context.
4. Request body (`request.sdk_call`) can safely reference those roots, e.g. `types.GenerateContentConfig`.

If an import line is invalid or module is missing, Uniapi raises an explicit runtime error.

### Common edit patterns

1. **Constrain models**

```json
"possible_parameters": {
  "model": ["gpt-image-1", "gpt-image-1-mini"]
}
```

2. **Expose provider-specific tuning**

```json
"possible_parameters": {
  "safety_tolerance": [1, 2, 3, 4, 5],
  "guidance": [1.5, 2.0, 2.5]
}
```

3. **Keep fixed constants out of runtime controls**

```json
"request": {
  "sdk_call": {
    "kwargs": {
      "output_format": "png",
      "prompt": "{{prompt}}"
    }
  }
}
```

No `output_format` entry in `possible_parameters` if you want it hard-fixed.

---

## 5) `response_handler` selection behavior

Service-level `response_handler` is optional:

- If defined, that filename is used.
- If omitted/empty, fallback is `<provider>_<service>.py`.

Example override:

```json
"response_handler": "BlackForest_FluxExpandPro.py"
```

---

## 6) Runtime expectations

- JSON syntax in `front_end/api_schemas.json` is strictly validated at load time (line/column errors are raised).
- Provider keys in `api_schemas.json` must exist in `json/apiconfig.example.json`.
- Service key must match inner `"service"`, and provider key must match inner `"provider"`.
- `import_modules` must be a list of non-empty strings.
- `api_provider` and `api_service` must map to an existing registry entry.
- Schema placeholders must align with node/runtime inputs.
- Response helper module must exist in `components/API/responses`.

Response helper implementation is user-owned by design.
