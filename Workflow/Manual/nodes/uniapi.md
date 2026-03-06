# PrimereApiProcessor (Uniapi) — Operator Guide

**Video walkthrough:** [https://www.youtube.com/watch?v=FcKcMQoU1rM](https://www.youtube.com/watch?v=FcKcMQoU1rM)

---

## Table of Contents

1. [Setup — provider config](#1-setup--provider-config)
2. [How it works — operating model](#2-how-it-works--operating-model)
3. [Schema workflow — snippet to JSON](#3-schema-workflow--snippet-to-json)
   - 3.1 [Write modes](#31-write-modes)
   - 3.2 [All parameters](#32-all-parameters)
   - 3.3 [What the helper extracts](#33-what-the-helper-extracts)
4. [Schema reference — editing `api_schemas.json`](#4-schema-reference--editing-api_schemasjson)
   - 4.1 [Minimal schema structure](#41-minimal-schema-structure)
   - 4.2 [Placeholders — `{{key}}` syntax](#42-placeholders--key-syntax)
   - 4.3 [`possible_parameters`](#43-possible_parameters)
   - 4.4 [`import_modules`](#44-import_modules)
   - 4.5 [`request_exclusions`](#45-request_exclusions)
   - 4.6 [URL-part and endpoint placeholders](#46-url-part-and-endpoint-placeholders)
   - 4.7 [Header authentication placeholders](#47-header-authentication-placeholders)
5. [Handlers](#5-handlers)
   - 5.1 [`response_handler`](#51-response_handler)
   - 5.2 [`reference_images_handler`](#52-reference_images_handler)
6. [Runtime rules and validation](#6-runtime-rules-and-validation)
7. [Debug outputs — understanding and using them](#7-debug-outputs--understanding-and-using-them)
8. [File save settings](#8-file-save-settings)

---

## 1) Setup — provider config

Before using Uniapi, rename:

```
json/apiconfig.example.json  →  json/apiconfig.json
```

Fill only providers you actually use (recommended for security and clarity):

```json
{
  "ProviderName": {
    "Name": "ProviderName",
    "APIKEY": "your-secure-api-key"
  }
}
```

### Critical provider-name matching rule

Provider key and `Name` value in `apiconfig.json` **must exactly match** the provider naming in `front_end/api_schemas.json`:

| File | Key | Field |
|---|---|---|
| `apiconfig.json` | top-level key | `"Name"` value |
| `api_schemas.json` | top-level key | `"provider"` value inside each service |

All four of these must be the same string. If they are not aligned exactly, provider/service resolution will fail.

---

## 2) How it works — operating model

`PrimereApiProcessor` is driven by a registry schema (`front_end/api_schemas.json`) generated from a request snippet (`terminal_helpers/snippet.py`) via the helper script (`terminal_helpers/api_snippet_to_json.py`).

**Practical flow:**

1. Capture or prepare one provider-service API call snippet in `terminal_helpers/snippet.py`.
2. Generate or update schema JSON with `api_snippet_to_json.py`.
3. Optionally edit the generated service schema (especially `possible_parameters`, placeholders, and optional `response_handler`).
4. Run the node using `api_provider` + `api_service` matching the registry entry.
5. Response parsing is delegated to `components/API/responses/<handler>.py`.

---

## 3) Schema workflow — snippet to JSON

Run from `terminal_helpers/`:

```bash
python api_snippet_to_json.py --provider <ProviderName> --service <ServiceName>
```

If `--provider` or `--service` are omitted, the script will prompt you to type them interactively instead of failing immediately.

---

### 3.1 Write modes

| Flag | Behavior |
|---|---|
| *(no flag)* | Upsert — merges/updates only the specified provider/service into `result.json` |
| `--replace` | Rewrites `result.json` from scratch with only the generated entry |

---

### 3.2 All parameters

#### `--provider <name>`

Top-level provider key (e.g. `Gemini`, `OpenAI`). Must match the key in `apiconfig.json` and `api_schemas.json`. Prompted interactively if omitted.

#### `--service <name>`

Service name nested under the provider (e.g. `Imagen`, `text2image`). Prompted interactively if omitted.

#### `--replace`

Replaces `result.json` entirely with only the newly generated entry. Without this flag, the script upserts — existing entries for other providers/services are preserved.

#### `--snippet <path>`

Path to a custom snippet file. By default the script looks for `snippet.py` in the current directory. Use this to point to any file anywhere:

```bash
python api_snippet_to_json.py --provider Gemini --service Imagen --snippet /path/to/my_call.py
```

`result.json` is always written to the current working directory regardless of where the snippet file is.

#### `--dry-run`

Builds and prints the generated schema to the terminal without writing `result.json` or creating any handler files. Use this to preview the output before committing:

```bash
python api_snippet_to_json.py --provider Gemini --service Imagen --dry-run
```

Can be combined with `--validate` — validation runs first, then the schema is printed without writing.

#### `--validate`

Checks the generated schema for issues before writing. Runs two types of checks:

**Non-blocking warnings** (printed, writing continues):

- Missing or empty required fields (`provider`, `service`, `response_handler`, `import_modules`, `possible_parameters`, `request`, `request.method`, `request.endpoint`)
- Conflicts — if the same `provider/service` already exists in `front_end/api_schemas.json`

**Hard error — cancels writing** (no `result.json` is touched):

- Provider name not found in `json/apiconfig.json` — because every provider used by the node must have credentials registered there. If the provider is missing, the schema would be unusable at runtime anyway.

```bash
python api_snippet_to_json.py --provider Gemini --service Imagen --validate
```

Example hard error output when provider is missing from `apiconfig.json`:

```
[VALIDATE] Schema looks good.
ERROR: Provider 'Gemini' not found in apiconfig.json.
  Registered providers: BlackForest, OpenAI
  Add 'Gemini' to json/apiconfig.json before writing this schema.
```

Validation output is printed to the terminal. Writing `result.json` is cancelled only on the provider hard error — non-blocking warnings do not stop the write. Combine with `--dry-run` to preview without writing regardless.

#### `--list`

Lists all `provider / service` pairs currently registered in `result.json` (the local working output file). No snippet or provider/service arguments needed:

```bash
python api_snippet_to_json.py --list
```

Example output:

```
Registered services in result.json:
  Gemini / Imagen
  OpenAI / DallE
```

#### `--prodlist`

Lists all `provider / service` pairs currently registered in `front_end/api_schemas.json` (the production schema file loaded by the node). No snippet or provider/service arguments needed:

```bash
python api_snippet_to_json.py --prodlist
```

Example output:

```
Registered services in front_end/api_schemas.json:
  BlackForest / FluxPro
  Gemini / Imagen
  OpenAI / DallE
```

Use `--list` to see your local draft entries and `--prodlist` to see what is actually live in the node.

---

### 3.3 What the helper extracts from the snippet

- `request.endpoint` from the call target.
- `request.sdk_call.args/kwargs` from snippet args/kwargs.
- Placeholders (`{{...}}`) for variable-like values.
- `possible_parameters` from detected placeholders, excluding internal/common keys.

### Fixed constants vs. variable placeholders

Fixed constants remain fixed in the `request` body and **do not** become `possible_parameters`:

```
"output_format": "png"     →  fixed in request, excluded from possible_parameters
"numeric_context": 5       →  fixed in request, excluded from possible_parameters
```

Variable-like values become placeholders and can appear in `possible_parameters`:

```
"image": reference_images
"mask": mask_images
"safety_tolerance": safety_tolerance
```

Type-marker strings are also treated as variable placeholders:

```
"guidance": "FLOAT"
"steps": "INT"
"prompt_upsampling": "STRING"
```

---

## 4) Schema reference — editing `api_schemas.json`

### 4.1 Minimal schema structure

```json
{
  "Provider": {
    "Service": {
      "provider": "Provider",
      "service": "Service",
      "response_handler": "Provider_Service.py",
      "import_modules": [
        "import module_name"
      ],
      "possible_parameters": {
        "model": ["model-a", "model-b"],
        "quality": ["low", "high"]
      },
      "request": {
        "method": "SDK",
        "endpoint": "client.images.generate",
        "sdk_call": {
          "args": ["access_url"],
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

---

### 4.2 Placeholders — `{{key}}` syntax

Placeholders in the form `{{key}}` are resolved at runtime from node/workflow inputs and `possible_parameters`. They can appear anywhere in the schema: `kwargs`, `args`, `endpoint`, and `headers`.

Any `None` value left after resolution is removed from the rendered request before the API call.

---

### 4.3 `possible_parameters`

`possible_parameters` is the editable parameter registry for that service.

- Keys represent service-level knobs expected in templates/placeholders.
- Values are option lists presented/consumed as selectable presets or defaults.
- **Omit** keys that are fixed constants in the request body.
- **Keep** keys for values you want configurable at runtime.

**Common patterns:**

Constrain models:

```json
"possible_parameters": {
  "model": ["gpt-image-1", "gpt-image-1-mini"]
}
```

Expose provider-specific tuning:

```json
"possible_parameters": {
  "safety_tolerance": [1, 2, 3, 4, 5],
  "guidance": [1.5, 2.0, 2.5]
}
```

Keep a fixed constant out of runtime controls (no `output_format` in `possible_parameters`):

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

---

### 4.4 `import_modules`

`import_modules` is a **service-level list of Python import lines** loaded before SDK request execution. This allows each provider/service to define its own runtime dependencies without hardcoding imports in `Uniapi.py`.

Example for Google Gemini services:

```json
"import_modules": [
  "from google import genai",
  "from google.genai import types"
]
```

**Supported line formats:**

- `import module`
- `import module as alias`
- `from package import symbol`
- `from package import symbol as alias`

**How it works at runtime:**

1. Node selects the schema by `api_provider` + `api_service`.
2. Uniapi reads `schema["import_modules"]`.
3. Import lines are loaded into SDK execution context.
4. Request body (`request.sdk_call`) can safely reference those roots, e.g. `types.GenerateContentConfig`.

If an import line is invalid or module is missing, Uniapi raises an explicit runtime error.

---

### 4.5 `request_exclusions`

Use `request_exclusions` at service level when some request fields are incompatible in specific cases. Exclusions are evaluated after placeholder resolution and applied to the rendered payload before the API call.

**Rule structure:**

- `when` (or `if`): condition object.
  - `path` (or `key`): key path checked in rendered payload.
  - `equals` (or `value`): value that must match.
- `remove`: one path string or list of paths to remove.

**Basic example:**

```json
"request_exclusions": [
  {
    "when": {"path": "model", "equals": "model-fast-preview"},
    "remove": ["config.image_size"]
  }
]
```

Result: if `model == "model-fast-preview"`, key `config.image_size` is removed before the SDK call.

**OR logic — multiple conditions for the same removal:**

```json
"request_exclusions": [
  {
    "when": {"path": "model", "equals": "model-fast-preview"},
    "remove": ["config.advanced_options"]
  },
  {
    "when": {"path": "model", "equals": "model-pro-preview"},
    "remove": ["config.advanced_options"]
  }
]
```

**Removing a whole nested object:**

Target the parent key to remove the entire block including all child fields:

```json
"request_exclusions": [
  {
    "when": {"path": "mode", "equals": "lite"},
    "remove": ["config.thinking_config"]
  }
]
```

---

### 4.6 URL-part and endpoint placeholders

#### URL-part placeholders

Parts of `request.sdk_call.args` (the URL string) can be templated with placeholders:

```json
"request": {
  "method": "SDK",
  "endpoint": "requests.post",
  "sdk_call": {
    "args": [
      "https://{{regions}}/{{version}}/{{model_name}}"
    ],
    "kwargs": {
      "headers": {
        "x-key": "{{api_key}}",
        "Content-Type": "application/json"
      },
      "json": {
        "prompt": "{{prompt}}"
      }
    }
  }
}
```

Then define selectable values in `possible_parameters`:

```json
"possible_parameters": {
  "regions": ["region_1", "region_2", "region_3"],
  "version": ["v1", "v2"],
  "model_name": ["model_1", "model_2"]
}
```

Runtime resolution: `render_from_schema(...)` resolves all placeholders including those inside `sdk_call.args`, so the final SDK call receives a fully rendered URL such as `https://region_1/v1/model_2`.

#### Endpoint placeholders

`request.endpoint` can also be templated to switch between SDK methods:

```json
"request": {
  "method": "SDK",
  "endpoint": "client.images.{{gen_method}}",
  "sdk_call": {
    "args": [],
    "kwargs": {
      "model": "{{model}}",
      "image": "{{reference_images}}",
      "prompt": "{{prompt}}"
    }
  }
},
"possible_parameters": {
  "gen_method": ["generate", "edit"]
}
```

Runtime behavior is the same as URL placeholders: selected value is injected before SDK execution, so endpoint becomes either `client.images.generate` or `client.images.edit`.

---

### 4.7 Header authentication placeholders

For providers that expect auth in headers, use an auth placeholder to keep schemas portable:

```json
"headers": {
  "x-key": "{{api_key}}",
  "Content-Type": "application/json"
}
```

This works in both locations:

1. `request.headers` (non-SDK HTTP mode)
2. `request.sdk_call.kwargs.headers` (SDK mode, e.g. `endpoint: "requests.post"`)

**Runtime resolution order for `{{api_key}}`:**

1. Explicit node/workflow input value (if provided)
2. Provider API key from `json/apiconfig.json` (with environment variable overrides)
3. Fallback defaults

You can also use env-token placeholders via a `$call` pattern for call-based schemas:

```json
"x-key": {
  "$call": "os.environ.get",
  "$args": ["{{ENV_API_KEY}}"],
  "$kwargs": {}
}
```

This resolves to `os.environ.get("ENV_API_KEY")` at runtime.

---

## 5) Handlers

### 5.1 `response_handler`

Service-level `response_handler` is optional:

- If defined, that filename is used.
- If omitted or empty, fallback is `<provider>_<service>.py`.

Example override:

```json
"response_handler": "BlackForest_FluxExpandPro.py"
```

#### Reusing one handler across multiple services

You can point many services (even from different providers) to the **same** `response_handler` filename. This is useful when output formats are similar and you want fewer files under `components/API/responses`.

```json
"response_handler": "Shared_Image_Response.py"
```

As long as `components/API/responses/Shared_Image_Response.py` exists and exposes `handle_response(...)`, Uniapi can reuse it for all mapped services.

#### Universal `handle_response()` signature

Uniapi forwards runtime context to response handlers. Recommended signature:

```python
def handle_response(
    api_result,
    schema=None,
    loaded_client=None,
    response_url=None,
    client=None,
    sdk_context=None,
):
    ...
```

| Parameter | Description |
|---|---|
| `api_result` | Raw API SDK/HTTP result returned by request execution |
| `schema` | Selected service schema from `api_schemas.json` |
| `loaded_client` | Provider root object used for the SDK call (module or client object) |
| `response_url` | First SDK positional route argument from `request.sdk_call.args[0]` when present |
| `client` | Base API client from `api_helper.create_api_client(...)` |
| `sdk_context` | Loaded import context dictionary |

---

### 5.2 `reference_images_handler`

Reference image conversion and upload is also schema-driven and provider-pluggable. Handler files are loaded from `components/API/references/`.

**Resolution order:**

1. `reference_images_handler` from service schema (if defined)
2. `<provider>.py`
3. `default.py`

**Schema entry example:**

```json
{
  "FakeProvider": {
    "FakeEditService": {
      "provider": "FakeProvider",
      "service": "FakeEditService",
      "reference_images_handler": "FakeProvider.py",
      "response_handler": "FakeProvider_FakeEditService.py",
      "request": {
        "method": "SDK",
        "endpoint": "client.images.edit",
        "sdk_call": {
          "args": [],
          "kwargs": {
            "prompt": "{{prompt}}",
            "image": "{{reference_images}}"
          }
        }
      }
    }
  }
}
```

Custom override to a shared handler:

```json
"reference_images_handler": "SharedUploadHandler.py"
```

**Minimal handler file:**

```python
def handle_reference_images(
    img_binary_api=None,
    temp_file_ref="",
    loaded_client_for_upload=None,
    **kwargs,
):
    output = img_binary_api if isinstance(img_binary_api, list) else []
    if temp_file_ref and hasattr(loaded_client_for_upload, "upload_file"):
        output.append(loaded_client_for_upload.upload_file(temp_file_ref))
    elif temp_file_ref:
        output.append(temp_file_ref)
    return output
```

**Notes:**

- If no reference image input is connected, Uniapi does not send `reference_images`.
- Any `None` value is removed from rendered request structures before the API call.

---

## 6) Runtime rules and validation

| Rule | Detail |
|---|---|
| JSON syntax | `front_end/api_schemas.json` is strictly validated at load time — line/column errors are raised |
| Provider keys | Every provider key in `api_schemas.json` must exist in `json/apiconfig.json` |
| Name consistency | Service key must match inner `"service"` field; provider key must match inner `"provider"` field |
| `import_modules` | Must be a list of non-empty strings |
| Provider/service | `api_provider` and `api_service` must map to an existing registry entry |
| Placeholders | Schema placeholders must align with node/runtime inputs |
| Response handler | Handler module must exist in `components/API/responses/` |

Response handler implementation is user-owned by design. Examples already exist in `components/API/responses/`.

---

## 7) Debug outputs — understanding and using them

`PrimereApiProcessor` exposes several outputs for inspection. These are most useful during schema development: they let you verify what the node resolved, what was sent, and what came back — without reading source code.

### The `debug_mode` toggle

When `debug_mode` is **ON**, the node builds the full request context but **stops before making the API call**. All debug outputs are populated with the resolved data so you can inspect the intended payload safely.

When `debug_mode` is **OFF** (production), the node executes the call normally and all outputs reflect actual sent and received data.

---

### Debug output reference

The node provides the following outputs, in order:

| # | Output | Type | Contains |
|---|---|---|---|
| 1 | `RESULT` | IMAGE | Final rendered image from the API (or `None` in debug mode) |
| 2 | `CLIENT` | APICLIENT | The initialized provider SDK client object |
| 3 | `PROVIDER` | STRING | Resolved provider name (e.g. `"OpenAI"`) |
| 4 | `SCHEMA` | TUPLE | The full selected service schema from `api_schemas.json` |
| 5 | `RENDERED` | TUPLE | The complete rendered request object — all fields including endpoint, method, headers, query, body, and sdk_call |
| 6 | `RAW_PAYLOAD` | TUPLE | Only the SDK `kwargs` dict (or HTTP `body` dict) — the exact data sent to the provider |
| 7 | `REQUEST_BODY` | TUPLE | The resolved placeholder values — what each `{{key}}` was filled with |
| 8 | `API_SCHEMAS` | TUPLE | Full debug bundle: schema, selected parameters, used values, rendered payload, API result, and any error |
| 9 | `API_RESULT` | TUPLE | Raw response returned by the provider SDK or HTTP call |

---

### Differences between RENDERED, RAW_PAYLOAD, and REQUEST_BODY

These three outputs answer different questions at different levels of abstraction:

**`REQUEST_BODY`** — *What inputs were resolved?*
Shows the flat dictionary of values that were matched to placeholders: which `{{key}}` resolved to what value. This is the input side of the rendering process. Use this to verify that node inputs, `possible_parameters`, and workflow values are being picked up correctly.

```
{ "prompt": "a red car", "model": "gpt-image-1", "width": 1024, "height": 1024 }
```

**`RENDERED`** — *What was the full request object after rendering?*
Shows the complete `RenderResult` object after all placeholders have been substituted, all exclusions applied, and all structure assembled. Includes endpoint, method, headers, query params, body, and sdk_call in full. Binary data (reference images) is sanitized to `[reference_images omitted]` for display. Use this to verify the full request structure, including nested objects and headers, before they reach the provider.

```
{ "endpoint": "client.images.edit", "method": "SDK", "headers": {...},
  "sdk_call": { "args": [], "kwargs": { "model": "gpt-image-1", "prompt": "a red car", ... } } }
```

**`RAW_PAYLOAD`** — *What exact data was handed to the provider?*
Shows only the `sdk_call.kwargs` dict (SDK mode) or `body` dict (HTTP mode) — stripped of all Uniapi envelope fields. This is the minimal "what was actually sent" view. Binary data is sanitized. Use this when debugging provider-side errors: if the provider rejects a request, compare `RAW_PAYLOAD` against the provider's API documentation.

```
{ "model": "gpt-image-1", "prompt": "a red car", "size": "1024x1024", "n": 1 }
```

---

### `API_SCHEMAS` — the full debug bundle

`API_SCHEMAS` is a single output combining all debug data into one inspectable object:

```
{
  "schema":               the full service schema,
  "selected_parameters":  resolved node/workflow inputs (without reference images),
  "used_values":          placeholder-to-value mapping (REQUEST_BODY content),
  "selected_service":     the service name that was matched,
  "rendered":             full rendered request (RENDERED content),
  "api_result":           raw provider response (API_RESULT content),
  "api_error":            error string if the call failed, otherwise null
}
```

Connect this to a `PrimereAnyOutput` node for a complete one-stop view during development.

---

### How debug outputs help when designing a new schema

When writing a new service schema from scratch or adapting an existing one, the debug outputs replace trial-and-error by showing you exactly what the system does with your schema at each step:

1. **Start with `REQUEST_BODY`** to confirm your `possible_parameters` keys and placeholder names are being picked up. If a value is missing here, the placeholder cannot be resolved and will be sent as a literal `{{key}}` string to the provider.

2. **Check `RENDERED`** to verify nested structure. If your schema has nested `kwargs` (e.g. `config.thinking_config`), this output shows whether the nesting survived rendering correctly and whether `request_exclusions` removed what they should.

3. **Compare `RAW_PAYLOAD`** against the provider's official API documentation. The provider receives exactly what is shown here (minus binary fields). If keys are present that should not be, add a `request_exclusions` rule. If keys are missing, check that the placeholder name in `kwargs` matches a key in `possible_parameters` or a standard node input.

4. **Use `debug_mode = ON`** during the design phase — the API call is never made, so there is no cost and no rate limit risk. Iterate on the schema until all three payload outputs look correct, then switch to production mode.

5. **Read `API_SCHEMAS`** after a failed production call. The `api_error` field contains the provider error message. Combined with `rendered` and `api_result` in the same object, you can diagnose whether the error is a structural problem (wrong key, wrong nesting, wrong type) or a credential/quota problem.

---

## 8) File save settings

File saving only runs when `auto_save_result` is ON and the API returned a valid result. If the response is `None` (no API error but no result), nothing is written.

### Output path

| Input | Default | Description |
|---|---|---|
| `output_path` | `[time(%Y-%m-%d)]` | Base output directory. Supports `[time(...)]` tokens. Relative paths are anchored to the ComfyUI output directory. Absolute paths are used as-is. |
| `subpath` | `Project` | Fixed subdirectory appended after provider/service/model dirs. Select `None` to skip. |
| `add_provider_to_path` | OFF | Adds the API provider name as a subdirectory (e.g. `Gemini`). |
| `add_service_to_path` | OFF | Adds the selected service name as a subdirectory (e.g. `Imagen`). |
| `add_model_to_path` | OFF | Adds the model identifier as a subdirectory. Reads `model_name` first, then `model`, then `version` from service parameters. |

Directory structure example with all path options enabled:

```
<output_path> / <provider> / <service> / <model> / <subpath> / <filename>
```

User-supplied strings (provider, service, model, subpath, output_path) are automatically sanitized before use as path components: spaces and special characters (` / \ . , ; - `) are replaced with `_`, consecutive underscores collapsed to one.

### Filename

| Input | Default | Description |
|---|---|---|
| `filename_prefix` | `API` | Base name for the saved file. |
| `filename_delimiter` | `_` | Separator between prefix, date, time, and counter parts. |
| `add_date_to_filename` | ON | Appends current date (`YYYY-MM-DD`) to filename. |
| `add_time_to_filename` | ON | Appends current time (`HHMMSS`) to filename. |
| `filename_number_padding` | `2` | Zero-padding width for the auto-increment counter. |
| `filename_number_start` | OFF | If ON, counter is placed before the prefix instead of after. |

### Image format

| Input | Default | Description |
|---|---|---|
| `image_extension` | `jpg` | Target image format. Options: `jpeg jpg png tiff gif bmp webp`. |
| `image_quality` | `95` | Compression quality for JPEG and WEBP. PNG, TIFF, GIF ignore this. |

### Non-image results

The actual file type is detected from the API response bytes (MIME detection), not assumed from `image_extension`. If the API returns audio, video, or text, the correct extension is used automatically:

| MIME type | Saved extension |
|---|---|
| `image/*` | uses `image_extension` input |
| `audio/*` | `.mp3` |
| `video/*` | `.mp4` |
| `text/*` | `.txt` (UTF-8) |
| other / unknown | extension from `image_extension` |

### Metadata files

| Input | Default | Description |
|---|---|---|
| `save_data_to_json` | OFF | Saves a `.json` file alongside the result containing provider, service, selected parameters, used values, and raw payload. |
| `save_data_to_txt` | OFF | Saves a `.txt` file alongside the result containing provider, service, and all used parameter values (flattened key: value lines). |

Both files share the same base path and filename as the saved result, only the extension differs.
