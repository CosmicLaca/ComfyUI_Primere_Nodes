import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { applyPrimereButtonStyle, showToast } from "./frontend_helper.js";

const TARGET_NODE_NAME = "PrimereAutoSamplerSettings";
const CONCEPT_JSON_URL = new URL("/extensions/ComfyUI_Primere_Nodes/model_concept.json", import.meta.url).href;

function modelNameToKey(modelPath) {
    const base = modelPath.split(/[\\/]/).pop();
    return base.replace(/\.[^/.]+$/, "");
}

function collectNodeData(node, includeLoraToggles = false) {
    const SKIP_KEYS = new Set(["concepts", "models", "runtime_concept"]);
    const widgets = node.widgets || [];

    const loraBooleans = new Set(
        widgets.filter((w) => w.type === "toggle" && w.name?.endsWith("_lora")).map((w) => w.name)
    );

    const suppressedPrefixes = [];
    for (const w of widgets) {
        if (loraBooleans.has(w.name)) continue;
        if (w.type === "toggle" && w.value === false) {
            suppressedPrefixes.push(w.name + "_");
        }
    }

    const data = {};
    for (const w of widgets) {
        if (!w.name || SKIP_KEYS.has(w.name)) continue;
        if (!includeLoraToggles && loraBooleans.has(w.name)) continue;
        if (w.value === null || w.value === undefined) continue;
        if (suppressedPrefixes.some((p) => w.name.startsWith(p))) continue;
        data[w.name] = w.value;
    }

    return data;
}

async function loadConceptValues(node, key) {
    let data;
    try {
        const response = await fetch(CONCEPT_JSON_URL + "?t=" + Date.now());
        if (!response.ok) {
            showToast("error", `No saved settings found. Save settings for "${key}" first.`);
            return;
        }
        data = await response.json();
    } catch (_) {
        return;
    }

    if (!data[key]) {
        showToast("error", `No saved settings for "${key}".`);
        return;
    }

    const saved = data[key];
    for (const w of node.widgets || []) {
        if (!w.name || !Object.prototype.hasOwnProperty.call(saved, w.name)) continue;
        const newValue = saved[w.name];
        if (w.options?.values) {
            const match = w.options.values.find((v) => String(v) === String(newValue));
            if (match !== undefined) w.value = match;
        } else {
            w.value = newValue;
        }
        w.callback?.(w.value);
    }

    if (saved.speed_lora === true && saved.speed_lora_name) {
        const stepMatch = saved.speed_lora_name.match(/(\d+)step/i);
        if (stepMatch) {
            const stepsWidget = node.widgets?.find((w) => w.name === "steps");
            if (stepsWidget) {
                stepsWidget.value = parseInt(stepMatch[1], 10);
                stepsWidget.callback?.(stepsWidget.value);
            }
        }
        if (saved.speed_lora_cfg != null) {
            const cfgWidget = node.widgets?.find((w) => w.name === "cfg");
            if (cfgWidget) {
                cfgWidget.value = saved.speed_lora_cfg;
                cfgWidget.callback?.(cfgWidget.value);
            }
        }
    }

    node.setDirtyCanvas?.(true, true);
}

function initializeSamplerNode(node) {
    if (node.__primereSamplerHooked) return;
    node.__primereSamplerHooked = true;

    const saveBtn = node.addWidget("button", "💾  Save node setting", null, async () => {
        const modelsWidget = node.widgets?.find((w) => w.name === "models");
        const conceptsWidget = node.widgets?.find((w) => w.name === "concepts");

        let saveKey = null;
        const modelVal = modelsWidget?.value;
        const conceptVal = conceptsWidget?.value;

        if (modelVal && modelVal !== "Auto") {
            saveKey = modelNameToKey(modelVal);
        } else if (conceptVal && conceptVal !== "Auto") {
            saveKey = conceptVal;
        }

        if (!saveKey) {
            showToast("error", "Cannot save: select a specific model or model type first.");
            return;
        }

        const data = collectNodeData(node, modelVal && modelVal !== "Auto");

        try {
            const response = await fetch("/primere_model_concept_save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ concept: saveKey, data }),
            });
            const result = response.ok ? await response.json() : null;
            if (result?.success) {
                showToast("success", `Settings saved for "${saveKey}".`);
            } else {
                showToast("error", `Failed to save settings for "${saveKey}".`);
            }
        } catch (error) {
            showToast("error", `Save error: ${error.message}`);
        }
    });

    saveBtn.serialize = false;
    saveBtn.options = saveBtn.options || {};
    saveBtn.options.serialize = false;
    applyPrimereButtonStyle(saveBtn);

    const originalOnWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (name, value, oldValue, widget) {
        originalOnWidgetChanged?.call(this, name, value, oldValue, widget);
        if (name === "concepts" && value !== "Auto") {
            loadConceptValues(this, value);
        }
    };

    node.conceptDisplayWidget = ComfyWidgets["STRING"](node, "runtime_concept", ["STRING", { multiline: true }], app).widget;
    node.conceptDisplayWidget.inputEl.readOnly = true;
    node.conceptDisplayWidget.inputEl.placeholder = "Runtime model type will appear here";
    node.conceptDisplayWidget.serialize = false;
    node.conceptDisplayWidget.options = node.conceptDisplayWidget.options || {};
    node.conceptDisplayWidget.options.serialize = false;

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        originalOnExecuted?.call(this, message);
        const displayKey = message?.active_concept?.[0];
        if (!displayKey) return;
        if (this.conceptDisplayWidget) {
            this.conceptDisplayWidget.value = displayKey;
        }
        const modelsWidget = this.widgets?.find((w) => w.name === "models");
        const conceptsWidget = this.widgets?.find((w) => w.name === "concepts");
        if (conceptsWidget?.value !== "Auto" || modelsWidget?.value !== "Auto") return;
        loadConceptValues(this, displayKey);
    };
}

app.registerExtension({
    name: "Primere.AutoSamplerSettings",

    setup() {
        app.api.addEventListener("primere.concept_setting", (event) => {
            const detail = event.detail;
            if (detail?.status === "missing") {
                showToast("error", `No saved settings for model type "${detail.concept}". Current node values will be used.`);
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== TARGET_NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, []);
            initializeSamplerNode(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            onConfigure?.apply(this, [config]);
            initializeSamplerNode(this);
        };
    },
});
