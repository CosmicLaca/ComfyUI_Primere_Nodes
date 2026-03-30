import { app } from "/scripts/app.js";
import { applyPrimereButtonStyle, showToast } from "./frontend_helper.js";

const TARGET_NODE_NAME = "PrimereRasterix";
const SETTINGS_JSON_URL = new URL("/extensions/ComfyUI_Primere_Nodes/rasterix_settings.json", import.meta.url).href;

const JSON_EXCLUDE_KEYS = new Set([
    "image",
    "model_concept",
    "model_name",
    "film_type",
    "show_histogram",
    "histogram_source",
    "histogram_channel",
    "histogram_style",
    "$$canvas-image-preview"
]);

function modelNameToKey(modelPath) {
    const base = modelPath.split(/[\\/]/).pop();
    return base.replace(/\.[^/.]+$/, "");
}

function collectNodeData(node) {
    const SKIP_KEYS = new Set(["concepts", "models", ...JSON_EXCLUDE_KEYS]);
    const widgets = node.widgets || [];

    const suppressedPrefixes = [];
    for (const w of widgets) {
        if (w.type === "toggle" && w.value === false) {
            suppressedPrefixes.push(w.name + "_");
        }
    }

    const data = {};
    for (const w of widgets) {
        if (!w.name || SKIP_KEYS.has(w.name)) continue;
        if (w.value === null || w.value === undefined) continue;
        if (suppressedPrefixes.some((p) => w.name.startsWith(p))) continue;
        data[w.name] = w.value;
    }

    return data;
}

async function loadSettingsValues(node, key, silent = false) {
    let data;
    try {
        const response = await fetch(SETTINGS_JSON_URL + "?t=" + Date.now());
        if (!response.ok) {
            if (!silent) showToast("error", `No saved settings found. Save settings for "${key}" first.`);
            return;
        }
        data = await response.json();
    } catch (_) {
        return;
    }

    if (!data[key]) {
        if (!silent) showToast("error", `No saved settings for "${key}".`);
        return;
    }

    const saved = data[key];
    const filmTypeWidget = node.widgets?.find((w) => w.name === "film_type");
    if (filmTypeWidget && filmTypeWidget.value !== "All") {
        filmTypeWidget.value = "All";
        filmTypeWidget.callback?.(filmTypeWidget.value);
    }

    const savedFilmRendering = saved.film_rendering;
    for (const w of node.widgets || []) {
        if (w.name === "film_type") continue;
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

    if (savedFilmRendering) {
        const filmRenderingWidget = node.widgets?.find((w) => w.name === "film_rendering");
        if (filmRenderingWidget) {
            if (filmRenderingWidget.options?.values && !filmRenderingWidget.options.values.includes(savedFilmRendering)) {
                filmRenderingWidget.options.values = [...filmRenderingWidget.options.values, savedFilmRendering];
            }
            filmRenderingWidget.value = savedFilmRendering;
            filmRenderingWidget.callback?.(filmRenderingWidget.value);
        }
    }

    node.setDirtyCanvas?.(true, true);
}

function initializeRasterixSettings(node) {
    if (node.__primereRasterixSettingsHooked) return;
    node.__primereRasterixSettingsHooked = true;

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

        const data = collectNodeData(node);

        try {
            const response = await fetch("/primere_rasterix_setting_save", {
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
            loadSettingsValues(this, value);
        }
        if (name === "models" && value !== "Auto") {
            const conceptsWidget = this.widgets?.find((w) => w.name === "concepts");
            if (conceptsWidget?.value === "Auto") {
                loadSettingsValues(this, modelNameToKey(value), true);
            }
        }
    };

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        originalOnExecuted?.call(this, message);
        const displayKey = message?.active_concept?.[0];
        if (!displayKey) return;
        const modelsWidget = this.widgets?.find((w) => w.name === "models");
        const conceptsWidget = this.widgets?.find((w) => w.name === "concepts");
        if (conceptsWidget?.value !== "Auto" || modelsWidget?.value !== "Auto") return;
        loadSettingsValues(this, displayKey, true);
    };
}

app.registerExtension({
    name: "Primere.RasterixSettings",

    setup() {
        app.api.addEventListener("primere.rasterix_setting", (event) => {
            const detail = event.detail;
            if (detail?.status === "missing" && detail.concept != null) {
                showToast("error", `No saved settings for model type "${detail.concept}". Current node values will be used.`);
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== TARGET_NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, []);
            initializeRasterixSettings(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            onConfigure?.apply(this, [config]);
            initializeRasterixSettings(this);
        };
    },
});
