import { app } from "/scripts/app.js";

const TARGET_NODE_NAME = "PrimereApiProcessor";
const SCHEMA_URL = new URL("/extensions/ComfyUI_Primere_Nodes/api_schemas.json", import.meta.url).href;

let schemaCache = null;
let schemaPromise = null;

async function loadSchemas() {
    if (schemaCache) {
        return schemaCache;
    }

    if (!schemaPromise) {
        schemaPromise = fetch(SCHEMA_URL)
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`Cannot load schema file (${response.status})`);
                }
                return response.json();
            })
            .then((json) => {
                schemaCache = json;
                return schemaCache;
            })
            .catch((error) => {
                console.error("[Primere UniApi] Failed to load json/api_schemas.json", error);
                return {};
            });
    }

    return schemaPromise;
}

function getWidget(node, widgetName) {
    return node.widgets?.find((widget) => widget?.name === widgetName);
}

function markDynamicWidget(node, widgetName) {
    const widget = getWidget(node, widgetName);
    if (widget) {
        widget.__primereUniApiDynamic = true;
    }
    return widget;
}
function ensureComboWidget(node, widgetName, values) {
    const normalizedValues = Array.isArray(values) && values.length > 0
        ? values.map((value) => String(value))
        : [`default_${widgetName}`];

    let widget = getWidget(node, widgetName);
    if (!widget) {
        widget = node.addWidget("combo", widgetName, normalizedValues[0], () => {}, { values: normalizedValues });
    }

    widget.options = widget.options || {};
    widget.options.values = normalizedValues;

    if (!normalizedValues.includes(String(widget.value))) {
        widget.value = normalizedValues[0];
    }

    return markDynamicWidget(node, widgetName);
}

function ensureBooleanWidget(node, widgetName, values) {
    const normalizedValues = Array.isArray(values) && values.length > 0
        ? values.map((value) => Boolean(value))
        : [false, true];

    const defaultValue = normalizedValues[0];
    let widget = getWidget(node, widgetName);

    if (!widget || widget.type !== "toggle") {
        removeWidgetByName(node, widgetName);
        widget = node.addWidget("toggle", widgetName, defaultValue, () => {}, {
            on: "ON",
            off: "OFF",
            label_on: "ON",
            label_off: "OFF",
        });
    }

    widget.options = widget.options || {};
    widget.options.on = "ON";
    widget.options.off = "OFF";
    widget.options.label_on = "ON";
    widget.options.label_off = "OFF";

    if (!normalizedValues.includes(Boolean(widget.value))) {
        widget.value = defaultValue;
    }

    return markDynamicWidget(node, widgetName);
}

function ensureIntegerWidget(node, widgetName, values, integerSpec = null) {
    const normalizedValues = Array.isArray(values) && values.length > 0
        ? values.map((value) => Number(value)).filter((value) => Number.isInteger(value))
        : [0, 1];

    const sortedValues = [...new Set(normalizedValues)].sort((a, b) => a - b);
    const min = integerSpec?.min ?? (sortedValues[0] ?? 0);
    const max = integerSpec?.max ?? (sortedValues[sortedValues.length - 1] ?? min);
    const step = integerSpec?.step ?? 1;
    const defaultValue = integerSpec?.defaultValue ?? min;

    let widget = getWidget(node, widgetName);
    if (!widget || widget.type !== "number") {
        removeWidgetByName(node, widgetName);
        widget = node.addWidget("number", widgetName, defaultValue, () => {}, {
            min,
            max,
            step,
            precision: 0,
        });
    }

    widget.options = widget.options || {};
    widget.options.min = min;
    widget.options.max = max;
    widget.options.step = step;
    widget.options.precision = 0;

    const currentValue = Number(widget.value);
    if (!Number.isInteger(currentValue) || currentValue < min || currentValue > max) {
        widget.value = defaultValue;
    }

    return markDynamicWidget(node, widgetName);
}

function detectParameterType(values) {
    if (!Array.isArray(values) || values.length === 0) {
        return "combo";
    }

    if (values.every((value) => typeof value === "boolean")) {
        return "boolean";
    }

    if (values.every((value) => Number.isInteger(value))) {
        return "integer";
    }

    return "combo";
}

function getUniformIntegerSpec(values) {
    if (!Array.isArray(values) || values.length === 0) {
        return null;
    }

    const uniqueSorted = [...new Set(values.map((value) => Number(value)))]
        .filter((value) => Number.isInteger(value))
        .sort((a, b) => a - b);

    if (uniqueSorted.length !== values.length || uniqueSorted.length === 0) {
        return null;
    }

    if (uniqueSorted.length === 1) {
        return {
            min: uniqueSorted[0],
            max: uniqueSorted[0],
            step: 1,
            defaultValue: uniqueSorted[0],
        };
    }

    const step = uniqueSorted[1] - uniqueSorted[0];
    if (!Number.isInteger(step) || step <= 0) {
        return null;
    }

    for (let i = 1; i < uniqueSorted.length; i += 1) {
        if ((uniqueSorted[i] - uniqueSorted[i - 1]) !== step) {
            return null;
        }
    }

    return {
        min: uniqueSorted[0],
        max: uniqueSorted[uniqueSorted.length - 1],
        step,
        defaultValue: uniqueSorted[0],
    };
}

function removeWidgetByName(node, widgetName) {
    if (!node.widgets) {
        return;
    }
    const widgetIndex = node.widgets.findIndex((widget) => widget?.name === widgetName);
    if (widgetIndex >= 0) {
        node.widgets.splice(widgetIndex, 1);
    }
}

function readServiceSchema(schemaRegistry, provider, service) {
    if (!schemaRegistry || typeof schemaRegistry !== "object") {
        return null;
    }
    const providerEntry = schemaRegistry?.[provider];
    if (!providerEntry || typeof providerEntry !== "object") {
        return null;
    }
    return providerEntry?.[service] || null;
}

function listProviders(schemaRegistry) {
    if (!schemaRegistry || typeof schemaRegistry !== "object") {
        return [];
    }
    return Object.keys(schemaRegistry);
}

function listServices(schemaRegistry, provider) {
    if (!schemaRegistry || typeof schemaRegistry !== "object") {
        return [];
    }
    const providerEntry = schemaRegistry?.[provider];
    if (!providerEntry || typeof providerEntry !== "object") {
        return [];
    }
    return Object.keys(providerEntry);
}

function listServiceParameterNames(schemaRegistry, provider, service) {
    const schema = readServiceSchema(schemaRegistry, provider, service);
    const possible = schema?.possible_parameters;
    if (!possible || typeof possible !== "object") {
        return [];
    }

    return Object.keys(possible).filter((name) => name !== "prompt");
}

function updateServiceWidget(node, schemaRegistry) {
    const providerWidget = getWidget(node, "api_provider");
    const serviceWidget = getWidget(node, "api_service");
    if (!providerWidget || !serviceWidget) {
        return;
    }

    const providers = listProviders(schemaRegistry);
    providerWidget.options = providerWidget.options || {};
    providerWidget.options.values = providers;
    if (!providers.includes(String(providerWidget.value))) {
        providerWidget.value = providers[0] || providerWidget.value;
    }

    const services = listServices(schemaRegistry, providerWidget.value);
    serviceWidget.options = serviceWidget.options || {};
    serviceWidget.options.values = services;
    if (!services.includes(String(serviceWidget.value))) {
        serviceWidget.value = services[0] || serviceWidget.value;
    }
}

function canonicalParameterKey(name) {
    return String(name ?? "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function getBlockedParameterNames(node) {
    const names = node.__primereOptionalInputNames;
    if (names instanceof Set) {
        return new Set([...names].map((name) => canonicalParameterKey(name)));
    }
    return new Set();
}

function updateParameterWidgets(node, schemaRegistry) {
    const providerWidget = getWidget(node, "api_provider");
    const serviceWidget = getWidget(node, "api_service");
    if (!providerWidget || !serviceWidget) {
        return;
    }

    const provider = String(providerWidget.value);
    const service = String(serviceWidget.value);
    const schema = readServiceSchema(schemaRegistry, provider, service);
    const possible = schema?.possible_parameters;

    const activeParams = new Set(listServiceParameterNames(schemaRegistry, provider, service));
    const blockedParams = getBlockedParameterNames(node);

    for (const widget of [...(node.widgets || [])]) {
        if (!widget || !widget.name || widget.__primereUniApiDynamic !== true) {
            continue;
        }
        if (!activeParams.has(widget.name) || blockedParams.has(canonicalParameterKey(widget.name))) {
            removeWidgetByName(node, widget.name);
        }
    }

    if (possible && typeof possible === "object") {
        for (const [paramName, values] of Object.entries(possible)) {
            if (paramName === "prompt") {
                continue;
            }
            if (blockedParams.has(canonicalParameterKey(paramName))) {
                removeWidgetByName(node, paramName);
                continue;
            }
            const paramType = detectParameterType(values);
            if (paramType === "boolean") {
                ensureBooleanWidget(node, paramName, values);
                continue;
            }
            /* if (paramType === "integer") {
                const integerSpec = getUniformIntegerSpec(values);
                if (integerSpec) {
                    ensureIntegerWidget(node, paramName, values, integerSpec);
                    continue;
                }
            } */
            ensureComboWidget(node, paramName, values);
        }
    }

    node.setDirtyCanvas?.(true, true);
    node.computeSize?.();
}

async function initializeUniApiNode(node) {
    if (!(node.__primereOptionalInputNames instanceof Set)) {
        node.__primereOptionalInputNames = new Set();
    }

    const schemaRegistry = await loadSchemas();
    updateServiceWidget(node, schemaRegistry);
    updateParameterWidgets(node, schemaRegistry);

    if (node.__primereUniApiWidgetHooked) {
        return;
    }
    node.__primereUniApiWidgetHooked = true;

    const originalOnWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (name, value, oldValue, widget) {
        originalOnWidgetChanged?.call(this, name, value, oldValue, widget);

        if (name !== "api_provider" && name !== "api_service") {
            return;
        }

        if (name === "api_provider") {
            updateServiceWidget(this, schemaRegistry);
        }
        updateParameterWidgets(this, schemaRegistry);
    };
}

app.registerExtension({
    name: "Primere.UniApiDynamicWidgets",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== TARGET_NODE_NAME) {
            return;
        }
        const optionalInputNames = new Set(Object.keys(nodeData?.input?.optional || {}));
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, []);
            this.__primereOptionalInputNames = new Set(optionalInputNames);
            initializeUniApiNode(this);
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            onConfigure?.apply(this, [config]);
            this.__primereOptionalInputNames = new Set(optionalInputNames);
            initializeUniApiNode(this);
        };
    },
});
