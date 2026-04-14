import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "PrimereCustomStyles";
const SOURCE_WIDGET = "style_source";

function removeDynamicStyleWidgets(node) {
    if (!Array.isArray(node.widgets) || !Array.isArray(node.__unistyle_widget_names)) return;

    const dynamicNames = new Set(node.__unistyle_widget_names);
    node.widgets = node.widgets.filter((widget) => !dynamicNames.has(widget.name));
    node.__unistyle_widget_names = [];
}

function addDynamicStyleWidgets(node, styles) {
    node.__unistyle_widget_names = [];

    for (const style of styles) {
        const values = Array.isArray(style.values) && style.values.length > 0 ? style.values : ["None"];
        const key = style.key;
        const strengthConfig = style.strength || {};

        node.addWidget("combo", key, values[0], () => {}, {
            values,
            serialize: true,
        });
        node.__unistyle_widget_names.push(key);

        node.addWidget("number", `${key}_strength`, Number(strengthConfig.default ?? 1), () => {}, {
            min: Number(strengthConfig.min ?? 0),
            max: Number(strengthConfig.max ?? 10),
            step: Number(strengthConfig.step ?? 0.01),
            precision: 2,
            serialize: true,
        });
        node.__unistyle_widget_names.push(`${key}_strength`);
    }

    node.setSize([Math.max(node.size[0], 340), Math.max(node.computeSize()[1], node.size[1])]);
    app.graph.setDirtyCanvas(true, true);
}

async function loadStyleData(styleFile) {
    const body = new FormData();
    if (styleFile) body.append("style_file", styleFile);

    const response = await api.fetchApi("/primere_unistyle_data", { method: "POST", body });
    if (!response.ok) {
        throw new Error(`Unistyle endpoint failed: ${response.status}`);
    }

    return response.json();
}

async function rebuildUnistyleWidgets(node, selectedFile) {
    const sourceWidget = node.widgets?.find((w) => w.name === SOURCE_WIDGET);
    if (!sourceWidget) return;

    const data = await loadStyleData(selectedFile || sourceWidget.value);
    if (!data || !Array.isArray(data.files)) return;

    sourceWidget.options.values = data.files;
    sourceWidget.value = data.selected_file || data.files[0] || sourceWidget.value;

    removeDynamicStyleWidgets(node);
    addDynamicStyleWidgets(node, Array.isArray(data.styles) ? data.styles : []);
}

app.registerExtension({
    name: "Primere.PrimereUniStyle",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const sourceWidget = this.widgets?.find((w) => w.name === SOURCE_WIDGET);
            if (!sourceWidget) return;

            this.__unistyle_widget_names = [];
            const originalSourceCallback = sourceWidget.callback;

            sourceWidget.callback = async (...args) => {
                const value = args[0];
                originalSourceCallback?.apply(sourceWidget, args);
                try {
                    await rebuildUnistyleWidgets(this, value);
                } catch (error) {
                    console.error("PrimereCustomStyles source switch failed", error);
                }
            };

            rebuildUnistyleWidgets(this, sourceWidget.value).catch((error) => {
                console.error("PrimereCustomStyles widget initialization failed", error);
            });
        };
    },
});
