import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { showToast } from "./frontend_helper.js";

const NODE_NAME = "PrimerePathSelector";
const PATH_ENDPOINT = "/primere_select_path";
const DEFAULT_DISPLAY = "No path selected";

function getNodeIdentifier(node) {
    if (!node || typeof node.id === "undefined" || node.id === null) {
        return "";
    }
    return String(node.id);
}

async function clearNodePath(node) {
    const nodeId = getNodeIdentifier(node);
    if (!nodeId) {
        return;
    }

    const body = new FormData();
    body.append("node_id", nodeId);
    body.append("clear", "1");

    try {
        await api.fetchApi(PATH_ENDPOINT, { method: "POST", body });
    } catch (error) {
        showToast("error", error?.message ?? "Failed to clear selected path");
    }

    if (node._primerePathDisplayWidget) {
        node._primerePathDisplayWidget.value = DEFAULT_DISPLAY;
    }
    app.graph?.setDirtyCanvas(true, true);
}

async function openPathDialog(node) {
    const nodeId = getNodeIdentifier(node);
    if (!nodeId) {
        showToast("error", "Node identifier is missing");
        return;
    }

    const selectWidget = node.widgets?.find((w) => w.name === "select_file");
    const selectFile = Boolean(selectWidget?.value ?? true);

    const body = new FormData();
    body.append("node_id", nodeId);
    body.append("select_file", selectFile ? "1" : "0");

    let response;
    try {
        response = await api.fetchApi(PATH_ENDPOINT, { method: "POST", body });
    } catch (error) {
        showToast("error", error?.message ?? "Failed to open path dialog");
        return;
    }

    let payload;
    try {
        payload = await response.json();
    } catch (error) {
        showToast("error", "Invalid response from path dialog");
        return;
    }

    if (!response.ok || payload?.error) {
        showToast("error", payload?.error ?? "Failed to open path dialog");
        return;
    }

    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Primere.PrimerePathSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const fileModeWidget = this.widgets?.find((w) => w.name === "select_file");
            if (fileModeWidget) {
                const originalCallback = fileModeWidget.callback;
                fileModeWidget.callback = (...args) => {
                    originalCallback?.apply(fileModeWidget, args);
                    void clearNodePath(this);
                };
            }

            this.addWidget("button", "Select path", null, () => openPathDialog(this), { serialize: false });

            this._primerePathDisplayWidget = ComfyWidgets["STRING"](
                this,
                "selected_path",
                ["STRING", { multiline: true }],
                app
            ).widget;
            this._primerePathDisplayWidget.value = DEFAULT_DISPLAY;
            this._primerePathDisplayWidget.inputEl.readOnly = true;
            this._primerePathDisplayWidget.serializeValue = async (node, index) => {
                node.widgets_values[index] = "";
                return "";
            };
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, [message]);
            if (!this._primerePathDisplayWidget) return;
            const display = message?.path_display?.[0] ?? message?.text?.[0] ?? DEFAULT_DISPLAY;
            this._primerePathDisplayWidget.value = display;
            app.canvas.setDirty(true);
        };
    },
});
