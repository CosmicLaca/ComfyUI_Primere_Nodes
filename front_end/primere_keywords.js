import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let loadedKeywordNode = null;

function getActiveCKPTSource() {
    const nodes = app.graph._nodes;
    if (nodes.some(n => n.type === 'PrimereVisualCKPT')) return 'PrimereVisualCKPT';
    if (nodes.some(n => n.type === 'PrimereCKPT'))       return 'PrimereCKPT';
    return null;
}

app.registerExtension({
    name: "Primere.PrimereKeywords",

    async setup() {
        const sourceType = getActiveCKPTSource();
        if (!sourceType) return;

        for (const node of app.graph._nodes) {
            if (node.type !== sourceType) continue;
            const widget = node.widgets?.find(w => w.name === 'base_model');
            if (widget?.value !== undefined) {
                sendPOSTModelName(widget.value);
            }
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "PrimereCKPT" || nodeData.name === "PrimereVisualCKPT") {
            const thisNodeType = nodeData.name;
            nodeType.prototype.onNodeCreated = function () {
                this.onWidgetChanged = function (name, value) {
                    if (name !== 'base_model') return;
                    if (getActiveCKPTSource() !== thisNodeType) return;
                    sendPOSTModelName(value);
                };
            };
        }

        if (nodeData.name === "PrimereModelKeyword") {
            nodeType.prototype.onNodeCreated = function () {
                this.name = 'PrimereKeywordSelector';
                loadedKeywordNode = this;
            };
        }
    },
});

api.addEventListener("ModelKeywordResponse", onModelKeywordResponse);

function onModelKeywordResponse(event) {
    if (!loadedKeywordNode) return;

    const responseValues = event.detail;
    const widget = loadedKeywordNode.widgets.find(w => !w.disabled && w.name === 'select_keyword');
    if (!widget) return;

    widget.options.values = responseValues;
    if (responseValues[3] !== undefined) {
        widget.value = responseValues[3];
    } else if (responseValues[1] !== undefined) {
        widget.value = responseValues[1];
    } else {
        widget.value = responseValues[0];
    }
}

function sendPOSTModelName(modelName) {
    const body = new FormData();
    body.append('modelName', modelName);
    api.fetchApi("/primere_keyword_parser", { method: "POST", body });
}
