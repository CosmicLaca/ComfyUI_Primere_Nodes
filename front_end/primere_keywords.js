import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let LoadedNodeKey = null;
const CKPTLoaderTypes = new Set(["PrimereCKPT", "PrimereVisualCKPT"]);

app.registerExtension({
    name: "Primere.PrimereKeywords",

    /* async init(app) {

    }, */

    async setup(app) {
        // Front-end/node widget values may still be hydrating during early setup.
        // Poll briefly to avoid pushing invalid values (e.g. NaN) on first load.
        for (let attempt = 0; attempt < 8; attempt++) {
            let sentAny = false;
            for (const node of app.canvas.visible_nodes) {
                if (!CKPTLoaderTypes.has(node.type)) {
                    continue;
                }
                const baseModelWidget = node.widgets?.find((widget) => widget?.name === 'base_model');
                if (!baseModelWidget) {
                    continue;
                }
                const modelValue = baseModelWidget.value;
                if (isValidModelName(modelValue)) {
                    sendPOSTModelName(modelValue);
                    sentAny = true;
                }
            }

            if (sentAny) {
                break;
            }
            await sleep(100);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (CKPTLoaderTypes.has(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                PrimereModelChange.apply(this, [this]);
            };
        }

        if (nodeData.name === "PrimereModelKeyword") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                PrimereKeywordList.apply(this, [this, 'PrimereKeywordSelector']);
            };
        }
    },
});

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function PrimereKeywordList(node, inputName) {
    node.name = inputName;

    const widget = {
        type: "primere_keyword_lister",
        name: `w${inputName}`,
        callback: () => {},
    };

    node.addWidget("combo", "select_keyword", 'Select in order', () => {
    },{
        values: ["None", "Select in order", "Random select"],
    });

    LoadedNodeKey = node;
    return {widget: widget};
}
function PrimereModelChange(node) {
    const previousWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function(name, value, old_value){
        if (typeof previousWidgetChanged === 'function') {
            previousWidgetChanged.apply(this, [name, value, old_value]);
        }
        if (name == 'base_model') {
            if (isValidModelName(value)) {
                sendPOSTModelName(value);
            }
        }
    };
}

function isValidModelName(value) {
    if (typeof value !== 'string') {
        return false;
    }
    const trimmed = value.trim();
    if (!trimmed || trimmed.toLowerCase() === 'nan') {
        return false;
    }
    return true;
}

api.addEventListener("ModelKeywordResponse", ModelKeywordResponse);
function ModelKeywordResponse(event) {
    if (LoadedNodeKey != null) {
        var ResponseText = event.detail;
        for (var iln = 0; iln < LoadedNodeKey.widgets.length; ++iln) {
            var wln = LoadedNodeKey.widgets[iln];
            if (!wln || wln.disabled)
                continue;

            if (wln.name == 'select_keyword') {
                wln.options.values = ResponseText;
                if (typeof ResponseText[3] === 'undefined') {
                    if (typeof ResponseText[1] === 'undefined') {
                        wln.value = ResponseText[0];
                    } else {
                        wln.value = ResponseText[1];
                    }
                } else {
                    wln.value = ResponseText[3];
                }
            }
        }
    }
}

function sendPOSTModelName(modelName) {
    const body = new FormData();
    body.append('modelName', modelName);
    api.fetchApi("/primere_keyword_parser", {method: "POST", body,});
}
