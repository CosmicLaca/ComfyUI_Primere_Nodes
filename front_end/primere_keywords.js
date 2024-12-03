import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let LoadedNodeKey = null;
let CKPTLoaderName = null;

app.registerExtension({
    name: "Primere.PrimereKeywords",

    /* async init(app) {

    }, */

    async setup(app) {
        await sleep(100);
        for (var its_1 = 0; its_1 < app.canvas.visible_nodes.length; ++its_1) {
            var wts_1 = app.canvas.visible_nodes[its_1];
            if (wts_1.type == CKPTLoaderName) {
                for (var its_2 = 0; its_2 < wts_1.widgets.length; ++its_2) {
                    var wts_2 = wts_1.widgets[its_2];
                    if (wts_2.name == 'base_model') {
                        var modelvalue = wts_2.value;
                        if (typeof modelvalue != 'undefined') {
                            sendPOSTModelName(wts_2.value);
                        }
                    }
                }
            }
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimereCKPT" || nodeData.name === 'PrimereVisualCKPT') {
            nodeType.prototype.onNodeCreated = function () {
                CKPTLoaderName = nodeData.name;
                PrimereModelChange.apply(this, [this, CKPTLoaderName]);
            };
        }

        if (nodeData.name === "PrimereModelKeyword") {
            nodeType.prototype.onNodeCreated = function () {
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
function PrimereModelChange(node, inputName) {
    node.onWidgetChanged = function(name, value, old_value){
        if (name == 'base_model') {
            sendPOSTModelName(value);
        }
    };
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