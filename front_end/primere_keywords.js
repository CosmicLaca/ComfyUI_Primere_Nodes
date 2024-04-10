import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let LoadedNodeKey = null;

app.registerExtension({
    name: "Primere.PrimereKeywords",

    async init(app) {

        const lcg = LGraphCanvas.prototype.processNodeWidgets;
        LGraphCanvas.prototype.processNodeWidgets = function(node, pos, event, active_widget) {
            if (event.type != LiteGraph.pointerevents_method + "up") {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (node.type != 'PrimereCKPT' && node.type != 'PrimereVisualCKPT') {
                return lcg.call(this, node, pos, event, active_widget);
            }

            var x = pos[0] - node.pos[0];
            var y = pos[1] - node.pos[1];
            var width = node.size[0];
            var that = this;
            var ref_window = this.getCanvasWindow();

            for (var i = 0; i < node.widgets.length; ++i) {
                var w = node.widgets[i];
                if (!w || w.disabled)
                    continue;

                var widget_height = w.computeSize ? w.computeSize(width)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                var widget_width = w.width || width;

                if (w.name != 'base_model') {
                    return lcg.call(this, node, pos, event, active_widget);
                }

                if (w != active_widget && (x < 6 || x > widget_width - 12 || y < w.last_y || y > w.last_y + widget_height || w.last_y === undefined))
                    continue

                if (w == active_widget || (x > 6 && x < widget_width - 12 && y > w.last_y && y < w.last_y + widget_height)) {
                    var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                    if (delta)
                        continue;

                    var values = w.options.values;
                    if (values && values.constructor === Function) {
                        values = w.options.values(w, node);
                    }

                    if (typeof values != 'undefined') {
                        function inner_clicked(v, option, event) {
                            sendPOSTModelName(v)
                            this.value = v;
                            that.dirty_canvas = true;

                            return false;
                        }

                        new LiteGraph.ContextMenu(values, {
                            scale: Math.max(1, this.ds.scale),
                            event: event,
                            className: "dark",
                            callback: inner_clicked.bind(w),
                            node: node,
                            widget: w,
                        }, ref_window);
                    }
                }
            }
            return lcg.call(this, node, pos, event, active_widget);
        }
    },

    /* async setup(app) {

    }, */

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimereCKPT" || nodeData.name === 'PrimereVisualCKPT') {
            nodeType.prototype.onNodeCreated = function () {
                PrimereModelChange.apply(this, [this, 'PrimereKeywordHandler']);
            };
        }

        if (nodeData.name === "PrimereModelKeyword") {
            nodeType.prototype.onNodeCreated = function () {
                PrimereKeywordList.apply(this, [this, 'PrimereKeywordSelector']);
            };
        }
    },
});


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
    node.name = inputName;

    node.onWidgetChanged = function(name, value, old_value){
        if (name == 'base_model') {
            sendPOSTModelName(value)
        }
    };
}

api.addEventListener("ModelKeywordResponse", ModelKeywordResponse);
function ModelKeywordResponse(event) {
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

function sendPOSTModelName(modelName) {
    const body = new FormData();
    body.append('modelName', modelName);
    api.fetchApi("/primere_keyword_parser", {method: "POST", body,});
}