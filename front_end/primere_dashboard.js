import { app } from "/scripts/app.js";

const GROUP_SIZE  = 3;
const PREFIX_POS  = "prompt_pos_";
const PREFIX_NEG  = "prompt_neg_";
const PREFIX_PRE  = "preferred_";

function propagateType(node, type) {
    node.outputs[0].type = type;
    node.outputs[1].type = type;
    node.outputs[2].type = type;
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name.startsWith(PREFIX_POS)) {
            node.inputs[i].type     = type;
            node.inputs[i + 1].type = type;
            node.inputs[i + 2].type = type;
        }
    }
}

function renumberInputs(node) {
    let slot = 1;
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name.startsWith(PREFIX_POS)) {
            node.inputs[i].name     = `${PREFIX_POS}${slot}`;
            node.inputs[i + 1].name = `${PREFIX_NEG}${slot}`;
            node.inputs[i + 2].name = `${PREFIX_PRE}${slot}`;
            slot++;
        }
    }
    return slot;
}

function cleanupEmptyGroups(node) {
    const groupStarts = [];
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name.startsWith(PREFIX_POS)) {
            groupStarts.push(i);
        }
    }
    let remaining = groupStarts.length;
    for (let g = groupStarts.length - 1; g >= 0; g--) {
        if (remaining <= 1) break;
        const i = groupStarts[g];
        if (node.inputs[i].link == null && node.inputs[i + 1].link == null) {
            node.removeInput(i + 2);
            node.removeInput(i + 1);
            node.removeInput(i);
            remaining--;
        }
    }
}

function updateSelectWidget(node) {
    if (!node.widgets) return;
    const widget     = node.widgets[0];
    const groupCount = node.inputs.filter(inp => inp.name.startsWith(PREFIX_POS)).length;
    const lastPos    = node.inputs[node.inputs.length - GROUP_SIZE];
    const lastNeg    = node.inputs[node.inputs.length - GROUP_SIZE + 1];
    const emptyLast  = (lastPos.link == null && lastNeg.link == null) ? -1 : 0;
    const max        = Math.max(1, groupCount + emptyLast);
    widget.options.max = max;
    widget.value = Math.max(1, Math.min(widget.value, max));
}

app.registerExtension({
    name: "Primere.Promptswitch",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== 'PrimerePromptSwitch') return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            origOnNodeCreated?.apply(this, arguments);
            if (!this.inputs.some(inp => inp.name.startsWith(PREFIX_POS))) {
                this.addInput(`${PREFIX_POS}1`, '*');
                this.addInput(`${PREFIX_NEG}1`, '*');
                this.addInput(`${PREFIX_PRE}1`, '*');
            }
        };

        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (!link_info) return;
            if (this._cleaningGroups) return;

            if (type === 2) {
                if (connected && index === 0 && this.outputs[0].type === '*') {
                    const originType = link_info.type;
                    if (originType !== '*') {
                        propagateType(this, originType);
                    }
                }
                return;
            }

            if (this.inputs[index].name === 'select') return;

            const firstPos = this.inputs.find(inp => inp.name.startsWith(PREFIX_POS));
            if (firstPos && firstPos.type === '*') {
                const originNode = app.graph.getNodeById(link_info.origin_id);
                const originType = originNode.outputs[link_info.origin_slot].type;
                if (originType !== '*') {
                    propagateType(this, originType);
                }
            }

            if (!connected) {
                const stackTrace   = new Error().stack;
                const isUserAction = !stackTrace.includes('LGraphNode.prototype.connect') &&
                                     !stackTrace.includes('LGraphNode.connect') &&
                                     !stackTrace.includes('loadGraphData');
                if (isUserAction) {
                    this._cleaningGroups = true;
                    cleanupEmptyGroups(this);
                    this._cleaningGroups = false;
                }
            }

            const nextSlot = renumberInputs(this);
            const lastPos  = this.inputs[this.inputs.length - GROUP_SIZE];
            const lastNeg  = this.inputs[this.inputs.length - GROUP_SIZE + 1];
            if (lastPos.name.startsWith(PREFIX_POS) && lastPos.link != null && lastNeg.link != null) {
                this.addInput(`${PREFIX_POS}${nextSlot}`, this.outputs[0].type);
                this.addInput(`${PREFIX_NEG}${nextSlot}`, this.outputs[1].type);
                this.addInput(`${PREFIX_PRE}${nextSlot}`, this.outputs[0].type);
            }

            updateSelectWidget(this);
        };
    },

});
