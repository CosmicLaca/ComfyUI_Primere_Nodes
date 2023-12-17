import { ComfyApp, app } from "/scripts/app.js";

app.registerExtension({
	name: "Primere.Promptswitch",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name == 'PrimerePromptSwitch') {
            var input_name_pos = "prompt_pos_";
            var input_name_neg = "prompt_neg_";
            var input_name_sub = "subpath_";
            var input_name_mod = "model_";
            var input_name_ori = "orientation_";
            var n = 0;

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if(!link_info)
                    return;

                if (type == 2) {
                    if (connected && index == 0){
                        if (this.outputs[0].type == '*'){
                            if (link_info.type == '*') {
                                app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
                            } else {
                                this.outputs[0].type = link_info.type;
                                this.outputs[1].type = origin_type;
                                this.outputs[2].type = origin_type;
                                this.outputs[3].type = origin_type;
                                this.outputs[4].type = origin_type;

                                for (let i in this.inputs) {
                                    n = parseInt(i);
                                    if (this.inputs[i].name.includes(input_name_pos) === true) {
                                        let input_i_pos = this.inputs[n];
                                        let input_i_neg = this.inputs[(n + 1)];
                                        let input_i_sub = this.inputs[(n + 2)];
                                        let input_i_mod = this.inputs[(n + 3)];
                                        let input_i_ori = this.inputs[(n + 4)];
                                        input_i_pos.type = link_info.type;
                                        input_i_neg.type = link_info.type;
                                        input_i_sub.type = link_info.type;
                                        input_i_mod.type = link_info.type;
                                        input_i_ori.type = link_info.type;
                                    }
                                }
                            }
                        }
                    }
                    return;
                } else {
                    //if (this.inputs[index].name.includes(input_name_neg) === true)
                        //return;

                    if (this.inputs[index].name == 'select')
                        return;

                    if (this.inputs[0].type == '*') {
                        const node = app.graph.getNodeById(link_info.origin_id);
                        let origin_type = node.outputs[link_info.origin_slot].type;

                        if (origin_type == '*') {
                            this.disconnectInput(link_info.target_slot);
                            return;
                        }

                        for (let i in this.inputs) {
                            n = parseInt(i);
                            if (this.inputs[i].name.includes(input_name_pos) === true) {
                                let input_i_pos = this.inputs[n];
                                let input_i_neg = this.inputs[(n + 1)];
                                let input_i_sub = this.inputs[(n + 2)];
                                let input_i_mod = this.inputs[(n + 3)];
                                let input_i_ori = this.inputs[(n + 4)];
                                input_i_pos.type = origin_type;
                                input_i_neg.type = origin_type;
                                input_i_sub.type = origin_type;
                                input_i_mod.type = origin_type;
                                input_i_ori.type = origin_type;
                            }
                        }

                        this.outputs[0].type = origin_type;
                        this.outputs[1].type = origin_type;
                        this.outputs[2].type = origin_type;
                        this.outputs[3].type = origin_type;
                        this.outputs[4].type = origin_type;
                    }
                }

                let select_slot = this.inputs.find(x => x.name == "select");
                let converted_count = 0;
                converted_count += select_slot ? 1 : 0;

                if (!connected && (this.inputs.length > 5 + converted_count)) {
                    const stackTrace = new Error().stack;

                    if (!stackTrace.includes('LGraphNode.prototype.connect') && // for touch device
                        !stackTrace.includes('LGraphNode.connect') && // for mouse device
                        !stackTrace.includes('loadGraphData') &&
                        this.inputs[index].name != 'select') {

                        let last_pos_slot = this.inputs[this.inputs.length - 2];
                        let last_neg_slot = this.inputs[this.inputs.length - 1];
                        if (last_pos_slot.link == undefined && last_neg_slot.link == undefined) {
                            this.removeInput(this.inputs.length - 1);
                            this.removeInput(this.inputs.length - 1);
                            this.removeInput(this.inputs.length - 1);
                            this.removeInput(this.inputs.length - 1);
                            this.removeInput(this.inputs.length - 1);
                        }
                    }
                }

				let slot_i = 1;
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].name.includes(input_name_pos) === true) {
                        let input_i_pos = this.inputs[i];
                        let input_i_neg = this.inputs[(i + 1)];
                        let input_i_sub = this.inputs[(i + 2)];
                        let input_i_mod = this.inputs[(i + 3)];
                        let input_i_ori = this.inputs[(i + 4)];
	                    input_i_pos.name = `${input_name_pos}${slot_i}`
                        input_i_neg.name = `${input_name_neg}${slot_i}`
                        input_i_sub.name = `${input_name_sub}${slot_i}`;
                        input_i_mod.name = `${input_name_mod}${slot_i}`;
                        input_i_ori.name = `${input_name_ori}${slot_i}`;
                        slot_i++;
                    }
                }

				let last_pos_slot = this.inputs[this.inputs.length - 5];
                let last_neg_slot = this.inputs[this.inputs.length - 4];
                if (last_pos_slot.name.includes(input_name_pos) === true && last_neg_slot.link != undefined && last_pos_slot.link != undefined) {
                        this.addInput(`${input_name_pos}${slot_i}`, this.outputs[0].type);
                        this.addInput(`${input_name_neg}${slot_i}`, this.outputs[1].type);
                        this.addInput(`${input_name_sub}${slot_i}`, this.outputs[2].type);
                        this.addInput(`${input_name_mod}${slot_i}`, this.outputs[3].type);
                        this.addInput(`${input_name_ori}${slot_i}`, this.outputs[4].type);
                }

                if (this.widgets) {
                    let last_pos_slot = this.inputs[this.inputs.length - 5];
                    let last_neg_slot = this.inputs[this.inputs.length - 4];

                    var additionalMax = 0;
                    if (last_pos_slot.link == undefined && last_neg_slot.link == undefined) {
                        additionalMax = -1;
                    }
                    var selectMax = Math.round((this.inputs.length / 5) + additionalMax);
                    this.widgets[0].value = Math.min(this.widgets[0].value, this.widgets[0].options.max);
                    this.widgets[0].options.max = selectMax;
                    if (this.widgets[0].options.max > 0 && this.widgets[0].value <= 0)
                        this.widgets[0].value = 1;
                    if (this.widgets[0].value > selectMax)
                        this.widgets[0].value = selectMax;
                }
            }
        }
	},
});
