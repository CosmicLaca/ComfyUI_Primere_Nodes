import { app } from "/scripts/app.js";

// Adds an upload button to the nodes
app.registerExtension({
	name: "Primere.PrimereMetaRead",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "PrimereMetaRead") {
			nodeData.input.required.upload = ["IMAGEUPLOAD"];
		}
	},
});