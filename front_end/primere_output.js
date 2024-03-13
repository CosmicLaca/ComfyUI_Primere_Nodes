import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
let hasShownAlertForUpdatingInt = false;

let ImagePath = null;
let WorkflowData = {};
const realPath = "extensions/Primere";

let PreviewTarget = 'Checkpoint';
let Convert = true;
let IMGType = true;
let MaxSide = -1;
let buttontitle = 'Image not available for save. Please load one.'
let SaveIsValid = false;
let TargetFileName = null;
let LoadedNode = null;

const NodenameByType = {
    'Checkpoint': 'PrimereVisualCKPT',
    'CSV Prompt': 'PrimereVisualStyle',
    'Lora': 'PrimereVisualLORA',
    'Lycoris': 'PrimereVisualLYCORIS',
    'Hypernetwork': 'PrimereVisualHypernetwork',
    'Embedding': 'PrimereVisualEmbedding'
}

app.registerExtension({
    name: "Primere.PrimereOutputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimereAnyOutput" || nodeData.name === "PrimereTextOutput") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.showValueWidget = ComfyWidgets["STRING"](this, "output", ["STRING", { multiline: true }], app).widget;
                this.showValueWidget.inputEl.readOnly = true;
                this.showValueWidget.serializeValue = async (node, index) => {
                    node.widgets_values[index] = "";
                    return "";
                };
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message]);
                this.showValueWidget.value = message.text[0];
            };
        }
    },
});

app.registerExtension({
    name: "Primere.PrimerePreviewImage",
    async init(app) {

        function PreviewHandler() {
            let head = document.getElementsByTagName('HEAD')[0];
            let js1 = document.createElement("script");
            js1.src = realPath + "/vendor/LoadImage/load-image.js";
            head.appendChild(js1);
            let js2 = document.createElement("script");
            js2.src = realPath + "/vendor/LoadImage/load-image-scale.js";
            head.appendChild(js2);
        }

        PreviewHandler();

        const lcg = LGraphCanvas.prototype.processNodeWidgets;
        LGraphCanvas.prototype.processNodeWidgets = function(node, pos, event, active_widget) {
            if (event.type != LiteGraph.pointerevents_method + "up") {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (node.type != 'PrimerePreviewImage') {
                return lcg.call(this, node, pos, event, active_widget);
            }

            var x = pos[0] - node.pos[0];
            var y = pos[1] - node.pos[1];
            var width = node.size[0];

            buttontitle = ButtonLabelCreator(node);

            for (var i = 0; i < node.widgets.length; ++i) {
                var w = node.widgets[i];
                if (!w || w.disabled)
                    continue;

                var widget_height = w.computeSize ? w.computeSize(width)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                var widget_width = w.width || width;
                var widget_name = node.widgets[i].name;

                console.log('++++++ --- ++++++++');
                console.log(widget_name);
                console.log(w.type)
                console.log(w)
                console.log('++++++ --- ++++++++');

                if (w.name == 'preview_target') {
                    //node.widgets[i].value = 'target change';
                }
                if (w.name == 'image_resize') {
                    //node.widgets[i].value = 'convert change';
                }
                if (w.type == 'button') {
                    w.name = buttontitle
                }

                if (w != active_widget && (x < 6 || x > widget_width - 12 || y < w.last_y || y > w.last_y + widget_height || w.last_y === undefined))
                    continue

                if (w == active_widget || (x > 6 && x < widget_width - 12 && y > w.last_y && y < w.last_y + widget_height)) {
                    var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                    if (delta)
                        continue;
                }
            }
            return lcg.call(this, node, pos, event, active_widget);
        }
    },

    async setup(app) {

    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimerePreviewImage") {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                ImagePath = nodeData.input.hidden['image_path'][0]
            }

            if (nodeData.input.hasOwnProperty('optional') === true) {

            }

            nodeType.prototype.onNodeCreated = function () {
                PrimerePreviewSaverWidget.apply(this, [this, 'PrimerePreviewSaver']);
            };
        }
    },
});

api.addEventListener("getVisualTargets", VisualDataReceiver);
function VisualDataReceiver(event) {
    WorkflowData = event.detail
    if (Object.keys(WorkflowData).length >= 1) {
        SaveIsValid = true;
    } else {
        SaveIsValid = false;
    }

    buttontitle = ButtonLabelCreator(LoadedNode);
    for (var iln = 0; iln < LoadedNode.widgets.length; ++iln) {
        var wln = LoadedNode.widgets[iln];
        if (!wln || wln.disabled)
            continue;
        if (wln.type == 'button') {
            wln.name = buttontitle
        }
    }
    console.log('----------- ÚJ KÉP ----------------------')
    console.log(WorkflowData)
    console.log(buttontitle)
    console.log('----------- ÚJ KÉP ----------------------')
}

function dataURLtoFile(dataurl, filename) {
    var arr = dataurl.split(','),
        mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[arr.length - 1]),
        n = bstr.length,
        u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
}

function downloadImage(url, extension, PreviewTarget) {
    fetch(url, {
        mode : 'no-cors',
    })
        .then(response => response.blob())
        .then(blob => {
        let blobUrl = window.URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.download = PreviewTarget + '.' + extension;
        a.href = blobUrl;
        document.body.appendChild(a);
        a.click();
        a.remove();
    })
}

class PreviewSaver {
    constructor(node) {
        console.log('-------- start -------------');
        // console.log(node['imgs'][0]['src']);
        // console.log(node['images'][0]['filename']);
        //console.log(ImagePath)
        console.log(PreviewTarget);
        console.log(Convert);
        console.log(IMGType);
        console.log(MaxSide);
        console.log(WorkflowData);
        console.log('-------- end -------------');

        var maxWidth = null;
        var maxHeight = null;

        if (Convert === true) {
            maxWidth = 350;
            maxHeight = 250;
        }

        var imgMime = "image/png";
        var extension = 'png';
        if (IMGType === true || Convert === true) {
            imgMime = "image/jpeg";
            extension = 'jpg';
        }

        if (MaxSide >= 64 && Convert === false) {
            maxWidth = MaxSide;
            maxHeight = MaxSide;
        }

        var SizeStringFN = '';
        if (MaxSide >= 64) {
            SizeStringFN = MaxSide + 'px_'
        }

        var ImageSource = node['imgs'][0]['src'];
        var ImageName = node['images'][0]['filename'];
        var SaveImageName = 'PreviewImage_' + SizeStringFN + (Math.random() + 1).toString(36).substring(5);

        if (TargetFileName !== null) {
            SaveImageName = TargetFileName;
        }

        fetch(ImageSource)
        .then((res) => res.blob())
        .then((blob) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                var file = dataURLtoFile(reader.result, ImageName);
                loadImage(file, function(img) {
                    var resampledOriginalImage = img.toDataURL(imgMime, 0.6);
                    if (Convert === false) {
                        downloadImage(resampledOriginalImage, extension, SaveImageName);
                    } else {
                        var resampledWidth = img.width;
                        var resampledHeight = img.height;
                        sendPOSTmessage(JSON.stringify({"PreviewTarget": PreviewTarget, "extension": extension, "ImageName": ImageName, "ImagePath": ImagePath, "SaveImageName": SaveImageName, "maxWidth": resampledWidth, "maxHeight": resampledHeight}));
                    }
                },
                {
                    maxWidth: maxWidth,
                    maxHeight: maxHeight,
                    canvas: true,
                    pixelRatio: 1,
                    downsamplingRatio: 0.6,
                    orientation: true,
                    imageSmoothingEnabled: 1
                });
            };
            reader.readAsDataURL(blob);
        });
    }
}

function ButtonLabelCreator(node) {
    for (var px = 0; px < node.widgets.length; ++px) {
       if (node.widgets[px].name == 'preview_target') {
           PreviewTarget = node.widgets[px].value;
       }
       if (node.widgets[px].name == 'image_resize') {
           Convert = node.widgets[px].value;
       }
       if (node.widgets[px].name == 'image_type') {
           IMGType = node.widgets[px].value;
       }
       if (node.widgets[px].name == 'largest_side') {
           MaxSide = node.widgets[px].value;
       }
    }

    var INIT_IMGTYPE_STRING = "";
    var INIT_IMGSIZE_STRING = "";
    if (IMGType === true) {
        INIT_IMGTYPE_STRING = ' JPG format'
    } else {
        INIT_IMGTYPE_STRING = ' PNG format'
    }
    if (MaxSide < 64) {
        INIT_IMGSIZE_STRING = " at original size";
    } else {
        INIT_IMGSIZE_STRING = " resized to " + MaxSide + 'px';
    }

    TargetFileName = null;
    SaveIsValid = false;
    if (Object.keys(WorkflowData).length < 1) {
        buttontitle = 'Image not available for save. Please load one.';
    } else {
        if (Convert === true) {
            if (WorkflowData[NodenameByType[PreviewTarget]].length < 1) {
                buttontitle = 'No resource selected for preview target: ' + PreviewTarget;
            } else {
                SaveIsValid = true;
                TargetFileName = WorkflowData[NodenameByType[PreviewTarget]][0];
                buttontitle = 'Save preview as:  [' + TargetFileName + '.jpg] to ' + PreviewTarget + ' folder.';
            }
        } else {
            SaveIsValid = true;
            buttontitle = 'Save image as' + INIT_IMGTYPE_STRING + INIT_IMGSIZE_STRING;
        }
    }

    return buttontitle;
}

function PrimerePreviewSaverWidget(node, inputName) {
    node.name = inputName;
    const widget = {
        type: "preview_saver_widget",
        name: `w${inputName}`,
        callback: () => {},
    };

    node.addWidget("button", buttontitle, null, () => {
        if (SaveIsValid === true) {
            node.PreviewSaver = new PreviewSaver(node);
        } else {
            alert('Current settings is invalid to save image.\n\nERROR: ' + buttontitle);
        }
        //alert('pushed...');
    });
    LoadedNode = node;
    return {widget: widget};
}

function sendPOSTmessage(message) {
    const body = new FormData();
    body.append('previewdata', message);
    api.fetchApi("/primere_preview_post", {method: "POST", body,});
}
