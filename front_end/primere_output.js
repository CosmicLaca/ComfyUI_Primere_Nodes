import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

let hasShownAlertForUpdatingInt = false;
let ImagePath = null;
const realPath = "extensions/Primere";

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
            if (node.type == 'PrimerePreviewImage') {
                console.log('==============================');
                console.log(node);
                console.log(node.type)
                console.log(event)
                console.log('==============================');
            } else {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (event.type != LiteGraph.pointerevents_method + "up") {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            var x = pos[0] - node.pos[0];
            var y = pos[1] - node.pos[1];
            var width = node.size[0];
            var that = this;

            var PreviewTarget = 'Checkpoint';
            var Convert = true;
            var IMGType = true;
            var MaxSide = -1;

            for (var p = 0; p < node.widgets.length; ++p) {
                if (node.widgets[p].name == 'preview_target') {
                    PreviewTarget = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'image_resize') {
                    Convert = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'image_type') {
                    IMGType = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'largest_side') {
                    MaxSide = node.widgets[p].value;
                }
            }

            console.log('++++++++++++++');
            console.log(PreviewTarget);
            console.log(Convert);
            console.log(IMGType);
            console.log(MaxSide);
            console.log('++++++++++++++');

            for (var i = 0; i < node.widgets.length; ++i) {
                var w = node.widgets[i];
                if (!w || w.disabled)
                    continue;

                //if (w.type != "boolean")
                //    continue

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
                    var IMGTYPE_STRING = "";
                    var IMGSIZE_STRING = "";
                    if (IMGType === true) {
                        IMGTYPE_STRING = ' as JPG format'
                    } else {
                        IMGTYPE_STRING = ' as PNG format'
                    }
                    if (MaxSide < 64) {
                        IMGSIZE_STRING = " at original size";
                    } else {
                        IMGSIZE_STRING = " resizing to " + MaxSide + 'px';
                    }
                    if (Convert === true) {
                        w.name = 'Save image to your ' + PreviewTarget + ' folder for preview' + IMGTYPE_STRING + IMGSIZE_STRING;
                    } else {
                        w.name = 'Save image without resizing' + IMGTYPE_STRING + IMGSIZE_STRING;
                    }
                }


                //node.setProperty(active_widget.options.property, 'teszt value');

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
            console.log('---------------------------------')
            console.log(nodeData)
            console.log(nodeType)
            console.log(app)
            console.log('---------------------------------')
            //const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                //const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                PrimerePreviewSaverWidget.apply(this, [this, 'PrimerePreviewSaver']);
                //return r;
            };
        }
    },
});

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

function downloadImage(url) {
    fetch(url, {
        mode : 'no-cors',
    })
        .then(response => response.blob())
        .then(blob => {
        let blobUrl = window.URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.download = 'testfilename.jpg';
        a.href = blobUrl;
        document.body.appendChild(a);
        a.click();
        a.remove();
    })
}

class PreviewSaver {
    constructor(node) {
        // console.log('-------- start -------------');
        // console.log(node['imgs'][0]['src']);
        // console.log(node['images'][0]['filename']);
        // console.log(ImagePath)
        // console.log('-------- end -------------');

        var ImageSource = node['imgs'][0]['src'];
        var ImageName = node['images'][0]['filename'];
        fetch(ImageSource)
        .then((res) => res.blob())
        .then((blob) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                var file = dataURLtoFile(reader.result, ImageName);
                loadImage(file, function(img) {
                    var resampledOriginalImage = img.toDataURL("image/jpeg", 0.6);
                    downloadImage(resampledOriginalImage);
                },
                {
                    // maxWidth:200,
                    maxHeight:250,
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

function PrimerePreviewSaverWidget(node, inputName) {
    node.name = inputName;
    const widget = {
        type: "preview_saver_widget",
        name: `w${inputName}`,
        callback: () => {},
    };

    $(document).ready(function () {
        //alert(' van jquery');
    });

    node.addWidget("button", "Save image as visual modal preview", inputName, () => {
      //node.p_painter.list_objects_panel__items.innerHTML = "";
      //node.p_painter.clearCanvas();
      node.painter = new PreviewSaver(node);
      //alert('megnyomtuk');
    });

    return {widget: widget};
}