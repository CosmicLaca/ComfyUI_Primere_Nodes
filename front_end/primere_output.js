import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
let hasShownAlertForUpdatingInt = false;

let outputEventListenerInit = false;
let ImagePath = null;
let WorkflowData = {};
const realPath = "extensions/Primere";
const prwPath = "extensions/PrimerePreviews";

let PreviewTarget = 'Checkpoint';
let PreviewTargetPreviousState = PreviewTarget;
let SaveMode = true;
let IMGType = 'jpeg';
let MaxSide = -1;
let TargetQuality = 95;
let buttontitle = 'Image not available for save. Please load one.'
let SaveIsValid = false;
let TargetFileName = null;
let LoadedNode = null;
let TargetSelValues = ["select target..."];
let SelectedTarget = null;
let PreviewExist = false;
let PrwSaveMode = 'Overwrite';

const NodenameByType = {
    'Checkpoint': 'PrimereVisualCKPT',
    'CSV Prompt': 'PrimereVisualStyle',
    'Lora': 'PrimereVisualLORA',
    'Lycoris': 'PrimereVisualLYCORIS',
    'Hypernetwork': 'PrimereVisualHypernetwork',
    'Embedding': 'PrimereVisualEmbedding'
}

const NodesubdirByType = {
    'Checkpoint': 'checkpoints',
    'CSV Prompt': 'styles',
    'Lora': 'loras',
    'Lycoris': 'lycoris',
    'Hypernetwork': 'hypernetworks',
    'Embedding': 'embeddings'
}

const OutputToNode = ['PrimereAnyOutput', 'PrimereTextOutput', 'PrimereAestheticCKPTScorer'];

app.registerExtension({
    name: "Primere.PrimereOutputs",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (OutputToNode.includes(nodeData.name) === true) {
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

        let callbackfunct = null;

        function PreviewHandler(app) {
            outputEventListenerInit = true;
            let head = document.getElementsByTagName('HEAD')[0];
            let js1 = document.createElement("script");
            js1.src = realPath + "/vendor/LoadImage/load-image.js";
            head.appendChild(js1);
            let js2 = document.createElement("script");
            js2.src = realPath + "/vendor/LoadImage/load-image-scale.js";
            head.appendChild(js2);

            $(document).on("click", 'div.graphdialog button', function(e) {
                for (var its_1 = 0; its_1 < app.canvas.visible_nodes.length; ++its_1) {
                    var wts_1 = app.canvas.visible_nodes[its_1];
                    if (wts_1.type == 'PrimerePreviewImage') {
                        ButtonLabelCreator(wts_1);
                        console.log('Buttontitle 1')
                        console.log(buttontitle)

                        /* for (var its_2 = 0; its_2 < wts_1.widgets.length; ++its_2) {
                            var wts_2 = wts_1.widgets[its_2];
                            if (wts_2.type == 'button') {
                                wts_2.name = buttontitle;
                            }
                        } */
                    }
                }
            });
        }

        if (outputEventListenerInit == false) {
            PreviewHandler(app);
        }

        const lcg = LGraphCanvas.prototype.processNodeWidgets;
        LGraphCanvas.prototype.processNodeWidgets = function (node, pos, event, active_widget) {
            if (event.type == 'pointermove' && node.type == 'PrimerePreviewImage') {
                return false;
            }

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
            var that = this;
            var ref_window = this.getCanvasWindow();
            var combo_hit = false

            TargetSelValues = TargetListCreator(node)

            let target_id = 0;
            let button_id = 0;

            for (var its = 0; its < node.widgets.length; ++its) {
                var wts = node.widgets[its];
                if (!wts || wts.disabled)
                    continue;

                if (wts.name == 'target_selection') {
                    target_id = its;
                    node.widgets[its].options.values = TargetSelValues;
                    if (PreviewTarget !== PreviewTargetPreviousState && TargetSelValues.length > 0) {
                        node.widgets[its].value = TargetSelValues[0];
                    }
                }
            }

            for (var i = 0; i < node.widgets.length; ++i) {
                var w = node.widgets[i];
                if (!w || w.disabled)
                    continue;

                var widget_height = w.computeSize ? w.computeSize(width)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                var widget_width = w.width || width;
                var widget_name = node.widgets[i].name;

                if (w.type == 'button') {
                    button_id = i;
                    ButtonLabelCreator(node);
                    console.log('Buttontitle 2')
                    console.log(buttontitle)
                    //w.name = buttontitle;
                }

                if (w != active_widget && (x < 6 || x > widget_width - 12 || y < w.last_y || y > w.last_y + widget_height || w.last_y === undefined))
                    continue

                if (w == active_widget || (x > 6 && x < widget_width - 12 && y > w.last_y && y < w.last_y + widget_height)) {
                    var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                    if (delta)
                        continue;

                    combo_hit = true;
                    var values = w.options.values;
                    if (values && values.constructor === Function) {
                        values = w.options.values(w, node);
                    }

                    if (typeof values != 'undefined') {
                        var values_list = values.constructor === Array ? values : Object.keys(values);
                        var text_values = values != values_list ? Object.values(values) : values;

                        function inner_clicked(v, option, event) {
                            console.log('callback v:')
                            console.log(v)
                            if (values != values_list)
                                v = text_values.indexOf(v);

                            this.value = v;
                            that.dirty_canvas = true;

                            TargetSelValues = TargetListCreator(node)

                            node.widgets[target_id].options.values = TargetSelValues;
                            if (PreviewTarget !== PreviewTargetPreviousState && TargetSelValues.length > 0) {
                                node.widgets[target_id].value = TargetSelValues[0];
                            }

                            ButtonLabelCreator(node);
                            console.log('Buttontitle 3')
                            console.log(buttontitle)

                            //node.widgets[button_id].name = buttontitle;
                            //return false;
                        }

                        new LiteGraph.ContextMenu(values, {
                            scale: Math.max(1, this.ds.scale),
                            event: event,
                            className: "dark",
                            callback: inner_clicked.bind(w),
                            node: node,
                            widget: w,
                        }, ref_window);
                    } else {
                        //return false;
                    }
                }
            }
            PreviewTargetPreviousState = PreviewTarget;
            return lcg.call(this, node, pos, event, active_widget);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimerePreviewImage") {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                ImagePath = nodeData.input.hidden['image_path'][0]
            }

            nodeType.prototype.onNodeCreated = function () {
                PrimerePreviewSaverWidget.apply(this, [this, 'PrimerePreviewSaver']);
            };
        }
    },
});

api.addEventListener("PreviewSaveResponse", PreviewSaveResponse);
function PreviewSaveResponse(event) {
    var ResponseText = event.detail;
    alert(ResponseText);
}

api.addEventListener("getVisualTargets", VisualDataReceiver);
function VisualDataReceiver(event) {
    WorkflowData = event.detail
    if (Object.keys(WorkflowData).length >= 1) {
        SaveIsValid = true;
    } else {
        SaveIsValid = false;
    }

    TargetSelValues = TargetListCreator(LoadedNode)

    for (var iln = 0; iln < LoadedNode.widgets.length; ++iln) {
        var wln = LoadedNode.widgets[iln];
        if (!wln || wln.disabled)
            continue;

        if (wln.name == 'target_selection') {
            wln.options.values = TargetSelValues;
            if (TargetSelValues.length > 0) {
                wln.value = TargetSelValues[0];
            }
        }

        if (wln.type == 'button') {
            function getFirstState(wln, LoadedNode) {
                //await sleep(150);
                ButtonLabelCreator(LoadedNode);
                //await sleep(200);
                    console.log('Buttontitle 4')
                    console.log(buttontitle)

                //wln.name = buttontitle
            }
            getFirstState(wln, LoadedNode);
        }
        // New image loaded
    }
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
        var maxWidth = null;
        var maxHeight = null;

        if (SaveMode === true) {
            maxWidth = 400;
            maxHeight = 220;
        }

        var imgMime = "image/jpeg";
        var extension = 'jpg';

        if (SaveMode === false) {
            if (IMGType === 'jpeg') {
                imgMime = "image/jpeg";
                extension = 'jpg';
            } else if (IMGType === 'png') {
                imgMime = "image/png";
                extension = 'png';
            } else if (IMGType === 'webp') {
                imgMime = "image/webp";
                extension = 'webp';
            }
        }

        if (MaxSide >= 64 && SaveMode === false) {
            maxWidth = MaxSide;
            maxHeight = MaxSide;
        }

        var SizeStringFN = '';
        if (MaxSide >= 64) {
            SizeStringFN = MaxSide + 'px_'
        }

        var ImageSource = node['imgs'][0]['src'];
        var ImageName = node['images'][0]['filename'];
        var SaveImageName = 'PreviewImage_' + SizeStringFN + '_QTY' + TargetQuality + '_' + (Math.random() + 1).toString(36).substring(5);

        if (TargetFileName !== null) {
            SaveImageName = TargetFileName;
        }

        fetch(ImageSource)
        .then((res) => res.blob())
        .then((blob) => {
            const reader = new FileReader();
            reader.onloadend = async () => {
                var file = dataURLtoFile(reader.result, ImageName);
                await loadImage(file, async function (img) {
                        if (typeof img.toDataURL === "function") {
                            var resampledOriginalImage = img.toDataURL(imgMime, TargetQuality);
                            if (SaveMode === false) {
                                downloadImage(resampledOriginalImage, extension, SaveImageName);
                            } else {
                                var resampledWidth = img.width;
                                var resampledHeight = img.height;

                                var ResponseText = await sendPOSTmessage(JSON.stringify({
                                    "PreviewTarget": PreviewTarget,
                                    "PreviewTargetOriginal": SelectedTarget,
                                    "extension": extension,
                                    "ImageName": ImageName,
                                    "ImagePath": ImagePath,
                                    "SaveImageName": SaveImageName,
                                    "maxWidth": resampledWidth,
                                    "maxHeight": resampledHeight,
                                    "TargetQuality": TargetQuality,
                                    "PrwSaveMode": PrwSaveMode
                                }));
                                alert(ResponseText);
                            }
                        } else {
                            alert('Source image: ' + ImageName + ' does not exist, maybe deleted.')
                        }
                    },
                    {
                        maxWidth: maxWidth,
                        maxHeight: maxHeight,
                        canvas: true,
                        pixelRatio: 1,
                        downsamplingRatio: TargetQuality,
                        orientation: true,
                        imageSmoothingEnabled: 1,
                        imageSmoothingQuality: 'high'
                    });
            };
            reader.readAsDataURL(blob);
        });
    }
}

function TargetListCreator(node) {
    ButtonLabelCreator(node);
    console.log('Buttontitle 5')
    console.log(buttontitle)


    if (WorkflowData[NodenameByType[PreviewTarget] + '_ORIGINAL'] !== undefined) {
        TargetSelValues = WorkflowData[NodenameByType[PreviewTarget] + '_ORIGINAL'];
        if (TargetSelValues.length == 0) {
            SaveIsValid = false;
            TargetSelValues = ['ERROR: Cannot list target for ' + PreviewTarget]
        }
    } else {
        SaveIsValid = false;
        TargetSelValues = ['ERROR: Cannot list target for ' + PreviewTarget]
    }
    return TargetSelValues;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function ButtonLabelCreator(node) {
    console.log('ButtonLabelCreator ----------------------')
    PreviewExist = false;

    for (var px = 0; px < node.widgets.length; ++px) {
        if (node.widgets[px].name == 'preview_target') {
            PreviewTarget = node.widgets[px].value;
        }
        if (node.widgets[px].name == 'image_save_as') {
            SaveMode = node.widgets[px].value;
        }
        if (node.widgets[px].name == 'image_type') {
            IMGType = node.widgets[px].value;
        }
        if (node.widgets[px].name == 'image_resize') {
            MaxSide = node.widgets[px].value;
        }
        if (node.widgets[px].name == 'target_selection') {
            SelectedTarget = node.widgets[px].value;
        }
        if (node.widgets[px].name == 'image_quality') {
            TargetQuality = node.widgets[px].value;
        }
        if (node.widgets[px].name == 'preview_save_mode') {
            PrwSaveMode = node.widgets[px].value;
        }
    }

    console.log('-------------- 1 s -------------------')
    console.log(PreviewTarget)
    console.log(SaveMode)
    console.log(IMGType)
    console.log(MaxSide)
    console.log(TargetQuality)
    console.log(PreviewTarget)
    console.log(PrwSaveMode)
    console.log('. . . . . . . . . . . . . . . . . . . . .')


    var INIT_IMGTYPE_STRING = "";
    var INIT_IMGSIZE_STRING = "";
    INIT_IMGTYPE_STRING = IMGType.toUpperCase() + ' format';
    if (MaxSide < 64) {
        INIT_IMGSIZE_STRING = "at original size";
    } else {
        INIT_IMGSIZE_STRING = "resized to " + MaxSide + 'px';
    }

    if (IMGType == 'png') {
        TargetQuality = 100;
    }

    TargetFileName = null;
    SaveIsValid = false;
    if (Object.keys(WorkflowData).length < 1) {
        buttontitle = 'Image not available for save. Please load one.';
    } else {
        if (SaveMode === true) {
            if (WorkflowData[NodenameByType[PreviewTarget]] !== undefined && SelectedTarget !== undefined) {
                if (WorkflowData[NodenameByType[PreviewTarget]].length < 1) {
                    buttontitle = 'No resource selected for preview target: [' + PreviewTarget + ']';
                } else {
                    SaveIsValid = true;
                    var targetIndex = WorkflowData[NodenameByType[PreviewTarget] + '_ORIGINAL'].indexOf(SelectedTarget);
                    if (targetIndex > -1) {
                        TargetFileName = WorkflowData[NodenameByType[PreviewTarget]][targetIndex];
                    }

                    let prwpath_new = SelectedTarget.replaceAll('\\', '/');
                    let dotLastIndex = prwpath_new.lastIndexOf('.');
                    if (dotLastIndex > 1) {
                        var finalName = prwpath_new.substring(0, dotLastIndex);
                    } else {
                        var finalName = prwpath_new;
                    }
                    finalName = finalName.replaceAll(' ', "_");
                    let previewName = finalName + '.jpg';
                    var imgsrc = prwPath + '/images/' + NodesubdirByType[PreviewTarget] + '/' + previewName;

                    let loadImage = src =>
                        new Promise((resolve, reject) => {
                            let img_check = new Image();
                            img_check.onload = () => resolve(img_check);
                            img_check.onerror = reject;
                            img_check.src = src;
                            if (img_check.height > 0) {
                                PreviewExist = true;
                            }
                        });

                    loadImage(imgsrc).then(image_prw_test =>
                        image_prw_test.complete
                    );

                    var imgExistLink = "";
                    if (PreviewExist === true) {
                        let splittedMode = PrwSaveMode.split(' ');
                        var prw_mode = '';
                        splittedMode.forEach(n => {
                            prw_mode += n[0]
                        });
                        imgExistLink = ' [' + prw_mode.toUpperCase() + ']';
                    } else {
                        imgExistLink = ' [C]';
                    }

                    buttontitle = 'Save preview as: [' + TargetFileName + '.jpg] to [' + PreviewTarget + '] folder.' + imgExistLink;
                }
            } else {
                buttontitle = 'Required node: [' + NodenameByType[PreviewTarget] + '] not available in workflow for target: [' + PreviewTarget + ']';
            }
        } else {
            /* const img_size = new Image();
            img_size.src = imgsrc
            img_size.onload = function() {
              console.log(imgsrc.width + ' x ' + imgsrc.height)
            } */

            console.log('--------------------- ImageSource')
            var ORIG_SIZE_STRING = "";

            var isExist = typeof node['imgs'];
            for (let i = 0; i < 20; i++) {
                isExist = typeof node['imgs'];
                if (isExist == "object") {
                    break;
                }
                //await sleep(100);
            }

            setTimeout(() => {
                if (typeof node['imgs'] != 'undefined') {
                    var ImageSource_getsize = node['imgs'][0]['src'];
                    const img_size = new Image();
                    img_size.src = ImageSource_getsize;
                    waitForImageToLoad(img_size).then(() => {
                        ORIG_SIZE_STRING = ' [' + img_size.width + ' X ' + img_size.height + '] '
                        console.log(ORIG_SIZE_STRING)
                    });
                }
            }, 500);

            //await sleep(500);
            SaveIsValid = true;
            buttontitle = 'Save image as ' + INIT_IMGTYPE_STRING + ' | ' + ORIG_SIZE_STRING + INIT_IMGSIZE_STRING + ' | QTY: ' + TargetQuality + '%';
            console.log(buttontitle)
            console.log('-------------- 1 e -------------------')

        }
    }
    console.log('-------------- 1 sx -------------------')
    console.log(buttontitle)

    for (var iln = 0; iln < LoadedNode.widgets.length; ++iln) {
        var wln = LoadedNode.widgets[iln];
        if (wln.type == 'button') {
            wln.name = buttontitle
        }
    }
    console.log('-------------- 1 ex -------------------')
    //return buttontitle;
}

function PrimerePreviewSaverWidget(node, inputName) {
    // node load
    node.name = inputName;
    const widget = {
        type: "preview_saver_widget",
        name: `w${inputName}`,
        callback: () => {},
    };

     node.onWidgetChanged = function(name, value, old_value){
        //alert('changed? +++')
        console.log('------------ widget ch: ---------------')
        console.log(name)
        console.log(value)
        console.log(old_value)
        ButtonLabelCreator(node);
        console.log('------------ w.ch end ---------------')
        return false;
    };

    node.addWidget("combo", "target_selection", 'select target...', () => {
    },{
        values: ["select target..."],
    });

    node.addWidget("button", buttontitle, null, () => {
        if (SaveIsValid === true) {
            node.PreviewSaver = new PreviewSaver(node);
        } else {
            alert('Current settings is invalid to save image.\n\nERROR: ' + buttontitle);
        }
        // button clicked
    });

    LoadedNode = node;
    return {widget: widget};
}

// ************************* sendPOSTmessage PreviewSaveResponse
function sendPOSTmessage(sourcetype) {
    console.log('------------------ ReadFileDate');
    return new Promise((resolve, reject) => {
        api.addEventListener("PreviewSaveResponse", (event) => resolve(event.detail), true);
        postImageData(sourcetype);
    });
}
function postImageData(sourcetype) {
    const body = new FormData();
    body.append('type', sourcetype);
    api.fetchApi("/primere_preview_post", {method: "POST", body,});
}

async function waitForImageToLoad(imageElement){
  return new Promise(resolve=>{imageElement.onload = resolve})
}

async function waitUntilEqual(condition1, condition2, time = 100) {
    while (condition1 != condition2) {
        await new Promise((resolve) => setTimeout(resolve, time));
    }
}