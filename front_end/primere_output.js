import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
let hasShownAlertForUpdatingInt = false;

let currentClass = false;
let outputEventListenerInit = false;
let ImagePath = null;
let WorkflowData = {};
const realPath = "/extensions/ComfyUI_Primere_Nodes";
const prwPath = "extensions/PrimerePreviews";
let ORIG_SIZE_STRING = "";
let PreviewTarget = 'Checkpoint';
let previewURL = null;

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

const OutputToNode = ['PrimereAnyOutput', 'PrimereTextOutput', 'PrimereAestheticCKPTScorer', 'PrimereFastSeed'];

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
                    if (typeof node.widgets_values != "undefined") {
                        node.widgets_values[index] = "";
                        return "";
                    }
                };
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message]);
                this.showValueWidget.value = message.text[0];
            };
        }

        if (nodeData.name === "PrimerePreviewImage") {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                ImagePath = nodeData.input.hidden['image_path'][0]
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                new MiniPreviewControl(this);
                PrimerePreviewSaverWidget.apply(this, [this, 'PrimerePreviewSaver']);
            }

            /* nodeType.prototype.onNodeCreated = function () {
                PrimerePreviewSaverWidget.apply(this, [this, 'PrimerePreviewSaver']);
            }; */
        }
    },
});


class MiniPreviewControl {
    constructor(node) {

    }
}

async function PrimerePreviewSaverWidget(node, inputName) {
    if (inputName == 'PrimerePreviewSaver') {
        node.name = inputName;
        const widget = {
            type: "preview_saver_widget",
            name: `w${inputName}`,
            callback: () => {
            },
        };

        node.onWidgetChanged = function (name, value, old_value) {
            if (name == 'preview_target') {
                PreviewTarget = value;
            }
            if (name == 'image_save_as') {
                SaveMode = value;
            }
            if (name == 'image_type') {
                IMGType = value;
            }
            if (name == 'image_resize') {
                MaxSide = value;
            }
            if (name == 'target_selection') {
                SelectedTarget = value;
            }
            if (name == 'image_quality') {
                TargetQuality = value;
            }
            if (name == 'preview_save_mode') {
                PrwSaveMode = value;
            }
            ButtonLabelCreator(node);
            return false;
        };

        node.addWidget("combo", "target_selection", 'select target...', () => {
        }, {
            values: ["select target..."],
        });

        node.addWidget("button", buttontitle, null, () => {
            if (SaveIsValid === true) {
                if (typeof node['imgs'] != "undefined") {
                    node.PreviewSaver = new PreviewSaver(node);
                } else {
                    SaveIsValid = false;
                    buttontitle = 'Image not available for save. Please load one.';
                    applyWidgetValues(LoadedNode, buttontitle, TargetSelValues);
                    alert('Current settings is invalid to save image.\n\nERROR: ' + buttontitle);
                }
            } else {
                alert('Current settings is invalid to save image.\n\nERROR: ' + buttontitle);
            }
        });

        LoadedNode = node;
        return {widget: widget};
    }
}

app.registerExtension({
    name: "Primere.PrimerePreviewImage",

    async init(app) {
        function PreviewHandler(app) {
            var previewbox = null;
            outputEventListenerInit = true;
            let head = document.getElementsByTagName('HEAD')[0];
            let js1 = document.createElement("script");
            js1.src = realPath + "/vendor/LoadImage/load-image.js";
            head.appendChild(js1);
            let js2 = document.createElement("script");
            js2.src = realPath + "/vendor/LoadImage/load-image-scale.js";
            head.appendChild(js2);

            let link = document.createElement('link');
            link.rel = 'stylesheet';
            link.type = 'text/css';
            link.href = realPath + '/css/visual.css';
            head.appendChild(link);

            previewbox = document.getElementById("primere_previewbox");
            if (!previewbox) {
                previewbox = document.createElement("div");
                previewbox.setAttribute('style', 'display:none;');
                previewbox.setAttribute("id", "primere_previewbox");
                previewbox.innerHTML = '<div class="preview_closebutton">X</div><img src="' + prwPath + '/images/missing.jpg" ' + 'class="privewbox_image">';
                document.body.appendChild(previewbox);
            }

            $(document).on('click', 'div#primere_previewbox div.preview_closebutton', function(e) {
                $('div#primere_previewbox').hide();
            });
            $(document).on('click', 'body', function(e) {
                checkPreviewExample();
            });
        }

        function showPreviewIfExist(coordX, coordY) {
            if (SaveIsValid == true && SaveMode == true && PreviewExist == true) {
                $('div#primere_previewbox').css({
                   top: coordY + 'px',
                   left: coordX + 'px',
                });
                $("div#primere_previewbox img.privewbox_image").attr("src", previewURL);
                $('div#primere_previewbox').show();
            }
        }

        function checkPreviewExample() {
            if ($('div#primere_previewbox').is(":visible")) {
                $('div#primere_previewbox').hide();
            }
        }

        if (outputEventListenerInit == false) {
            PreviewHandler(app);
        }

        /* const lcg = LGraphCanvas.prototype.processNodeWidgets;
        LGraphCanvas.prototype.processNodeWidgets = function (node, pos, event, active_widget) {
            if (event.type == 'pointermove' && node.type == 'PrimerePreviewImage') {
                return false;
            }

            if (event.type != LiteGraph.pointerevents_method + "up") {
                if (event.type == LiteGraph.pointerevents_method + "down" && node.type == 'PrimerePreviewImage') {
                    if ((pos[1] - node.pos[1]) >= 214 && (pos[1] - node.pos[1]) <= 234) {
                        if ((pos[0] - node.pos[0]) >= 10 && (pos[0] - node.pos[0]) <= node.size[0]) {
                            showPreviewIfExist(event.clientX, event.clientY);
                        }
                    } else {
                        checkPreviewExample();
                    }
                }
                if (event.type == LiteGraph.pointerevents_method + "down") {
                    if (node.type != 'PrimerePreviewImage') {
                        checkPreviewExample();
                    }
                }
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (node.type != 'PrimerePreviewImage') {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            currentClass = node.type;
            return lcg.call(this, node, pos, event, active_widget);
        } */
    },
});

api.addEventListener("getVisualTargets", VisualDataReceiver);
async function VisualDataReceiver(event) { // 01
    WorkflowData = event.detail

    await sleep(1000);
    var img = document.querySelector('img')

    function loaded() {
        let newLoadedURL = img.src
        ButtonLabelCreator(LoadedNode, newLoadedURL)
    }

    if (img.complete) {
      loaded(img)
    } else {
      img.addEventListener('load', loaded)
      img.addEventListener('error', function() {
          console.log('new image loaded - ERROR')
      })
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
            reader.onloadend = () => {
                var file = dataURLtoFile(reader.result, ImageName);
                loadImage(file, function (img) {
                        if (typeof img.toDataURL === "function") {
                            var resampledOriginalImage = img.toDataURL(imgMime, TargetQuality);
                            if (SaveMode === false) {
                                downloadImage(resampledOriginalImage, extension, SaveImageName);
                            } else {
                                var resampledWidth = img.width;
                                var resampledHeight = img.height;

                                sendPOSTmessage(JSON.stringify({
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

function ButtonLabelCreator(node, url = false) {
    PreviewExist = false;
    previewURL = null;

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

    TargetSelValues = TargetListCreator(node);
    if (typeof TargetSelValues == "object") {
        var targetIndexChanged = 0;
        if (TargetSelValues.includes(SelectedTarget)) {
            targetIndexChanged = TargetSelValues.indexOf(SelectedTarget)
        }
        SelectedTarget = TargetSelValues[targetIndexChanged];
    }

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
        if (SaveMode === true) {
            buttontitle = SelectedTarget;
            applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
        } else {
            buttontitle = 'Image not available for save. Please load one.';
            SaveIsValid = true;
            if (url != false) {
                ;(async () => {
                    const img = await getMeta(url);
                    ORIG_SIZE_STRING = '[' + img.naturalHeight + ' X ' + img.naturalWidth + ']'
                    buttontitle = 'Save image as ' + INIT_IMGTYPE_STRING + ' | ' + ORIG_SIZE_STRING + ' ' + INIT_IMGSIZE_STRING + ' | QTY: ' + TargetQuality + '%';
                    applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
                })();
            } else {
                //buttontitle = '+Save image as ' + INIT_IMGTYPE_STRING + ' | ' + ORIG_SIZE_STRING + ' ' + INIT_IMGSIZE_STRING + ' | QTY: ' + TargetQuality + '%';
                applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
            }
        }
    } else {
        if (SaveMode === true) {
            if (url != false) {
                ;(async () => {
                    const img = await getMeta(url);
                    ORIG_SIZE_STRING = '[' + img.naturalHeight + ' X ' + img.naturalWidth + ']'
                })();
            }

            if (WorkflowData[NodenameByType[PreviewTarget]] !== undefined && SelectedTarget !== undefined) {
                if (WorkflowData[NodenameByType[PreviewTarget]].length < 1) {
                    buttontitle = 'No resource selected for preview target: [' + PreviewTarget + ']';
                    applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
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

                    ;(async () => {
                        const img = await getMeta(imgsrc);
                        if (typeof img != "undefined" && img != false && img.naturalHeight > 0) {
                            previewURL = imgsrc;
                            PreviewExist = true;
                        } else {
                            previewURL = null;
                            PreviewExist = false;
                        }

                        /* if (PreviewExist === true) {
                            node.addWidget("button", "View preview", null, () => {
                                return false;
                            });
                        } */

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
                        applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
                     })();
                }
            } else {
                buttontitle = 'Required node: [' + NodenameByType[PreviewTarget] + '] not available in workflow for target: [' + PreviewTarget + ']';
                applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
            }
        } else {
            SaveIsValid = true;
            if (url != false) {
                ;(async () => {
                    const img = await getMeta(url);
                    ORIG_SIZE_STRING = '[' + img.naturalHeight + ' X ' + img.naturalWidth + ']'
                    buttontitle = 'Save image as ' + INIT_IMGTYPE_STRING + ' | ' + ORIG_SIZE_STRING + ' ' + INIT_IMGSIZE_STRING + ' | QTY: ' + TargetQuality + '%';
                    applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
                })();
            } else {
                buttontitle = 'Save image as ' + INIT_IMGTYPE_STRING + ' | ' + ORIG_SIZE_STRING + ' ' + INIT_IMGSIZE_STRING + ' | QTY: ' + TargetQuality + '%';
                applyWidgetValues(LoadedNode, buttontitle, TargetSelValues)
            }
        }
    }
}

function applyWidgetValues(LoadedNode, buttontitle, TargetSelValues) {
    for (var iln = 0; iln < LoadedNode.widgets.length; ++iln) {
        var wln = LoadedNode.widgets[iln];
        if (wln.type == 'button') {
            wln.name = buttontitle
        }
        if (wln.name == 'target_selection') {
            wln.options.values = TargetSelValues;
            if (TargetSelValues.length > 0) {
                var targetIndexChanged = 0;
                if (TargetSelValues.includes(SelectedTarget)) {
                    targetIndexChanged = TargetSelValues.indexOf(SelectedTarget)
                }
                wln.value = TargetSelValues[targetIndexChanged];
            }
        }
    }
}

// ************************* sendPOSTmessage PreviewSaveResponse
function sendPOSTmessage(message) {
    const body = new FormData();
    body.append('previewdata', message);
    api.fetchApi("/primere_preview_post", {method: "POST", body,});
}
api.addEventListener("PreviewSaveResponse", PreviewSaveResponse);
function PreviewSaveResponse(event) {
    var ResponseText = event.detail;
    alert(ResponseText);
}

const getMeta = (url) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = url;
}).catch(function(error) {
    return false;
});

//await sleep(2000);
//await waitUntil(() => variable === true);
function sleep(ms){
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitUntil(condition, time = 100) {
    while (!condition()) {
        await new Promise((resolve) => setTimeout(resolve, time));
    }
}

async function waitUntilEqual(condition1, condition2, time = 100) {
    while (condition1 != condition2) {
        await new Promise((resolve) => setTimeout(resolve, time));
    }
}

async function waitForImageToLoad(imageElement){
  return new Promise(resolve=>{imageElement.onload = resolve})
}