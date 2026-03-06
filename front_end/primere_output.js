import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
let hasShownAlertForUpdatingInt = false;

let currentClass = false;
let outputEventListenerInit = false;
let ImagePath = null;
const realPath = "/extensions/ComfyUI_Primere_Nodes";
const prwPath = "/extensions/ComfyUI_Primere_Nodes";

let pendingSaveNode = null;

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

// ── Per-node state ────────────────────────────────────────────────────────────

function initNodeState() {
    return {
        workflowData: {},
        origSizeString: '',
        saveIsValid: false,
        targetFileName: null,
        targetSelValues: ['select target...'],
        selectedTarget: null,
        previewExist: false,
        previewURL: null,
    };
}

function ensureNodeState(node) {
    if (!node.psState) node.psState = initNodeState();
    return node.psState;
}

function getWidgetValues(node) {
    const state = ensureNodeState(node);
    const vals = {
        previewTarget: 'Checkpoint',
        saveMode: true,
        imgType: 'jpeg',
        maxSide: -1,
        targetQuality: 95,
        prwSaveMode: 'Overwrite',
    };
    for (const w of node.widgets) {
        switch (w.name) {
            case 'preview_target':   vals.previewTarget  = w.value; break;
            case 'image_save_as':    vals.saveMode       = w.value; break;
            case 'image_type':       vals.imgType        = w.value; break;
            case 'image_resize':     vals.maxSide        = w.value; break;
            case 'image_quality':    vals.targetQuality  = w.value; break;
            case 'preview_save_mode': vals.prwSaveMode   = w.value; break;
            case 'target_selection': state.selectedTarget = w.value; break;
        }
    }
    return vals;
}

// ── Node registration ─────────────────────────────────────────────────────────

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
                app.canvas.setDirty(true);
            };
        }

        if (nodeData.name === "PrimerePreviewImage") {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                ImagePath = nodeData.input.hidden['image_path'][0];
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                new MiniPreviewControl(this);
                PrimerePreviewSaverWidget.apply(this, [this, 'PrimerePreviewSaver']);
            };
        }
    },
});


class MiniPreviewControl {
    constructor(node) {
        node.onMouseDown = function(event, pos, graphcanvas) {
            if (event.type == 'pointerdown') {
                if (pos[1] >= 218 && pos[1] <= 248) {
                    showPreviewIfExist(node, event.clientX, event.clientY);
                } else {
                    checkPreviewExample();
                }
            }
        };
    }
}

function showPreviewIfExist(node, coordX, coordY) {
    const state = ensureNodeState(node);
    const wv = getWidgetValues(node);
    if (state.saveIsValid && wv.saveMode && state.previewExist) {
        const previewBox = document.querySelector('div#primere_previewbox');
        const previewImage = document.querySelector('div#primere_previewbox img.privewbox_image');
        if (!previewBox || !previewImage) return;
        previewBox.style.top = coordY + 'px';
        previewBox.style.left = coordX + 'px';
        previewImage.src = state.previewURL;
        previewBox.style.display = 'block';
    }
}

function checkPreviewExample() {
    const previewBox = document.querySelector('div#primere_previewbox');
    if (previewBox && previewBox.style.display !== 'none') {
        previewBox.style.display = 'none';
    }
}

async function PrimerePreviewSaverWidget(node, inputName) {
    if (inputName == 'PrimerePreviewSaver') {
        node.name = inputName;
        ensureNodeState(node);

        const widget = {
            type: "preview_saver_widget",
            name: `w${inputName}`,
            callback: () => {},
        };

        node.onWidgetChanged = function (name, value, old_value) {
            ButtonLabelCreator(node);
            return false;
        };

        node.addWidget("combo", "target_selection", 'select target...', () => {}, {
            values: ["select target..."],
        });

        const saveBtn = node.addWidget("button", '⛔ Image not available for save. Please load one.', null, () => {
            const state = ensureNodeState(node);
            if (state.saveIsValid === true) {
                if (typeof node['imgs'] != "undefined") {
                    pendingSaveNode = node;
                    node.PreviewSaver = new PreviewSaver(node);
                } else {
                    state.saveIsValid = false;
                    const errTitle = '⛔ Image not available for save. Please load one.';
                    applyWidgetValues(node, errTitle, state.targetSelValues);
                    alert('Current settings is invalid to save image.\n\nERROR: ' + errTitle);
                }
            } else {
                const btn = node.widgets.find(w => w.type === 'button');
                alert('Current settings is invalid to save image.\n\nERROR: ' + (btn ? btn.name : 'Unknown'));
            }
        });

        const BTN_HEIGHT = 32;
        const BTN_COLOR = "#771a1a";
        const BTN_COLOR_ACTIVE = "#932424";
        const BTN_RADIUS = 6;
        const BTN_FONT = "bold 15px sans-serif";

        saveBtn.computeSize = () => [0, BTN_HEIGHT];
        saveBtn.draw = function (ctx, node, widget_width, y) {
            ctx.save();
            const margin = 15;
            ctx.fillStyle = this.clicked ? BTN_COLOR_ACTIVE : BTN_COLOR;
            if (this.clicked) {
                this.clicked = false;
                node.setDirtyCanvas?.(true);
            }
            ctx.beginPath();
            ctx.roundRect(margin, y, widget_width - margin * 2, BTN_HEIGHT, BTN_RADIUS);
            ctx.fill();
            ctx.fillStyle = "#dad570";
            ctx.font = BTN_FONT;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(this.name, widget_width * 0.5, y + BTN_HEIGHT * 0.5);
            ctx.restore();
        };

        return { widget: widget };
    }
}

app.registerExtension({
    name: "Primere.PrimerePreviewImage",

    async init(app) {
        function PreviewHandler(app) {
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

            let previewbox = document.getElementById("primere_previewbox");
            if (!previewbox) {
                previewbox = document.createElement("div");
                previewbox.setAttribute('style', 'display:none;');
                previewbox.setAttribute("id", "primere_previewbox");
                previewbox.innerHTML = '<div class="preview_closebutton">X</div><img src="' + prwPath + '/images/missing.jpg" ' + 'class="privewbox_image">';
                document.body.appendChild(previewbox);
            }

            document.addEventListener('click', function(e) {
                if (e.target.closest('div#primere_previewbox div.preview_closebutton')) {
                    checkPreviewExample();
                    return;
                }
                if (e.target.closest('body')) {
                    checkPreviewExample();
                }
            });
        }

        if (outputEventListenerInit == false) {
            PreviewHandler(app);
        }
    },
});

// ── Event: node executed ──────────────────────────────────────────────────────

api.addEventListener("getVisualTargets", VisualDataReceiver);
function VisualDataReceiver(event) {
    const data = event.detail;

    let targetNode = null;
    const nodeId = data['node_id'];
    if (nodeId && app.graph) {
        targetNode = app.graph.getNodeById(parseInt(nodeId));
    }
    if (!targetNode && app.graph && app.graph._nodes) {
        targetNode = app.graph._nodes.find(n => n.type === 'PrimerePreviewImage');
    }
    if (!targetNode) return;

    const state = ensureNodeState(targetNode);
    state.workflowData = data;

    const imgs = data['SaveImages'];
    if (!imgs || imgs.length === 0) return;

    const newLoadedURL = window.location.origin + '/view?filename=' + imgs[0]['filename'] + '&type=' + imgs[0]['type'] + '&subfolder=' + imgs[0]['subfolder'];

    UrlExists(newLoadedURL, function (status) {
        if (status === 200) {
            ButtonLabelCreator(targetNode, newLoadedURL);
        } else {
            state.origSizeString = '[Unknown dimensions]';
            console.log('new image loaded - ERROR status ' + status + ': - ' + newLoadedURL);
        }
    });
}

// ── Utility ───────────────────────────────────────────────────────────────────

function UrlExists(url, cb) {
    fetch(url, {
        method: 'HEAD',
    }).then((response) => {
        if (typeof cb === 'function') {
            cb.apply(this, [response.status]);
        }
    }).catch(() => {
        if (typeof cb === 'function') {
            cb.apply(this, [0]);
        }
    });
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
    });
}

// ── PreviewSaver ──────────────────────────────────────────────────────────────

class PreviewSaver {
    constructor(node) {
        const state = ensureNodeState(node);
        const wv = getWidgetValues(node);

        let maxWidth = null;
        let maxHeight = null;
        if (wv.saveMode === true) {
            maxWidth = 400;
            maxHeight = 220;
        }

        let imgMime = "image/jpeg";
        let extension = 'jpg';
        if (wv.saveMode === false) {
            if (wv.imgType === 'jpeg') {
                imgMime = "image/jpeg";
                extension = 'jpg';
            } else if (wv.imgType === 'png') {
                imgMime = "image/png";
                extension = 'png';
            } else if (wv.imgType === 'webp') {
                imgMime = "image/webp";
                extension = 'webp';
            }
        }

        if (wv.maxSide >= 64 && wv.saveMode === false) {
            maxWidth = wv.maxSide;
            maxHeight = wv.maxSide;
        }

        const sizeStringFN = wv.maxSide >= 64 ? wv.maxSide + 'px_' : '';

        const imageSource = node['imgs'][0]['src'];
        const imageName = node['images'][0]['filename'];
        let saveImageName = 'PreviewImage_' + sizeStringFN + '_QTY' + wv.targetQuality + '_' + (Math.random() + 1).toString(36).substring(5);

        if (state.targetFileName !== null) {
            saveImageName = state.targetFileName;
        }

        fetch(imageSource)
        .then((res) => res.blob())
        .then((blob) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const file = dataURLtoFile(reader.result, imageName);
                loadImage(file, function (img) {
                    if (typeof img.toDataURL === "function") {
                        const resampledImage = img.toDataURL(imgMime, wv.targetQuality);
                        if (wv.saveMode === false) {
                            downloadImage(resampledImage, extension, saveImageName);
                        } else {
                            sendPOSTmessage(JSON.stringify({
                                "PreviewTarget": wv.previewTarget,
                                "PreviewTargetOriginal": state.selectedTarget,
                                "extension": extension,
                                "ImageName": imageName,
                                "ImagePath": ImagePath,
                                "SaveImageName": saveImageName,
                                "maxWidth": img.width,
                                "maxHeight": img.height,
                                "TargetQuality": wv.targetQuality,
                                "PrwSaveMode": wv.prwSaveMode,
                            }));
                        }
                    } else {
                        alert('Source image: ' + imageName + ' does not exist, maybe deleted.');
                    }
                }, {
                    maxWidth: maxWidth,
                    maxHeight: maxHeight,
                    canvas: true,
                    pixelRatio: 1,
                    downsamplingRatio: wv.targetQuality / 100,
                    orientation: true,
                    imageSmoothingEnabled: true,
                    imageSmoothingQuality: 'high',
                });
            };
            reader.readAsDataURL(blob);
        })
        .catch((err) => {
            alert('Failed to load source image: ' + err.message);
        });
    }
}

// ── Button label / target list ────────────────────────────────────────────────

function TargetListCreator(node) {
    const state = ensureNodeState(node);
    const wv = getWidgetValues(node);
    const workflowData = state.workflowData;
    const origKey = NodenameByType[wv.previewTarget] + '_ORIGINAL';

    if (workflowData[origKey] !== undefined) {
        state.targetSelValues = workflowData[origKey];
        if (state.targetSelValues.length === 0) {
            state.saveIsValid = false;
            state.targetSelValues = ['ERROR: Cannot list target for ' + wv.previewTarget];
        }
    } else {
        state.saveIsValid = false;
        state.targetSelValues = ['ERROR: Cannot list target for ' + wv.previewTarget];
    }
    return state.targetSelValues;
}

function ButtonLabelCreator(node, url = false) {
    const state = ensureNodeState(node);
    state.previewExist = false;
    state.previewURL = null;

    const wv = getWidgetValues(node);

    state.targetSelValues = TargetListCreator(node);
    if (typeof state.targetSelValues === "object") {
        let targetIndexChanged = 0;
        if (state.targetSelValues.includes(state.selectedTarget)) {
            targetIndexChanged = state.targetSelValues.indexOf(state.selectedTarget);
        }
        state.selectedTarget = state.targetSelValues[targetIndexChanged];
    }

    const imgTypeString = wv.imgType.toUpperCase() + ' format';
    const imgSizeString = wv.maxSide < 64 ? "at original size" : "resized to " + wv.maxSide + 'px';
    const targetQuality = wv.imgType === 'png' ? 100 : wv.targetQuality;

    state.targetFileName = null;
    state.saveIsValid = false;
    const workflowData = state.workflowData;

    if (Object.keys(workflowData).length < 1) {
        if (wv.saveMode === true) {
            applyWidgetValues(node, state.selectedTarget, state.targetSelValues);
        } else {
            state.saveIsValid = true;
            if (url !== false) {
                getMeta(url).then(img => {
                    if (img) state.origSizeString = '[' + img.naturalHeight + ' X ' + img.naturalWidth + ']';
                    const title = '💾 Save image as ' + imgTypeString + ' | ' + state.origSizeString + ' ' + imgSizeString + ' | QTY: ' + targetQuality + '%';
                    applyWidgetValues(node, title, state.targetSelValues);
                });
            } else {
                applyWidgetValues(node, '⛔ Image not available for save. Please load one.', state.targetSelValues);
            }
        }
    } else {
        if (wv.saveMode === true) {
            if (url !== false) {
                getMeta(url).then(img => {
                    if (img) state.origSizeString = '[' + img.naturalHeight + ' X ' + img.naturalWidth + ']';
                });
            }

            const nodeName = NodenameByType[wv.previewTarget];
            if (workflowData[nodeName] !== undefined && state.selectedTarget !== undefined) {
                if (workflowData[nodeName].length < 1) {
                    const title = '❌ No resource selected for preview target: [' + wv.previewTarget + ']';
                    applyWidgetValues(node, title, state.targetSelValues);
                } else {
                    state.saveIsValid = true;
                    const origKey = nodeName + '_ORIGINAL';
                    const targetIndex = workflowData[origKey].indexOf(state.selectedTarget);
                    if (targetIndex > -1) {
                        state.targetFileName = workflowData[nodeName][targetIndex];
                    }

                    let prwpathNew = state.selectedTarget.replaceAll('\\', '/');
                    const dotLastIndex = prwpathNew.lastIndexOf('.');
                    const finalName = (dotLastIndex > 1 ? prwpathNew.substring(0, dotLastIndex) : prwpathNew).replaceAll(' ', '_');
                    const imgsrc = prwPath + '/images/' + NodesubdirByType[wv.previewTarget] + '/' + finalName + '.jpg';

                    fetch(imgsrc, { method: 'HEAD' })
                        .then(response => {
                            state.previewExist = response.ok;
                            state.previewURL = response.ok ? imgsrc : null;

                            const prwMode = wv.prwSaveMode.split(' ').map(n => n[0]).join('').toUpperCase();
                            const imgExistLink = state.previewExist ? ' [' + prwMode + ']' : ' [C]';
                            const title = '🏙️ Save preview as: [' + state.targetFileName + '.jpg] to [' + wv.previewTarget + '] folder.' + imgExistLink;
                            applyWidgetValues(node, title, state.targetSelValues);
                        })
                        .catch(() => {
                            state.previewExist = false;
                            state.previewURL = null;
                            const title = '🏙️ Save preview as: [' + state.targetFileName + '.jpg] to [' + wv.previewTarget + '] folder. [C]';
                            applyWidgetValues(node, title, state.targetSelValues);
                        });
                }
            } else {
                const title = '❌ Required node: [' + NodenameByType[wv.previewTarget] + '] not available in workflow for target: [' + wv.previewTarget + ']';
                applyWidgetValues(node, title, state.targetSelValues);
            }
        } else {
            state.saveIsValid = true;
            if (url !== false) {
                getMeta(url).then(img => {
                    if (img) state.origSizeString = '[' + img.naturalHeight + ' X ' + img.naturalWidth + ']';
                    const title = '💾 Save image as ' + imgTypeString + ' | ' + state.origSizeString + ' ' + imgSizeString + ' | QTY: ' + targetQuality + '%';
                    applyWidgetValues(node, title, state.targetSelValues);
                });
            } else {
                const title = '💾 Save image as ' + imgTypeString + ' | ' + state.origSizeString + ' ' + imgSizeString + ' | QTY: ' + targetQuality + '%';
                applyWidgetValues(node, title, state.targetSelValues);
            }
        }
    }
}

function applyWidgetValues(node, buttontitle, targetSelValues) {
    const state = ensureNodeState(node);
    for (const w of node.widgets) {
        if (w.type == 'button') {
            w.name = buttontitle;
        }
        if (w.name == 'target_selection') {
            w.options.values = targetSelValues;
            if (targetSelValues.length > 0) {
                let targetIndexChanged = 0;
                if (targetSelValues.includes(state.selectedTarget)) {
                    targetIndexChanged = targetSelValues.indexOf(state.selectedTarget);
                }
                w.value = targetSelValues[targetIndexChanged];
                state.selectedTarget = w.value;
            }
        }
    }
    app.canvas.setDirty(true);
}

// ── POST save ─────────────────────────────────────────────────────────────────

function sendPOSTmessage(message) {
    const body = new FormData();
    body.append('previewdata', message);
    api.fetchApi("/primere_preview_post", {method: "POST", body,});
}

api.addEventListener("PreviewSaveResponse", PreviewSaveResponse);
function PreviewSaveResponse(event) {
    alert(event.detail);
    if (pendingSaveNode) {
        ButtonLabelCreator(pendingSaveNode);
        pendingSaveNode = null;
    }
}

// ── getMeta ───────────────────────────────────────────────────────────────────

const getMeta = (url) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = url;
}).catch(function() {
    return false;
});
