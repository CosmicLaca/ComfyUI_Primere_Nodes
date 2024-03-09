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
  },
    async setup(app) {

    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimerePreviewImage") {
        if (nodeData.input.hasOwnProperty('hidden') === true) {
            ImagePath = nodeData.input.hidden['image_path'][0]
        }
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
        //const getBase64StringFromDataURL = (dataURL) => dataURL.replace('data:', '').replace(/^.+,/, '');
        fetch(ImageSource)
        .then((res) => res.blob())
        .then((blob) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                //const base64 = getBase64StringFromDataURL(reader.result);
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