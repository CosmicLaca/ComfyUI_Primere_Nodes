import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const realPath = "extensions/Primere";
const prwPath = "extensions/PrimerePreviews";
const validClasses = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualStyle', 'PrimereVisualLYCORIS'];
const versionClasses = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualLYCORIS', 'PrimereVisualEmbedding']
let lastDirObject = {};
let currentClass = false;
let hiddenWidgets = {};
let checkpointVersions = false;
const SDXLVersionTags = ['SDXL_2048', 5120, 1280, 'SDXL'];
let networkVersions = {};

function createCardElement(checkpoint, container, SelectedModel, ModelType) {
    let checkpoint_new = checkpoint.replaceAll('\\', '/');
    let dotLastIndex = checkpoint_new.lastIndexOf('.');
    if (dotLastIndex > 1) {
        var finalName = checkpoint_new.substring(0, dotLastIndex);
    } else {
        var finalName = checkpoint_new;
    }

    let pathLastIndex = finalName.lastIndexOf('/');
    let ckptName = finalName.substring(pathLastIndex + 1);
    let versionWidget = '';

    if (checkpointVersions !== false && checkpointVersions != null) {
        if (checkpointVersions.hasOwnProperty(ckptName) === true) {
            var versionString;
            var titleText;
            if (SDXLVersionTags.includes(checkpointVersions[ckptName])) {
                titleText = 'SDXL checkpoint. Select right version of additional networks.';
                versionString = 'sdxl';
            } else if (checkpointVersions[ckptName] == 'Unknown') {
                titleText = 'Unknown checkpoint version. You can enter right version manually to the version cache file. README.md helps.';
                versionString = 'unknown';
            } else {
                titleText = 'SD1.5 checkpoint. Select right version of additional networks.';
                versionString = 'sd1';
            }
            versionWidget = '<div class="ckpt-version ' + versionString + '-ckpt" title="' + titleText + '"></div>';
        }
    }

    finalName = finalName.replaceAll(' ', "_");
    let previewName = finalName + '.jpg';

    var card_html = '<div class="checkpoint-name background-' + versionString + '">' + ckptName.replaceAll('_', " ") + '</div>' + versionWidget;
    var imgsrc = prwPath + '/images/' + ModelType + '/' + previewName;
    var missingimgsrc = prwPath + '/images/missing.jpg';

    let supportedImageExtensions = ['.preview.jpg', '.jpeg', '.preview.jpeg', '.png', '.preview.png'];
    let alternativeImgSources = []
    for (let ending of supportedImageExtensions) {
        var alternativeImgSrc = prwPath + '/images/' + ModelType + '/' + finalName + ending;
        alternativeImgSources.push(alternativeImgSrc);
    }
    let currentAttempt = 0;

	var card = document.createElement("div");
	card.classList.add('visual-ckpt', 'version-' + versionString);
    if (SelectedModel === checkpoint) {
        card.classList.add('visual-ckpt-selected');
    }

    const img = new Image();
    img.onload = () => {
        const width = img.width;
        if (width > 0) {
            card_html += '<img src="' + img.src + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '">';
            card.innerHTML = card_html;
            container.appendChild(card);
            // console.log('Image loaded successfully with image source: ' + img.src + ' ...')
        }
        currentAttempt = 0;
    };

    img.onerror = () => {
        // console.error('Image error detected with image source: ' + img.src + '. Attempting alternative image sources');
        if (currentAttempt < alternativeImgSources.length) {
            currentAttempt++;
            img.src = alternativeImgSources[currentAttempt - 1];
        } else {
            card_html += '<img src="' + missingimgsrc + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '">';
            card.innerHTML = card_html;
            container.appendChild(card);
            currentAttempt = 0;
        }
        // card_html += '<img src="' + missingimgsrc + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '">';
        // card.innerHTML = card_html;
        // container.appendChild(card);
    };
    img.src = imgsrc;
}


app.registerExtension({
    name: "Primere.VisualMenu",

    init() {
        /* Promise.all([
          fetch('extensions/Primere/keywords/lora-keyword.txt').then(x => x.text()),
          fetch('extensions/Primere/keywords/model-keyword.txt').then(x => x.text())
        ]).then(([Lora, Model]) => {
          console.log(Lora);
          console.log(Model);
        }); */

        let callbackfunct = null;
        function ModalHandler() {
            let head = document.getElementsByTagName('HEAD')[0];
            let link = document.createElement('link');
            link.rel = 'stylesheet';
            link.type = 'text/css';
            link.href = realPath + '/css/visual.css';
            head.appendChild(link);

            let js = document.createElement("script");
            js.src = realPath + "/jquery/jquery-1.9.0.min.js";
            head.appendChild(js);

            js.onload = function(e) {
                $(document).ready(function () {
                    var modal = null;
                    $('body').on("click", 'button.modal-closer', function() {
                        modal = document.getElementById("primere_visual_modal");
                        modal.setAttribute('style','display: none; width: 80%; height: 70%;')
                    });

                    $('body').on("click", 'div.primere-modal-content div.visual-ckpt img', function() {
                        var ckptName = $(this).data('ckptname');
                        modal = document.getElementById("primere_visual_modal");
                        modal.setAttribute('style','display: none; width: 80%; height: 70%;')
                        apply_modal(ckptName);
                    });

                    var subdirName ='All';
                    var filteredCheckpoints = 0;
                    $('body').on("click", 'div.subdirtab button.subdirfilter', function() {
                        $('div.subdirtab input').val('');
                        subdirName = $(this).data('ckptsubdir');
                        if (currentClass !== false) {
                            lastDirObject[currentClass] = subdirName
                        }

                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).find('img').each(function (img_index, img_obj) {
                            var ImageCheckpoint = $(img_obj).data('ckptname');
                            if (subdirName === 'Root') {
                                let isSubdirExist = ImageCheckpoint.lastIndexOf('\\');
                                if (isSubdirExist > 1 && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                    $(img_obj).parent().hide();
                                } else {
                                    $(img_obj).parent().show();
                                    filteredCheckpoints++;
                                }
                            } else {
                                if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                    $(img_obj).parent().hide();
                                } else {
                                    $(img_obj).parent().show();
                                    filteredCheckpoints++;
                                }
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-name').text(subdirName);
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        $('div.subdirtab button').removeClass("selected_path");
                        $(this).addClass('selected_path');
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });

                    $('body').on("click", 'div.subdirtab button.verfilter', function() {
                        $('div.subdirtab input').val('');
                        var versionName = $(this).data('ckptver').toLowerCase();
                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).each(function () {
                            if ($(this).hasClass('version-' + versionName)) {
                                $(this).show();
                                filteredCheckpoints++;
                            } else {
                                $(this).hide();
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-name').text(versionName);
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints);
                        $('div.subdirtab button').removeClass("selected_path");
                        $(this).addClass('selected_path');
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });

                    $('body').on("keyup", 'div.subdirtab input', function() {
                        var filter = $(this).val();
                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).find('img').each(function (img_index, img_obj) {
                            var ImageCheckpoint = $(img_obj).data('ckptname');
                            let dotLastIndex = ImageCheckpoint.lastIndexOf('.');
                            if (dotLastIndex > 1) {
                                var finalFilter = ImageCheckpoint.substring(0, dotLastIndex);
                            } else {
                                var finalFilter = ImageCheckpoint;
                            }
                            if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                if (finalFilter.toLowerCase().indexOf(filter.toLowerCase()) >= 0 || $(img_obj).parent().closest(".visual-ckpt-selected").length > 0) {
                                    $(img_obj).parent().show();
                                    filteredCheckpoints++;
                                } else {
                                    $(img_obj).parent().hide();
                                }
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });

                    $('body').on("click", 'div.subdirtab button.filter_clear', function() {
                        $('div.subdirtab input').val('');
                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).find('img').each(function (img_index, img_obj) {
                            var ImageCheckpoint = $(img_obj).data('ckptname');
                            if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                $(img_obj).parent().show();
                                filteredCheckpoints++;
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });
                });
            };
        }

        function apply_modal(Selected) {
            if (Selected && typeof callbackfunct == 'function') {
                callbackfunct(Selected);
                sendPOSTModelName(Selected);
                return false;
            }
        }

        function setup_visual_modal(combo_name, AllModels, ShowHidden, SelectedModel, ModelType, node) {
            var container = null;
            var modal = null;
            var modalExist = true;

            modal = document.getElementById("primere_visual_modal");
            if (!modal) {
                modalExist = false;
				modal = document.createElement("div");
				modal.classList.add("comfy-modal");
				modal.setAttribute("id","primere_visual_modal");
				modal.innerHTML='<div class="modal_header"><button type="button" class="modal-closer">Close modal</button> <h3 class="visual_modal_title">' + combo_name.replace("_"," ") + ' :: <label class="ckpt-name">All</label> :: <label class="ckpt-counter"></label></h3></div>';

                let subdir_container = document.createElement("div");
                subdir_container.classList.add("subdirtab");

				let container = document.createElement("div");
				container.classList.add("primere-modal-content", "ckpt-container", "ckpt-grid-layout");
                modal.appendChild(subdir_container);
				modal.appendChild(container);

				document.body.appendChild(modal);
			} else {
                $('div#primere_visual_modal div.modal_header h3.visual_modal_title').html(combo_name.replace("_"," ") + ' :: <label class="ckpt-name">All</label> :: <label class="ckpt-counter"></label>');
            }

            container = modal.getElementsByClassName("ckpt-container")[0];
			container.innerHTML = "";

            var subdirArray = ['All'];
            for (var checkpoints of AllModels) {
                let pathLastIndex = checkpoints.lastIndexOf('\\');
                let ckptSubdir = checkpoints.substring(0, pathLastIndex);
                if (ckptSubdir === '') {
                    ckptSubdir = 'Root';
                }
                if (subdirArray.indexOf(ckptSubdir) === -1) {
                    subdirArray.push(ckptSubdir);
                }
            }

            var subdir_tabs = modal.getElementsByClassName("subdirtab")[0];
            var menu_html = '';
            var version_html = '';

            for (var subdir of subdirArray) {
                var addWhiteClass = '';
                let firstletter = subdir.charAt(0);
                var subdirName = subdir;
                if (firstletter === '.') {
                    subdirName = subdir.substring(1);
                }
                if ((firstletter === '.' && ShowHidden === true) || firstletter !== '.') {
                    if (lastDirObject.hasOwnProperty(currentClass) === true && subdir === lastDirObject[currentClass]) {
                        addWhiteClass = ' selected_path';
                    } else if (lastDirObject.hasOwnProperty(currentClass) === false && subdir === 'All') {
                        addWhiteClass = ' selected_path';
                    }
                    menu_html += '<button type="button" data-ckptsubdir="' + subdir + '" class="subdirfilter' + addWhiteClass + '">' + subdirName + '</button>';
                }
            }

            if (checkpointVersions !== false && checkpointVersions != null) {
                //console.log(checkpointVersions);
                $.each(checkpointVersions, function(ver_index, ver_value) {
                    var addWhiteClass = '';
                    var versionName = 'Unknown'
                    if (SDXLVersionTags.includes(ver_value)) {
                        versionName ='SDXL'
                    } else if (!SDXLVersionTags.includes(ver_value) && ver_value !== 'Unknown') {
                        versionName ='SD1'
                    }
                    if (!version_html.includes('data-ckptver="' + versionName + '"')) {
                        version_html += '<button type="button" data-ckptver="' + versionName + '" class="verfilter' + addWhiteClass + '">' + versionName + '</button>';
                    }
                });
                version_html += '<label> | </label>';
            }

            subdir_tabs.innerHTML = menu_html + '<label> | </label> ' + version_html + ' <input type="text" name="ckptfilter" placeholder="filter"> <button type="button" class="filter_clear">Clear filter</button>';

            var CKPTElements = 0;
            for (var checkpoint of AllModels) {
                let firstletter = checkpoint.charAt(0);
                if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && ShowHidden === true) || !checkpoint.match('^NSFW')))  {
                    CKPTElements++;
                    createCardElement(checkpoint, container, SelectedModel, ModelType)
                }
            }


            var TimingBase =AllModels.length;
            var mtimeout = TimingBase * 5;
            if (modalExist === false) {
                mtimeout = TimingBase * 10;
            }

            setTimeout(function(mtimeout) {
                $('div.subdirtab input').val('');
                if (lastDirObject.hasOwnProperty(currentClass)) {
                    subdirName = lastDirObject[currentClass];
                    var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                    var filteredCheckpoints = 0;
                    $(imageContainers).find('img').each(function (img_index, img_obj) {
                        var ImageCheckpoint = $(img_obj).data('ckptname');
                        if (subdirName === 'Root') {
                            let isSubdirExist = ImageCheckpoint.lastIndexOf('\\');
                            if (isSubdirExist > 1 && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                $(img_obj).parent().show();
                                filteredCheckpoints++;
                            }
                        } else {
                            if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                $(img_obj).parent().show();
                                filteredCheckpoints++;
                            }
                        }
                    });
                    $('div#primere_visual_modal div.modal_header label.ckpt-name').text(subdirName);
                    $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                } else {
                    $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(CKPTElements - 1);
                }

                $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                modal.setAttribute('style','display: block; width: 80%; height: 70%;');
            }, mtimeout);

        }

        ModalHandler();

        let subdirname = '';
        let modaltitle = '';
        let nodematch = '';
        let isnumeric_end = true;
        const lcg = LGraphCanvas.prototype.processNodeWidgets;

        LGraphCanvas.prototype.processNodeWidgets = function(node, pos, event, active_widget) {
            //console.log(node);
            if (!validClasses.includes(node.type)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (node.type == 'PrimereVisualCKPT') {
                subdirname = 'checkpoints';
                modaltitle = 'Select checkpoint';
                nodematch = '^base_model';
                isnumeric_end = false;
                checkpointVersions = false;
                if (hiddenWidgets.hasOwnProperty('PrimereVisualCKPT') === true) {
                    $.each(hiddenWidgets['PrimereVisualCKPT'], function(index, value) {
                        if (index == 'cached_model') {
                            checkpointVersions = value[0];
                        }
                    });
                }
            }

            if (node.type == 'PrimereVisualLORA') {
                subdirname = 'loras';
                modaltitle = 'Select LoRA';
                nodematch = '^lora_';
                isnumeric_end = true;
                checkpointVersions = false;
                if (hiddenWidgets.hasOwnProperty('PrimereVisualLORA') === true) {
                    $.each(hiddenWidgets['PrimereVisualLORA'], function(index, value) {
                        if (index == 'cached_lora') {
                            checkpointVersions = value[0];
                        }
                    });
                }
            }

            if (node.type == 'PrimereVisualEmbedding') {
                subdirname = 'embeddings';
                modaltitle = 'Select embedding';
                nodematch = '^embedding_';
                isnumeric_end = true;
                checkpointVersions = false;
                if (hiddenWidgets.hasOwnProperty('PrimereVisualEmbedding') === true) {
                    $.each(hiddenWidgets['PrimereVisualEmbedding'], function(index, value) {
                        if (index == 'cached_embedding') {
                            checkpointVersions = value[0];
                        }
                    });
                }
            }

            if (node.type == 'PrimereVisualHypernetwork') {
                subdirname = 'hypernetworks';
                modaltitle = 'Select hypernetwork';
                nodematch = '^hypernetwork_';
                isnumeric_end = true;
                checkpointVersions = false;
            }

            if (node.type == 'PrimereVisualStyle') {
                subdirname = 'styles';
                modaltitle = 'Select style';
                nodematch = '^styles';
                isnumeric_end = false;
                checkpointVersions = false;
            }

            if (node.type == 'PrimereVisualLYCORIS') {
                subdirname = 'lycoris';
                modaltitle = 'Select LYCORIS';
                nodematch = '^lycoris_';
                isnumeric_end = true;
                checkpointVersions = false;
                if (hiddenWidgets.hasOwnProperty('PrimereVisualLYCORIS') === true) {
                    $.each(hiddenWidgets['PrimereVisualLYCORIS'], function(index, value) {
                        if (index == 'cached_lyco') {
                            checkpointVersions = value[0];
                        }
                    });
                }
            }

            if (event.type != LiteGraph.pointerevents_method + "down") {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            currentClass = node.type;

            var x = pos[0] - node.pos[0];
            var y = pos[1] - node.pos[1];
            var width = node.size[0];
            var that = this;

            var ShowHidden = false;
            var ShowModal = false;

            for (var p = 0; p < node.widgets.length; ++p) {
                if (node.widgets[p].name == 'show_hidden') {
                    ShowHidden = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'show_modal') {
                    ShowModal = node.widgets[p].value;
                }
            }

            if (ShowModal === false) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            for (var i = 0; i < node.widgets.length; ++i) {
                var w = node.widgets[i];
                if (!w || w.disabled)
                    continue;

                if (w.type != "combo")
                    continue

                var widget_height = w.computeSize ? w.computeSize(width)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                var widget_width = w.width || width;
                var widget_name = node.widgets[i].name;

                if (w != active_widget && (x < 6 || x > widget_width - 12 || y < w.last_y || y > w.last_y + widget_height || w.last_y === undefined))
                    continue

                if (w == active_widget || (x > 6 && x < widget_width - 12 && y > w.last_y && y < w.last_y + widget_height)) {
                    var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                    if (delta)
                        continue;

                    if (widget_name.match(nodematch) && $.isNumeric(widget_name.substr(-1)) === isnumeric_end) {
                        var AllModels = node.widgets[i].options.values;
                        var SelectedModel = node.widgets[i].value;

                        callbackfunct = inner_clicked.bind(w);
                        setup_visual_modal(modaltitle, AllModels, ShowHidden, SelectedModel, subdirname, node);

                        function inner_clicked(v, option, event) {
                            inner_value_change(this, v);
                            that.dirty_canvas = true;
                            return false;
                        }

                        function inner_value_change(widget, value) {
                            if (widget.type == "number") {
                                value = Number(value);
                            }
                            widget.value = value;
                            if (widget.options && widget.options.property && node.properties[widget.options.property] !== undefined) {
                                node.setProperty(widget.options.property, value);
                            }
                            if (widget.callback) {
                                widget.callback(widget.value, that, node, pos, event);
                            }
                        }
                        return null;
                    }
                }
            }
            return lcg.call(this, node, pos, event, active_widget);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (versionClasses.includes(nodeData.name)) {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                hiddenWidgets[nodeData.name] = nodeData.input.hidden;
            }
            //console.log(hiddenWidgets);
        }
    },
});

function sendPOSTModelName(modelName) {
    const body = new FormData();
    body.append('modelName', modelName);
    api.fetchApi("/primere_keyword_parser", {method: "POST", body,});
}
