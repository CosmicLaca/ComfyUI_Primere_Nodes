import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let eventListenerInit = false;
const realPath = "/extensions/ComfyUI_Primere_Nodes"; //"extensions/Primere";
const prwPath = "extensions/PrimerePreviews";
const validClasses = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualStyle', 'PrimereVisualLYCORIS', 'PrimereVisualPromptOrganizerCSV'];
const stackedClasses = ['PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualLYCORIS'];
let widget_name = "";
let lastDirObject = {};
let currentClass = false;
let hiddenWidgets = {};
let ShowHidden = false;
let FilterType = 'Subdir';
let SelectedModel = 'SelectedModel';
let sortType = 'name';
let operator = 'ASC';
let PreviewPath = true;
let aeScoreMin = 400;
let aeScoreMax = 900;
let nodeHelper = {};
let cache_key = '';
let source_subdirname = '';
let AutoFilter = false;
let StackVersion = 'Auto';
let ModelVersion = 'SD1';

// API
let SetupCacheData = {};
let VersionCacheData = {};
let supportedModels = [];
let ModelsByVersion = {};
let AllPath = [];
let ModelList = {};
let AscoreDataResponse = {};
let STimeDataResponse = {};
let FileDateResponse = {};
let FileLinkResponse = {};
let RawImageDataResponse = {};

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
        let modaltitle = '';
        let nodematch = '';
        let isnumeric_end = false;

        const lcg = LGraphCanvas.prototype.processNodeWidgets;
        LGraphCanvas.prototype.processNodeWidgets = function (node, pos, event, active_widget) { // 01
            if (event.type == 'pointermove' && validClasses.includes(node.type)) {
                return false;
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (event.type != LiteGraph.pointerevents_method + "down") {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!validClasses.includes(node.type)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (stackedClasses.includes(node.type)) {
                isnumeric_end = true;
            } else {
                isnumeric_end = false;
            }

            currentClass = node.type;

            var x = pos[0] - node.pos[0];
            var y = pos[1] - node.pos[1];
            var width = node.size[0];
            var that = this;
            var ShowModal = false;

            AutoFilter = null;
            StackVersion = null;
            ModelVersion = null;

            for (var p = 0; p < node.widgets.length; ++p) {
                if (node.widgets[p].name == 'show_hidden') {
                    ShowHidden = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'show_modal') {
                    ShowModal = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'preview_path') {
                    PreviewPath = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'aescore_percent_min') {
                    if (node.widgets[p].value >= 0) {
                        aeScoreMin = node.widgets[p].value;
                    }
                }
                if (node.widgets[p].name == 'aescore_percent_max') {
                    if (node.widgets[p].value > 0) {
                        aeScoreMax = node.widgets[p].value;
                    }
                }
                if (node.widgets[p].name == 'auto_filter') {
                    AutoFilter = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'stack_version') {
                    StackVersion = node.widgets[p].value;
                }
                if (node.widgets[p].name == 'model_version') {
                    ModelVersion = node.widgets[p].value;
                }
            }

            if (ShowModal === false) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            nodeHelper = hiddenWidgets[currentClass]
            source_subdirname = nodeHelper['subdir'];
            cache_key = nodeHelper['cache_key'];

            if (eventListenerInit == false) {
                ModalHandler();
            }

            if ($('div.primere-modal-content div.visual-ckpt').length) {
                $('div.primere-modal-content.ckpt-container').empty();
            }

            for (var i = 0; i < node.widgets.length; ++i) {
                var w = node.widgets[i];
                if (!w || w.disabled)
                    continue;

                if (w.type != "combo")
                    continue;

                var widget_height = w.computeSize ? w.computeSize(width)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                var widget_width = w.width || width;
                widget_name = node.widgets[i].name;

                if (w != active_widget && (x < 6 || x > widget_width - 12 || y < w.last_y || y > w.last_y + widget_height || w.last_y === undefined))
                    continue

                if (w == active_widget || (x > 6 && x < widget_width - 12 && y > w.last_y && y < w.last_y + widget_height)) {
                    var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
                    if (delta)
                        continue;

                    if (widget_name.match(nodematch) && $.isNumeric(widget_name.substr(-1)) === isnumeric_end) {
                        SelectedModel = node.widgets[i].value;
                        setup_visual_modal(modaltitle, 'AllModels', ShowHidden, SelectedModel, source_subdirname, node, PreviewPath);

                        callbackfunct = inner_clicked.bind(w);
                        function inner_clicked(v, option, event) {
                            inner_value_change(this, v);
                            that.dirty_canvas = true;
                            return false;
                        }

                        function inner_value_change(widget, value) {
                            if (typeof nodeHelper['sortbuttons'] === "object" && typeof nodeHelper['sortbuttons'][0] === "object" && nodeHelper['sortbuttons'][0].indexOf("Path") == -1) {
                                for (var i = 0; i < node.widgets.length; ++i) {
                                    var widget_type = node.widgets[i].type;
                                    var widget_value = node.widgets[i].value;
                                    if (widget_type == 'combo' && widget_value != 'None') {
                                        widget_name = node.widgets[i].name;
                                        node.widgets[i].value = 'None';
                                    }
                                }
                            }
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

            function ModalHandler() { // 02
                eventListenerInit = true;
                let head = document.getElementsByTagName('HEAD')[0];
                //let link = document.createElement('link');
                //link.rel = 'stylesheet';
                //link.type = 'text/css';
                //link.href = realPath + '/css/visual.css';
                //head.appendChild(link);

                let js = document.createElement("script");
                js.src = realPath + "/jquery/jquery-1.9.0.min.js";
                head.appendChild(js);

                js.onload = function(e) {
                    $(document).ready(function () {
                        var modal = null;

                        $('body').on("click", 'button.modal-closer', async function () { // modal close
                            modal = document.getElementById("primere_visual_modal");
                            modal.setAttribute('style', 'display: none; width: 80%; height: 70%;')
                            var lastDirValue = 'All'
                            if (lastDirObject.hasOwnProperty(currentClass) === true) {
                                lastDirValue = lastDirObject[currentClass];
                            }

                            if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
                                if (AutoFilter !== true) {
                                    await categoryHandler(lastDirValue, 'add', 'last_visual_category' + '_' + cache_key);
                                    await categoryHandler(FilterType, 'add', 'last_visual_category_type' + '_' + cache_key);
                                    var filter = $('body div.subdirtab input').val();
                                    await categoryHandler(filter, 'add', 'last_visual_filter' + '_' + cache_key);
                                    await categoryHandler(sortType, 'add', 'last_visual_sort_type' + '_' + cache_key);
                                    await categoryHandler(operator, 'add', 'last_visual_sort_operator' + '_' + cache_key);
                                }
                            }
                        });

                        $('body').on("click", 'div.primere-modal-content div.visual-ckpt img', async function () { // image choosen
                            var ckptName = $(this).data('ckptname');
                            modal = document.getElementById("primere_visual_modal");
                            modal.setAttribute('style', 'display: none; width: 80%; height: 70%;')
                            var lastDirValue = 'All'
                            if (lastDirObject.hasOwnProperty(currentClass) === true) {
                                lastDirValue = lastDirObject[currentClass];
                            }

                            if (source_subdirname == 'styles') {
                                let pathLastIndex = ckptName.lastIndexOf('\\');
                                ckptName = ckptName.substring(pathLastIndex + 1);
                            }

                            apply_modal(ckptName);

                            if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
                                if (AutoFilter !== true) {
                                    await categoryHandler(lastDirValue, 'add', 'last_visual_category' + '_' + cache_key);
                                    await categoryHandler(FilterType, 'add', 'last_visual_category_type' + '_' + cache_key);
                                    var filter = $('body div.subdirtab input').val();
                                    await categoryHandler(filter, 'add', 'last_visual_filter' + '_' + cache_key);
                                    await categoryHandler(sortType, 'add', 'last_visual_sort_type' + '_' + cache_key);
                                    await categoryHandler(operator, 'add', 'last_visual_sort_operator' + '_' + cache_key);
                                }
                            }
                        });

                        var subdirName = 'All';
                        var filteredCheckpoints = 0;
                        $('body').on("click", 'div.subdirtab button.subdirfilter', async function () { // subdir filter
                            $('div.subdirtab input').val('');
                            subdirName = $(this).data('ckptsubdir');
                            if (currentClass !== false) {
                                lastDirObject[currentClass] = subdirName
                            }

                            var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                            filteredCheckpoints = 0;
                            $(imageContainers).each(function (cont_index, cont_obj) {
                                if ($(cont_obj).find('img').parent().closest(".visual-ckpt-selected").length === 0) {
                                    cont_obj.remove();
                                }
                            });

                            if (source_subdirname != 'styles') {
                                ModelsByVersion = await getModelData(cache_key + '_version');
                            }
                            ModelList = await getModelDatabyPath(source_subdirname, subdirName);

                            if (source_subdirname == 'styles') {
                                if (ModelList.includes(SelectedModel) || ModelList.includes(subdirName + '\\' + SelectedModel)) {
                                    var index_pre = (ModelList.indexOf(SelectedModel) + ModelList.indexOf(subdirName + '\\' + SelectedModel)) + 1;
                                    if (index_pre !== -1) {
                                        await ModelList.splice(index_pre, 1);
                                    }
                                }
                            }

                            for (var checkpoint of ModelList) {
                            //$(ModelList).each(async function (checkpoint_index, checkpoint) {
                                let firstletter = checkpoint.charAt(0);

                                var filterpass = true;
                                if (SelectedModel == checkpoint) {
                                    filterpass = false;
                                }

                                if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && ShowHidden === true) || !checkpoint.match('^NSFW')) && filterpass == true) {
                                    let pathLastIndex = checkpoint.lastIndexOf('\\');
                                    let ckptName_full = checkpoint.substring(pathLastIndex + 1);
                                    let dotLastIndex = ckptName_full.lastIndexOf('.');
                                    var ckptName = ckptName_full.substring(0, dotLastIndex);
                                    var CategoryName = 'Unknown';

                                    $.each(ModelsByVersion, function (ver_index, ver_value) {
                                        if (ver_value.includes(ckptName)) {
                                            CategoryName = ver_index
                                        }
                                    });

                                    var container = $('div.primere-modal-content.ckpt-container')[0];
                                    if (subdirName === 'Root') {
                                        let isSubdirExist = checkpoint.lastIndexOf('\\');
                                        if (isSubdirExist < 0) {
                                            filteredCheckpoints++;
                                            await createCardElement(checkpoint, container, SelectedModel, source_subdirname, CategoryName)
                                        }
                                    } else {
                                        filteredCheckpoints++;
                                        await createCardElement(checkpoint, container, SelectedModel, source_subdirname, CategoryName)
                                    }
                                }
                                $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints);
                            }
                            //await sleep(600);
                            previewSorter(operator, sortType);

                            $('div#primere_visual_modal div.modal_header label.ckpt-name').text(subdirName);
                            $('div#primere_visual_modal div.modal_header label.ckpt-ver').text('Subdir');
                            $('div.subdirtab button').not('button.preview_sort').not('button.preview_sort_direction').removeClass("selected_path");
                            $(this).addClass('selected_path');
                            FilterType = 'Subdir';
                        });

                        $('body').on("click", 'div.subdirtab button.verfilter', async function () { // model version filter
                            $('div.subdirtab input').val('');
                            var versionName = $(this).data('ckptver');
                            if (currentClass !== false) {
                                lastDirObject[currentClass] = versionName
                            }
                            var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                            filteredCheckpoints = 0;
                            $(imageContainers).each(function (cont_index, cont_obj) {
                                if ($(cont_obj).find('img').parent().closest(".visual-ckpt-selected").length === 0) {
                                    cont_obj.remove();
                                }
                            });

                            ModelList = await getModelDatabyVersion(source_subdirname, cache_key + '_version', versionName);

                            for (var checkpoint of ModelList) {
                                let firstletter = checkpoint.charAt(0);
                                var filterpass = true;
                                if (SelectedModel == checkpoint) {
                                    filterpass = false;
                                }
                                if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && ShowHidden === true) || !checkpoint.match('^NSFW')) && filterpass == true) {
                                    filteredCheckpoints++;
                                    var container = $('div.primere-modal-content.ckpt-container')[0];
                                    await createCardElement(checkpoint, container, SelectedModel, source_subdirname, versionName)
                                }
                                $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints);
                            }

                            //await sleep(600);
                            previewSorter(operator, sortType);

                            $('div#primere_visual_modal div.modal_header label.ckpt-name').text(versionName);
                            $('div#primere_visual_modal div.modal_header label.ckpt-ver').text('Version');
                            $('div.subdirtab button').not('button.preview_sort').not('button.preview_sort_direction').removeClass("selected_path");
                            $(this).addClass('selected_path');
                            FilterType = 'Version';
                        });

                        $('body').on("keyup", 'div.subdirtab input', function() { // keyword filter
                            var filter = $(this).val();
                            previewFilter(filter);
                        });

                        $('body').on("click", 'div.subdirtab button.filter_clear', function() { // keyword inut clear
                            $('div.subdirtab input').val('');
                            var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                            filteredCheckpoints = 0;
                            $(imageContainers).find('img').each(function (img_index, img_obj) {
                                $(img_obj).parent().show();
                                filteredCheckpoints++;
                            });
                            $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        });

                        $('body').on("click", 'div.subdirtab button.preview_sort', function() { // preview sort type
                            sortType = $(this).data('sortsource');
                            operator = $('button.preview_sort_direction').text();
                            previewSorter(operator, sortType);
                            $('div.subdirtab button.preview_sort').removeClass("selected_path");
                            $(this).addClass('selected_path');
                        });

                        $('body').on("click", 'div.subdirtab button.preview_sort_direction', function () { // preview sort direction
                            operator = $('button.preview_sort_direction').text();
                            if (operator == 'ASC') {
                                $('button.preview_sort_direction').text('DESC');
                                operator = 'DESC';
                                $('div.subdirtab button.preview_sort_direction').addClass("selected_path");
                            } else {
                                $('button.preview_sort_direction').text('ASC');
                                operator = 'ASC';
                                $('div.subdirtab button.preview_sort_direction').removeClass("selected_path");
                            }
                            previewSorter(operator, sortType);
                        });
                    });
                };
            }

            function apply_modal(Selected) { // apply
                if (Selected && typeof callbackfunct == 'function') {
                    callbackfunct(Selected);
                    sendPOSTModelName(Selected);
                    return false;
                }
            }

            return lcg.call(this, node, pos, event, active_widget);
        }
    },

    async setup() {

    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) { // 0
        if (validClasses.includes(nodeData.name)) {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                hiddenWidgets[nodeData.name] = nodeData.input.hidden;
            }
        }
    },
});

async function setup_visual_modal(combo_name, AllModels, ShowHidden, SelectedModel, ModelType, node, PreviewPath) { //3
    var container = null;
    var modal = null;

    modal = document.getElementById("primere_visual_modal");
    if (!modal) {
        modal = document.createElement("div");
        modal.classList.add("comfy-modal");
        modal.setAttribute("id","primere_visual_modal");
        modal.innerHTML = '<div class="modal_header"><button type="button" class="modal-closer">Close modal</button> <h3 class="visual_modal_title">' + combo_name.replace("_"," ") + '<label class="ckpt-ver">Subdir</label> :: <label class="ckpt-name">All</label> :: <label class="ckpt-counter"></label></h3></div>';

        let subdir_container = document.createElement("div");
        subdir_container.classList.add("subdirtab");

        let container = document.createElement("div");
        container.classList.add("primere-modal-content", "ckpt-container", "ckpt-grid-layout");
        modal.appendChild(subdir_container);
        modal.appendChild(container);

        document.body.appendChild(modal);
    } else {
        $('div#primere_visual_modal div.modal_header h3.visual_modal_title').html(combo_name.replace("_"," ") + '<label class="ckpt-ver">Subdir</label> :: <label class="ckpt-name">All</label> :: <label class="ckpt-counter"></label>');
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
    var type_html = '';
    var version_html = '';

    if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1 && AutoFilter !== true) {
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
                }
                menu_html += '<button type="button" data-ckptsubdir="' + subdir + '" class="subdirfilter' + addWhiteClass + '">' + subdirName + '</button>';
            }
        }
    }

    var LastCat = 'All';
    var LastCatType = 'Subdir';
    var savedfilter = "";

    SetupCacheData = await getCacheByKey("setup")

    if (SetupCacheData != null && typeof SetupCacheData === "object") {
        if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
            if (SetupCacheData.hasOwnProperty("last_visual_category" + '_' + cache_key)) {
                LastCat = SetupCacheData["last_visual_category" + '_' + cache_key];
            }
            if (SetupCacheData.hasOwnProperty("last_visual_category_type" + '_' + cache_key)) {
                LastCatType = SetupCacheData["last_visual_category_type" + '_' + cache_key];
            }
            if (SetupCacheData.hasOwnProperty("last_visual_filter" + '_' + cache_key)) {
                savedfilter = SetupCacheData['last_visual_filter' + '_' + cache_key];
            }
            if (SetupCacheData.hasOwnProperty("last_visual_sort_type" + '_' + cache_key)) {
                sortType = SetupCacheData['last_visual_sort_type' + '_' + cache_key];
            }
            if (SetupCacheData.hasOwnProperty("last_visual_sort_operator" + '_' + cache_key)) {
                operator = SetupCacheData['last_visual_sort_operator' + '_' + cache_key];
            }
        }
    }

    if (AutoFilter === true && StackVersion != null && ModelVersion != null) {
        if (StackVersion == 'Auto') {
            LastCat = ModelVersion;
            LastCatType = 'Version';
        } else if (StackVersion != 'Any' && StackVersion != 'Auto') {
            LastCat = StackVersion;
            LastCatType = 'Version';
        }
    }

    if (typeof nodeHelper['sortbuttons'] === "object" && typeof nodeHelper['sortbuttons'][0] === "object" && nodeHelper['sortbuttons'][0].indexOf("Path") == -1) {
        LastCat = widget_name
    }

    supportedModels = [];
    ModelsByVersion = {};
    if (ModelType != 'styles') {
        supportedModels = await getSupportedModels();
        ModelsByVersion = await getModelData(cache_key + '_version');
        VersionCacheData = await getCacheByKey(cache_key + '_version')
    }

    AllPath = await getAllPath(source_subdirname);

    if (LastCatType == 'Subdir') {
        ModelList = await getModelDatabyPath(source_subdirname, LastCat);
    }
    if (LastCatType == 'Version') {
        ModelList = await getModelDatabyVersion(source_subdirname, cache_key + '_version', LastCat);
    }

    if (typeof nodeHelper['cache_key'] !== "undefined" && nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof nodeHelper['sortbuttons'] === "object" && typeof nodeHelper['sortbuttons'][0] === "object" && nodeHelper['sortbuttons'][0].length > 0) {
            if (nodeHelper['sortbuttons'][0].indexOf("aScore") > -1) {
                AscoreDataResponse = await ReadAScores(nodeHelper['cache_key']);
            }
        }
    }

    if (typeof nodeHelper['cache_key'] !== "undefined" && nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof nodeHelper['sortbuttons'] === "object" && typeof nodeHelper['sortbuttons'][0] === "object" && nodeHelper['sortbuttons'][0].length > 0) {
            if (nodeHelper['sortbuttons'][0].indexOf("STime") > -1) {
                STimeDataResponse = await ReadSTimes(nodeHelper['cache_key']);
            }
        }
    }

    if (typeof nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof nodeHelper['sortbuttons'] === "object" && typeof nodeHelper['sortbuttons'][0] === "object" && nodeHelper['sortbuttons'][0].length > 0) {
            if (nodeHelper['sortbuttons'][0].indexOf("Date") > -1) {
                FileDateResponse = await ReadFileDate(nodeHelper['subdir']);
            }
        }
    }

    if (nodeHelper['subdir'] == 'checkpoints' && nodeHelper['sortbuttons'][0].indexOf("Symlink") > -1) {
        FileLinkResponse = await ReadFileSymlink(nodeHelper['subdir']);
    }

    RawImageDataResponse = {}
    RawImageDataResponse = await modelImageData(source_subdirname, PreviewPath);

    if (ModelsByVersion !== false && ModelsByVersion != null && Object.keys(ModelsByVersion).length > 0 && source_subdirname != 'styles' && AutoFilter !== true) {
        version_html = createTypeMenu(ModelsByVersion, supportedModels, LastCat, LastCatType);
    }

    if (AllPath !== false && AllPath != null && AllPath.length > 0 && AutoFilter !== true) {
        if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
            type_html = createPathMenu(AllPath, ShowHidden, LastCat, LastCatType);
        }
    }

    subdir_tabs.innerHTML = menu_html + type_html + version_html + ' <input type="text" name="ckptfilter" placeholder="filter"> <button type="button" class="filter_clear">Clear filter</button>';

    var sortbuttons = nodeHelper['sortbuttons'];
    var sort_string = "";
    if (typeof sortbuttons === "object" && typeof sortbuttons[0] === "object" && sortbuttons[0].length > 0) {
        sort_string = createSortButtons(sortbuttons[0]);
    }
    subdir_tabs.innerHTML += sort_string;

    var fondModel = null;
    if (ModelList.includes(SelectedModel) || ModelList.includes(LastCat + '\\' + SelectedModel)) {
        var index_pre = (ModelList.indexOf(SelectedModel) + ModelList.indexOf(LastCat + '\\' + SelectedModel)) + 1;
        fondModel = ModelList[index_pre];
        if (index_pre !== -1) {
            await ModelList.splice(index_pre, 1);
        }
    }

    if (ModelType == 'styles' && fondModel != null) {
        SelectedModel = fondModel;
    }

    if (!ModelList.includes(SelectedModel)) {
        if (fondModel == null) {
            await ModelList.unshift(SelectedModel)
        } else {
            await ModelList.unshift(fondModel)
        }
    }

    $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(ModelList.length);
    $('div#primere_visual_modal div.modal_header label.ckpt-name').text(LastCat);
    $('div#primere_visual_modal div.modal_header label.ckpt-ver').text(LastCatType);

    var CKPTElements = 0;
    modal.setAttribute('style','display: block; width: 80%; height: 70%;');

    for (var checkpoint of ModelList) {
        let firstletter = checkpoint.charAt(0);
        let pathLastIndex = checkpoint.lastIndexOf('\\');
        let ckptName_full = checkpoint.substring(pathLastIndex + 1);
        let dotLastIndex = ckptName_full.lastIndexOf('.');
        var ckptName = ckptName_full.substring(0, dotLastIndex);

        if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && ShowHidden === true) || !checkpoint.match('^NSFW')))  {
            var CategoryName = 'Unknown';

            $.each(ModelsByVersion, function(ver_index, ver_value) {
                if (ver_value.includes(ckptName)) {
                    CategoryName = ver_index
                }
            });

            if (LastCat === 'Root') {
                let isSubdirExist = checkpoint.lastIndexOf('\\');
                if (isSubdirExist < 0 || checkpoint == SelectedModel) {
                    CKPTElements++;
                    await createCardElement(checkpoint, container, SelectedModel, ModelType, CategoryName)
                }
            } else {
                CKPTElements++;
                await createCardElement(checkpoint, container, SelectedModel, ModelType, CategoryName)
            }
        }
        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(CKPTElements);
    }

    if (savedfilter.length > 0) {
        $('body div.subdirtab input').val(savedfilter);
        previewFilter(savedfilter);
    }
    $('button.preview_sort_direction').text(operator);
    if (operator != 'ASC') {
        $('div.subdirtab button.preview_sort_direction').addClass("selected_path");
    }
    $('div.subdirtab button.preview_sort[data-sortsource="' + sortType + '"]').addClass('selected_path');
    previewSorter(operator, sortType);

    if (!$('#primere_visual_modal div.subdirtab button.selected_path').length) {
        $('#primere_visual_modal div.subdirtab button[data-ckptsubdir="All"]').addClass('selected_path');
    }
}

function previewSorter(operator, sortType) {
    if (operator == 'ASC') {
        $('div.primere-modal-content.ckpt-container.ckpt-grid-layout').find('div.visual-ckpt').not('.visual-ckpt-selected').sort(function (a, b) {
            //alert($(a).attr('data-' + sortType) + ' -> ' + isNaN(parseInt($(a).attr('data-' + sortType))));
            if (isNaN(parseInt($(a).attr('data-' + sortType))) && isNaN(parseInt($(b).attr('data-' + sortType)))) {
                return $(a).attr('data-' + sortType).toUpperCase() > $(b).attr('data-' + sortType).toUpperCase();
            } else {
                return parseInt($(a).attr('data-' + sortType)) > parseInt($(b).attr('data-' + sortType));
            }
        }).appendTo('div.primere-modal-content.ckpt-container.ckpt-grid-layout');
    } else {
        $('div.primere-modal-content.ckpt-container.ckpt-grid-layout').find('div.visual-ckpt').not('.visual-ckpt-selected').sort(function (a, b) {
            if (isNaN(parseInt($(a).attr('data-' + sortType))) && isNaN(parseInt($(b).attr('data-' + sortType)))) {
                return $(a).attr('data-' + sortType).toUpperCase() < $(b).attr('data-' + sortType).toUpperCase();
            } else {
                return parseInt($(a).attr('data-' + sortType)) < parseInt($(b).attr('data-' + sortType));
            }
        }).appendTo('div.primere-modal-content.ckpt-container.ckpt-grid-layout');
    }
}

function previewFilter(filterString) {
    var imageContainers = $('div.primere-modal-content div.visual-ckpt');
    var filteredCheckpoints = 0;
    $(imageContainers).find('img').each(function (img_index, img_obj) {
        var versiontext = $(img_obj).data('ckptver');
        var ImageCheckpoint = $(img_obj).data('ckptname') + '_' + versiontext;
        if (ImageCheckpoint.toLowerCase().indexOf(filterString.toLowerCase()) >= 0 || $(img_obj).parent().closest(".visual-ckpt-selected").length > 0) {
            $(img_obj).parent().show();
            filteredCheckpoints++;
        } else {
            $(img_obj).parent().hide();
        }
        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
    });
}

async function createCardElement(checkpoint, container, SelectedModel, ModelType, CategoryName) {
    var card = document.createElement("div");

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
    let aesthWidget = '';
    let stimeWidget = '';
    let symlinkWidget = '';
    var path_only = checkpoint.substring(0, checkpoint.indexOf("\\"));

    if (ckptName == 'None') {
        return;
    }

    if (CategoryName) {
        var titleText = CategoryName + ' checkpoint. Select right version of additional networks.';
        var versionString = CategoryName;
        if (ModelType != 'styles') {
            versionWidget = '<div class="ckpt-version ' + versionString + '-ckpt" title="' + titleText + '"></div>';
        } else {
            var SelModelIndex = SelectedModel.replaceAll(' ', "_");
            if (RawImageDataResponse.hasOwnProperty(SelectedModel.replaceAll(' ', "_")) === true && checkpoint == SelectedModel) {
                var selected_full = RawImageDataResponse[SelectedModel.replaceAll(' ', "_")];
                var selful_first = selected_full.charAt(0);
                var substringStart = 0;
                if (selful_first == '\\') {
                    substringStart = 1;
                }
                path_only = selected_full.substring(substringStart, selected_full.lastIndexOf("\\"));
            }

            if (path_only == "") {
                path_only = 'Root';
            }
            versionWidget = '<div class="ckpt-version" title="Saved prompt on ' + path_only + ' category">&nbsp' + path_only + '&nbsp</div>';
        }
    }

    finalName = finalName.replaceAll(' ', "_");
    finalName = finalName.substring(pathLastIndex + 1);

    var card_html = '<div class="checkpoint-name background-' + versionString + '">' + ckptName.replaceAll('_', " ") + '</div>' + versionWidget;
    var missingimgsrc = prwPath + '/images/missing.jpg';
    card.classList.add('visual-ckpt', 'version-' + versionString);

    if (FileDateResponse.hasOwnProperty(ckptName) === true) {
        var timestamp = FileDateResponse[ckptName];
        $(card).attr('data-date', timestamp);
    } else {
        $(card).attr('data-date', 0);
    }

    if (FileLinkResponse.hasOwnProperty(ckptName) === true) {
        var unetname = FileLinkResponse[ckptName];
        $(card).attr('data-symlink', unetname);
        var unetnameShort = unetname;
        if (unetname == 'diffusion_models') { unetnameShort = 'DiMo'}
        if (unetname == 'diffusers') { unetnameShort = 'Diff'}
        symlinkWidget = '<div class="visual-symlink-type" title="Symlinked from: ' + unetname + '">' + unetnameShort + '</div>';
        card_html += symlinkWidget;
    } else {
        $(card).attr('data-symlink', "");
    }

    if (SelectedModel === checkpoint) {
        card.classList.add('visual-ckpt-selected');
    }

    if (AscoreDataResponse != null && AscoreDataResponse.hasOwnProperty(ckptName) === true) {
        var aestString = AscoreDataResponse[ckptName];
        var aestArray = aestString.split("|");
        var aestAVGValue = Math.floor(aestArray[1] / aestArray[0]);
        $(card).attr('data-ascore', aestAVGValue);
        var aeScorePercentLine = Math.floor(((aestAVGValue - aeScoreMin) / (aeScoreMax - aeScoreMin)) * 100);
        if (aeScorePercentLine < 0) {
            aeScorePercentLine = 0;
        }
        if (aeScorePercentLine > 100) {
            aeScorePercentLine = 100;
        }
        aesthWidget = '<div class="visual-aesthetic-score" title="' + aestAVGValue + ' / ' + aeScorePercentLine + '%">' + aestAVGValue + '<span> - ' + aeScorePercentLine + '%</span><hr style="width: ' + aeScorePercentLine + '%"></div>';
        card_html += aesthWidget;
    } else {
        $(card).attr('data-ascore', 0);
    }

    if (STimeDataResponse != null && STimeDataResponse.hasOwnProperty(ckptName) === true) {
        var stimeString = STimeDataResponse[ckptName];
        var stimestArray = stimeString.split("|");
        var stimestAVGValue = Math.floor(stimestArray[1] / stimestArray[0]);
        $(card).attr('data-stime', stimestAVGValue);
        stimeWidget = '<div class="visual-stime" title="Average sampling time: ' + stimestAVGValue + ' sec">' + stimestAVGValue + 's</div>';
        card_html += stimeWidget;
    } else {
        $(card).attr('data-stime', 0);
    }

    $(card).attr('data-name', ckptName);
    if (CategoryName == 'StableCascade') { CategoryName = 'Cascade'; }
    $(card).attr('data-version', CategoryName);
    $(card).attr('data-path', path_only);

    if (RawImageDataResponse != null &&  Object.keys(RawImageDataResponse).length > 0) {
        if (RawImageDataResponse.hasOwnProperty(finalName) === true) {
            if (PreviewPath == false) {
                var imgsrc = 'data:image/jpeg;charset=utf-8;base64,  ' + RawImageDataResponse[finalName];
                card_html += '<img src="data:image/jpeg;charset=utf-8;base64,  ' + RawImageDataResponse[finalName] + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '" data-ckptver="' + CategoryName + '">';
            } else {
                var imgsrc = prwPath + '/images/' + RawImageDataResponse[finalName];
                card_html += '<img src="' + prwPath + '/images/' + RawImageDataResponse[finalName] + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '" data-ckptver="' + CategoryName + '">';
            }
            card.innerHTML = card_html;
            container.appendChild(card);
        } else {
            var imgsrc = missingimgsrc;
            card_html += '<img src="' + missingimgsrc + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '" data-ckptver="' + CategoryName + '">';
            card.innerHTML = card_html;
            container.appendChild(card);
        }
    } else {
        var imgsrc = missingimgsrc;
        card_html += '<img src="' + missingimgsrc + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '" data-ckptver="' + CategoryName + '">';
        card.innerHTML = card_html;
        container.appendChild(card);
    }

    const img = new Image();
    img.src = imgsrc;
    waitForImageToLoad(img).then(() => {
        return true;
    });
}


function createSortButtons(buttondata) {
    var sort_html = '<label class="sort_by_label"><br> Sort by: </label>';
    $.each(buttondata, function() {
        sort_html += '<button type="button" class="preview_sort" data-sortsource="' + this.toLowerCase() + '">' + this + '</button>';
    });
    sort_html += '<label> | </label><button type="button" class="preview_sort_direction">ASC</button>';
    return sort_html;
}

function createTypeMenu(ModelsByVersion, supportedModels, LastCat, LastCatType) {
    var version_html = '';

    $.each(ModelsByVersion, function(ver_index, ver_value) {
        var addWhiteClass = '';
        if (supportedModels.includes(ver_index)) {
            if (!version_html.includes('data-ckptver="' + ver_index + '"')) {
                if (ver_index === LastCat && LastCatType == 'Version') {
                    addWhiteClass = ' selected_path';
                }
                version_html += '<button type="button" data-ckptver="' + ver_index + '" class="verfilter' + addWhiteClass + '">' + ver_index + '</button>';
            }
        }
    });

    version_html += '<label> | </label>';
    return version_html;
}

function createPathMenu(AllPath, ShowHidden, LastCat, LastCatType) {
    var menu_html = '';
    for (var subdir of AllPath) {
        var addWhiteClass = "";
        let firstletter = subdir.charAt(0);
        var subdirName = subdir;
        if (firstletter === '.') {
            subdirName = subdir.substring(1);
        }
        if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((subdir.match('^NSFW') && ShowHidden === true) || !subdir.match('^NSFW')))  {
            if (subdir === LastCat && LastCatType == 'Subdir') {
                addWhiteClass = ' selected_path';
            }
            menu_html += '<button type="button" data-ckptsubdir="' + subdir + '" class="subdirfilter' + addWhiteClass + '">' + subdirName + '</button>';
        }
    }
    return menu_html + '<label> <br> </label>';
}

function sendPOSTModelName(modelName) { // ModelKeywordResponse
    const body = new FormData();
    body.append('modelName', modelName);
    api.fetchApi("/primere_keyword_parser", {method: "POST", body,});
}

// ************************* categoryHandler LastCategoryResponse
function categoryHandler(setupValue, method, setupKey) {
    return new Promise((resolve, reject) => {
        api.addEventListener("LastCategoryResponse", (event) => resolve(event.detail), true);
        postCategoryHandler(setupValue, method, setupKey);
    });
}
function postCategoryHandler(setupValue, method, setupKey) {
    const body = new FormData();
    body.append('setupValue', setupValue);
    body.append('setupMethod', method);
    body.append('setupKey', setupKey);
    api.fetchApi("/primere_category_handler", {method: "POST", body,});
}

// ************************* getSupportedModels SupportedModelsResponse
function getSupportedModels() {
    return new Promise((resolve, reject) => {
        api.addEventListener("SupportedModelsResponse", (event) => resolve(event.detail), true);
        postSupportedModels();
    });
}
function postSupportedModels() {
    const body = new FormData();
    body.append('models', 'get');
    api.fetchApi("/primere_supported_models", {method: "POST", body,});
}

// ************************* getAllPath AllPathResponse
function getAllPath(source_subdirname) {
    return new Promise((resolve, reject) => {
        api.addEventListener("AllPathResponse", (event) => resolve(event.detail), true);
        postAllPath(source_subdirname);
    });
}
function postAllPath(source_subdirname) {
    const body = new FormData();
    body.append('sourceType', source_subdirname);
    api.fetchApi("/primere_modelpaths", {method: "POST", body,});
}

// ************************* getModelData CategoryListResponse
function getModelData(message) {
    return new Promise((resolve, reject) => {
        api.addEventListener("CategoryListResponse", (event) => resolve(event.detail), true);
        postModelData(message);
    });
}
function postModelData(message) {
    const body = new FormData();
    body.append('cache_key', message);
    api.fetchApi("/primere_get_category", {method: "POST", body,});
}

// ************************* getModelDatabyPath SourceListResponse
function getModelDatabyPath(subdir, type) {
    return new Promise((resolve, reject) => {
        api.addEventListener("SourceListResponse", (event) => resolve(event.detail), true);
        postModelDatabyPath(subdir, type);
    });
}
function postModelDatabyPath(subdir, type) {
    const body = new FormData();
    body.append('subdir', subdir);
    body.append('type', type);
    api.fetchApi("/primere_get_subdir", {method: "POST", body,});
}

// ************************* getModelDatabyVersion VersionListResponse
function getModelDatabyVersion(subdir, cachekey, version) {
    return new Promise((resolve, reject) => {
        api.addEventListener("VersionListResponse", (event) => resolve(event.detail), true);
        postModelVersion(subdir, cachekey, version);
    });
}

function postModelVersion(subdir, cachekey, version) {
    const body = new FormData();
    body.append('subdir', subdir);
    body.append('cachekey', cachekey);
    body.append('version', version);
    api.fetchApi("/primere_get_version", {method: "POST", body,});
}

// ************************* getCacheByKey CacheByKey
function getCacheByKey(chachekey) {
    return new Promise((resolve, reject) => {
        api.addEventListener("CacheByKey", (event) => resolve(event.detail), true);
        postCacheKey(chachekey);
    });
}
function postCacheKey(chachekey) {
    const body = new FormData();
    body.append('chachekey', chachekey);
    api.fetchApi("/primere_get_cache", {method: "POST", body,});
}

// ************************* ReadAScores AscoreData
function ReadAScores(type) {
    return new Promise((resolve, reject) => {
        api.addEventListener("AscoreData", (event) => resolve(event.detail), true);
        postAscoreData(type);
    });
}
function postAscoreData(type) {
    const body = new FormData();
    body.append('type', type);
    api.fetchApi("/primere_get_ascores", {method: "POST", body,});
}

// ************************* ReadSTimes STimeData
function ReadSTimes(type) {
    return new Promise((resolve, reject) => {
        api.addEventListener("STimeData", (event) => resolve(event.detail), true);
        postSTimeData(type);
    });
}
function postSTimeData(type) {
    const body = new FormData();
    body.append('type', type);
    api.fetchApi("/primere_get_stime", {method: "POST", body,});
}

// ************************* modelImageData CollectedImageData
function modelImageData(SubdirName, PreviewPath) {
    return new Promise((resolve, reject) => {
        api.addEventListener("CollectedImageData", (event) => resolve(event.detail), true);
        postModelImageData(SubdirName, PreviewPath);
    });
}
function postModelImageData(SubdirName, PreviewPath) {
    const body = new FormData();
    body.append('SubdirName', SubdirName);
    body.append('PreviewPath', PreviewPath);
    api.fetchApi("/primere_get_images", {method: "POST", body,});
}

// ************************* ReadFileDate FileDateData
function ReadFileDate(sourcetype) {
    return new Promise((resolve, reject) => {
        api.addEventListener("FileDateData", (event) => resolve(event.detail), true);
        postReadFileDate(sourcetype);
    });
}
function postReadFileDate(sourcetype) {
    const body = new FormData();
    body.append('type', sourcetype);
    api.fetchApi("/primere_get_filedates", {method: "POST", body,});
}

// ************************* ReadFileDate ReadFileSymlink
function ReadFileSymlink(sourcetype) {
    return new Promise((resolve, reject) => {
        api.addEventListener("FileLinkData", (event) => resolve(event.detail), true);
        postReadFileSymlink(sourcetype);
    });
}
function postReadFileSymlink(sourcetype) {
    const body = new FormData();
    body.append('type', sourcetype);
    api.fetchApi("/primere_get_filelinks", {method: "POST", body,});
}

//await sleep(2000);
//await waitUntil(() => variable === true);
function sleep(ms) {
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