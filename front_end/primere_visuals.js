import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let eventListenerInit = false;
const realPath = "/extensions/ComfyUI_Primere_Nodes"; //"extensions/Primere";
const prwPath = "/extensions/ComfyUI_Primere_Nodes"; //"extensions/PrimerePreviews";
const validClasses = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualStyle', 'PrimereVisualLYCORIS', 'PrimereVisualPromptOrganizerCSV'];
const stackedClasses = ['PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualLYCORIS'];
let widget_name = "";
let widget_object = {};
let node_object = {};
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

function setText(selector, value) {
    const el = document.querySelector(selector);
    if (el) {
        el.textContent = value;
    }
}

function getFilterInput() {
    return document.querySelector('body div.subdirtab input');
}

function clearModalCardsKeepingSelected() {
    const cards = document.querySelectorAll('div.primere-modal-content div.visual-ckpt');
    for (const card of cards) {
        if (!card.classList.contains('visual-ckpt-selected')) {
            card.remove();
        }
    }
}

function clearSelectedPathButtons() {
    const buttons = document.querySelectorAll('div.subdirtab button');
    for (const button of buttons) {
        if (!button.classList.contains('preview_sort') && !button.classList.contains('preview_sort_direction')) {
            button.classList.remove('selected_path');
        }
    }
}

app.registerExtension({
    name: "Primere.VisualMenu",

    /* init() {
        Promise.all([
          fetch('extensions/Primere/keywords/lora-keyword.txt').then(x => x.text()),
          fetch('extensions/Primere/keywords/model-keyword.txt').then(x => x.text())
        ]).then(([Lora, Model]) => {
          console.log(Lora);
          console.log(Model);
        });
    }, */

    /* async setup() {

    }, */

    async beforeRegisterNodeDef(nodeType, nodeData, app) { // 0
        if (validClasses.includes(nodeData.name)) {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                hiddenWidgets[nodeData.name] = nodeData.input.hidden;
            }
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                new ModalControl(this);
            }
        }
    },
});


class ModalControl {
    constructor(node) {
        let callbackfunct = null;
        let modaltitle = '';
        let nodematch = '';
        let isnumeric_end = false;

        for (var i = 0; i < node.widgets.length; ++i) {
            var w = node.widgets[i];
            if (!w || w.disabled)
                continue;

            if (w.type != "combo")
                continue;

            let nx = i;
            node.widgets[i].mouse = async function (event, pos, node) {
                if (event.type == 'pointermove' && validClasses.includes(node.type)) {
                    return false;
                }

                if (stackedClasses.includes(node.type)) {
                    isnumeric_end = true;
                } else {
                    isnumeric_end = false;
                }

                currentClass = node.type;
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

                widget_name = node.widgets[nx].name;
                widget_object = node.widgets[nx];
                node_object = node;

                if (ShowModal === true) {
                    const contextMenu = document.querySelector('div.litegraph.litecontextmenu.litemenubar-panel');
                    if (contextMenu) {
                        contextMenu.style.display = 'none';
                    }

                    nodeHelper = hiddenWidgets[currentClass]
                    source_subdirname = nodeHelper['subdir'];
                    cache_key = nodeHelper['cache_key'];

                    if (eventListenerInit == false) {
                        ModalHandler();
                    }

                    if (document.querySelector('div.primere-modal-content div.visual-ckpt')) {
                        const ckptContainer = document.querySelector('div.primere-modal-content.ckpt-container');
                        if (ckptContainer) {
                            ckptContainer.innerHTML = '';
                        }
                    }

                    const widgetNumericEnd = /\d$/.test(String(widget_object?.name ?? ''));
                    //const widgetNumericEnd = Number.isFinite(Number(widget_object.name.substr(-1)));
                    if (widget_object.name.match(nodematch) && widgetNumericEnd === isnumeric_end) {
                        await sleep(200);
                        SelectedModel = node_object.widgets[nx].value;
                        // setup_visual_modal(modaltitle, 'AllModels', ShowHidden, SelectedModel, source_subdirname, node_object, PreviewPath);
                        await new Promise((resolve) => requestAnimationFrame(resolve));
                        setup_visual_modal(modaltitle, 'AllModels', ShowHidden, SelectedModel, source_subdirname, node_object, PreviewPath);
                        callbackfunct = inner_clicked.bind(widget_object);

                        function inner_clicked(value, option, event) {
                            inner_value_change(widget_object, value);
                            app.canvas.setDirty(true);
                            return false;
                        }

                        function inner_value_change(widget_object, value) {
                            if (typeof nodeHelper['sortbuttons'] === "object" && typeof nodeHelper['sortbuttons'][0] === "object" && nodeHelper['sortbuttons'][0].indexOf("Path") == -1) {
                                for (var ic = 0; ic < node_object.widgets.length; ++ic) {
                                    if (node_object.widgets[ic].type == 'combo' && node_object.widgets[ic].value != 'None') {
                                        node_object.widgets[ic].value = 'None';
                                    }
                                }
                            }

                            if (widget_object.type == "number") {
                                value = Number(value);
                            }
                            widget_object.value = value;
                            if (widget_object.options && widget_object.options.property && node_object.properties[widget_object.options.property] !== undefined) {
                                node_object.setProperty(widget_object.options.property, value);
                            }
                            if (widget_object.callback) {
                                widget_object.callback(widget_object.value, this, node_object, pos, event);
                            }
                        }

                        return null;
                    }
                }
            }
        }

        function ModalHandler() { // 02
            eventListenerInit = true;

            document.body.addEventListener("click", async function (event) {
                let modal = null;

                const closeButton = event.target.closest('button.modal-closer');
                if (closeButton) {
                    modal = document.getElementById("primere_visual_modal");
                    if (modal) {
                        modal.setAttribute('style', 'display: none; width: 80%; height: 70%;');
                    }
                    var lastDirValue = 'All';
                    if (lastDirObject.hasOwnProperty(currentClass) === true) {
                        lastDirValue = lastDirObject[currentClass];
                    }

                    if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
                        if (AutoFilter !== true) {
                            await categoryHandler(lastDirValue, 'add', 'last_visual_category' + '_' + cache_key);
                            await categoryHandler(FilterType, 'add', 'last_visual_category_type' + '_' + cache_key);
                            const filter = getFilterInput()?.value ?? '';
                            await categoryHandler(filter, 'add', 'last_visual_filter' + '_' + cache_key);
                            await categoryHandler(sortType, 'add', 'last_visual_sort_type' + '_' + cache_key);
                            await categoryHandler(operator, 'add', 'last_visual_sort_operator' + '_' + cache_key);
                        }
                    }
                    return;
                }

                const selectedImage = event.target.closest('div.primere-modal-content div.visual-ckpt img');
                if (selectedImage) {
                    var ckptName = selectedImage.dataset.ckptname;
                    modal = document.getElementById("primere_visual_modal");
                    if (modal) {
                        modal.setAttribute('style', 'display: none; width: 80%; height: 70%;');
                    }
                    var lastDirValue = 'All';
                    if (lastDirObject.hasOwnProperty(currentClass) === true) {
                        lastDirValue = lastDirObject[currentClass];
                    }

                    if (source_subdirname == 'styles') {
                        let pathLastIndex = ckptName.lastIndexOf('\\');
                        ckptName = ckptName.substring(pathLastIndex + 1);
                    }

                    if (ckptName && typeof callbackfunct == 'function') {
                        callbackfunct(ckptName);
                        sendPOSTModelName(ckptName);
                    }

                    if (typeof nodeHelper['sortbuttons'] !== "object" || typeof nodeHelper['sortbuttons'][0] !== "object" || nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
                        if (AutoFilter !== true) {
                            await categoryHandler(lastDirValue, 'add', 'last_visual_category' + '_' + cache_key);
                            await categoryHandler(FilterType, 'add', 'last_visual_category_type' + '_' + cache_key);
                            const filter = getFilterInput()?.value ?? '';
                            await categoryHandler(filter, 'add', 'last_visual_filter' + '_' + cache_key);
                            await categoryHandler(sortType, 'add', 'last_visual_sort_type' + '_' + cache_key);
                            await categoryHandler(operator, 'add', 'last_visual_sort_operator' + '_' + cache_key);
                        }
                    }
                    return;
                }

                var subdirName = 'All';
                var filteredCheckpoints = 0;

                const subdirButton = event.target.closest('div.subdirtab button.subdirfilter');
                if (subdirButton) {
                    const filterInput = getFilterInput();
                    if (filterInput) {
                        filterInput.value = '';
                    }
                    subdirName = subdirButton.dataset.ckptsubdir;
                    if (currentClass !== false) {
                        lastDirObject[currentClass] = subdirName;
                    }

                    filteredCheckpoints = 0;
                    clearModalCardsKeepingSelected();

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

                            for (const [ver_index, ver_value] of Object.entries(ModelsByVersion)) {
                                if (ver_value.includes(ckptName)) {
                                    CategoryName = ver_index;
                                }
                            }

                            var container = document.querySelector('div.primere-modal-content.ckpt-container');
                            if (subdirName === 'Root') {
                                let isSubdirExist = checkpoint.lastIndexOf('\\');
                                if (isSubdirExist < 0) {
                                    filteredCheckpoints++;
                                    await createCardElement(checkpoint, container, SelectedModel, source_subdirname, CategoryName);
                                }
                            } else {
                                filteredCheckpoints++;
                                await createCardElement(checkpoint, container, SelectedModel, source_subdirname, CategoryName);
                            }
                        }
                        setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(filteredCheckpoints));
                    }

                    previewSorter(operator, sortType);

                    setText('div#primere_visual_modal div.modal_header label.ckpt-name', subdirName);
                    setText('div#primere_visual_modal div.modal_header label.ckpt-ver', 'Subdir');
                    clearSelectedPathButtons();
                    subdirButton.classList.add('selected_path');
                    FilterType = 'Subdir';
                    return;
                }

                const versionButton = event.target.closest('div.subdirtab button.verfilter');
                if (versionButton) {
                    const filterInput = getFilterInput();
                    if (filterInput) {
                        filterInput.value = '';
                    }
                    var versionName = versionButton.dataset.ckptver;

                    if (currentClass !== false) {
                        lastDirObject[currentClass] = versionName;
                    }

                    filteredCheckpoints = 0;
                    clearModalCardsKeepingSelected();

                    ModelList = await getModelDatabyVersion(source_subdirname, cache_key + '_version', versionName);

                    for (var checkpoint of ModelList) {
                        let firstletter = checkpoint.charAt(0);
                        var filterpass = true;
                        if (SelectedModel == checkpoint) {
                            filterpass = false;
                        }
                        if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && ShowHidden === true) || !checkpoint.match('^NSFW')) && filterpass == true) {
                            filteredCheckpoints++;
                            var container = document.querySelector('div.primere-modal-content.ckpt-container');
                            await createCardElement(checkpoint, container, SelectedModel, source_subdirname, versionName);
                        }
                        setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(filteredCheckpoints));
                    }

                    previewSorter(operator, sortType);

                    setText('div#primere_visual_modal div.modal_header label.ckpt-name', versionName);
                    setText('div#primere_visual_modal div.modal_header label.ckpt-ver', 'Version');
                    clearSelectedPathButtons();
                    versionButton.classList.add('selected_path');
                    FilterType = 'Version';
                    return;
                }

                const clearButton = event.target.closest('div.subdirtab button.filter_clear');
                if (clearButton) {
                    const filterInput = getFilterInput();
                    if (filterInput) {
                        filterInput.value = '';
                    }
                    const images = document.querySelectorAll('div.primere-modal-content div.visual-ckpt img');
                    filteredCheckpoints = 0;
                    for (const image of images) {
                        const parent = image.parentElement;
                        if (parent) {
                            parent.style.display = '';
                            filteredCheckpoints++;
                        }
                    }
                    setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(filteredCheckpoints - 1));
                    return;
                }

                const sortButton = event.target.closest('div.subdirtab button.preview_sort');
                if (sortButton) {
                    sortType = sortButton.dataset.sortsource;
                    operator = document.querySelector('button.preview_sort_direction')?.textContent ?? 'ASC';
                    previewSorter(operator, sortType);
                    const sortButtons = document.querySelectorAll('div.subdirtab button.preview_sort');
                    for (const button of sortButtons) {
                        button.classList.remove('selected_path');
                    }
                    sortButton.classList.add('selected_path');
                    return;
                }

                const sortDirectionButton = event.target.closest('div.subdirtab button.preview_sort_direction');
                if (sortDirectionButton) {
                    operator = sortDirectionButton.textContent;
                    if (operator == 'ASC') {
                        sortDirectionButton.textContent = 'DESC';
                        operator = 'DESC';
                        sortDirectionButton.classList.add('selected_path');
                    } else {
                        sortDirectionButton.textContent = 'ASC';
                        operator = 'ASC';
                        sortDirectionButton.classList.remove('selected_path');
                    }
                    previewSorter(operator, sortType);
                }
            });

            document.body.addEventListener("keyup", function (event) {
                const filterInput = event.target.closest('div.subdirtab input');
                if (filterInput) {
                    previewFilter(filterInput.value);
                }
            });
        }
    }
}

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
        const title = document.querySelector('div#primere_visual_modal div.modal_header h3.visual_modal_title');
        if (title) {
            title.innerHTML = combo_name.replace("_", " ") + '<label class="ckpt-ver">Subdir</label> :: <label class="ckpt-name">All</label> :: <label class="ckpt-counter"></label>';
        }
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
        VersionCacheData = await getCacheByKey(cache_key + '_version');
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
            await ModelList.unshift(SelectedModel);
        } else {
            await ModelList.unshift(fondModel);
        }
    }

    setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(ModelList.length));
    setText('div#primere_visual_modal div.modal_header label.ckpt-name', LastCat);
    setText('div#primere_visual_modal div.modal_header label.ckpt-ver', LastCatType);

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

            for (const [ver_index, ver_value] of Object.entries(ModelsByVersion)) {
                if (ver_value.includes(ckptName)) {
                    CategoryName = ver_index;
                }
            }

            if (LastCat === 'Root') {
                let isSubdirExist = checkpoint.lastIndexOf('\\');
                if (isSubdirExist < 0 || checkpoint == SelectedModel) {
                    CKPTElements++;
                    await createCardElement(checkpoint, container, SelectedModel, ModelType, CategoryName);
                }
            } else {
                CKPTElements++;
                await createCardElement(checkpoint, container, SelectedModel, ModelType, CategoryName);
            }
        }
        setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(CKPTElements));
    }

    if (savedfilter.length > 0) {
        const filterInput = getFilterInput();
        if (filterInput) {
            filterInput.value = savedfilter;
        }
        previewFilter(savedfilter);
    }
    const sortDirectionButton = document.querySelector('button.preview_sort_direction');
    if (sortDirectionButton) {
        sortDirectionButton.textContent = operator;
    }
    if (operator != 'ASC') {
        document.querySelector('div.subdirtab button.preview_sort_direction')?.classList.add("selected_path");
    }
    document.querySelector('div.subdirtab button.preview_sort[data-sortsource="' + sortType + '"]')?.classList.add('selected_path');
    previewSorter(operator, sortType);

    if (!document.querySelector('#primere_visual_modal div.subdirtab button.selected_path')) {
        document.querySelector('#primere_visual_modal div.subdirtab button[data-ckptsubdir="All"]')?.classList.add('selected_path');
    }
}

function previewSorter(operator, sortType) {
    const container = document.querySelector('div.primere-modal-content.ckpt-container.ckpt-grid-layout');
    if (!container) {
        return;
    }
    const cards = Array.from(container.querySelectorAll('div.visual-ckpt:not(.visual-ckpt-selected)'));
    cards.sort(function (a, b) {
        var aVal = a.dataset[sortType] ?? '';
        var bVal = b.dataset[sortType] ?? '';

        if (isNaN(parseInt(aVal)) && isNaN(parseInt(bVal))) {
            aVal = String(aVal).toUpperCase();
            bVal = String(bVal).toUpperCase();
            if (operator == 'ASC') {
                return aVal < bVal ? -1 : (aVal > bVal ? 1 : 0);
            }
            return aVal > bVal ? -1 : (aVal < bVal ? 1 : 0);
        }

        if (operator == 'ASC') {
            return parseInt(aVal) - parseInt(bVal);
        }
        return parseInt(bVal) - parseInt(aVal);
    });

    for (const card of cards) {
        container.appendChild(card);
    }
}

function previewFilter(filterString) {
    var imageContainers = document.querySelectorAll('div.primere-modal-content div.visual-ckpt img');
    var filteredCheckpoints = 0;
    for (const img_obj of imageContainers) {
        var versiontext = img_obj.dataset.ckptver;
        var ImageCheckpoint = img_obj.dataset.ckptname + '_' + versiontext;
        const parentCard = img_obj.parentElement;
        const selectedCard = parentCard?.closest('.visual-ckpt-selected');
        if (ImageCheckpoint.toLowerCase().indexOf(filterString.toLowerCase()) >= 0 || selectedCard) {
            if (parentCard) {
                parentCard.style.display = '';
            }
            filteredCheckpoints++;
        } else {
            if (parentCard) {
                parentCard.style.display = 'none';
            }
        }
        setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(filteredCheckpoints - 1));
    }
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
        card.dataset.date = String(timestamp);
    } else {
        card.dataset.date = '0';
    }

    if (FileLinkResponse.hasOwnProperty(ckptName) === true) {
        var unetname = FileLinkResponse[ckptName];
        card.dataset.symlink = unetname;
        var unetnameShort = unetname;
        if (unetname == 'diffusion_models') { unetnameShort = 'DiMo'}
        if (unetname == 'diffusers') { unetnameShort = 'Diff'}
        symlinkWidget = '<div class="visual-symlink-type" title="Symlinked from: ' + unetname + '">' + unetnameShort + '</div>';
        card_html += symlinkWidget;
    } else {
        card.dataset.symlink = "";
    }

    if (SelectedModel === checkpoint) {
        card.classList.add('visual-ckpt-selected');
    }

    if (AscoreDataResponse != null && AscoreDataResponse.hasOwnProperty(ckptName) === true) {
        var aestString = AscoreDataResponse[ckptName];
        var aestArray = aestString.split("|");
        var aestAVGValue = Math.floor(aestArray[1] / aestArray[0]);
        card.dataset.ascore = String(aestAVGValue);
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
        card.dataset.ascore = '0';
    }

    if (STimeDataResponse != null && STimeDataResponse.hasOwnProperty(ckptName) === true) {
        var stimeString = STimeDataResponse[ckptName];
        var stimestArray = stimeString.split("|");
        var stimestAVGValue = Math.floor(stimestArray[1] / stimestArray[0]);
        card.dataset.stime = String(stimestAVGValue);
        stimeWidget = '<div class="visual-stime" title="Average sampling time: ' + stimestAVGValue + ' sec">' + stimestAVGValue + 's</div>';
        card_html += stimeWidget;
    } else {
        card.dataset.stime = '0';
    }

    card.dataset.name = ckptName;
    if (CategoryName == 'StableCascade') { CategoryName = 'Cascade'; }
    card.dataset.version = CategoryName;
    card.dataset.path = path_only;

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
    for (const buttonName of buttondata) {
        sort_html += '<button type="button" class="preview_sort" data-sortsource="' + buttonName.toLowerCase() + '">' + buttonName + '</button>';
    }
    sort_html += '<label> | </label><button type="button" class="preview_sort_direction">ASC</button>';
    return sort_html;
}

function createTypeMenu(ModelsByVersion, supportedModels, LastCat, LastCatType) {
    var version_html = '';

    for (const [ver_index, ver_value] of Object.entries(ModelsByVersion)) {
        var addWhiteClass = '';
        if (supportedModels.includes(ver_index)) {
            if (!version_html.includes('data-ckptver="' + ver_index + '"')) {
                if (ver_index === LastCat && LastCatType == 'Version') {
                    addWhiteClass = ' selected_path';
                }
                version_html += '<button type="button" data-ckptver="' + ver_index + '" class="verfilter' + addWhiteClass + '">' + ver_index + '</button>';
            }
        }
    }

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