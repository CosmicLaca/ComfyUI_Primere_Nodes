import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const prwPath = "/extensions/ComfyUI_Primere_Nodes";
const validClasses = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualStyle', 'PrimereVisualLYCORIS', 'PrimereVisualPromptOrganizerCSV'];
const stackedClasses = ['PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualLYCORIS'];

const state = {
    eventListenerInit: false,
    callbackfunct: null,
    widget_name: "",
    widget_object: {},
    node_object: {},
    lastDirObject: {},
    currentClass: false,
    hiddenWidgets: {},
    ShowHidden: false,
    FilterType: 'Subdir',
    SelectedModel: 'SelectedModel',
    sortType: 'name',
    operator: 'ASC',
    PreviewPath: true,
    aeScoreMin: 550,
    aeScoreMax: 800,
    nodeHelper: {},
    cache_key: '',
    source_subdirname: '',
    AutoFilter: false,
    StackVersion: 'Auto',
    ModelVersion: 'SD1',
    SetupCacheData: {},
    VersionCacheData: {},
    supportedModels: [],
    ModelsByVersion: {},
    AllPath: [],
    ModelList: {},
    AscoreDataResponse: {},
    STimeDataResponse: {},
    FileDateResponse: {},
    SimilarityDataResponse: {},
    FileLinkResponse: {},
    RawImageDataResponse: {},
};

function apiPost(endpoint, eventName, params = {}) {
    return new Promise((resolve) => {
        const timer = setTimeout(() => resolve(null), 10000);
        api.addEventListener(eventName, (event) => {
            clearTimeout(timer);
            resolve(event.detail);
        }, { once: true });
        const body = new FormData();
        for (const [key, value] of Object.entries(params)) body.append(key, value);
        api.fetchApi(endpoint, { method: "POST", body });
    });
}

const BADGE_OVERRIDES = {
    'SD1': 'SD 1.x', 'SD2': 'SD 2.x', 'SD3': 'SD 3.x',
    'StableCascade': 'Cascade', 'KwaiKolors': 'Kolors',
    'StableAudio': 'S.Audio', 'Playground': 'PG', 'PixartSigma': 'Σ',
    'Illustrious': 'Ill', 'Unknown': '?',
};

function typeHash(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++) h = (h * 33 ^ str.charCodeAt(i)) >>> 0;
    return h;
}

function badgeLabel(type) {
    if (BADGE_OVERRIDES[type]) return BADGE_OVERRIDES[type];
    return type.length < 10 ? type : type.substring(0, 6);
}

function injectTypeStyles(types) {
    const existing = document.getElementById('primere-type-styles');
    if (existing) existing.remove();
    const rules = types.map(type => {
        const h = typeHash(type);
        const hue = h % 360;
        const satBg    = 30 + (h >> 4)  % 20;
        const satBadge = 45 + (h >> 8)  % 20;
        const label = badgeLabel(type).replace(/'/g, "\\'");
        return `#primere_visual_modal .background-${type}{background:hsl(${hue},${satBg}%,80%);}` +
               `#primere_visual_modal .ckpt-version.${type}-ckpt::before{content:'${label}';background:hsl(${hue},${satBadge}%,22%);color:#ffffff;}`;
    }).join('\n');
    const style = document.createElement('style');
    style.id = 'primere-type-styles';
    style.textContent = rules;
    document.head.appendChild(style);
}

const categoryHandler      = (setupValue, method, setupKey) => apiPost('/primere_category_handler',  'LastCategoryResponse',   { setupValue, setupMethod: method, setupKey });
const getSupportedModels   = ()                              => apiPost('/primere_supported_models',  'SupportedModelsResponse', { models: 'get' });
const getAllPath            = (sourceType)                   => apiPost('/primere_modelpaths',         'AllPathResponse',         { sourceType });
const getModelData         = (message)                      => apiPost('/primere_get_category',       'CategoryListResponse',    { cache_key: message });
const getModelDatabyPath   = (subdir, type)                 => apiPost('/primere_get_subdir',         'SourceListResponse',      { subdir, type });
const getModelDatabyVersion= (subdir, cachekey, version)    => apiPost('/primere_get_version',        'VersionListResponse',     { subdir, cachekey, version });
const getCacheByKey        = (chachekey)                    => apiPost('/primere_get_cache',           'CacheByKey',             { chachekey });
const ReadAScores          = (type)                         => apiPost('/primere_get_ascores',         'AscoreData',             { type });
const ReadSTimes           = (type)                         => apiPost('/primere_get_stime',           'STimeData',              { type });
const modelImageData       = (SubdirName, PreviewPath)      => apiPost('/primere_get_images',          'CollectedImageData',      { SubdirName, PreviewPath });
const ReadFileDate         = (sourcetype)                   => apiPost('/primere_get_filedates',       'FileDateData',           { type: sourcetype });
const ReadSimilarity       = (SubdirName, PreviewPath, SelectedModel) => apiPost('/primere_get_similarity', 'SimilarityData',    { SubdirName, PreviewPath, SelectedModel });
const ReadFileSymlink      = (sourcetype)                   => apiPost('/primere_get_filelinks',       'FileLinkData',           { type: sourcetype });

function sendPOSTModelName(modelName) {
    const body = new FormData();
    body.append('modelName', modelName);
    api.fetchApi("/primere_keyword_parser", { method: "POST", body });
}

function setText(selector, value) {
    const el = document.querySelector(selector);
    if (el) el.textContent = value;
}

function getFilterInput() {
    return document.querySelector('body div.subdirtab input');
}

function clearModalCardsKeepingSelected() {
    const cards = document.querySelectorAll('div.primere-modal-content div.visual-ckpt');
    for (const card of cards) {
        if (!card.classList.contains('visual-ckpt-selected')) card.remove();
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

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (validClasses.includes(nodeData.name)) {
            if (nodeData.input.hasOwnProperty('hidden') === true) {
                state.hiddenWidgets[nodeData.name] = nodeData.input.hidden;
            }
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                new ModalControl(this);
            }
        }
    },

    loadedGraphNode(node) {
        if (validClasses.includes(node.type)) {
            new ModalControl(node);
        }
    },
});

function ModalHandler() {
    state.eventListenerInit = true;

    document.body.addEventListener("click", async function (event) {
        let modal = null;

        const closeButton = event.target.closest('button.modal-closer');
        if (closeButton) {
            modal = document.getElementById("primere_visual_modal");
            if (modal) modal.setAttribute('style', 'display: none; width: 80%; height: 70%;');

            var lastDirValue = 'All';
            if (state.lastDirObject.hasOwnProperty(state.currentClass) === true) lastDirValue = state.lastDirObject[state.currentClass];

            if (typeof state.nodeHelper['sortbuttons'] !== "object" || typeof state.nodeHelper['sortbuttons'][0] !== "object" || state.nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
                if (state.AutoFilter !== true) {
                    await categoryHandler(lastDirValue, 'add', 'last_visual_category' + '_' + state.cache_key);
                    await categoryHandler(state.FilterType, 'add', 'last_visual_category_type' + '_' + state.cache_key);
                    const filter = getFilterInput()?.value ?? '';
                    await categoryHandler(filter, 'add', 'last_visual_filter' + '_' + state.cache_key);
                    await categoryHandler(state.sortType, 'add', 'last_visual_sort_type' + '_' + state.cache_key);
                    await categoryHandler(state.operator, 'add', 'last_visual_sort_operator' + '_' + state.cache_key);
                }
            }
            return;
        }

        const selectedImage = event.target.closest('div.primere-modal-content div.visual-ckpt img');
        if (selectedImage) {
            var ckptName = selectedImage.dataset.ckptname;
            modal = document.getElementById("primere_visual_modal");
            if (modal) modal.setAttribute('style', 'display: none; width: 80%; height: 70%;');

            var lastDirValue = 'All';
            if (state.lastDirObject.hasOwnProperty(state.currentClass) === true) lastDirValue = state.lastDirObject[state.currentClass];

            if (state.source_subdirname == 'styles') {
                let pathLastIndex = ckptName.lastIndexOf('\\');
                ckptName = ckptName.substring(pathLastIndex + 1);
            }

            if (ckptName && typeof state.callbackfunct == 'function') {
                state.callbackfunct(ckptName);
                sendPOSTModelName(ckptName);
            }

            if (typeof state.nodeHelper['sortbuttons'] !== "object" || typeof state.nodeHelper['sortbuttons'][0] !== "object" || state.nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
                if (state.AutoFilter !== true) {
                    await categoryHandler(lastDirValue, 'add', 'last_visual_category' + '_' + state.cache_key);
                    await categoryHandler(state.FilterType, 'add', 'last_visual_category_type' + '_' + state.cache_key);
                    const filter = getFilterInput()?.value ?? '';
                    await categoryHandler(filter, 'add', 'last_visual_filter' + '_' + state.cache_key);
                    await categoryHandler(state.sortType, 'add', 'last_visual_sort_type' + '_' + state.cache_key);
                    await categoryHandler(state.operator, 'add', 'last_visual_sort_operator' + '_' + state.cache_key);
                }
            }
            return;
        }

        var subdirName = 'All';
        var filteredCheckpoints = 0;

        const subdirButton = event.target.closest('div.subdirtab button.subdirfilter');
        if (subdirButton) {
            const filterInput = getFilterInput();
            if (filterInput) filterInput.value = '';
            subdirName = subdirButton.dataset.ckptsubdir;
            if (state.currentClass !== false) state.lastDirObject[state.currentClass] = subdirName;

            filteredCheckpoints = 0;
            clearModalCardsKeepingSelected();

            if (state.source_subdirname != 'styles') {
                state.ModelsByVersion = await getModelData(state.cache_key + '_version');
            }
            state.ModelList = await getModelDatabyPath(state.source_subdirname, subdirName);

            if (state.source_subdirname == 'styles') {
                if (state.ModelList.includes(state.SelectedModel) || state.ModelList.includes(subdirName + '\\' + state.SelectedModel)) {
                    var index_pre = (state.ModelList.indexOf(state.SelectedModel) + state.ModelList.indexOf(subdirName + '\\' + state.SelectedModel)) + 1;
                    if (index_pre !== -1) await state.ModelList.splice(index_pre, 1);
                }
            }

            for (var checkpoint of state.ModelList) {
                let firstletter = checkpoint.charAt(0);
                var filterpass = true;
                if (state.SelectedModel == checkpoint) filterpass = false;

                if (((firstletter === '.' && state.ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && state.ShowHidden === true) || !checkpoint.match('^NSFW')) && filterpass == true) {
                    let pathLastIndex = checkpoint.lastIndexOf('\\');
                    let ckptName_full = checkpoint.substring(pathLastIndex + 1);
                    let dotLastIndex = ckptName_full.lastIndexOf('.');
                    var ckptName = ckptName_full.substring(0, dotLastIndex);
                    var CategoryName = 'Unknown';

                    for (const [ver_index, ver_value] of Object.entries(state.ModelsByVersion)) {
                        if (ver_value.includes(ckptName)) CategoryName = ver_index;
                    }

                    var container = document.querySelector('div.primere-modal-content.ckpt-container');
                    if (subdirName === 'Root') {
                        let isSubdirExist = checkpoint.lastIndexOf('\\');
                        if (isSubdirExist < 0) {
                            filteredCheckpoints++;
                            await createCardElement(checkpoint, container, state.SelectedModel, state.source_subdirname, CategoryName);
                        }
                    } else {
                        filteredCheckpoints++;
                        await createCardElement(checkpoint, container, state.SelectedModel, state.source_subdirname, CategoryName);
                    }
                }
                setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(filteredCheckpoints));
            }

            previewSorter(state.operator, state.sortType);
            setText('div#primere_visual_modal div.modal_header label.ckpt-name', subdirName);
            setText('div#primere_visual_modal div.modal_header label.ckpt-ver', 'Subdir');
            clearSelectedPathButtons();
            subdirButton.classList.add('selected_path');
            state.FilterType = 'Subdir';
            return;
        }

        const versionButton = event.target.closest('div.subdirtab button.verfilter');
        if (versionButton) {
            const filterInput = getFilterInput();
            if (filterInput) filterInput.value = '';
            var versionName = versionButton.dataset.ckptver;

            if (state.currentClass !== false) state.lastDirObject[state.currentClass] = versionName;

            filteredCheckpoints = 0;
            clearModalCardsKeepingSelected();

            state.ModelList = await getModelDatabyVersion(state.source_subdirname, state.cache_key + '_version', versionName);

            for (var checkpoint of state.ModelList) {
                let firstletter = checkpoint.charAt(0);
                var filterpass = true;
                if (state.SelectedModel == checkpoint) filterpass = false;
                if (((firstletter === '.' && state.ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && state.ShowHidden === true) || !checkpoint.match('^NSFW')) && filterpass == true) {
                    filteredCheckpoints++;
                    var container = document.querySelector('div.primere-modal-content.ckpt-container');
                    await createCardElement(checkpoint, container, state.SelectedModel, state.source_subdirname, versionName);
                }
            }
            setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(filteredCheckpoints));

            previewSorter(state.operator, state.sortType);
            setText('div#primere_visual_modal div.modal_header label.ckpt-name', versionName);
            setText('div#primere_visual_modal div.modal_header label.ckpt-ver', 'Version');
            clearSelectedPathButtons();
            versionButton.classList.add('selected_path');
            state.FilterType = 'Version';
            return;
        }

        const clearButton = event.target.closest('div.subdirtab button.filter_clear');
        if (clearButton) {
            const filterInput = getFilterInput();
            if (filterInput) filterInput.value = '';
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
            state.sortType = sortButton.dataset.sortsource;
            state.operator = document.querySelector('button.preview_sort_direction')?.textContent ?? 'ASC';
            previewSorter(state.operator, state.sortType);
            const sortButtons = document.querySelectorAll('div.subdirtab button.preview_sort');
            for (const button of sortButtons) button.classList.remove('selected_path');
            sortButton.classList.add('selected_path');
            return;
        }

        const sortDirectionButton = event.target.closest('div.subdirtab button.preview_sort_direction');
        if (sortDirectionButton) {
            state.operator = sortDirectionButton.textContent;
            if (state.operator == 'ASC') {
                sortDirectionButton.textContent = 'DESC';
                state.operator = 'DESC';
                sortDirectionButton.classList.add('selected_path');
            } else {
                sortDirectionButton.textContent = 'ASC';
                state.operator = 'ASC';
                sortDirectionButton.classList.remove('selected_path');
            }
            previewSorter(state.operator, state.sortType);
        }
    });

    document.body.addEventListener("keyup", function (event) {
        const filterInput = event.target.closest('div.subdirtab input');
        if (filterInput) previewFilter(filterInput.value);
    });
}

class ModalControl {
    constructor(node) {
        const modaltitle = '';
        const nodematch = '';

        for (var i = 0; i < node.widgets.length; ++i) {
            var w = node.widgets[i];
            if (!w || w.disabled) continue;
            if (w.type != "combo") continue;

            let nx = i;
            node.widgets[i].mouse = async function (event, pos, node) {
                if (event.type == 'pointerup' && (pos[0] < 35 || pos[0] > node.size[0] - 35)) return false;
                if (event.type == 'pointermove' && validClasses.includes(node.type)) return false;
                if (event.type == 'pointerdown' && (pos[0] < 35 || pos[0] > node.size[0] - 35)) return false;

                const isnumeric_end = stackedClasses.includes(node.type);

                state.currentClass = node.type;
                var ShowModal = false;
                state.AutoFilter = null;
                state.StackVersion = null;
                state.ModelVersion = null;

                for (var p = 0; p < node.widgets.length; ++p) {
                    if (node.widgets[p].name == 'show_hidden') state.ShowHidden = node.widgets[p].value;
                    if (node.widgets[p].name == 'show_modal') ShowModal = node.widgets[p].value;
                    if (node.widgets[p].name == 'preview_path') state.PreviewPath = node.widgets[p].value;
                    if (node.widgets[p].name == 'aescore_percent_min' && node.widgets[p].value >= 0) state.aeScoreMin = node.widgets[p].value;
                    if (node.widgets[p].name == 'aescore_percent_max' && node.widgets[p].value > 0) state.aeScoreMax = node.widgets[p].value;
                    if (node.widgets[p].name == 'auto_filter') state.AutoFilter = node.widgets[p].value;
                    if (node.widgets[p].name == 'stack_version') state.StackVersion = node.widgets[p].value;
                    if (node.widgets[p].name == 'model_version') state.ModelVersion = node.widgets[p].value;
                }

                state.widget_name = node.widgets[nx].name;
                state.widget_object = node.widgets[nx];
                state.node_object = node;

                if (ShowModal === true) {
                    const contextMenu = document.querySelector('div.litegraph.litecontextmenu.litemenubar-panel');
                    if (contextMenu) contextMenu.style.display = 'none';

                    state.nodeHelper = state.hiddenWidgets[state.currentClass];
                    if (!state.nodeHelper) return;
                    state.source_subdirname = state.nodeHelper['subdir'];
                    state.cache_key = state.nodeHelper['cache_key'];

                    if (state.eventListenerInit == false) ModalHandler();

                    if (document.querySelector('div.primere-modal-content div.visual-ckpt')) {
                        const ckptContainer = document.querySelector('div.primere-modal-content.ckpt-container');
                        if (ckptContainer) ckptContainer.innerHTML = '';
                    }

                    const widgetNumericEnd = /\d$/.test(String(state.widget_object?.name ?? ''));
                    if (state.widget_object.name.match(nodematch) && widgetNumericEnd === isnumeric_end) {
                        await sleep(200);
                        state.SelectedModel = state.node_object.widgets[nx].value;
                        await new Promise((resolve) => requestAnimationFrame(resolve));
                        setup_visual_modal(modaltitle, 'AllModels', state.ShowHidden, state.SelectedModel, state.source_subdirname, state.node_object, state.PreviewPath);
                        state.callbackfunct = inner_clicked.bind(state.widget_object);

                        function inner_clicked(value, option, event) {
                            inner_value_change(state.widget_object, value);
                            app.canvas.setDirty(true);
                            return false;
                        }

                        function inner_value_change(widget_object, value) {
                            if (typeof state.nodeHelper['sortbuttons'] === "object" && typeof state.nodeHelper['sortbuttons'][0] === "object" && state.nodeHelper['sortbuttons'][0].indexOf("Path") == -1) {
                                for (var ic = 0; ic < state.node_object.widgets.length; ++ic) {
                                    if (state.node_object.widgets[ic].type == 'combo' && state.node_object.widgets[ic].value != 'None') {
                                        state.node_object.widgets[ic].value = 'None';
                                    }
                                }
                            }
                            if (widget_object.type == "number") value = Number(value);
                            widget_object.value = value;
                            if (widget_object.options && widget_object.options.property && state.node_object.properties[widget_object.options.property] !== undefined) {
                                state.node_object.setProperty(widget_object.options.property, value);
                            }
                            if (widget_object.callback) {
                                widget_object.callback(widget_object.value, this, state.node_object, pos, event);
                            }
                        }

                        return null;
                    }
                }
            }
        }
    }
}

async function setup_visual_modal(combo_name, AllModels, ShowHidden, SelectedModel, ModelType, node, PreviewPath) {
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
        if (ckptSubdir === '') ckptSubdir = 'Root';
        if (subdirArray.indexOf(ckptSubdir) === -1) subdirArray.push(ckptSubdir);
    }

    var subdir_tabs = modal.getElementsByClassName("subdirtab")[0];
    var menu_html = '';
    var type_html = '';
    var version_html = '';

    if (typeof state.nodeHelper['sortbuttons'] !== "object" || typeof state.nodeHelper['sortbuttons'][0] !== "object" || state.nodeHelper['sortbuttons'][0].indexOf("Path") > -1 && state.AutoFilter !== true) {
        for (var subdir of subdirArray) {
            var addWhiteClass = '';
            let firstletter = subdir.charAt(0);
            var subdirName = subdir;
            if (firstletter === '.') subdirName = subdir.substring(1);
            if ((firstletter === '.' && ShowHidden === true) || firstletter !== '.') {
                if (state.lastDirObject.hasOwnProperty(state.currentClass) === true && subdir === state.lastDirObject[state.currentClass]) {
                    addWhiteClass = ' selected_path';
                }
                menu_html += '<button type="button" data-ckptsubdir="' + subdir + '" class="subdirfilter' + addWhiteClass + '">' + subdirName + '</button>';
            }
        }
    }

    var LastCat = 'All';
    var LastCatType = 'Subdir';
    var savedfilter = "";

    state.SetupCacheData = await getCacheByKey("setup");

    if (state.SetupCacheData != null && typeof state.SetupCacheData === "object") {
        if (typeof state.nodeHelper['sortbuttons'] !== "object" || typeof state.nodeHelper['sortbuttons'][0] !== "object" || state.nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
            if (state.SetupCacheData.hasOwnProperty("last_visual_category" + '_' + state.cache_key)) {
                LastCat = state.SetupCacheData["last_visual_category" + '_' + state.cache_key];
            }
            if (state.SetupCacheData.hasOwnProperty("last_visual_category_type" + '_' + state.cache_key)) {
                LastCatType = state.SetupCacheData["last_visual_category_type" + '_' + state.cache_key];
            }
            if (state.SetupCacheData.hasOwnProperty("last_visual_filter" + '_' + state.cache_key)) {
                savedfilter = state.SetupCacheData['last_visual_filter' + '_' + state.cache_key];
            }
            if (state.SetupCacheData.hasOwnProperty("last_visual_sort_type" + '_' + state.cache_key)) {
                state.sortType = state.SetupCacheData['last_visual_sort_type' + '_' + state.cache_key];
            }
            if (state.SetupCacheData.hasOwnProperty("last_visual_sort_operator" + '_' + state.cache_key)) {
                state.operator = state.SetupCacheData['last_visual_sort_operator' + '_' + state.cache_key];
            }
        }
    }

    if (state.AutoFilter === true && state.StackVersion != null && state.ModelVersion != null) {
        if (state.StackVersion == 'Auto') {
            LastCat = state.ModelVersion;
            LastCatType = 'Version';
        } else if (state.StackVersion != 'Any' && state.StackVersion != 'Auto') {
            LastCat = state.StackVersion;
            LastCatType = 'Version';
        }
    }

    if (typeof state.nodeHelper['sortbuttons'] === "object" && typeof state.nodeHelper['sortbuttons'][0] === "object" && state.nodeHelper['sortbuttons'][0].indexOf("Path") == -1) {
        LastCat = state.widget_name;
    }

    state.supportedModels = [];
    state.ModelsByVersion = {};
    if (ModelType != 'styles') {
        state.supportedModels = await getSupportedModels();
        injectTypeStyles(state.supportedModels);
        state.ModelsByVersion = await getModelData(state.cache_key + '_version');
        state.VersionCacheData = await getCacheByKey(state.cache_key + '_version');
    }

    state.AllPath = await getAllPath(state.source_subdirname);

    if (LastCatType == 'Subdir') state.ModelList = await getModelDatabyPath(state.source_subdirname, LastCat);
    if (LastCatType == 'Version') state.ModelList = await getModelDatabyVersion(state.source_subdirname, state.cache_key + '_version', LastCat);

    if (typeof state.nodeHelper['cache_key'] !== "undefined" && state.nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof state.nodeHelper['sortbuttons'] === "object" && typeof state.nodeHelper['sortbuttons'][0] === "object" && state.nodeHelper['sortbuttons'][0].length > 0) {
            if (state.nodeHelper['sortbuttons'][0].indexOf("aScore") > -1) {
                state.AscoreDataResponse = await ReadAScores(state.nodeHelper['cache_key']);
            }
        }
    }

    if (typeof state.nodeHelper['cache_key'] !== "undefined" && state.nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof state.nodeHelper['sortbuttons'] === "object" && typeof state.nodeHelper['sortbuttons'][0] === "object" && state.nodeHelper['sortbuttons'][0].length > 0) {
            if (state.nodeHelper['sortbuttons'][0].indexOf("STime") > -1) {
                state.STimeDataResponse = await ReadSTimes(state.nodeHelper['cache_key']);
            }
        }
    }

    if (typeof state.nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof state.nodeHelper['sortbuttons'] === "object" && typeof state.nodeHelper['sortbuttons'][0] === "object" && state.nodeHelper['sortbuttons'][0].length > 0) {
            if (state.nodeHelper['sortbuttons'][0].indexOf("Date") > -1) {
                state.FileDateResponse = await ReadFileDate(state.nodeHelper['subdir']);
            }
        }
    }

    if (typeof state.nodeHelper['sortbuttons'] !== "undefined") {
        if (typeof state.nodeHelper['sortbuttons'] === "object" && typeof state.nodeHelper['sortbuttons'][0] === "object" && state.nodeHelper['sortbuttons'][0].length > 0) {
            if (state.nodeHelper['sortbuttons'][0].indexOf("Similarity") > -1) {
                state.SimilarityDataResponse = await ReadSimilarity(state.source_subdirname, PreviewPath, SelectedModel);
            }
        }
    }

    if (state.nodeHelper['subdir'] == 'checkpoints' && state.nodeHelper['sortbuttons'][0].indexOf("Symlink") > -1) {
        state.FileLinkResponse = await ReadFileSymlink(state.nodeHelper['subdir']);
    }

    state.RawImageDataResponse = {};
    state.RawImageDataResponse = await modelImageData(state.source_subdirname, PreviewPath);

    if (state.ModelsByVersion !== false && state.ModelsByVersion != null && Object.keys(state.ModelsByVersion).length > 0 && state.source_subdirname != 'styles' && state.AutoFilter !== true) {
        version_html = createTypeMenu(state.ModelsByVersion, state.supportedModels, LastCat, LastCatType);
    }

    if (state.AllPath !== false && state.AllPath != null && state.AllPath.length > 0 && state.AutoFilter !== true) {
        if (typeof state.nodeHelper['sortbuttons'] !== "object" || typeof state.nodeHelper['sortbuttons'][0] !== "object" || state.nodeHelper['sortbuttons'][0].indexOf("Path") > -1) {
            type_html = createPathMenu(state.AllPath, ShowHidden, LastCat, LastCatType);
        }
    }

    subdir_tabs.innerHTML = menu_html + type_html + version_html + ' <input type="text" name="ckptfilter" placeholder="filter" class="filter_input"> <button type="button" class="filter_clear">Clear filter</button>';

    var sortbuttons = state.nodeHelper['sortbuttons'];
    var sort_string = "";
    if (typeof sortbuttons === "object" && typeof sortbuttons[0] === "object" && sortbuttons[0].length > 0) {
        sort_string = createSortButtons(sortbuttons[0]);
    }
    subdir_tabs.innerHTML += sort_string;

    var fondModel = null;
    if (state.ModelList.includes(SelectedModel) || state.ModelList.includes(LastCat + '\\' + SelectedModel)) {
        var index_pre = (state.ModelList.indexOf(SelectedModel) + state.ModelList.indexOf(LastCat + '\\' + SelectedModel)) + 1;
        fondModel = state.ModelList[index_pre];
        if (index_pre !== -1) await state.ModelList.splice(index_pre, 1);
    }

    if (ModelType == 'styles' && fondModel != null) SelectedModel = fondModel;

    if (!state.ModelList.includes(SelectedModel)) {
        if (fondModel == null) {
            await state.ModelList.unshift(SelectedModel);
        } else {
            await state.ModelList.unshift(fondModel);
        }
    }

    setText('div#primere_visual_modal div.modal_header label.ckpt-counter', String(state.ModelList.length));
    setText('div#primere_visual_modal div.modal_header label.ckpt-name', LastCat);
    setText('div#primere_visual_modal div.modal_header label.ckpt-ver', LastCatType);

    var CKPTElements = 0;
    modal.setAttribute('style','display: block; width: 80%; height: 70%;');

    for (var checkpoint of state.ModelList) {
        let firstletter = checkpoint.charAt(0);
        let pathLastIndex = checkpoint.lastIndexOf('\\');
        let ckptName_full = checkpoint.substring(pathLastIndex + 1);
        let dotLastIndex = ckptName_full.lastIndexOf('.');
        var ckptName = ckptName_full.substring(0, dotLastIndex);

        if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((checkpoint.match('^NSFW') && ShowHidden === true) || !checkpoint.match('^NSFW'))) {
            var CategoryName = 'Unknown';

            for (const [ver_index, ver_value] of Object.entries(state.ModelsByVersion)) {
                if (ver_value.includes(ckptName)) CategoryName = ver_index;
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
        if (filterInput) filterInput.value = savedfilter;
        previewFilter(savedfilter);
    }
    const sortDirectionButton = document.querySelector('button.preview_sort_direction');
    if (sortDirectionButton) sortDirectionButton.textContent = state.operator;
    if (state.operator != 'ASC') {
        document.querySelector('div.subdirtab button.preview_sort_direction')?.classList.add("selected_path");
    }
    document.querySelector('div.subdirtab button.preview_sort[data-sortsource="' + state.sortType + '"]')?.classList.add('selected_path');
    previewSorter(state.operator, state.sortType);

    if (!document.querySelector('#primere_visual_modal div.subdirtab button.selected_path')) {
        document.querySelector('#primere_visual_modal div.subdirtab button[data-ckptsubdir="All"]')?.classList.add('selected_path');
    }
}

function previewSorter(operator, sortType) {
    const container = document.querySelector('div.primere-modal-content.ckpt-container.ckpt-grid-layout');
    if (!container) return;
    const cards = Array.from(container.querySelectorAll('div.visual-ckpt:not(.visual-ckpt-selected)'));
    cards.sort(function (a, b) {
        var aVal = a.dataset[sortType] ?? '';
        var bVal = b.dataset[sortType] ?? '';

        if (isNaN(parseInt(aVal)) && isNaN(parseInt(bVal))) {
            aVal = String(aVal).toUpperCase();
            bVal = String(bVal).toUpperCase();
            if (operator == 'ASC') return aVal < bVal ? -1 : (aVal > bVal ? 1 : 0);
            return aVal > bVal ? -1 : (aVal < bVal ? 1 : 0);
        }

        if (operator == 'ASC') return parseInt(aVal) - parseInt(bVal);
        return parseInt(bVal) - parseInt(aVal);
    });

    for (const card of cards) container.appendChild(card);
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
            if (parentCard) parentCard.style.display = '';
            filteredCheckpoints++;
        } else {
            if (parentCard) parentCard.style.display = 'none';
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

    if (ckptName == 'None') return;

    var versionString = 'Unknown';
    if (CategoryName) {
        var titleText = CategoryName + ' checkpoint. Select right version of additional networks.';
        versionString = CategoryName;
        if (ModelType != 'styles') {
            versionWidget = '<div class="ckpt-version ' + versionString + '-ckpt" title="' + titleText + '"></div>';
        } else {
            var SelModelIndex = SelectedModel.replaceAll(' ', "_");
            if (state.RawImageDataResponse.hasOwnProperty(SelectedModel.replaceAll(' ', "_")) === true && checkpoint == SelectedModel) {
                var selected_full = state.RawImageDataResponse[SelectedModel.replaceAll(' ', "_")];
                var selful_first = selected_full.charAt(0);
                var substringStart = 0;
                if (selful_first == '\\') substringStart = 1;
                path_only = selected_full.substring(substringStart, selected_full.lastIndexOf("\\"));
            }
            if (path_only == "") path_only = 'Root';
            versionWidget = '<div class="ckpt-version" title="Saved prompt on ' + path_only + ' category">&nbsp' + path_only + '&nbsp</div>';
        }
    }

    finalName = finalName.replaceAll(' ', "_");
    finalName = finalName.substring(pathLastIndex + 1);

    var card_html = '<div class="checkpoint-name background-' + versionString + '">' + ckptName.replaceAll('_', " ") + '</div>' + versionWidget;
    var missingimgsrc = prwPath + '/images/missing.jpg';
    card.classList.add('visual-ckpt', 'version-' + versionString);

    if (state.FileDateResponse.hasOwnProperty(ckptName) === true) {
        card.dataset.date = String(state.FileDateResponse[ckptName]);
    } else {
        card.dataset.date = '0';
    }

    if (state.FileLinkResponse.hasOwnProperty(ckptName) === true) {
        var unetname = state.FileLinkResponse[ckptName];
        card.dataset.symlink = unetname;
        var unetnameShort = unetname;
        if (unetname == 'diffusion_models') unetnameShort = 'DiMo';
        if (unetname == 'diffusers') unetnameShort = 'Diff';
        symlinkWidget = '<div class="visual-symlink-type" title="Symlinked from: ' + unetname + '">' + unetnameShort + '</div>';
        card_html += symlinkWidget;
    } else {
        card.dataset.symlink = "";
    }

    if (SelectedModel === checkpoint) card.classList.add('visual-ckpt-selected');

    if (state.AscoreDataResponse != null && state.AscoreDataResponse.hasOwnProperty(ckptName) === true) {
        var aestString = state.AscoreDataResponse[ckptName];
        var aestArray = aestString.split("|");
        var aestAVGValue = Math.floor(aestArray[1] / aestArray[0]);
        card.dataset.ascore = String(aestAVGValue);
        var aeScorePercentLine = Math.floor(((aestAVGValue - state.aeScoreMin) / (state.aeScoreMax - state.aeScoreMin)) * 100);
        if (aeScorePercentLine < 0) aeScorePercentLine = 0;
        if (aeScorePercentLine > 100) aeScorePercentLine = 100;
        aesthWidget = '<div class="visual-aesthetic-score" title="' + aestAVGValue + ' / ' + aeScorePercentLine + '%">' + aestAVGValue + '<span> - ' + aeScorePercentLine + '%</span><hr style="width: ' + aeScorePercentLine + '%"></div>';
        card_html += aesthWidget;
    } else {
        card.dataset.ascore = '0';
    }

    if (state.STimeDataResponse != null && state.STimeDataResponse.hasOwnProperty(ckptName) === true) {
        var stimeString = state.STimeDataResponse[ckptName];
        var stimestArray = stimeString.split("|");
        var stimestAVGValue = Math.floor(stimestArray[1] / stimestArray[0]);
        card.dataset.stime = String(stimestAVGValue);
        stimeWidget = '<div class="visual-stime" title="Average sampling time: ' + stimestAVGValue + ' sec">' + stimestAVGValue + 's</div>';
        card_html += stimeWidget;
    } else {
        card.dataset.stime = '0';
    }

    if (state.SimilarityDataResponse != null && state.SimilarityDataResponse.hasOwnProperty(ckptName) === true) {
        card.dataset.similarity = String(Math.floor(state.SimilarityDataResponse[ckptName] * 100));
    } else {
        card.dataset.similarity = '0';
    }

    card.dataset.name = ckptName;
    if (CategoryName == 'StableCascade') CategoryName = 'Cascade';
    card.dataset.version = CategoryName;
    card.dataset.path = path_only;

    if (state.RawImageDataResponse != null && Object.keys(state.RawImageDataResponse).length > 0) {
        if (state.RawImageDataResponse.hasOwnProperty(finalName) === true) {
            if (state.PreviewPath == false) {
                var imgsrc = 'data:image/jpeg;charset=utf-8;base64,  ' + state.RawImageDataResponse[finalName];
                card_html += '<img src="data:image/jpeg;charset=utf-8;base64,  ' + state.RawImageDataResponse[finalName] + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '" data-ckptver="' + CategoryName + '">';
            } else {
                var imgsrc = prwPath + '/images/' + state.RawImageDataResponse[finalName];
                card_html += '<img src="' + prwPath + '/images/' + state.RawImageDataResponse[finalName] + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '" data-ckptver="' + CategoryName + '">';
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
    waitForImageToLoad(img).then(() => { return true; });
}

function createSortButtons(buttondata) {
    var sort_html = '<label class="sort_by_label"> | Sort by: </label>';
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
                if (ver_index === LastCat && LastCatType == 'Version') addWhiteClass = ' selected_path';
                version_html += '<button type="button" data-ckptver="' + ver_index + '" class="verfilter' + addWhiteClass + '">' + ver_index + '</button>';
            }
        }
    }
    version_html += '<label> <hr> </label>';
    return version_html;
}

function createPathMenu(AllPath, ShowHidden, LastCat, LastCatType) {
    var menu_html = '';
    for (var subdir of AllPath) {
        var addWhiteClass = "";
        let firstletter = subdir.charAt(0);
        var subdirName = subdir;
        if (firstletter === '.') subdirName = subdir.substring(1);
        if (((firstletter === '.' && ShowHidden === true) || firstletter !== '.') && ((subdir.match('^NSFW') && ShowHidden === true) || !subdir.match('^NSFW'))) {
            if (subdir === LastCat && LastCatType == 'Subdir') addWhiteClass = ' selected_path';
            menu_html += '<button type="button" data-ckptsubdir="' + subdir + '" class="subdirfilter' + addWhiteClass + '">' + subdirName + '</button>';
        }
    }
    return menu_html + '<label> <hr> </label>';
}

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

async function waitForImageToLoad(imageElement) {
    return new Promise(resolve => { imageElement.onload = resolve });
}
