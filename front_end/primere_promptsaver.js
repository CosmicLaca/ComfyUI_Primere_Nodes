import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let PrompteventListenerInit = false;
const realPath = "/extensions/ComfyUI_Primere_Nodes";
const validClasses = ['PrimerePrompt'];
let PositivePrompt = '';
let NegativePrompt = '';
let SubPath = '';
let Model = 'None';
let Orientation = 'None';
let PromptData = {}

app.registerExtension({
    name: "Primere.PromptSaver",

    async beforeRegisterNodeDef(nodeType, nodeData, app) { // 0
        if (validClasses.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                PrimerePromptSaverWidget.apply(this, [this, 'PrimerePromptSaver']);
            }
        }
    },
});

async function PrimerePromptSaverWidget(node, inputName) {
    if (inputName == 'PrimerePromptSaver') {
        node.name = inputName;
        const widget = {
            type: "preview_saver_widget",
            name: `w${inputName}`,
            callback: () => {
            },
        };

        node.addWidget("button", '💾 Save prompt to file...', null, () => {
            node.PromptSaver = new PromptSaver(node);
        });

        LoadedNode = node;
        return {widget: widget};
    }
}

class PromptSaver {
    constructor(node) {
        if (PrompteventListenerInit == false) {
            PromptModalHandler();
        }

        function PromptModalHandler() {
            var scripts = document.getElementsByTagName('script');
            var jqueryCheck = false;
            $(scripts).each(function(index, value) {
                if (value.src.includes('jquery-1.9.0.min.js')) {
                    jqueryCheck = true;
                    return false;
                }
            });

            PrompteventListenerInit = true;
            let head = document.getElementsByTagName('HEAD')[0];
            let js = document.createElement("script");
            if (jqueryCheck === false) {
                js.src = realPath + "/jquery/jquery-1.9.0.min.js";
                head.appendChild(js);
            }

            //js.onload = function(e) {
                $(document).ready(function () {
                    var promptmodal = null;
                    PromptData['replace'] = 0;

                    $('body').on("click", 'div#primere_promptsaver_modal button.promptmodal-closer', function () { // modal close
                        promptmodal = document.getElementById("primere_promptsaver_modal");
                        promptmodal.setAttribute('style', 'display: none; width: 30%; height: 60%;')
                    });

                    $('body').on("change", 'div#primere_promptsaver_modal select#name', function () { // name list change
                        if (this.value == '') {
                            $('div#primere_promptsaver_modal input#prompt_name').show();
                            $('span.prompt_replace_warning').hide()
                        } else {
                            $('div#primere_promptsaver_modal input#prompt_name').hide();
                            $('div#primere_promptsaver_modal input#prompt_name').val('');
                            $('span.prompt_replace_warning').show()
                        }
                    });

                    $('body').on("change", 'div#primere_promptsaver_modal select#preferred_subpath', function () { // preferred_subpath list change
                        if (this.value == '') {
                            $('div#primere_promptsaver_modal input#prompt_path').show();
                        } else {
                            $('div#primere_promptsaver_modal input#prompt_path').hide();
                            $('div#primere_promptsaver_modal input#prompt_path').val('');
                        }
                    });

                    $('body').on("click", 'div#primere_promptsaver_modal button.prompt_saver_button', async function () { // save
                        PromptData['prompt'] = $('div#primere_promptsaver_modal textarea[name="positive_prompt"]').val();
                        PromptData['negative_prompt'] = $('div#primere_promptsaver_modal textarea[name="negative_prompt"]').val();
                        PromptData['preferred_model'] = $('div#primere_promptsaver_modal input[name="model"]').val();
                        PromptData['preferred_orientation'] = $('div#primere_promptsaver_modal input[name="orientation"]').val();

                        if ($('div#primere_promptsaver_modal input#prompt_name').is(":visible")) {
                            PromptData['replace'] = 0;
                            PromptData['name'] = $('div#primere_promptsaver_modal input#prompt_name').val();
                            if ($('div#primere_promptsaver_modal select#name option').filter(function () {
                                return $(this).val().toLowerCase() == PromptData['name'].toLowerCase();
                            }).length) {
                                if (PromptData['name'].length < 1) {
                                    alert('Required prompt name missing...')
                                } else {
                                    alert('Prompt name already exist...')
                                }
                                return false;
                            }
                        } else {
                            PromptData['replace'] = 1;
                            PromptData['name'] = $('div#primere_promptsaver_modal select#name').val();
                        }

                        if ($('div#primere_promptsaver_modal input#prompt_path').is(":visible")) {
                            PromptData['preferred_subpath'] = $('div#primere_promptsaver_modal input#prompt_path').val();
                        } else {
                            PromptData['preferred_subpath'] = $('div#primere_promptsaver_modal select#preferred_subpath').val();
                        }

                        if (PromptData['prompt'].length < 3 || PromptData['name'].length < 1) {
                            alert('Required data missing...')
                            return false;
                        }

                        //alert(JSON.stringify(PromptData));
                        var isSaved = await savePromptData('stylecsv', 'styles', 'csv', JSON.stringify(PromptData));
                        if (isSaved == false) {
                            alert('Cannot save new prompt to CSV file.');
                        } else {
                            alert('New prompt: [' + PromptData['name'] + '] added to CSV file.')
                            promptmodal = document.getElementById("primere_promptsaver_modal");
                            promptmodal.setAttribute('style', 'display: none; width: 30%; height: 60%;')
                        }
                    });
                });
            //}
        }

        for (var px = 0; px < node.widgets.length; ++px) {
            if (node.widgets[px].name == 'positive_prompt') {
                PositivePrompt = node.widgets[px].value;
            }
            if (node.widgets[px].name == 'negative_prompt') {
                NegativePrompt = node.widgets[px].value;
            }
            if (node.widgets[px].name == 'subpath') {
                SubPath = node.widgets[px].value;
            }
            if (node.widgets[px].name == 'model') {
                Model = node.widgets[px].value;
            }
            if (node.widgets[px].name == 'orientation') {
                Orientation = node.widgets[px].value;
            }
        }

        if (PositivePrompt.length < 3) {
            alert('⛔ Positive prompt required...');
            return false;
        }

        setup_promptsaver_modal();
    }
}

async function setup_promptsaver_modal() {
    var container = null;
    var modal = null;
    modal = document.getElementById("primere_promptsaver_modal");
    if (!modal) {
        modal = document.createElement("div");
        modal.classList.add("comfy-modal");
        modal.setAttribute("id","primere_promptsaver_modal");
        modal.innerHTML = '<div class="promptmodal_header"><div class="prompt_modal_title">Save prompt to external file</div></div>';

        let container = document.createElement("div");
        container.classList.add("primere-promptsaver-modal-content", "prompt-container");
        modal.appendChild(container);

        document.body.appendChild(modal);
    } else {
        $('div#primere_promptsaver_modal div.promptmodal_header h3.prompt_modal_title').html('💾 Save prompt to file...');
    }

    var PromptData = await getPromptData('stylecsv', 'styles', 'csv', ['name', 'preferred_subpath']);

    container = modal.getElementsByClassName("prompt-container")[0];
    container.innerHTML = "<p></p><label>Prompt name:</label><select name='name' id='name'>";
    $('select#name').append($('<option>', {
        value: '',
        text : 'Add new name'
    }));
    $.each(PromptData['name'], function (i, item) {
        $('select#name').append($('<option>', {
            value: item,
            text: item
        }));
    });
    container.innerHTML += "<input type='text' name='prompt_name' id='prompt_name' value='' placeholder='Enter new prompt name'>";
    container.innerHTML += "<span class='prompt_replace_warning'>This setting will replace the existing prompt by selected name!</span>";
    $(container).find('span.prompt_replace_warning').hide();

    var category_match = false;
    container.innerHTML += "<label>Prompt category (subpath):</label><select name='preferred_subpath' id='preferred_subpath'>";
    $('select#preferred_subpath').append($('<option>', {
        value: '',
        text : 'Add new category'
    }));
    $.each(PromptData['preferred_subpath'], function (i, item) {
        if (SubPath == item) {
            category_match = true;
            $('select#preferred_subpath').append($('<option>', {
                value: item,
                text: item
            }).attr('selected', true));
        } else {
            $('select#preferred_subpath').append($('<option>', {
                value: item,
                text: item
            }));
        }
    });

    if (category_match == true) {
        container.innerHTML += "<input type='text' name='prompt_path' id='prompt_path' value='' placeholder='Enter new prompt category (subpath)'>";
        $(container).find('input#prompt_path').hide();
    } else {
        container.innerHTML += "<input type='text' name='prompt_path' id='prompt_path' value='" + SubPath + "' placeholder='Enter new prompt category (subpath)'>";
    }

    container.innerHTML += "<label>Positive prompt:</label><textarea name='positive_prompt' rows=4 cols=50>" + PositivePrompt + "</textarea>";
    container.innerHTML += "<label>Negative prompt:</label><textarea name='negative_prompt' rows=4 cols=50>" + NegativePrompt + "</textarea>";
    container.innerHTML += "<label>Preferred Model:</label><input type='text' name='model' value='" + Model + "' readonly='readonly'>";
    container.innerHTML += "<label>Preferred Orientation:</label><input type='text' name='orientation' value='" + Orientation + "' readonly='readonly'>";
    container.innerHTML += '<button type="button" class="promptmodal-closer">Close without save</button>';
    container.innerHTML += "<button type='button' class='prompt_saver_button'>Save prompt to external CSV file</button>";

    modal.setAttribute('style','display: block; width: 30%; height: 60%;');
}

// ************************* getPromptData PromptDataResponse
function getPromptData(folder, name, type, keys) {
    return new Promise((resolve, reject) => {
        api.addEventListener("PromptDataResponse", (event) => resolve(event.detail), true);
        postPromptDataList(folder, name, type, keys);
    });
}
function postPromptDataList(folder, name, type, keys) {
    const body = new FormData();
    body.append('folder', folder);
    body.append('name', name);
    body.append('type', type);
    body.append('keys', keys);
    api.fetchApi("/primere_prompt_data", {method: "POST", body,});
}

// ************************* savePromptData PromptDataSaveResponse
function savePromptData(folder, name, type, data) {
    return new Promise((resolve, reject) => {
        api.addEventListener("PromptDataSaveResponse", (event) => resolve(event.detail), true);
        postPromptSave(folder, name, type, data);
    });
}
function postPromptSave(folder, name, type, data) {
    const body = new FormData();
    body.append('folder', folder);
    body.append('name', name);
    body.append('type', type);
    body.append('promptdata', data);
    api.fetchApi("/primere_prompt_saver", {method: "POST", body,});
}