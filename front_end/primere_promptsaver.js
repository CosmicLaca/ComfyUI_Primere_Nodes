import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let PrompteventListenerInit = false;
const validClasses = ['PrimerePrompt'];
let PositivePrompt = '';
let NegativePrompt = '';
let SubPath = '';
let Model = 'None';
let Orientation = 'None';
let PromptData = {};

function isVisible(element) {
    return !!(element && element.offsetParent !== null);
}

function setVisible(element, visible) {
    if (element) {
        element.style.display = visible ? '' : 'none';
    }
}

function appendOption(selectElement, value, text, selected = false) {
    if (!selectElement) {
        return;
    }
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    option.selected = selected;
    selectElement.appendChild(option);
}

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
            PrompteventListenerInit = true;
            PromptData['replace'] = 0;

            document.body.addEventListener('click', async function (event) {
                const promptmodal = document.getElementById("primere_promptsaver_modal");
                if (!promptmodal) {
                    return;
                }

                if (event.target.closest('div#primere_promptsaver_modal button.promptmodal-closer')) {
                    promptmodal.setAttribute('style', 'display: none; width: 30%; height: 60%;');
                    return;
                }

                if (!event.target.closest('div#primere_promptsaver_modal button.prompt_saver_button')) {
                    return;
                }

                PromptData['prompt'] = promptmodal.querySelector('textarea[name="positive_prompt"]')?.value ?? '';
                PromptData['negative_prompt'] = promptmodal.querySelector('textarea[name="negative_prompt"]')?.value ?? '';
                PromptData['preferred_model'] = promptmodal.querySelector('input[name="model"]')?.value ?? '';
                PromptData['preferred_orientation'] = promptmodal.querySelector('input[name="orientation"]')?.value ?? '';

                const promptNameInput = promptmodal.querySelector('input#prompt_name');
                const promptNameSelect = promptmodal.querySelector('select#name');
                const promptPathInput = promptmodal.querySelector('input#prompt_path');
                const promptPathSelect = promptmodal.querySelector('select#preferred_subpath');

                if (isVisible(promptNameInput)) {
                    PromptData['replace'] = 0;
                    PromptData['name'] = promptNameInput?.value ?? '';
                    const promptNameLower = PromptData['name'].toLowerCase();
                    const optionExists = Array.from(promptNameSelect?.options ?? []).some((option) => option.value.toLowerCase() === promptNameLower);
                    if (optionExists && PromptData['name'].length > 0) {
                        if (confirm('The [' + PromptData['name'] + '] prompt already exist. Replace existing prompt by this name?')) {
                            PromptData['replace'] = 1;
                        } else {
                            return false;
                        }
                    }
                } else {
                    PromptData['replace'] = 1;
                    PromptData['name'] = promptNameSelect?.value ?? '';
                }

                if (isVisible(promptPathInput)) {
                    PromptData['preferred_subpath'] = promptPathInput?.value ?? '';
                } else {
                    PromptData['preferred_subpath'] = promptPathSelect?.value ?? '';
                }

                if (PromptData['prompt'].length < 3 || PromptData['name'].length < 1) {
                    alert('Required data missing...');
                    return false;
                }

                const isSaved = await savePromptData('stylecsv', 'styles', 'csv', JSON.stringify(PromptData));
                if (isSaved == false) {
                    alert('Cannot save new prompt to CSV file.');
                } else {
                    alert('New prompt: [' + PromptData['name'] + '] added to CSV file.');
                    promptmodal.setAttribute('style', 'display: none; width: 30%; height: 60%;');
                }
            });

            document.body.addEventListener('change', function (event) {
                const promptmodal = document.getElementById("primere_promptsaver_modal");
                if (!promptmodal) {
                    return;
                }

                const nameSelect = event.target.closest('div#primere_promptsaver_modal select#name');
                if (nameSelect) {
                    const promptNameInput = promptmodal.querySelector('input#prompt_name');
                    const replaceWarning = promptmodal.querySelector('span.prompt_replace_warning');
                    if (nameSelect.value == '') {
                        setVisible(promptNameInput, true);
                        setVisible(replaceWarning, false);
                    } else {
                        setVisible(promptNameInput, false);
                        if (promptNameInput) {
                            promptNameInput.value = '';
                        }
                        setVisible(replaceWarning, true);
                    }
                    return;
                }

                const subpathSelect = event.target.closest('div#primere_promptsaver_modal select#preferred_subpath');
                if (subpathSelect) {
                    const promptPathInput = promptmodal.querySelector('input#prompt_path');
                    if (subpathSelect.value == '') {
                        setVisible(promptPathInput, true);
                    } else {
                        setVisible(promptPathInput, false);
                        if (promptPathInput) {
                            promptPathInput.value = '';
                        }
                    }
                }
            });
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
        modal.setAttribute("id", "primere_promptsaver_modal");
        modal.innerHTML = '<div class="promptmodal_header"><div class="prompt_modal_title">Save prompt to external file</div></div>';

        let container = document.createElement("div");
        container.classList.add("primere-promptsaver-modal-content", "prompt-container");
        modal.appendChild(container);

        document.body.appendChild(modal);
    } else {
        const modalTitle = document.querySelector('div#primere_promptsaver_modal div.promptmodal_header .prompt_modal_title');
        if (modalTitle) {
            modalTitle.textContent = '💾 Save prompt to file...';
        }
    }

    var PromptData = await getPromptData('stylecsv', 'styles', 'csv', ['name', 'preferred_subpath']);

    container = modal.getElementsByClassName("prompt-container")[0];
    container.innerHTML = "<p></p><label>Prompt name:</label><select name='name' id='name'>";

    const promptNameSelect = container.querySelector('select#name');
    appendOption(promptNameSelect, '', 'Add new name');
    for (const item of PromptData['name']) {
        appendOption(promptNameSelect, item, item);
    }

    container.innerHTML += "<input type='text' name='prompt_name' id='prompt_name' value='' placeholder='Enter new prompt name'>";
    container.innerHTML += "<span class='prompt_replace_warning'>This setting will replace the existing prompt by selected name!</span>";
    setVisible(container.querySelector('span.prompt_replace_warning'), false);

    var category_match = false;
    container.innerHTML += "<label>Prompt category (subpath):</label><select name='preferred_subpath' id='preferred_subpath'>";

    const promptSubpathSelect = container.querySelector('select#preferred_subpath');
    appendOption(promptSubpathSelect, '', 'Add new category');
    for (const item of PromptData['preferred_subpath']) {
        if (SubPath == item) {
            category_match = true;
            appendOption(promptSubpathSelect, item, item, true);
        } else {
            appendOption(promptSubpathSelect, item, item);
        }
    }

    if (category_match == true) {
        container.innerHTML += "<input type='text' name='prompt_path' id='prompt_path' value='' placeholder='Enter new prompt category (subpath)'>";
        setVisible(container.querySelector('input#prompt_path'), false);
    } else {
        container.innerHTML += "<input type='text' name='prompt_path' id='prompt_path' value='" + SubPath + "' placeholder='Enter new prompt category (subpath)'>";
    }

    container.innerHTML += "<label>Positive prompt:</label><textarea name='positive_prompt' rows=4 cols=50>" + PositivePrompt + "</textarea>";
    container.innerHTML += "<label>Negative prompt:</label><textarea name='negative_prompt' rows=4 cols=50>" + NegativePrompt + "</textarea>";
    container.innerHTML += "<label>Preferred Model:</label><input type='text' name='model' value='" + Model + "' readonly='readonly'>";
    container.innerHTML += "<label>Preferred Orientation:</label><input type='text' name='orientation' value='" + Orientation + "' readonly='readonly'>";
    container.innerHTML += '<button type="button" class="promptmodal-closer">Close without save</button>';
    container.innerHTML += "<button type='button' class='prompt_saver_button'>Save prompt to external CSV file</button>";

    modal.setAttribute('style', 'display: block; width: 30%; height: 60%;');
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
