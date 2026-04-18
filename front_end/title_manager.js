import { app } from "/scripts/app.js";

// Add node names here to enable backend-driven title separators on those nodes.
const TITLE_MANAGED_NODES = [
    "PrimereRasterix", "PrimereModelControl", "PrimereMetaSave", "PrimereRasterixLens", "PrimereMetaHandler", "PrimereCLIPEncoder"
];

const titleConfigCache = {};
const titleConfigPromise = {};
const TITLE_WIDGET_HEIGHT = 30;
const TITLE_TOTAL_CHARS = 40;
const TITLE_PREVIEW_HOVER_DELAY_MS = 1000;

const titlePreviewHoverState = {
    token: 0,
    key: null,
};

function hexToRgb(hex) {
    const clean = String(hex || "").trim().replace("#", "");
    if (![3, 6].includes(clean.length)) return null;
    const full = clean.length === 3 ? clean.split("").map(c => c + c).join("") : clean;
    const n = parseInt(full, 16);
    if (Number.isNaN(n)) return null;
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function bestTextColor(bgHex) {
    const rgb = hexToRgb(bgHex);
    if (!rgb) return "#F4F4F4";
    const luma = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255.0;
    return luma > 0.6 ? "#1B1B1B" : "#F4F4F4";
}

async function getNodeTitleConfig(nodeName) {
    if (titleConfigCache[nodeName]) return titleConfigCache[nodeName];
    if (!titleConfigPromise[nodeName]) {
        const q = new URLSearchParams({ node_name: nodeName });
        titleConfigPromise[nodeName] = fetch(`/primere_titles?${q.toString()}`)
            .then(resp => resp.json())
            .then(data => {
                const sections = Array.isArray(data?.sections) ? data.sections : [];
                titleConfigCache[nodeName] = sections;
                return sections;
            })
            .catch(() => []);
    }
    return titleConfigPromise[nodeName];
}

function makeTitleWidget(node, section) {
    const rawTitle = String(section?.title || "").trim();
    if (!rawTitle) return null;

    const widget = node.addWidget("button", formatTitleLine(rawTitle), null, () => {}, { serialize: false });
    widget.options = widget.options || {};
    widget.options.serialize = false;
    widget.serialize = false;
    widget.__primereTitleWidget = true;

    const bgColor = section?.color ? String(section.color) : null;
    const textColor = section?.text_color ? String(section.text_color) : bestTextColor(bgColor);
    const label = section?.label ? String(section.label) : "";
    const sectionName = section?.name ? String(section.name) : "";

    const titleText = widget.name;
    const color = bgColor || "#3A3A3A";
    const tColor = textColor || "#F4F4F4";

    widget.draw = function (ctx, nodeObj, widgetWidth, y, h) {
        const x = 15;
        const w = widgetWidth - 30;
        const radius = 4;
        const hh = Math.max(14, h - 6);
        const yy = y + 3;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x + radius, yy);
        ctx.lineTo(x + w - radius, yy);
        ctx.quadraticCurveTo(x + w, yy, x + w, yy + radius);
        ctx.lineTo(x + w, yy + hh - radius);
        ctx.quadraticCurveTo(x + w, yy + hh, x + w - radius, yy + hh);
        ctx.lineTo(x + radius, yy + hh);
        ctx.quadraticCurveTo(x, yy + hh, x, yy + hh - radius);
        ctx.lineTo(x, yy + radius);
        ctx.quadraticCurveTo(x, yy, x + radius, yy);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        ctx.fillStyle = tColor;
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(titleText, widgetWidth / 2, yy + hh / 2 + 0.5);
        ctx.restore();
    };

    widget.__primereTitleMeta = {
        label,
        name: sectionName,
    };
    return {
        widget,
        before: String(section?.before || ""),
        after: String(section?.after || ""),
    };
}

function formatTitleLine(rawTitle, totalChars = TITLE_TOTAL_CHARS) {
    const title = String(rawTitle || "").trim();
    if (!title) return "";

    const minSide = 1;
    const spaces = 2; // one before and one after title
    const available = totalChars - title.length - spaces;

    if (available < minSide * 2) {
        return `${"─".repeat(minSide)} ${title} ${"─".repeat(minSide)}`;
    }

    const left = Math.floor(available / 2);
    const right = available - left;
    return `${"─".repeat(left)} ${title} ${"─".repeat(right)}`;
}

function insertTitleWidgets(node, sections) {
    if (!node?.widgets || node.__primereTitleManagerApplied) return;
    if (!Array.isArray(sections) || sections.length === 0) return;

    const created = [];
    for (const section of sections) {
        const item = makeTitleWidget(node, section);
        if (!item) continue;
        created.push(item);
    }
    for (const item of created.reverse()) {
        const from = node.widgets.indexOf(item.widget);
        if (from >= 0) node.widgets.splice(from, 1);
        let insertAt = node.widgets.length;
        if (item.before) {
            const beforeIndex = node.widgets.findIndex(w => w.name === item.before);
            if (beforeIndex >= 0) insertAt = beforeIndex;
        } else if (item.after) {
            const afterIndex = node.widgets.findIndex(w => w.name === item.after);
            if (afterIndex >= 0) insertAt = afterIndex + 1;
        }
        node.widgets.splice(insertAt, 0, item.widget);
    }

    node.__primereTitleManagerApplied = true;
    node.setDirtyCanvas?.(true, true);
}

function waitForHoverDelay(delayMs = TITLE_PREVIEW_HOVER_DELAY_MS) {
    return new Promise((resolve) => setTimeout(resolve, delayMs));
}

function clearPendingTitlePreview() {
    titlePreviewHoverState.token += 1;
    titlePreviewHoverState.key = null;
}

async function scheduleTitlePreview(sectionName, x, y, hoverKey) {
    titlePreviewHoverState.token += 1;
    const token = titlePreviewHoverState.token;
    titlePreviewHoverState.key = hoverKey;
    await waitForHoverDelay();
    if (titlePreviewHoverState.token !== token || titlePreviewHoverState.key !== hoverKey) return;
    showTitlePreview(sectionName, x, y);
}

function ensureTitleTooltip() {
    let box = document.querySelector("div#primere_title_hover");
    if (box) return box;
    box = document.createElement("div");
    box.id = "primere_title_hover";
    /* box.style.cssText = [
        "display:none",
        "position:fixed",
        "z-index:99999",
        "max-width:360px",
        "padding:10px 12px",
        "border-radius:8px",
        "background:rgba(22,24,28,0.96)",
        "border:1px solid rgba(255,255,255,0.12)",
        "box-shadow:0 8px 24px rgba(0,0,0,0.35)",
        "color:#EDEFF4",
        "font:12px/1.4 Arial, sans-serif",
        "pointer-events:none",
        "white-space:normal",
    ].join(";"); */
    document.body.appendChild(box);
    return box;
}

function hideTitleTooltip() {
    const box = ensureTitleTooltip();
    box.style.display = "none";
}

function ensureTitlePreviewBox() {
    let box = document.querySelector("div#primere_previewbox_title");
    if (!box) {
        box = document.createElement("div");
        box.id = "primere_previewbox_title";
        /* box.style.cssText = [
            "display:none",
            "position:fixed",
            "z-index:99998",
            "padding:6px",
            "border-radius:8px",
            "background:rgba(18,20,24,0.96)",
            "border:1px solid rgba(255,255,255,0.12)",
            "box-shadow:0 8px 24px rgba(0,0,0,0.35)",
            "pointer-events:none",
        ].join(";"); */
        const closeButton = document.createElement("div");
        closeButton.className = "preview_closebutton";
        closeButton.textContent = "X";

        const img = document.createElement("img");
        img.className = "previewbox_image";
        /* img.style.cssText = [
            "display:block",
            "max-width:560px",
            "max-height:180px",
            "border-radius:4px",
        ].join(";"); */
        box.appendChild(closeButton);
        box.appendChild(img);
        document.body.appendChild(box);
    }

    if (!box.__primereCloseHandlerBound) {
        box.addEventListener("click", (event) => {
            if (event.target?.closest("div.preview_closebutton")) {
                hideTitlePreview();
                clearPendingTitlePreview();
            }
        });
        box.__primereCloseHandlerBound = true;
    }

    return box;
}

function hideTitlePreview() {
    const box = ensureTitlePreviewBox();
    box.style.display = "none";
}

function showTitlePreview(sectionName, x, y) {
    if (!sectionName) {
        hideTitlePreview();
        return;
    }
    const box = ensureTitlePreviewBox();
    const img = box.querySelector("img.previewbox_image");
    if (!img) return;

    const src = `/extensions/ComfyUI_Primere_Nodes/images/sections_titles/${encodeURIComponent(sectionName)}.jpg?t=${Date.now()}`;
    img.onload = () => {
        box.style.left = `${x + 12}px`;
        box.style.top = `${y + 12}px`;
        box.style.display = "block";
    };
    img.onerror = () => {
        hideTitlePreview();
    };
    img.src = src;
}

function showTitleTooltip(text, x, y) {
    const box = ensureTitleTooltip();
    box.textContent = text;
    box.style.left = `${x + 12}px`;
    box.style.top = `${y + 12}px`;
    box.style.display = "block";
}

function handleTitleHover(node, event, pos) {
    if (!event || event.type !== "pointermove" || !Array.isArray(node?.widgets)) {
        hideTitleTooltip();
        return;
    }

    const nodeWidth = Number(node?.size?.[0] || 300);
    const xMin = 15;
    const xMax = Math.max(xMin + 20, nodeWidth - 15);
    const rightHalfStart = xMin + (xMax - xMin) / 2;

    for (const w of node.widgets) {
        const meta = w?.__primereTitleMeta;
        if (!meta?.label && !meta?.name) continue;
        const y0 = Number(w.last_y || 0);
        const y1 = y0 + TITLE_WIDGET_HEIGHT;
        const insideY = pos[1] >= y0 && pos[1] <= y1;
        const insideLeftHalf = pos[0] >= xMin && pos[0] < rightHalfStart;
        const insideRightHalf = pos[0] >= rightHalfStart && pos[0] <= xMax;
        if (insideY && insideRightHalf) {
            clearPendingTitlePreview();
            hideTitlePreview();
            showTitleTooltip(meta.label, event.clientX, event.clientY);
            return;
        }
        if (insideY && insideLeftHalf && meta.name) {
            hideTitleTooltip();
            //showTitlePreview(meta.name, event.clientX, event.clientY);
            const hoverKey = `${node.id || "node"}:${meta.name}`;
            if (titlePreviewHoverState.key !== hoverKey) {
                hideTitlePreview();
                scheduleTitlePreview(meta.name, event.clientX, event.clientY, hoverKey);
            }
            return;
        }
    }

    clearPendingTitlePreview();
    hideTitleTooltip();
    hideTitlePreview();
}

function attachTitleHoverHandlers(node) {
    if (!node || node.__primereTitleHoverBound) return;
    const prevMove = node.onMouseMove;
    const prevLeave = node.onMouseLeave;

    node.onMouseMove = function (event, pos, graphcanvas) {
        if (typeof prevMove === "function") {
            prevMove.call(this, event, pos, graphcanvas);
        }
        handleTitleHover(this, event, pos);
    };

    node.onMouseLeave = function (event, pos, graphcanvas) {
        if (typeof prevLeave === "function") {
            prevLeave.call(this, event, pos, graphcanvas);
        }
        clearPendingTitlePreview();
        hideTitleTooltip();
        hideTitlePreview();
    };

    node.__primereTitleHoverBound = true;
}

function attachTitleSerializationGuard(node) {
    if (!node || node.__primereTitleSerializeGuardBound) return;
    const prevOnSerialize = node.onSerialize;
    node.onSerialize = function (o) {
        if (typeof prevOnSerialize === "function") {
            prevOnSerialize.call(this, o);
        }
        if (!o || !Array.isArray(o.widgets_values) || !Array.isArray(this.widgets)) return;
        const titleIndexes = new Set();
        for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i]?.__primereTitleWidget) {
                titleIndexes.add(i);
            }
        }
        if (titleIndexes.size > 0) {
            o.widgets_values = o.widgets_values.filter((_, idx) => !titleIndexes.has(idx));
        }
    };
    node.__primereTitleSerializeGuardBound = true;
}

app.registerExtension({
    name: "Primere.TitleManager",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TITLE_MANAGED_NODES.includes(nodeData.name)) return;

        const applyTitleManager = (node) => {
            getNodeTitleConfig(nodeData.name).then((sections) => {
                insertTitleWidgets(node, sections);
                attachTitleHoverHandlers(node);
                attachTitleSerializationGuard(node);
            });
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
            const node = this;
            setTimeout(() => applyTitleManager(node), 0);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            if (config && Array.isArray(config.widgets_values)) {
                config.widgets_values = config.widgets_values.filter((v) => {
                    if (typeof v !== "string") return true;
                    return !/^─+\s.*\s─+$/.test(v.trim());
                });
            }
            onConfigure ? onConfigure.apply(this, [config]) : undefined;
            const node = this;
            setTimeout(() => applyTitleManager(node), 0);
        };

    },
});
