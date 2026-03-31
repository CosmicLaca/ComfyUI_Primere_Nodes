import { app } from "/scripts/app.js";

// Add node names here to enable backend-driven title separators on those nodes.
const TITLE_MANAGED_NODES = [
    "PrimereRasterix",
];

const titleConfigCache = {};
const titleConfigPromise = {};

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
        titleConfigPromise[nodeName] = fetch(`/primere_rasterix_titles?${q.toString()}`)
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

    const widget = node.addWidget("button", `──── ${rawTitle} ────`, null, () => {}, { serialize: false });
    widget.options = widget.options || {};
    widget.options.serialize = false;

    const bgColor = section?.color ? String(section.color) : null;
    const textColor = section?.text_color ? String(section.text_color) : bestTextColor(bgColor);

    if (bgColor || section?.text_color) {
        const label = widget.name;
        widget.draw = function (ctx, nodeObj, widgetWidth, y, h) {
            const x = 15;
            const w = widgetWidth - 30;
            const radius = 4;
            const hh = h - 6;
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
            ctx.fillStyle = bgColor || "#3A3A3A";
            ctx.fill();

            ctx.fillStyle = textColor || "#F4F4F4";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(label, widgetWidth / 2, yy + hh / 2 + 0.5);
            ctx.restore();
        };
    }
    return widget;
}

function insertTitleWidgets(node, sections) {
    if (!node?.widgets || node.__primereTitleManagerApplied) return;
    if (!Array.isArray(sections) || sections.length === 0) return;

    const created = [];
    for (const section of sections) {
        const widget = makeTitleWidget(node, section);
        if (!widget) continue;
        created.push({
            widget,
            before: String(section?.before || ""),
            after: String(section?.after || ""),
        });
    }

    for (const item of created.reverse()) {
        const from = node.widgets.indexOf(item.widget);
        if (from >= 0) node.widgets.splice(from, 1);
        let insertAt = node.widgets.length;

        if (item.before) {
            const beforeIndex = node.widgets.findIndex(w => w.name === item.before);
            if (beforeIndex >= 0) {
                insertAt = beforeIndex;
            }
        } else if (item.after) {
            const afterIndex = node.widgets.findIndex(w => w.name === item.after);
            if (afterIndex >= 0) {
                insertAt = afterIndex + 1;
            }
        }

        node.widgets.splice(insertAt, 0, item.widget);
    }

    node.__primereTitleManagerApplied = true;
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Primere.TitleManager",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TITLE_MANAGED_NODES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
            const node = this;
            getNodeTitleConfig(nodeData.name).then((sections) => {
                insertTitleWidgets(node, sections);
            });
        };
    },
});
