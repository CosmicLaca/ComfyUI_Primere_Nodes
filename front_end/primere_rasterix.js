import { app } from "/scripts/app.js";

const CB_TONES    = ["highlights", "midtones", "shadows"];
const HS_CHANNELS = ["master", "red", "green", "blue"];
const ST_ZONES    = ["highlights", "midtones", "shadows", "blacks"];
const SH_MODES    = ["fine", "medium", "broad"];

const cbDefault = () => ({
    highlights: { cyan_red: 0, magenta_green: 0, yellow_blue: 0 },
    midtones:   { cyan_red: 0, magenta_green: 0, yellow_blue: 0 },
    shadows:    { cyan_red: 0, magenta_green: 0, yellow_blue: 0 },
});

const hsDefault = () => ({
    master: { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
    red:    { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
    green:  { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
    blue:   { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
});

const stDefault = () => ({
    highlights: 0,
    midtones:   0,
    shadows:    0,
    blacks:     0,
});

const shDefault = () => ({
    fine:   { shade_level: 0, shade_radius: 0 },
    medium: { shade_level: 0, shade_radius: 0 },
    broad:  { shade_level: 0, shade_radius: 0 },
});

async function rasterixLoad() {
    try {
        const resp = await fetch('/primere_rasterix_read');
        if (!resp.ok) return { color_balance: cbDefault(), hue_saturation: hsDefault(), selective_tone: stDefault(), shade: shDefault() };
        const data = await resp.json();

        const cb = data.color_balance || {};
        for (const t of CB_TONES)
            if (!cb[t]) cb[t] = { cyan_red: 0, magenta_green: 0, yellow_blue: 0 };

        const hs = data.hue_saturation || {};
        for (const ch of HS_CHANNELS)
            if (!hs[ch]) hs[ch] = { hue: 0, saturation: 0, lightness: 0, vibrance: 0 };

        const st = data.selective_tone || {};
        for (const z of ST_ZONES)
            if (st[z] === undefined) st[z] = 0;

        const sh = data.shade || {};
        for (const m of SH_MODES)
            if (!sh[m]) sh[m] = { shade_level: 0, shade_radius: 0 };

        return { color_balance: cb, hue_saturation: hs, selective_tone: st, shade: sh };
    } catch {
        return { color_balance: cbDefault(), hue_saturation: hsDefault(), selective_tone: stDefault(), shade: shDefault() };
    }
}

async function rasterixSave(section, data) {
    try {
        await fetch('/primere_rasterix_save', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ section, data }),
        });
    } catch (e) {
        console.warn('[Primere Rasterix] save failed:', e);
    }
}

app.registerExtension({
    name: "Primere.Rasterix",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PrimereRasterix") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated ? onNodeCreated.apply(this, []) : undefined;

            const node = this;
            const fw   = (name) => node.widgets?.find(w => w.name === name);

            // Color Balance
            const wTone  = fw("color_balance_tone");
            const wCR    = fw("color_balance_cyan_red");
            const wMG    = fw("color_balance_magenta_green");
            const wYB    = fw("color_balance_yellow_blue");

            // Hue / Saturation
            const wHsCh  = fw("hsl_channel");
            const wHsHue = fw("hsl_hue");
            const wHsSat = fw("hsl_saturation");
            const wHsLit = fw("hsl_lightness");
            const wHsVib = fw("hsl_vibrance");

            // Selective Tone
            const wStZone = fw("selective_tone_zone");
            const wStVal  = fw("selective_tone_value");

            // Shade
            const wShMode = fw("detail_mode");
            const wShLvl  = fw("shade_level");
            const wShRad  = fw("shade_radius");

            let cbStore  = cbDefault();
            let hsStore  = hsDefault();
            let stStore  = stDefault();
            let shStore  = shDefault();

            let prevTone  = wTone?.value   ?? "midtones";
            let prevHsCh  = wHsCh?.value   ?? "master";
            let prevStZn  = wStZone?.value  ?? "midtones";
            let prevShMd  = wShMode?.value  ?? "medium";
            let updating  = false;

            function updateHistogramDisplay(showInput) {
                const fname = showInput ? "input_histogram.jpg" : "output_histogram.jpg";
                const url   = `/extensions/ComfyUI_Primere_Nodes/images/${fname}?t=${Date.now()}`;
                if (!node.imgs) node.imgs = [new Image()];
                node.imgs[0].onload = () => app.canvas?.setDirty(true);
                node.imgs[0].src = url;
                app.canvas?.setDirty(true);
            }

            node.onExecuted = function() {
                const showInput = fw("show_input_histogram")?.value ?? false;
                if (node.imgs?.[0]) {
                    node.imgs[0].onload = () => app.canvas?.setDirty(true);
                    node.imgs[0].src = `/extensions/ComfyUI_Primere_Nodes/images/${showInput ? "input_histogram.jpg" : "output_histogram.jpg"}?t=${Date.now()}`;
                } else {
                    updateHistogramDisplay(showInput);
                }
            };

            // ── Color Balance ─────────────────────────────────────────────────
            function applyCbSliders(tone) {
                if (!wCR) return;
                updating = true;
                const v  = cbStore[tone] || { cyan_red: 0, magenta_green: 0, yellow_blue: 0 };
                wCR.value = v.cyan_red;
                wMG.value = v.magenta_green;
                wYB.value = v.yellow_blue;
                updating  = false;
                app.canvas?.setDirty(true);
            }
            function captureCbSliders(tone) {
                if (!wCR) return;
                cbStore[tone] = { cyan_red: wCR.value, magenta_green: wMG.value, yellow_blue: wYB.value };
                rasterixSave('color_balance', cbStore);
            }

            // ── Hue / Saturation ──────────────────────────────────────────────
            function applyHsSliders(ch) {
                if (!wHsHue) return;
                updating = true;
                const v  = hsStore[ch] || { hue: 0, saturation: 0, lightness: 0, vibrance: 0 };
                wHsHue.value = v.hue;
                wHsSat.value = v.saturation;
                wHsLit.value = v.lightness;
                wHsVib.value = v.vibrance;
                updating = false;
                app.canvas?.setDirty(true);
            }
            function captureHsSliders(ch) {
                if (!wHsHue) return;
                hsStore[ch] = { hue: wHsHue.value, saturation: wHsSat.value, lightness: wHsLit.value, vibrance: wHsVib.value };
                rasterixSave('hue_saturation', hsStore);
            }

            // ── Selective Tone ────────────────────────────────────────────────
            function applyStSlider(zone) {
                if (!wStVal) return;
                updating = true;
                wStVal.value = stStore[zone] ?? 0;
                updating = false;
                app.canvas?.setDirty(true);
            }
            function captureStSlider(zone) {
                if (!wStVal) return;
                stStore[zone] = wStVal.value;
                rasterixSave('selective_tone', stStore);
            }

            // ── Shade ─────────────────────────────────────────────────────────
            function applyShSliders(mode) {
                if (!wShLvl) return;
                updating = true;
                const v  = shStore[mode] || { shade_level: 0, shade_radius: 0 };
                wShLvl.value = v.shade_level;
                wShRad.value = v.shade_radius;
                updating = false;
                app.canvas?.setDirty(true);
            }
            function captureShSliders(mode) {
                if (!wShLvl) return;
                shStore[mode] = { shade_level: wShLvl.value, shade_radius: wShRad.value };
                rasterixSave('shade', shStore);
            }

            // ── Initial load ──────────────────────────────────────────────────
            rasterixLoad().then(loaded => {
                cbStore = loaded.color_balance;
                hsStore = loaded.hue_saturation;
                stStore = loaded.selective_tone;
                shStore = loaded.shade;
                applyCbSliders(prevTone);
                applyHsSliders(prevHsCh);
                applyStSlider(prevStZn);
                applyShSliders(prevShMd);
            });

            // ── Widget change handler ─────────────────────────────────────────
            node.onWidgetChanged = function (name, value) {
                if (updating) return;

                if (name === "color_balance_tone") {
                    captureCbSliders(prevTone);
                    applyCbSliders(value);
                    prevTone = value;
                } else if (
                    name === "color_balance_cyan_red"      ||
                    name === "color_balance_magenta_green" ||
                    name === "color_balance_yellow_blue"
                ) {
                    captureCbSliders(wTone?.value ?? prevTone);

                } else if (name === "hsl_channel") {
                    captureHsSliders(prevHsCh);
                    applyHsSliders(value);
                    prevHsCh = value;
                } else if (
                    name === "hsl_hue"        ||
                    name === "hsl_saturation" ||
                    name === "hsl_lightness"  ||
                    name === "hsl_vibrance"
                ) {
                    captureHsSliders(wHsCh?.value ?? prevHsCh);

                } else if (name === "selective_tone_zone") {
                    captureStSlider(prevStZn);
                    applyStSlider(value);
                    prevStZn = value;
                } else if (name === "selective_tone_value") {
                    captureStSlider(wStZone?.value ?? prevStZn);

                } else if (name === "detail_mode") {
                    captureShSliders(prevShMd);
                    applyShSliders(value);
                    prevShMd = value;
                } else if (
                    name === "shade_level" ||
                    name === "shade_radius"
                ) {
                    captureShSliders(wShMode?.value ?? prevShMd);

                } else if (name === "show_input_histogram") {
                    updateHistogramDisplay(value);
                }
            };
        };
    },
});
