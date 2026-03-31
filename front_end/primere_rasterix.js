import { app } from "/scripts/app.js";

const CB_TONES      = ["highlights", "midtones", "shadows"];
const HS_CHANNELS   = ["master", "red", "green", "blue"];
const ST_ZONES      = ["highlights", "midtones", "shadows", "blacks"];
const SH_MODES      = ["fine", "medium", "broad"];
const POST_CHANNELS = ["Red", "Green", "Blue"];

function buildFilmPresetMap(allPresets) {
    const byType = {};
    for (const name of allPresets) {
        const match = String(name).match(/_([A-Z][A-Z0-9]*)$/);
        if (!match) continue;
        const type = match[1];
        if (!Array.isArray(byType[type])) byType[type] = [];
        byType[type].push(name);
    }
    return byType;
}

const cbDefault   = () => ({
    highlights: { cyan_red: 0, magenta_green: 0, yellow_blue: 0 },
    midtones:   { cyan_red: 0, magenta_green: 0, yellow_blue: 0 },
    shadows:    { cyan_red: 0, magenta_green: 0, yellow_blue: 0 },
});
const hsDefault   = () => ({
    master: { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
    red:    { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
    green:  { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
    blue:   { hue: 0, saturation: 0, lightness: 0, vibrance: 0 },
});
const stDefault   = () => ({ highlights: 0, midtones: 0, shadows: 0, blacks: 0 });
const shDefault   = () => ({
    fine:   { shade_level: 0, shade_radius: 0 },
    medium: { shade_level: 0, shade_radius: 0 },
    broad:  { shade_level: 0, shade_radius: 0 },
});
const postDefault = () => ({ Red: 255, Green: 255, Blue: 255 });

async function rasterixLoad() {
    try {
        const resp = await fetch('/primere_rasterix_read');
        if (!resp.ok) return {
            color_balance: cbDefault(), hue_saturation: hsDefault(),
            selective_tone: stDefault(), shade: shDefault(), posterize: postDefault(),
        };
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

        const post = data.posterize || {};
        for (const ch of POST_CHANNELS)
            if (post[ch] === undefined) post[ch] = 255;

        return { color_balance: cb, hue_saturation: hs, selective_tone: st, shade: sh, posterize: post };
    } catch {
        return {
            color_balance: cbDefault(), hue_saturation: hsDefault(),
            selective_tone: stDefault(), shade: shDefault(), posterize: postDefault(),
        };
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
        const rasterixNodes = [
            "PrimereRasterix", "PrimereSelectiveTone", "PrimereColorBalance",
            "PrimereHSL", "PrimereShadeDetailer", "PrimereHistogram",
            "PrimereFilmRendering", "PrimerePosterize",
        ];
        if (!rasterixNodes.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated ? onNodeCreated.apply(this, []) : undefined;

            const node = this;
            const fw   = (name) => node.widgets?.find(w => w.name === name);

            const wTone  = fw("color_balance_tone");
            const wCR    = fw("color_balance_cyan_red");
            const wMG    = fw("color_balance_magenta_green");
            const wYB    = fw("color_balance_yellow_blue");

            const wHsCh  = fw("hsl_channel");
            const wHsHue = fw("hsl_hue");
            const wHsSat = fw("hsl_saturation");
            const wHsLit = fw("hsl_lightness");
            const wHsVib = fw("hsl_vibrance");

            const wStZone = fw("selective_tone_zone");
            const wStVal  = fw("selective_tone_value");

            const wShMode = fw("detail_mode");
            const wShLvl  = fw("shade_level");
            const wShRad  = fw("shade_radius");

            const wPostCh  = fw("channels");
            const wPostSh  = fw("shades");

            let cbStore   = cbDefault();
            let hsStore   = hsDefault();
            let stStore   = stDefault();
            let shStore   = shDefault();
            let postStore = postDefault();

            let prevTone   = wTone?.value    ?? "midtones";
            let prevHsCh   = wHsCh?.value    ?? "master";
            let prevStZn   = wStZone?.value  ?? "midtones";
            let prevShMd   = wShMode?.value  ?? "medium";
            let prevPostCh = wPostCh?.value  ?? "Red";
            let updating   = false;
            let histogramDebounceTimer = null;

            function applyFilmTypeFilter() {
                const filmTypeWidget      = fw("film_type");
                const filmRenderingWidget = fw("film_rendering");
                if (!filmTypeWidget || !filmRenderingWidget) return;

                if (!Array.isArray(node.__primereFilmAllPresets) || node.__primereFilmAllPresets.length === 0) {
                    const initialValues = filmRenderingWidget.options?.values || [];
                    node.__primereFilmAllPresets = [...initialValues];
                    node.__primereFilmByType = buildFilmPresetMap(node.__primereFilmAllPresets);
                }

                const selectedType = String(filmTypeWidget.value || "All");
                const allValues    = node.__primereFilmAllPresets;
                const byType       = node.__primereFilmByType || {};
                const nextValues   = selectedType === "All"
                    ? allValues
                    : (byType[selectedType]?.length > 0 ? byType[selectedType] : allValues);

                filmRenderingWidget.options         = filmRenderingWidget.options || {};
                filmRenderingWidget.options.values  = [...nextValues];

                if (!nextValues.includes(filmRenderingWidget.value)) {
                    filmRenderingWidget.value = nextValues[0] || filmRenderingWidget.value;
                    filmRenderingWidget.callback?.(filmRenderingWidget.value);
                }

                app.canvas?.setDirty(true);
            }

            applyFilmTypeFilter();

            function currentNodeId() {
                return String(node?.id ?? "global");
            }

            function histogramFileUrl(showInput, channel, style) {
                const q = new URLSearchParams({
                    node_id: currentNodeId(),
                    histogram_source: String(!!showInput),
                    histogram_channel: channel || "RGB",
                    histogram_style: style || "bars",
                });
                return `/primere_rasterix_histogram_image?${q.toString()}`;
            }

            function updateHistogramDisplay(showInput, channel, style) {
                const url = `${histogramFileUrl(showInput, channel, style)}&t=${Date.now()}`;
                const img    = new Image();
                img.onload = () => {
                    if (!node.imgs) node.imgs = [img];
                    else node.imgs[0] = img;
                    app.canvas?.setDirty(true);
                };
                img.onerror = () => {};
                img.src = url;
            }

            function showHistogramOffImage() {
                const url = `/extensions/ComfyUI_Primere_Nodes/images/No_histogram_08.jpg?t=${Date.now()}`;
                const img = new Image();
                img.onload = () => {
                    if (!node.imgs) node.imgs = [img];
                    else node.imgs[0] = img;
                    app.canvas?.setDirty(true);
                };
                img.onerror = () => {};
                img.src = url;
            }

            async function histogramFileExists(showInput, channel, style) {
                const url = `${histogramFileUrl(showInput, channel, style)}&t=${Date.now()}`;
                try {
                    const headResp = await fetch(url, { method: "HEAD" });
                    if (headResp.ok) return true;
                } catch (_) {}
                try {
                    const getResp = await fetch(url, { method: "GET" });
                    return getResp.ok;
                } catch (_) {
                    return false;
                }
            }

            async function generateHistogram(showInput, channel, style) {
                try {
                    await fetch('/primere_rasterix_histogram_generate', {
                        method:  'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body:    JSON.stringify({
                            node_id:           currentNodeId(),
                            histogram_source:  showInput,
                            histogram_channel: channel || "RGB",
                            histogram_style:   style   || "bars",
                            precision:         fw("precision")?.value ?? false,
                        }),
                    });
                } catch (e) {
                    console.warn('[Primere Rasterix] histogram render failed:', e);
                }
            }

            function scheduleHistogramGenerate(showInput, channel, style, delayMs = 1000) {
                if (histogramDebounceTimer) clearTimeout(histogramDebounceTimer);
                histogramDebounceTimer = setTimeout(async () => {
                    await generateHistogram(showInput, channel, style);
                    updateHistogramDisplay(showInput, channel, style);
                }, delayMs);
            }

            async function requestHistogramSwitch(showInput, channel, style) {
                if (await histogramFileExists(showInput, channel, style)) {
                    if (histogramDebounceTimer) clearTimeout(histogramDebounceTimer);
                    updateHistogramDisplay(showInput, channel, style);
                    return;
                }
                scheduleHistogramGenerate(showInput, channel, style, 1000);
            }

            function currentHistState() {
                return {
                    enabled:   fw("show_histogram")?.value    ?? false,
                    showInput: fw("histogram_source")?.value  ?? false,
                    channel:   fw("histogram_channel")?.value ?? "RGB",
                    style:     fw("histogram_style")?.value   ?? "bars",
                };
            }

            node.onExecuted = async function () {
                const { enabled, showInput, channel, style } = currentHistState();
                if (!enabled) { showHistogramOffImage(); return; }
                if (await histogramFileExists(showInput, channel, style)) {
                    updateHistogramDisplay(showInput, channel, style);
                    return;
                }
                await generateHistogram(showInput, channel, style);
                updateHistogramDisplay(showInput, channel, style);
            };

            // Color Balance
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

            // Hue / Saturation
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

            // Selective Tone
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

            // Shade
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

            // Posterize
            function applyPostSlider(ch) {
                if (!wPostSh) return;
                updating = true;
                wPostSh.value = postStore[ch] ?? 255;
                updating = false;
                app.canvas?.setDirty(true);
            }
            function capturePostSlider(ch) {
                if (!wPostSh) return;
                postStore[ch] = wPostSh.value;
                rasterixSave('posterize', postStore);
            }

            // Initial load
            rasterixLoad().then(loaded => {
                cbStore   = loaded.color_balance;
                hsStore   = loaded.hue_saturation;
                stStore   = loaded.selective_tone;
                shStore   = loaded.shade;
                postStore = loaded.posterize;
                applyCbSliders(prevTone);
                applyHsSliders(prevHsCh);
                applyStSlider(prevStZn);
                applyShSliders(prevShMd);
                applyPostSlider(prevPostCh);
            });

            // Widget change handler
            node.onWidgetChanged = function (name, value) {
                if (updating) return;

                if (name === "film_type") {
                    applyFilmTypeFilter();

                } else if (name === "color_balance_tone") {
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

                } else if (name === "channels") {
                    capturePostSlider(prevPostCh);
                    applyPostSlider(value);
                    prevPostCh = value;
                } else if (name === "shades") {
                    capturePostSlider(wPostCh?.value ?? prevPostCh);

                } else if (name === "histogram_source") {
                    const { enabled, channel, style } = currentHistState();
                    if (enabled) requestHistogramSwitch(value, channel, style);
                } else if (name === "histogram_channel") {
                    const { enabled, showInput, style } = currentHistState();
                    if (enabled) requestHistogramSwitch(showInput, value, style);
                } else if (name === "histogram_style") {
                    const { enabled, showInput, channel } = currentHistState();
                    if (enabled) requestHistogramSwitch(showInput, channel, value);
                } else if (name === "show_histogram") {
                    const { showInput, channel, style } = currentHistState();
                    if (value) {
                        requestHistogramSwitch(showInput, channel, style);
                    } else {
                        if (histogramDebounceTimer) clearTimeout(histogramDebounceTimer);
                        showHistogramOffImage();
                    }
                }
            };
        };
    },
});
