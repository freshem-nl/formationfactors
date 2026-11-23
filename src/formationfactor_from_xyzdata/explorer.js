"use strict";
const rows = JSON.parse(document.getElementById("dataset").textContent);
const $ = id => document.getElementById(id);
const X = "ec_groundwater_25_ms_cm", Y = "ec_bulk_25_ms_cm";
const excluded = new Set(), selected = new Set(), fitCache = new Map();
const state = {filtered: [], retained: [], reasons: new Map(), baseline: null, fit: null, view: null};
const NS = "http://www.w3.org/2000/svg";
const box = {left: 80, right: 975, top: 20, bottom: 425};
let visible = [], drag = null;
let chlorideView = null, chlorideVisible = [];
const number = v => Number.isFinite(v) ? v.toPrecision(4) : "\u2014";
const unique = values => [...new Set(values)].sort();
const cluster = row => JSON.stringify([row.log_source, row.log_id]);

function sumStats(data, center) {
    const s = [0, 0, 0, 0, 0];
    for (const r of data) {
        const x = r[X] - center, y = r[Y];
        s[0]++; s[1] += x; s[2] += y; s[3] += x*x; s[4] += x*y;
    }
    return s;
}
function fromSums(s, center) {
    const variance = s[3] - s[1]*s[1]/s[0];
    if (!(variance > 1e-14 * Math.max(s[3], 1e-20))) return null;
    const slope = (s[4] - s[1]*s[2]/s[0]) / variance;
    return {slope, intercept: (s[2] - slope*s[1])/s[0] - slope*center};
}
function fixedFromSums(s, center, intercept) {
    const sx = s[1] + center*s[0];
    const sxx = s[3] + 2*center*s[1] + center*center*s[0];
    if (!(sxx > 0)) return null;
    return {intercept, slope: (s[4] + center*s[2] - intercept*sx)/sxx};
}
function quantile(sorted, q) {
    const index = (sorted.length - 1)*q, lo = Math.floor(index), f = index-lo;
    return sorted[lo]*(1-f) + sorted[Math.ceil(index)]*f;
}
function median(values) {
    return quantile([...values].sort((a,b) => a-b), 0.5);
}
function randomGenerator() {
    let seed = 42;
    return () => {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
function regression(data, fixedMinimum = false) {
    const key = `${fixedMinimum}:` + data.map(r => r.pair_id).join("|");
    if (fitCache.has(key)) return fitCache.get(key);
    const result = {n: data.length, logs: new Set(data.map(cluster)).size, draws: [], note: "", fixedMinimum};
    const minimum = data.length ? Math.min(...data.map(r => r[Y])) : null;
    const center = data.length ? data.reduce((s,r) => s+r[X], 0)/data.length : 0;
    const solve = sums => fixedMinimum ? fixedFromSums(sums, center, minimum) : fromSums(sums, center);
    const fit = data.length >= 3 ? solve(sumStats(data, center)) : null;
    if (!fit) {
        result.note = "Need at least 3 observations and an estimable groundwater-EC predictor.";
        return result;
    }
    Object.assign(result, fit);
    const mean = data.reduce((s,r) => s+r[Y], 0)/data.length;
    const sse = data.reduce((s,r) => s+(r[Y]-fit.intercept-fit.slope*r[X])**2, 0);
    const sst = data.reduce((s,r) => s+(r[Y]-mean)**2, 0);
    result.r2 = sst > 0 ? 1-sse/sst : null;
    result.rmse = Math.sqrt(sse/data.length);
    result.minX = Math.min(...data.map(r => r[X]));
    result.maxX = Math.max(...data.map(r => r[X]));
    if (result.logs < 3) {
        result.note = "Fewer than 3 logs: line shown, but cluster-bootstrap uncertainty is unavailable.";
    } else {
        const groups = new Map(), strata = new Map();
        for (const r of data) {
            const id = cluster(r);
            if (!groups.has(id)) groups.set(id, []);
            groups.get(id).push(r);
        }
        for (const group of groups.values()) {
            const source = group[0].log_source;
            if (!strata.has(source)) strata.set(source, []);
            strata.get(source).push(sumStats(group, center));
        }
        const random = randomGenerator();
        for (let b=0; b<1500; b++) {
            const total = [0,0,0,0,0];
            for (const sums of strata.values()) {
                for (let i=0; i<sums.length; i++) {
                    const sample = sums[Math.floor(random()*sums.length)];
                    for (let j=0; j<5; j++) total[j] += sample[j];
                }
            }
            const draw = solve(total);
            if (draw && Number.isFinite(draw.slope) && Number.isFinite(draw.intercept)) result.draws.push(draw);
        }
        if (result.draws.length < 1425) {
            result.note = `Only ${result.draws.length}/1500 valid bootstrap draws: confidence interval suppressed.`;
            result.draws = [];
        } else {
            const slopes = result.draws.map(d => d.slope).sort((a,b) => a-b);
            const intercepts = result.draws.map(d => d.intercept).sort((a,b) => a-b);
            result.ci = [quantile(slopes, .025), quantile(slopes, .975)];
            result.interceptCI = [quantile(intercepts, .025), quantile(intercepts, .975)];
        }
    }
    if (fitCache.size >= 16) fitCache.delete(fitCache.keys().next().value);
    fitCache.set(key, result);
    return result;
}

function physicalParameters(fit) {
    const result = {
        formation_factor: null,
        surface_conductivity_ms_cm: Number.isFinite(fit.intercept) ? fit.intercept : null,
        surface_conductivity_s_m: Number.isFinite(fit.intercept) ? fit.intercept/10 : null,
        formation_factor_ci_low: null,
        formation_factor_ci_high: null,
        surface_conductivity_ci_low_ms_cm: fit.interceptCI?.[0] ?? null,
        surface_conductivity_ci_high_ms_cm: fit.interceptCI?.[1] ?? null,
        surface_conductivity_ci_low_s_m: fit.interceptCI ? fit.interceptCI[0]/10 : null,
        surface_conductivity_ci_high_s_m: fit.interceptCI ? fit.interceptCI[1]/10 : null,
        physical_parameter_note: ""
    };
    const notes = [];
    if (!Number.isFinite(fit.slope) || !Number.isFinite(fit.intercept)) {
        result.physical_parameter_note = "No estimable regression: physical parameters unavailable.";
        return result;
    }
    if (fit.slope === 0) {
        result.physical_parameter_note = "Zero slope: no finite formation factor.";
        return result;
    }
    result.formation_factor = 1/fit.slope;
    if (result.formation_factor < 1) notes.push("F < 1: not physically plausible under this model.");
    if (result.surface_conductivity_ms_cm < 0)
        notes.push("Negative surface conductivity: the unconstrained fit is not a physical calibration.");
    if (fit.draws.length && fit.ci?.[0] > 0) {
        const factors = fit.draws.filter(d => d.slope !== 0)
            .map(d => 1/d.slope).filter(Number.isFinite).sort((a,b) => a-b);
        if (factors.length >= .95 * fit.draws.length) {
            result.formation_factor_ci_low = quantile(factors, .025);
            result.formation_factor_ci_high = quantile(factors, .975);
        } else notes.push("Formation-factor CI unavailable: too many undefined reciprocals.");
    } else {
        notes.push(fit.ci ? "Formation-factor CI suppressed: slope CI reaches or falls below zero." :
            "Bootstrap uncertainty unavailable for physical parameters.");
    }
    if (fit.fixedMinimum) notes.push("Surface conductivity fixed at subset minimum; uncertainty is conditional on this observed minimum.");
    result.physical_parameter_note = notes.join(" ");
    return result;
}

function fillOptions(id, values, allLabel) {
    const select = $(id), previous = select.value;
    select.replaceChildren(new Option(allLabel, ""));
    for (const value of values) select.add(new Option(value, value));
    if (values.includes(previous)) select.value = previous;
}
function refreshOptions() {
    const prefix = $("assignment").value;
    fillOptions("formation", unique(rows.map(r => r[`${prefix}_formation`])), "All formations");
    const formationRows = rows.filter(r => !$("formation").value || r[`${prefix}_formation`] === $("formation").value);
    fillOptions("unit", unique(formationRows.map(r => r[`${prefix}_regis_layer`] || "No REGIS assignment")), "All REGIS layers");
}
function recompute(resetView = false) {
    const distance = $("distance").valueAsNumber, capInput = $("cap"), cap = capInput.valueAsNumber;
    if (!$("distance").value || !Number.isFinite(distance) || distance < 0 || distance > 250 ||
        (capInput.value && (!Number.isFinite(cap) || cap < 0))) {
        $("messages").textContent = "Enter an XY distance between 0 and 250 m and a nonnegative conductivity ceiling.";
        $("messages").className = "notice warning";
        return;
    }
    const prefix = $("assignment").value;
    state.filtered = rows.filter(r =>
        (!$("formation").value || r[`${prefix}_formation`] === $("formation").value) &&
        (!$("unit").value || (r[`${prefix}_regis_layer`] || "No REGIS assignment") === $("unit").value) &&
        (!$("source").value || r.log_source === $("source").value) &&
        r.xy_distance_m <= distance &&
        (!$("inside").checked || r.screen_gap_m === 0) &&
        (!$("same").checked || String(r.same_regis_layer).toLowerCase() === "true")
    );
    state.baseline = regression(state.filtered);
    state.reasons = new Map();
    const base = state.baseline, k = Number($("residual").value);
    let residualMedian, scale;
    if (k && base.slope !== undefined) {
        const residuals = state.filtered.map(r => r[Y]-base.intercept-base.slope*r[X]);
        residualMedian = median(residuals);
        scale = 1.4826 * median(residuals.map(v => Math.abs(v-residualMedian)));
    }
    for (const r of state.filtered) {
        const reasons = [];
        if (excluded.has(r.pair_id)) reasons.push("manual");
        if (capInput.value && r[Y] > cap) reasons.push("bulk EC ceiling");
        if (k && scale > 1e-14 && Math.abs(r[Y]-base.intercept-base.slope*r[X]-residualMedian) > k*scale)
            reasons.push("residual rule");
        if (reasons.length) state.reasons.set(r.pair_id, reasons.join("; "));
    }
    state.retained = state.filtered.filter(r => !state.reasons.has(r.pair_id));
    state.fit = regression(state.retained);
    state.fixedBaseline = regression(state.filtered, true);
    state.fixedFit = regression(state.retained, true);
    if (resetView) {state.view = null; chlorideView = null;}
    $("n-filtered").textContent = state.filtered.length.toLocaleString();
    $("n-retained").textContent = state.retained.length.toLocaleString();
    $("n-logs").textContent = state.fit.logs.toLocaleString();
    $("n-excluded").textContent = state.reasons.size.toLocaleString();
    $("scope").textContent = $("formation").value || "All formations";
    const notes = [];
    if (!state.filtered.length) notes.push("No observations match these filters. Broaden the selection.");
    if (base.note) notes.push("Before: " + base.note);
    if (state.fit.note) notes.push("After: " + state.fit.note);
    if (state.fixedFit.note) notes.push("Fixed minimum after: " + state.fixedFit.note);
    if (k && !(scale > 1e-14)) notes.push("Residual rule not applied: insufficient variation or zero residual MAD.");
    if (excluded.size) notes.push(`${excluded.size} manual exclusion(s) stored across all filters.`);
    const hidden = state.filtered.filter(r => ($("logx").checked && r[X] <= 0) || ($("logy").checked && r[Y] <= 0)).length;
    if (hidden) notes.push(`${hidden} nonpositive value(s) hidden on logarithmic axes, but still included in the original-unit fit.`);
    $("messages").textContent = notes.join(" ");
    $("messages").className = notes.length ? "notice" : "";
    updateSummary();
    updateSelection();
    renderTable();
    renderChart();
}
function updateSummary() {
    const body = $("fit-table").tBodies[0];
    const parameters = $("parameter-table").tBodies[0];
    body.replaceChildren();
    parameters.replaceChildren();
    for (const [label, fit] of modelComparisons()) {
        const tr = body.insertRow();
        for (const text of [label, `${fit.n} / ${fit.logs}`, number(fit.intercept), number(fit.slope),
            fit.ci ? fit.ci.map(number).join(" to ") : "Unavailable", number(fit.r2), number(fit.rmse)])
            tr.insertCell().textContent = text;
        const p = physicalParameters(fit), parameterRow = parameters.insertRow();
        const surfaceCI = p.surface_conductivity_ci_low_ms_cm === null ? "Unavailable" :
            [p.surface_conductivity_ci_low_ms_cm, p.surface_conductivity_ci_high_ms_cm].map(number).join(" to ");
        for (const text of [label, number(p.formation_factor),
            p.formation_factor_ci_low === null ? "Unavailable" :
                [p.formation_factor_ci_low, p.formation_factor_ci_high].map(number).join(" to "),
            number(p.surface_conductivity_ms_cm), surfaceCI, number(p.surface_conductivity_s_m),
            p.physical_parameter_note || "No point-estimate physicality warning; not a validation."])
            parameterRow.insertCell().textContent = text;
    }
    const fit = state.fit;
    $("equation").textContent = fit.slope === undefined ? "No estimable retained-data relation." :
        `Retained free: EC bulk,25 = ${number(fit.intercept)} + ${number(fit.slope)} \u00d7 EC groundwater,25 (mS/cm)`;
    const fixed = state.fixedFit;
    $("fixed-equation").textContent = fixed.slope === undefined ? "No estimable fixed-minimum relation." :
        `Retained fixed minimum: EC bulk,25 = ${number(fixed.intercept)} + ${number(fixed.slope)} \u00d7 EC groundwater,25 (mS/cm)`;
}
function modelComparisons() {
    return [["Free / before",state.baseline],["Free / after",state.fit],
        ["Fixed minimum / before",state.fixedBaseline],["Fixed minimum / after",state.fixedFit]];
}
function updateSelection() {
    $("selected-count").textContent = `${selected.size} selected`;
    $("exclude-selected").disabled = !selected.size;
    $("restore").disabled = !excluded.size;
}
function renderTable() {
    const needle = $("search").value.toLowerCase(), prefix = $("assignment").value;
    const found = state.filtered.filter(r => [r.pair_id, r.log_id, r.locationcode, r[`${prefix}_regis_layer`]]
        .join(" ").toLowerCase().includes(needle)).sort((a,b) => b[Y]-a[Y]);
    $("table-count").textContent = `(${Math.min(200, found.length)} of ${found.length})`;
    const body = $("observations").tBodies[0];
    body.replaceChildren();
    for (const r of found.slice(0,200)) {
        const tr = body.insertRow();
        if (selected.has(r.pair_id)) tr.className = "selected-row";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox"; checkbox.checked = selected.has(r.pair_id);
        checkbox.setAttribute("aria-label", `Select pair ${r.pair_id}`);
        checkbox.addEventListener("change", () => toggleSelection(r.pair_id));
        tr.addEventListener("click", event => {
            if (event.target !== checkbox) toggleSelection(r.pair_id);
        });
        tr.insertCell().append(checkbox);
        for (const text of [r.pair_id, r.log_id, r.log_source, number(r[X]), number(r[Y]), number(r.chloride_g_l),
            r[`${prefix}_regis_layer`] || "Unassigned", number(r.xy_distance_m),
            state.reasons.get(r.pair_id) || "retained"]) tr.insertCell().textContent = text;
    }
}
function toggleSelection(id) {
    if (selected.has(id)) selected.delete(id); else selected.add(id);
    updateSelection(); renderTable(); renderChart();
}
function svgElement(tag, attributes = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
}
const tx = x => $("logx").checked ? Math.log10(x) : x;
const ty = y => $("logy").checked ? Math.log10(y) : y;
const inverseX = x => $("logx").checked ? 10**x : x;
const inverseY = y => $("logy").checked ? 10**y : y;
function autoView() {
    const valid = state.filtered.filter(r => Number.isFinite(tx(r[X])) && Number.isFinite(ty(r[Y])));
    if (!valid.length) return [0,1,0,1];
    const xs = valid.map(r => tx(r[X])), ys = valid.map(r => ty(r[Y]));
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const dx = (x1-x0 || Math.abs(x0)*.1 || 1)*.06, dy = (y1-y0 || Math.abs(y0)*.1 || 1)*.08;
    if (!$("logx").checked) x0 = 0; else x0 -= dx;
    if (!$("logy").checked) y0 = Math.max(0, y0-dy); else y0 -= dy;
    return [x0, x1+dx, y0, y1+dy];
}
function scales() {
    const [x0,x1,y0,y1] = state.view || autoView();
    return {
        px: x => box.left+(tx(x)-x0)/(x1-x0)*(box.right-box.left),
        py: y => box.bottom-(ty(y)-y0)/(y1-y0)*(box.bottom-box.top),
        ux: p => x0+(p-box.left)/(box.right-box.left)*(x1-x0),
        uy: p => y0+(box.bottom-p)/(box.bottom-box.top)*(y1-y0),
        view: [x0,x1,y0,y1]
    };
}
function renderChart(chloride = false) {
    let chartScales = scales();
    if (chloride) {
        const values = state.filtered;
        const maxX = values.length ? Math.max(...values.map(r => r.chloride_g_l)) : 1;
        const maxY = values.length ? Math.max(...values.map(r => r[Y])) : 1;
        const view = chlorideView || [0,Math.max(maxX*1.06, .001),0,Math.max(maxY*1.08,.001)];
        const [a,b,c,d] = view;
        chartScales = {
            px:x=>box.left+((360*x-450)/1000-a)/(b-a)*(box.right-box.left),
            py:y=>box.bottom-(y-c)/(d-c)*(box.bottom-box.top), view
        };
    }
    const svg = $(chloride ? "chloride-plot" : "plot"), {px,py,view:[x0,x1,y0,y1]} = chartScales;
    const clipId = chloride ? "chloride-clip" : "plot-clip";
    svg.replaceChildren();
    const defs = svgElement("defs"), clip = svgElement("clipPath", {id:clipId});
    clip.append(svgElement("rect", {x:box.left,y:box.top,width:box.right-box.left,height:box.bottom-box.top}));
    defs.append(clip); svg.append(defs);
    svg.append(svgElement("rect", {x:box.left,y:box.top,width:box.right-box.left,height:box.bottom-box.top,fill:"#fbfdfd"}));
    for (let i=0;i<=6;i++) {
        const xv = chloride ? x0+(x1-x0)*i/6 : inverseX(x0+(x1-x0)*i/6);
        const yv = chloride ? y0+(y1-y0)*i/6 : inverseY(y0+(y1-y0)*i/6);
        const x = px(chloride ? 1.25+(1000/360)*xv : xv), y = py(yv);
        svg.append(svgElement("line",{x1:x,x2:x,y1:box.top,y2:box.bottom,stroke:"#e3eaed"}));
        svg.append(svgElement("line",{x1:box.left,x2:box.right,y1:y,y2:y,stroke:"#e3eaed"}));
        svg.append(svgElement("text",{x,y:box.bottom+25,"text-anchor":"middle",fill:"#526a78","font-size":12},Number(xv.toPrecision(3)).toString()));
        svg.append(svgElement("text",{x:box.left-12,y:y+4,"text-anchor":"end",fill:"#526a78","font-size":12},Number(yv.toPrecision(3)).toString()));
    }
    svg.append(svgElement("text",{x:525,y:485,"text-anchor":"middle",fill:"#203343","font-size":15},chloride ? "Groundwater chloride (g/L)" : "EC groundwater at 25 \u00b0C (mS/cm)"));
    svg.append(svgElement("text",{transform:"translate(19 225) rotate(-90)","text-anchor":"middle",fill:"#203343","font-size":15},"EC bulk at 25 \u00b0C (mS/cm)"));
    const chart = svgElement("g",{"clip-path":`url(#${clipId})`});
    const fit = state.fit;
    for (const [bandFit, color] of [[fit,"#087f86"],[state.fixedFit,"#9251a0"]]) {
        if (bandFit.draws.length) {
            const lower = [], upper = [];
            for (let i=0;i<=60;i++) {
                const x = bandFit.minX+(bandFit.maxX-bandFit.minX)*i/60;
                const predicted = bandFit.draws.map(d => d.intercept+d.slope*x).sort((a,b) => a-b);
                const lo = quantile(predicted,.025), hi = quantile(predicted,.975);
                if ([px(x),py(lo),py(hi)].every(Number.isFinite)) {
                    lower.push([px(x),py(lo)]); upper.push([px(x),py(hi)]);
                }
            }
            if (lower.length) chart.append(svgElement("polygon",{points:[...lower,...upper.reverse()].map(p => p.join(",")).join(" "),fill:color,opacity:.12}));
        }
    }
    const chartVisible = [];
    for (const r of state.filtered) {
        const x = px(r[X]), y = py(r[Y]), out = state.reasons.has(r.pair_id);
        if (!Number.isFinite(x) || !Number.isFinite(y) || x<box.left || x>box.right || y<box.top || y>box.bottom) continue;
        chartVisible.push({r,x,y});
        if (out) chart.append(svgElement("path",{d:`M${x-4},${y-4}l8,8m-8,0l8,-8`,stroke:"#8b969d",opacity:.65,"stroke-width":1.5}));
        else chart.append(svgElement("circle",{cx:x,cy:y,r:3.6,fill:r.log_source==="LAS"?"#087f86":"#be5b31",opacity:.5}));
        if (selected.has(r.pair_id)) chart.append(svgElement("circle",{cx:x,cy:y,r:7,fill:"none",stroke:"#162e3e","stroke-width":2}));
    }
    for (const [model, dashed] of [[state.baseline,true],[fit,false],[state.fixedBaseline,true],[state.fixedFit,false]]) {
        if (model.slope === undefined) continue;
        const points = [];
        for (let i=0;i<=200;i++) {
            const start = chloride ? 1.25 : $("logx").checked ? model.minX : 0;
            const x = start+(model.maxX-start)*i/200, y = model.intercept+model.slope*x;
            if (Number.isFinite(px(x)) && Number.isFinite(py(y))) points.push(`${px(x)},${py(y)}`);
        }
        chart.append(svgElement("polyline",{points:points.join(" "),fill:"none",stroke:model.fixedMinimum?"#9251a0":dashed?"#7b838c":"#173c48","stroke-width":dashed?2:2.8,"stroke-dasharray":dashed?"7,5":"none"}));
    }
    svg.append(chart);
    if (chloride) {
        chlorideVisible = chartVisible;
        $("chloride-equation").textContent = [
            ["Retained free",state.fit],["Retained fixed minimum",state.fixedFit]
        ].filter(([,f])=>f.slope!==undefined).map(([label,f]) =>
            `${label}: EC bulk,25 = ${number(f.intercept+1.25*f.slope)} + ${number(1000/360*f.slope)} \u00d7 chloride (g/L)`
        ).join("; ");
    } else {
        visible = chartVisible;
        renderChart(true);
    }
}
function plotPoint(event) {
    const r = $("plot").getBoundingClientRect();
    return {x:(event.clientX-r.left)*1000/r.width,y:(event.clientY-r.top)*500/r.height};
}
function nearest(p) {
    let distance = 100, result = null;
    for (const point of visible) {
        const d = (point.x-p.x)**2+(point.y-p.y)**2;
        if (d<distance) {distance=d;result=point;}
    }
    return result;
}
function insideBox(p) {return p.x>=box.left && p.x<=box.right && p.y>=box.top && p.y<=box.bottom;}
function clearSelection() {selected.clear();updateSelection();renderTable();renderChart();}
function applyZoom(view) {
    if (view.every(Number.isFinite) && view[1]-view[0]>1e-10 && view[3]-view[2]>1e-10) state.view = view;
    renderChart();
}
function download(name, records) {
    if (!records.length) return;
    const keys = unique(records.flatMap(Object.keys));
    // Prevent formula execution when imported into spreadsheet applications.
    const quote = value => {
        let text = String(value ?? "");
        if (typeof value === "string" && /^[=+\-@\t\r]/.test(text)) text = "'" + text;
        return '"' + text.replaceAll('"','""') + '"';
    };
    const csv = [keys.map(quote).join(","), ...records.map(r => keys.map(k => quote(r[k])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
    const link = document.createElement("a"); link.href=url;link.download=name;link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
}
for (const id of ["assignment","formation","unit","source","distance","inside","same"]) {
    $(id).addEventListener("change",() => {
        if (id==="assignment") {$("formation").value="";$("unit").value="";}
        if (id==="formation") $("unit").value="";
        refreshOptions();selected.clear();recompute(true);
    });
}
for (const id of ["cap","residual"]) $(id).addEventListener("change",() => recompute());
for (const id of ["logx","logy"]) $(id).addEventListener("change",() => recompute(true));
$("search").addEventListener("input",renderTable);
$("reset-view").onclick = () => {state.view=null;renderChart();};
$("reset-chloride-view").onclick = () => {chlorideView=null;renderChart(true);};
$("chloride-plot").addEventListener("dblclick",()=>{chlorideView=null;renderChart(true);});
$("chloride-plot").addEventListener("click",event=>{
    const rect=$("chloride-plot").getBoundingClientRect();
    const x=(event.clientX-rect.left)*1000/rect.width,y=(event.clientY-rect.top)*500/rect.height;
    const point=chlorideVisible.reduce((best,p)=>{
        const distance=(p.x-x)**2+(p.y-y)**2;
        return distance<best.distance?{distance,p}:best;
    },{distance:100,p:null}).p;
    if(point)toggleSelection(point.r.pair_id);
});
$("chloride-plot").addEventListener("wheel",event=>{
    const rect=$("chloride-plot").getBoundingClientRect();
    const p={x:(event.clientX-rect.left)*1000/rect.width,y:(event.clientY-rect.top)*500/rect.height};
    if(!insideBox(p)||!state.filtered.length)return;
    event.preventDefault();
    const view=chlorideView||[0,Math.max(.001,Math.max(...state.filtered.map(r=>r.chloride_g_l))*1.06),
        0,Math.max(.001,Math.max(...state.filtered.map(r=>r[Y]))*1.08)];
    const cx=view[0]+(p.x-box.left)/(box.right-box.left)*(view[1]-view[0]);
    const cy=view[2]+(box.bottom-p.y)/(box.bottom-box.top)*(view[3]-view[2]);
    const f=event.deltaY>0?1.2:1/1.2;
    const next=[cx+(view[0]-cx)*f,cx+(view[1]-cx)*f,cy+(view[2]-cy)*f,cy+(view[3]-cy)*f];
    if(next.every(Number.isFinite)&&next[1]-next[0]>1e-10&&next[3]-next[2]>1e-10)chlorideView=next;
    renderChart(true);
},{passive:false});
$("clear-selection").onclick = clearSelection;
$("exclude-selected").onclick = () => {for (const id of selected) excluded.add(id);selected.clear();recompute();};
$("restore").onclick = () => {excluded.clear();recompute();};
$("reset-all").onclick = () => {
    excluded.clear();selected.clear();
    $("assignment").value="chloride";
    for (const id of ["formation","unit","source","cap","residual","search"]) $(id).value="";
    for (const id of ["inside","same","logx","logy"]) $(id).checked=false;
    $("distance").value="250";$("mode").value="zoom";refreshOptions();recompute(true);
};
$("download-data").onclick = () => download("groundwater_bulk_ec_experiment.csv",
    state.filtered.map(r => ({...r,excluded:state.reasons.has(r.pair_id),exclusion_reason:state.reasons.get(r.pair_id)||""})));
$("download-fits").onclick = () => download("regression_comparison.csv",
    modelComparisons().map(([subset,f]) => ({
        subset,n_pairs:f.n,n_logs:f.logs,intercept_ms_cm:f.intercept,slope_dimensionless:f.slope,
        intercept_ci_low:f.interceptCI?.[0],intercept_ci_high:f.interceptCI?.[1],
        slope_ci_low:f.ci?.[0],slope_ci_high:f.ci?.[1],r_squared:f.r2,residual_rmse_ms_cm:f.rmse,
        ...physicalParameters(f),
        model:f.fixedMinimum ? "fixed subset minimum" : "free intercept",
        x_variable:X,y_variable:Y,temperature_c:25,ec25_over_ec11:1.36,
        groundwater_ec25_formula:"(1000 * chloride_g_l + 450) / 360",
        valid_bootstrap_draws:f.draws.length,note:f.note,
        formation:$("formation").value,layer:$("unit").value,assignment_at:$("assignment").value,
        source:$("source").value,max_xy_m:$("distance").value,
        inside_screen:$("inside").checked,same_layer:$("same").checked,
        bulk_ec_ceiling_ms_cm:$("cap").value,residual_rule:$("residual").value
    })));
$("plot").addEventListener("wheel",event => {
    const p = plotPoint(event);if (!insideBox(p)) return;
    event.preventDefault();
    const s = scales(), f = event.deltaY>0?1.2:1/1.2, cx=s.ux(p.x),cy=s.uy(p.y);
    applyZoom([cx+(s.view[0]-cx)*f,cx+(s.view[1]-cx)*f,cy+(s.view[2]-cy)*f,cy+(s.view[3]-cy)*f]);
},{passive:false});
$("plot").addEventListener("dblclick",() => {state.view=null;renderChart();});
$("plot").addEventListener("pointerdown",event => {
    if (event.button!==0) return;
    const p=plotPoint(event);if (!insideBox(p)) return;
    drag=p;$("plot").setPointerCapture(event.pointerId);$("tooltip").hidden=true;
    $("plot").append(svgElement("rect",{id:"brush",x:p.x,y:p.y,width:0,height:0,fill:"#087f86","fill-opacity":.12,stroke:"#087f86","stroke-dasharray":"4,3"}));
});
$("plot").addEventListener("pointermove",event => {
    const p=plotPoint(event), tip=$("tooltip");
    if (drag) {
        const brush=$("brush");if (!brush) return;
        const x=Math.max(box.left,Math.min(box.right,p.x)),y=Math.max(box.top,Math.min(box.bottom,p.y));
        brush.setAttribute("x",Math.min(x,drag.x));brush.setAttribute("y",Math.min(y,drag.y));
        brush.setAttribute("width",Math.abs(x-drag.x));brush.setAttribute("height",Math.abs(y-drag.y));
        return;
    }
    const point=nearest(p);tip.hidden=!point;
    if (!point) return;
    const r=point.r;
    tip.textContent=`${r.log_source} / ${r.log_id} / pair ${r.pair_id}\nEC groundwater,25: ${number(r[X])} mS/cm\nEC bulk,25: ${number(r[Y])} mS/cm\nChloride: ${number(r.chloride_g_l)} g/L\nREGIS: ${r.chloride_regis_layer||"unassigned"} / ${r.log_regis_layer||"unassigned"}\nXY separation: ${number(r.xy_distance_m)} m\nScreen gap: ${number(r.screen_gap_m)} m\n${state.reasons.get(r.pair_id)||"Retained"}`;
    const rect=$("plot").getBoundingClientRect();
    tip.style.left=`${Math.max(0,Math.min(event.clientX-rect.left+15,rect.width-285))}px`;
    tip.style.top=`${event.clientY-rect.top+15}px`;
});
$("plot").addEventListener("pointerleave",() => {$("tooltip").hidden=true;});
$("plot").addEventListener("pointercancel",() => {drag=null;$("brush")?.remove();});
$("plot").addEventListener("pointerup",event => {
    if (!drag) return;
    const end=plotPoint(event), start=drag;drag=null;$("brush")?.remove();
    const x0=Math.max(box.left,Math.min(start.x,end.x)),x1=Math.min(box.right,Math.max(start.x,end.x));
    const y0=Math.max(box.top,Math.min(start.y,end.y)),y1=Math.min(box.bottom,Math.max(start.y,end.y));
    if (Math.hypot(end.x-start.x,end.y-start.y)<5) {
        const point=nearest(end);if (point) toggleSelection(point.r.pair_id);
    } else if ($("mode").value==="select") {
        for (const p of visible) if (p.x>=x0 && p.x<=x1 && p.y>=y0 && p.y<=y1) selected.add(p.r.pair_id);
        updateSelection();renderTable();renderChart();
    } else if (x1-x0>5 && y1-y0>5) {
        const s=scales();applyZoom([s.ux(x0),s.ux(x1),s.uy(y1),s.uy(y0)]);
    }
});
window.addEventListener("error",event => {
    $("messages").className="notice warning";
    $("messages").textContent=`Application error: ${event.message}. Reload the page to reset the experiment.`;
});
refreshOptions();
recompute(true);
