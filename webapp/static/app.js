// ToM Lens viewer. Plain SVG, no external libraries, so it runs offline.
"use strict";

const QT_LABEL = {
  pooled: "All belief questions",
  binding: "Name binding (leaver vs actor, same story)",
  reality: "Reality (0th order)",
  belief_leaver: "Leaver's belief (1st order)",
  belief_actor: "Actor's belief (1st order)",
  nested_actor_about_leaver: "Actor about leaver (2nd order)",
  nested_leaver_about_actor: "Leaver about actor (2nd order)",
};
const SHORT = { reality: "R", belief_leaver: "L", belief_actor: "A",
  nested_actor_about_leaver: "A→L", nested_leaver_about_actor: "L→A" };
const SERIES = ["--series-1", "--series-2", "--series-3"];
const state = { models: [], results: {}, colors: {} };

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const api = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); };
const fmtPct = (x) => (x == null ? "–" : `${(100 * x).toFixed(1)}%`);
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const tip = $("#tooltip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${x}px`; tip.style.top = `${ev.clientY + 14}px`; }
function hideTip() { tip.style.display = "none"; }
function colorOf(m) { return css(state.colors[m] || "--series-1"); }

// ---------------------------------------------------------------- tabs ----
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === b.dataset.tab));
  if (b.dataset.tab === "map") drawMaps();
  if (b.dataset.tab === "lens") drawLens();
}));

// ------------------------------------------------------------ line chart --
// series: [{name, color, dash, points:[[x,y]], band:[[x,lo,hi]]}]
function lineChart(host, series, opt) {
  host.innerHTML = "";
  const W = host.clientWidth || 600, H = opt.height || 260;
  const m = { l: 44, r: 16, t: 10, b: 34 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.label || "chart" }, host);
  const all = series.flatMap((s) => s.points.map((p) => p[1]).concat((s.band || []).flatMap((b) => [b[1], b[2]])));
  let [y0, y1] = opt.yDomain || [Math.min(...all), Math.max(...all)];
  if (opt.yRef != null) { y0 = Math.min(y0, opt.yRef); y1 = Math.max(y1, opt.yRef); }
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.06; if (!opt.yDomain) { y0 -= pad; y1 += pad; }
  const X = (x) => m.l + x * (W - m.l - m.r), Y = (y) => m.t + (1 - (y - y0) / (y1 - y0)) * (H - m.t - m.b);
  const g = el("g", { class: "axis" }, svg);
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (i / 4) * (y1 - y0);
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline" }, g);
    el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, g).textContent = opt.yFmt ? opt.yFmt(v) : v.toFixed(2);
  }
  for (let i = 0; i <= 4; i++) {
    el("text", { x: X(i / 4), y: H - 12, "text-anchor": "middle" }, g).textContent = (opt.xFmt || ((v) => v.toFixed(2)))(i / 4);
  }
  el("text", { x: (m.l + W - m.r) / 2, y: H, "text-anchor": "middle" }, g).textContent = opt.xLabel || "";
  if (opt.yRef != null) el("line", { x1: m.l, x2: W - m.r, y1: Y(opt.yRef), y2: Y(opt.yRef), class: "refline" }, svg);
  for (const s of series) {
    if (s.band && s.band.length) {
      const d = s.band.map((b, i) => `${i ? "L" : "M"}${X(b[0])},${Y(b[2])}`).join("") +
        s.band.slice().reverse().map((b) => `L${X(b[0])},${Y(b[1])}`).join("") + "Z";
      el("path", { d, fill: s.color, opacity: 0.12, stroke: "none" }, svg);
    }
    el("path", { d: s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join(""), fill: "none",
      stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round" }, svg);
    for (const p of s.points) el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.5, fill: s.color }, svg);
  }
  // crosshair + tooltip
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--text-muted"), "stroke-width": 1, opacity: 0 }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect(); const x = ((ev.clientX - r.left) / r.width * W - m.l) / (W - m.l - m.r);
    cross.setAttribute("x1", X(Math.max(0, Math.min(1, x)))); cross.setAttribute("x2", X(Math.max(0, Math.min(1, x))));
    cross.setAttribute("opacity", 0.6);
    const rows = series.map((s) => {
      let best = s.points[0]; for (const p of s.points) if (Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
      return `<div><span style="color:${s.color}">■</span> ${s.name}: <b>${(opt.yFmt || ((v) => v.toFixed(3)))(best[1])}</b> <span style="color:var(--text-muted)">${best[2] || ""}</span></div>`;
    });
    showTip(ev, rows.join(""));
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  if (opt.legend !== false) {
    const lg = document.createElement("div"); lg.className = "legend";
    lg.innerHTML = series.map((s) => `<span><i style="background:${s.color};${s.dash ? "opacity:.6" : ""}"></i>${s.name}</span>`).join("");
    host.appendChild(lg);
  }
}

const depth = (i, n) => (n ? i / n : 0);

// --------------------------------------------------------------- overview --
function drawCards() {
  $("#model-cards").innerHTML = state.models.map((m) => {
    const r = state.results[m.name];
    const fb = r.behaviour["test|belief_leaver|FB"];
    return `<div class="card"><div><span class="swatch" style="background:${colorOf(m.name)}"></span><span class="name">${m.name}</span></div>
      <div class="big">${fmtPct(m.test_acc)}</div>
      <div class="sub">held-out accuracy · leaver false belief ${fmtPct(fb && fb.acc)}</div>
      <div class="sub">${m.n_layers} layers · d=${m.d_model} · ${m.n_prompts.toLocaleString()} prompts</div></div>`;
  }).join("");
}

function drawAlign() {
  const qt = $("#align-qtype").value;
  const series = state.models.map((m) => {
    const a = state.results[m.name].alignment; const n = a.layers.length - 1;
    if (!a.raw[qt]) return null;
    return { name: m.name, color: colorOf(m.name),
      points: a.raw[qt].map((v, i) => [depth(i, n), v, `layer ${i}`]),
      band: a.raw_ci[qt].map((c, i) => [depth(i, n), c[0], c[1]]) };
  }).filter(Boolean);
  lineChart($("#align-chart"), series, { yRef: 0.5, xLabel: "relative depth (0 = embeddings, 1 = last layer)",
    yFmt: (v) => v.toFixed(2), label: "triplet alignment by depth" });
}

function drawMetric() {
  const series = [];
  for (const m of state.models) {
    const a = state.results[m.name].alignment; const n = a.layers.length - 1;
    series.push({ name: `${m.name} learned`, color: colorOf(m.name), points: a.learned.map((v, i) => [depth(i, n), v, `layer ${i}`]) });
    series.push({ name: `${m.name} control`, color: colorOf(m.name), dash: "5 4", points: a.control.map((v, i) => [depth(i, n), v, `layer ${i}`]) });
  }
  lineChart($("#metric-chart"), series, { yRef: 0.5, xLabel: "relative depth", label: "learned metric accuracy" });
}

function drawLensCurves() {
  const key = $("#lens-key").value;
  const series = state.models.filter((m) => state.results[m.name].lens[key]).map((m) => {
    const c = state.results[m.name].lens[key]; const n = c.length - 1;
    return { name: m.name, color: colorOf(m.name), points: c.map((v, i) => [depth(i, n), v, `layer ${i}`]) };
  });
  lineChart($("#lenscurve-chart"), series, { yRef: 0, xLabel: "relative depth", label: "logit lens margin" });
}

function drawBehaviour() {
  const split = $("#beh-split").value;
  const keys = Object.keys(state.results[state.models[0].name].behaviour)
    .filter((k) => k.startsWith(split + "|") && k.split("|").length === 3);
  const head = `<tr><th>Question</th><th>Belief</th>${state.models.map((m) => `<th class="num">${m.name}</th>`).join("")}<th class="num">n</th></tr>`;
  const rows = keys.map((k) => {
    const [, qt, fb] = k.split("|");
    const cells = state.models.map((m) => { const b = state.results[m.name].behaviour[k];
      return `<td class="num">${fmtPct(b && b.acc)}<span class="bar" style="width:${b ? 40 * b.acc : 0}px;background:${colorOf(m.name)}"></span></td>`; });
    const n = state.results[state.models[0].name].behaviour[k].n;
    return `<tr><td>${QT_LABEL[qt]}</td><td><span class="tag ${fb === "FB" ? "fb" : ""}">${fb === "FB" ? "false" : "true"}</span></td>${cells.join("")}<td class="num">${n}</td></tr>`;
  });
  const tot = state.models.map((m) => `<td class="num"><b>${fmtPct((state.results[m.name].behaviour[`${split}|all`] || {}).acc)}</b></td>`);
  $("#beh-table").innerHTML = `<table>${head}${rows.join("")}<tr><td><b>All</b></td><td></td>${tot.join("")}<td></td></tr></table>`;
}

// ------------------------------------------------------------- cognitive map
const FB_OF = (qt, v) => (qt === "belief_leaver" && v === "away") || (qt === "nested_leaver_about_actor" && v === "away") ||
  (qt === "nested_actor_about_leaver" && v !== "return");

async function drawMap(sel, host, meta) {
  const name = $(sel).value; const mdl = state.models.find((m) => m.name === name);
  const rel = +$("#map-depth").value / 100; const layer = Math.round(rel * mdl.n_layers);
  const d = await api(`/api/mds/${encodeURIComponent(name)}?layer=${layer}`);
  const view = $("#map-view").value;
  const box = $(host); box.innerHTML = "";
  const S = box.clientWidth || 400; const pad = 28;
  const svg = el("svg", { viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": `MDS ${name} layer ${layer}` }, box);
  const P = (v) => pad + (v + 1) / 2 * (S - 2 * pad);
  el("line", { x1: P(0), x2: P(0), y1: pad, y2: S - pad, class: "gridline" }, svg);
  el("line", { y1: P(0), y2: P(0), x1: pad, x2: S - pad, class: "gridline" }, svg);
  const surf = css("--surface-1");
  if (view === "cloud") {
    d.points.forEach((p, i) => {
      const [x, y] = d.cloud[i]; const c = css(p.false_belief ? "--fb" : "--tb");
      const shape = p.qtype === "belief_leaver"
        ? el("circle", { cx: P(x), cy: P(-y), r: 4.5, fill: c, stroke: surf, "stroke-width": 1.5 }, svg)
        : el("path", { d: `M${P(x)},${P(-y) - 5.5}l5,9h-10z`, fill: c, stroke: surf, "stroke-width": 1.5 }, svg);
      shape.addEventListener("mousemove", (ev) => showTip(ev, `<b>${p.qid}</b><br>${QT_LABEL[p.qtype]}<br>variant: ${p.variant} · ${p.skeleton}<br>${p.false_belief ? "false" : "true"} belief`));
      shape.addEventListener("mouseleave", hideTip);
    });
    $(meta).textContent = `${name} · layer ${layer} of ${mdl.n_layers} · stress ${d.cloud_stress.toFixed(3)} · ${d.points.length} prompts`;
  } else {
    d.centroid_keys.forEach(([qt, v], i) => {
      const [x, y] = d.centroids[i]; const fb = FB_OF(qt, v); const c = css(fb ? "--fb" : "--tb");
      const dot = el("circle", { cx: P(x), cy: P(-y), r: 7, fill: c, stroke: surf, "stroke-width": 2 }, svg);
      el("text", { x: P(x) + 10, y: P(-y) + 4, fill: css("--text-secondary"), "font-size": 11 }, svg).textContent = `${SHORT[qt]}·${v}`;
      dot.addEventListener("mousemove", (ev) => showTip(ev, `<b>${QT_LABEL[qt]}</b><br>variant: ${v}<br>${fb ? "false" : "true"} belief`));
      dot.addEventListener("mouseleave", hideTip);
    });
    $(meta).textContent = `${name} · layer ${layer} of ${mdl.n_layers} · stress ${d.centroid_stress.toFixed(3)} · 15 condition means`;
  }
}

function drawMaps() {
  $("#map-depth-val").textContent = `${$("#map-depth").value}%`;
  $("#map-legend").innerHTML = $("#map-view").value === "cloud"
    ? `<span><i class="dot" style="background:var(--tb)"></i>true belief</span><span><i class="dot" style="background:var(--fb)"></i>false belief</span><span>● leaver's belief</span><span>▲ actor about leaver</span>`
    : `<span><i class="dot" style="background:var(--tb)"></i>true belief</span><span><i class="dot" style="background:var(--fb)"></i>false belief</span><span>R reality · L leaver · A actor · A→L / L→A second order</span>`;
  drawMap("#map-a", "#map-chart-a", "#map-meta-a");
  drawMap("#map-b", "#map-chart-b", "#map-meta-b");
}

// ------------------------------------------------------------ token lens --
const showcase = {};
function mix(t) {               // sequential ramp, t in [0,1]
  const a = css("--seq-lo"), b = css("--seq-hi");
  const h = (s) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
  const [r1, g1, b1] = h(a), [r2, g2, b2] = h(b);
  return `rgb(${Math.round(r1 + (r2 - r1) * t)},${Math.round(g1 + (g2 - g1) * t)},${Math.round(b1 + (b2 - b1) * t)})`;
}
function inkOn(t) { return t > 0.55 ? (matchMedia("(prefers-color-scheme: dark)").matches ? "#0b0b0b" : "#ffffff") : "var(--text-primary)"; }

async function lensPanel(model, qid, T, host, drawStory) {
  const box = $(host);
  let d;
  try { d = await api(`/api/token/${encodeURIComponent(model)}/${encodeURIComponent(qid)}?T=${T}&k=4`); }
  catch (e) { box.innerHTML = `<p class="note">This prompt is not in ${model}'s showcase set.</p>`; return; }
  if (drawStory) {
    const maxS = 8;
    const toks = d.tokens.map((t, i) => {
      const s = d.surprisal_bits[i]; const v = s == null ? 0 : Math.min(1, s / maxS);
      const txt = t.replace(/</g, "&lt;");
      return txt === "\n" || txt.trim() === "" && t.includes("\n") ? "<br>" :
        `<span class="tok" style="background:${mix(v)};color:${inkOn(v)}" data-s="${s}">${txt}</span>`;
    }).join(" ");
    $("#lens-story").innerHTML = `<div class="lens-sub">Tokens shaded by ${model}'s surprisal (bits, 0 → ${maxS}+). Answer: <b>${d.answer}</b> · foil: ${d.foil} · ${d.false_belief ? "false" : "true"} belief · ${QT_LABEL[d.qtype]} · variant ${d.variant}</div>${toks}`;
    $("#lens-story").querySelectorAll(".tok").forEach((e) => {
      e.addEventListener("mousemove", (ev) => showTip(ev, `${e.textContent}: ${e.dataset.s === "null" ? "–" : (+e.dataset.s).toFixed(2) + " bits"}`));
      e.addEventListener("mouseleave", hideTip);
    });
  }
  const rows = d.layers.map((L, i) => {
    const top = L.top.map((t) => `<td class="p"><span class="cell" style="background:${mix(t.p)};color:${inkOn(t.p)}" title="logit ${t.logit}">${t.token.replace(/</g, "&lt;").trim() || "␣"} ${(100 * t.p).toFixed(0)}%</span></td>`).join("");
    const win = L.answer_p > L.foil_p;
    return `<tr><td>${i === 0 ? "emb" : i}</td><td class="num" style="${win ? "font-weight:600" : ""}">${(100 * L.answer_p).toFixed(1)}%</td><td class="num">${(100 * L.foil_p).toFixed(1)}%</td>${top}</tr>`;
  }).join("");
  box.innerHTML = `<div class="lens-sub">Logit lens at the last token, T = ${d.T.toFixed(2)}${d.renormalised_over_topk ? " · probabilities renormalised over the stored top-k" : " · exact over the full vocabulary"}</div>
    <div class="table-wrap"><table class="lens-table"><tr><th>Layer</th><th class="num">P(${d.answer})</th><th class="num">P(${d.foil})</th><th colspan="4">Top-4 next tokens</th></tr>${rows}</table></div>`;
}

async function drawLens() {
  const qid = $("#lens-prompt").value; if (!qid) return;
  const T = $("#lens-temp").value; $("#lens-temp-val").textContent = (+T).toFixed(2);
  await lensPanel($("#lens-a").value, qid, T, "#lens-a-body", true);
  await lensPanel($("#lens-b").value, qid, T, "#lens-b-body", false);
}

async function fillPrompts() {
  const a = $("#lens-a").value, b = $("#lens-b").value;
  for (const m of [a, b]) if (!showcase[m]) showcase[m] = await api(`/api/showcase/${encodeURIComponent(m)}`);
  const inB = new Set(showcase[b].map((s) => s.qid));
  const list = showcase[a].filter((s) => inB.has(s.qid));
  const use = list.length ? list : showcase[a];
  const prev = $("#lens-prompt").value;
  $("#lens-prompt").innerHTML = use.map((s) => `<option value="${s.qid}">${s.group} · ${s.variant} · ${QT_LABEL[s.qtype]}${s.false_belief ? " · FB" : ""}</option>`).join("");
  if (use.some((s) => s.qid === prev)) $("#lens-prompt").value = prev;
}

// ------------------------------------------------------------------ init --
async function init() {
  state.models = await api("/api/models");
  if (!state.models.length) { document.querySelector("main").innerHTML = "<p class='note'>No results yet. Run <code>python -m tomlens.run --tiny gpt llama gru</code>.</p>"; return; }
  state.models.forEach((m, i) => { state.colors[m.name] = SERIES[i % SERIES.length]; });
  for (const m of state.models) state.results[m.name] = await api(`/api/results/${encodeURIComponent(m.name)}`);
  $("#align-qtype").innerHTML = ["pooled", "belief_leaver", "nested_actor_about_leaver", "nested_leaver_about_actor", "binding"]
    .map((q) => `<option value="${q}">${QT_LABEL[q]}</option>`).join("");
  const lensKeys = Object.keys(state.results[state.models[0].name].lens);
  $("#lens-key").innerHTML = lensKeys.map((k) => { const [q, b] = k.split("|"); return `<option value="${k}">${QT_LABEL[q]} · ${b === "FB" ? "false" : "true"} belief</option>`; }).join("");
  if (lensKeys.includes("belief_leaver|FB")) $("#lens-key").value = "belief_leaver|FB";
  const opts = state.models.map((m) => `<option>${m.name}</option>`).join("");
  for (const s of ["#map-a", "#map-b", "#lens-a", "#lens-b"]) $(s).innerHTML = opts;
  if (state.models.length > 1) { $("#map-b").selectedIndex = 1; $("#lens-b").selectedIndex = 1; }
  drawCards(); drawAlign(); drawMetric(); drawLensCurves(); drawBehaviour();
  $("#align-qtype").onchange = drawAlign; $("#lens-key").onchange = drawLensCurves; $("#beh-split").onchange = drawBehaviour;
  for (const s of ["#map-a", "#map-b", "#map-view"]) $(s).onchange = drawMaps;
  $("#map-depth").oninput = drawMaps;
  for (const s of ["#lens-a", "#lens-b"]) $(s).onchange = async () => { await fillPrompts(); drawLens(); };
  $("#lens-prompt").onchange = drawLens; $("#lens-temp").oninput = drawLens;
  await fillPrompts();
  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { drawAlign(); drawMetric(); drawLensCurves(); }, 150); });
}
init();
