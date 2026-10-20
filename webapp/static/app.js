"use strict";

const DEFAULT_PLACE = { name: "Montreal", region: "Quebec, Canada", lat: 45.5088, lon: -73.5878 };
const $ = (id) => document.getElementById(id);

// ---- small helpers ------------------------------------------------------------------------------
const store = {
  get(k) { try { return localStorage.getItem("tw:" + k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem("tw:" + k, v); } catch { /* private mode */ } },
};
const state = { unit: store.get("unit") === "F" ? "F" : "C", place: null, data: null, reqId: 0 };
const conv = (c) => (state.unit === "F" ? (c * 9) / 5 + 32 : c);
const minus = (s) => String(s).replace(/^-/, "−"); // a true minus sign, not a hyphen
const deg = (c) => (c == null ? "–" : `${minus(Math.round(conv(c)) || 0)}°`);
const miss = (c) => (c == null ? "–" : `${(state.unit === "F" ? (c * 9) / 5 : c).toFixed(1)}°`);

// Times arrive as local wall-clock strings ("2026-10-20T15:00"); treat them as UTC so the
// browser's own time zone and daylight saving never shift them.
const parse = (s) => {
  const [d, t = "00:00"] = s.split("T");
  const [y, m, day] = d.split("-").map(Number);
  const [h, min] = t.split(":").map(Number);
  return new Date(Date.UTC(y, m - 1, day, h, min));
};
const fmt = (date, opts) => new Intl.DateTimeFormat("en-US", { ...opts, timeZone: "UTC" }).format(date);
const weekday = (d) => fmt(d, { weekday: "short" });
const longDay = (d) => fmt(d, { weekday: "long" });
const monthDay = (d) => fmt(d, { month: "short", day: "numeric" });
const clock = (d) => fmt(d, { hour: "numeric", minute: d.getUTCMinutes() ? "2-digit" : undefined });
const sameDay = (a, b) => a.toISOString().slice(0, 10) === b.toISOString().slice(0, 10);

// Diverging temperature ramp in °C: blue (cold) -> neutral gray (mild) -> red (hot).
const RAMP = [
  [-25, "#0d366b"], [-15, "#1c5cab"], [-5, "#3987e5"], [4, "#86b6ef"],
  [12, "#aaa69c"], [19, "#ee9468"], [26, "#e2602d"], [33, "#d63f3e"], [40, "#a3262a"],
];
function tempColor(c) {
  if (c <= RAMP[0][0]) return RAMP[0][1];
  for (let i = 1; i < RAMP.length; i++) {
    const [t1, c1] = RAMP[i];
    if (c <= t1) {
      const [t0, c0] = RAMP[i - 1];
      return mix(c0, c1, (c - t0) / (t1 - t0));
    }
  }
  return RAMP[RAMP.length - 1][1];
}
function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return "#" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

// ---- loading a place --------------------------------------------------------------------------
function initialPlace() {
  const q = new URLSearchParams(location.search);
  const lat = parseFloat(q.get("lat")), lon = parseFloat(q.get("lon"));
  if (Number.isFinite(lat) && Number.isFinite(lon)) {
    return { lat, lon, name: q.get("name") || "Selected place", region: q.get("region") || "" };
  }
  try { const saved = JSON.parse(store.get("place")); if (saved && Number.isFinite(saved.lat)) return saved; } catch {}
  return DEFAULT_PLACE;
}

async function loadPlace(place, { push = false } = {}) {
  state.place = place;
  store.set("place", JSON.stringify(place));
  if (push) {
    const q = new URLSearchParams({ lat: place.lat.toFixed(4), lon: place.lon.toFixed(4), name: place.name });
    if (place.region) q.set("region", place.region);
    history.replaceState(null, "", "?" + q.toString());
  }
  const key = `${place.lat.toFixed(3)},${place.lon.toFixed(3)}`;
  $("place-name").textContent = place.name;
  $("place-region").textContent = place.region || `${place.lat.toFixed(2)}, ${place.lon.toFixed(2)}`;
  document.title = `${place.name} | Timestral Weather`;
  $("error").hidden = true;
  if (state.dataKey !== key) $("now-temp").textContent = "–"; // never show another place's numbers
  $("summary").textContent = "Forecasting…";
  $("now-note").textContent = "";
  document.body.classList.add("is-loading");
  const id = ++state.reqId;
  const slow = setTimeout(() => {
    if (id === state.reqId) $("now-note").textContent = "Waking up the model. The first forecast takes a little longer.";
  }, 4000);
  try {
    const r = await fetch(`api/forecast?lat=${place.lat}&lon=${place.lon}`);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || "The forecast couldn’t be loaded. Try again in a minute.");
    if (id !== state.reqId) return;
    state.data = body;
    state.dataKey = key;
    document.body.classList.remove("is-empty");
    render({ animate: true });
  } catch (e) {
    if (id !== state.reqId) return;
    if (state.dataKey !== key) document.body.classList.add("is-empty");
    $("summary").textContent = "";
    $("now-note").textContent = "";
    $("error-text").textContent = e.message.startsWith("Failed to fetch")
      ? "Can’t reach the forecast server. Check your connection and try again." : e.message;
    $("error").hidden = false;
  } finally {
    clearTimeout(slow);
    if (id === state.reqId) document.body.classList.remove("is-loading");
  }
}

// ---- rendering ----------------------------------------------------------------------------------
function render({ animate = false } = {}) {
  const d = state.data;
  if (!d) return;
  const now = parse(d.now.time);
  $("now-temp").textContent = deg(d.now.temperature);
  $("now-note").textContent = `Now, ${clock(now)} local time. Forecast by ${d.model}.`;
  $("summary").textContent = summarize(d.daily, now);
  $("days").classList.toggle("animate", animate);
  renderDays(d.daily, now);
  renderForecastChart(d, now);
  renderRecord(d.track_record);
}

function summarize(daily, now) {
  if (!daily.length) return "";
  const highs = daily.map((x) => x.high);
  const lo = Math.min(...highs), hi = Math.max(...highs);
  const warm = daily[highs.indexOf(hi)];
  const cold = daily.reduce((a, b) => (b.low < a.low ? b : a));
  const name = (x) => (sameDay(parse(x.date), now) ? "today" : longDay(parse(x.date)));
  const first = Math.round(conv(hi)) - Math.round(conv(lo)) <= 2
    ? `Steady highs around ${deg((hi + lo) / 2)} this week.`
    : `Highs from ${deg(lo)} to ${deg(hi)} this week, warmest ${name(warm)}.`;
  return `${first} Coldest point: ${deg(cold.low)} ${name(cold) === "today" ? "tonight" : "on " + name(cold)}.`;
}

function renderDays(daily, now) {
  const all = daily.flatMap((x) => [x.low, x.high, x.service_low, x.service_high]).filter((v) => v != null);
  const min = Math.min(...all), max = Math.max(...all);
  const span = Math.max(max - min, 6);
  const pad = (span - (max - min)) / 2;
  const lo = min - pad, range = span;
  const pos = (v) => ((v - lo) / range) * 100; // % from bottom
  const ol = $("days");
  ol.innerHTML = "";
  daily.forEach((x, i) => {
    const date = parse(x.date);
    const li = document.createElement("li");
    li.className = "day";
    li.style.setProperty("--i", i);
    const label = sameDay(date, now) ? "Today" : weekday(date);
    li.setAttribute("aria-label",
      `${sameDay(date, now) ? "Today" : longDay(date)}: high ${deg(x.high)}, low ${deg(x.low)}. ` +
      `Weather service: high ${deg(x.service_high)}, low ${deg(x.service_low)}.`);
    const fillBottom = pos(x.low), fillTop = pos(x.high);
    const svc = x.service_low != null && x.service_high != null
      ? `<span class="thermo-service" style="bottom:${pos(x.service_low)}%;height:${Math.max(pos(x.service_high) - pos(x.service_low), 1.5)}%"></span>` : "";
    li.innerHTML = `
      <span class="day-name" aria-hidden="true">${label}</span>
      <span class="day-date" aria-hidden="true">${monthDay(date)}</span>
      <span class="day-hi" aria-hidden="true">${deg(x.high)}</span>
      <span class="thermo" aria-hidden="true">
        <span class="thermo-track"></span>
        <span class="thermo-fill" style="bottom:${fillBottom}%;height:${Math.max(fillTop - fillBottom, 3)}%;
          background:linear-gradient(to top, ${tempColor(x.low)}, ${tempColor(x.high)})"></span>
        ${svc}
      </span>
      <span class="day-lo" aria-hidden="true">${deg(x.low)}</span>`;
    ol.appendChild(li);
  });
}

function renderForecastChart(d, now) {
  const times = [...d.recent.time, ...d.forecast.time].map(parse);
  const nR = d.recent.time.length, nF = d.forecast.time.length;
  const nowIdx = nR - 1;
  const lastObs = d.recent.observed[nowIdx];
  const observed = [...d.recent.observed, ...Array(nF).fill(null)];
  const pad = Array(nR).fill(null);
  const median = [...pad, ...d.forecast.median];
  const service = [...pad, ...d.forecast.service];
  const low = [...pad, ...d.forecast.low];
  const high = [...pad, ...d.forecast.high];
  median[nowIdx] = service[nowIdx] = low[nowIdx] = high[nowIdx] = lastObs; // join the lines at "now"
  $("hourly-key").innerHTML = keyHtml(true);
  lineChart($("chart-forecast"), {
    times, nowIdx, nowLabel: "Now",
    band: { low, high },
    series: [
      { key: "observed", label: "Observed", values: observed },
      { key: "timestral", label: "Timestral", values: median, direct: true },
      { key: "service", label: "Weather service", values: service, direct: true },
    ],
    summary: `Hourly forecast for the next 7 days. Timestral ranges from ${deg(Math.min(...d.forecast.median.filter(Number.isFinite)))} to ${deg(Math.max(...d.forecast.median.filter(Number.isFinite)))}.`,
  });
}

function renderRecord(tr) {
  const issued = parse(tr.issued_at);
  $("record-intro").textContent =
    `We reran this forecast as if it were ${longDay(issued)}, ${monthDay(issued)}, ${clock(issued)}, ` +
    `using only what was known then, and compared it with the temperatures that followed.`;
  $("score-t").textContent = miss(tr.mae.timestral);
  $("score-s").textContent = miss(tr.mae.service);
  const t = tr.mae.timestral, s = tr.mae.service;
  let verdict = "";
  if (t != null && s != null) {
    if (Math.abs(t - s) < 0.05) verdict = "Both were about equally close this week.";
    else if (t < s) verdict = `Timestral was ${Math.round((1 - t / s) * 100)}% closer than the weather service.`;
    else verdict = `The weather service was closer this week, by ${miss(s - t + 0)} on average.`;
  }
  $("verdict").textContent = verdict;
  $("record-key").innerHTML = keyHtml(true);
  lineChart($("chart-record"), {
    times: tr.time.map(parse),
    band: { low: tr.low, high: tr.high },
    series: [
      { key: "observed", label: "Observed", values: tr.observed },
      { key: "timestral", label: "Timestral", values: tr.median, direct: true },
      { key: "service", label: "Weather service", values: tr.service, direct: true },
    ],
    summary: `Last week's forecast against observed temperatures. Average miss: Timestral ${miss(t)}, weather service ${miss(s)}.`,
  });
  $("misses").innerHTML = tr.daily.map((x) => {
    const tc = x.timestral != null && x.service != null && x.timestral < x.service;
    const sc = x.timestral != null && x.service != null && x.service < x.timestral;
    const date = parse(x.date);
    return `<tr><td>${weekday(date)}, ${monthDay(date)}</td>
      <td class="${tc ? "closer" : ""}">${miss(x.timestral)}</td>
      <td class="${sc ? "closer" : ""}">${miss(x.service)}</td></tr>`;
  }).join("");
}

function keyHtml(withObserved) {
  return (withObserved ? `<span class="key-ink"></span>Observed<span class="gap"></span>` : "") +
    `<span class="key-solid"></span>Timestral<span class="gap"></span>` +
    `<span class="key-band"></span>Likely range<span class="gap"></span>` +
    `<span class="key-dash"></span>Weather service`;
}

// ---- line chart (hand-built SVG) ---------------------------------------------------------------
const charts = new Map();

function lineChart(el, cfg) {
  charts.set(el, cfg);
  drawChart(el);
}

function drawChart(el) {
  const cfg = charts.get(el);
  if (!cfg) return;
  const W = el.clientWidth;
  const narrow = W < 560;
  const H = Math.round(Number(el.dataset.height) * (narrow ? 0.78 : 1));
  const m = { top: 18, right: narrow ? 8 : 104, bottom: 30, left: 34 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const n = cfg.times.length;

  const vals = [...cfg.series.flatMap((s) => s.values), ...cfg.band.low, ...cfg.band.high]
    .filter((v) => v != null).map(conv);
  let yMin = Math.min(...vals), yMax = Math.max(...vals);
  const step = niceStep((yMax - yMin) / 4);
  yMin = Math.floor(yMin / step) * step; yMax = Math.ceil(yMax / step) * step;
  if (yMax === yMin) yMax += step;
  const x = (i) => m.left + (i / (n - 1)) * iw;
  const y = (v) => m.top + (1 - (conv(v) - yMin) / (yMax - yMin)) * ih;

  const parts = [];
  parts.push(`<g class="grid axis">`);
  for (let v = yMin; v <= yMax + 1e-9; v += step) {
    const yy = m.top + (1 - (v - yMin) / (yMax - yMin)) * ih;
    parts.push(`<line x1="${m.left}" x2="${m.left + iw}" y1="${yy}" y2="${yy}"/>`,
      `<text x="${m.left - 8}" y="${yy + 4}" text-anchor="end">${minus(Math.round(v) || 0)}°</text>`);
  }
  parts.push(`</g>`);

  // day separators at local midnight, day names centered at noon
  parts.push(`<g class="axis">`);
  cfg.times.forEach((t, i) => {
    const h = t.getUTCHours();
    if (h === 0 && i > 0) parts.push(`<line class="daysep" x1="${x(i)}" x2="${x(i)}" y1="${m.top}" y2="${m.top + ih}"/>`);
    if (h === 12) parts.push(`<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${weekday(t)}</text>`);
  });
  parts.push(`</g>`);

  // likely-range band
  const bi = [];
  for (let i = 0; i < n; i++) if (cfg.band.low[i] != null && cfg.band.high[i] != null) bi.push(i);
  if (bi.length) {
    const top = bi.map((i) => `${x(i)},${y(cfg.band.high[i])}`);
    const bot = bi.slice().reverse().map((i) => `${x(i)},${y(cfg.band.low[i])}`);
    parts.push(`<polygon class="band" points="${top.join(" ")} ${bot.join(" ")}"/>`);
  }
  if (cfg.nowIdx != null) {
    parts.push(`<line class="nowline" x1="${x(cfg.nowIdx)}" x2="${x(cfg.nowIdx)}" y1="${m.top - 6}" y2="${m.top + ih}"/>`,
      `<text class="nowlabel" x="${x(cfg.nowIdx) + 6}" y="${m.top + 4}">${cfg.nowLabel}</text>`);
  }
  const ends = [];
  for (const s of cfg.series) {
    parts.push(`<path class="series s-${s.key}" d="${pathFor(s.values, x, y)}"/>`);
    let last = -1;
    s.values.forEach((v, i) => { if (v != null) last = i; });
    if (s.direct && last >= 0) ends.push({ label: s.label, y: y(s.values[last]), x: x(last) });
  }
  if (!narrow) { // direct labels at the right end, nudged apart
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
    ends.forEach((e) => parts.push(`<text class="label" x="${e.x + 8}" y="${e.y + 4}">${e.label}</text>`));
  }
  parts.push(`<g class="hover" visibility="hidden"><line class="cross" y1="${m.top}" y2="${m.top + ih}"/>` +
    cfg.series.map((s) => `<circle class="dot" r="4.5" data-k="${s.key}" style="fill:var(--${s.key === "observed" ? "ink" : s.key})"/>`).join("") +
    `</g>`);

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${cfg.summary}">${parts.join("")}</svg>`;
  el.tabIndex = 0;
  attachHover(el, cfg, { x, y, m, iw, n });
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 0.5))));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

function pathFor(values, x, y) {
  let d = "", pen = false;
  values.forEach((v, i) => {
    if (v == null) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

function attachHover(el, cfg, g) {
  const svg = el.querySelector("svg");
  const layer = svg.querySelector(".hover");
  const cross = layer.querySelector(".cross");
  const tip = $("tooltip");
  let idx = null;

  const show = (i) => {
    idx = Math.max(0, Math.min(g.n - 1, i));
    const xx = g.x(idx);
    cross.setAttribute("x1", xx); cross.setAttribute("x2", xx);
    layer.setAttribute("visibility", "visible");
    const rows = [];
    for (const s of cfg.series) {
      const v = s.values[idx];
      const dot = layer.querySelector(`[data-k="${s.key}"]`);
      if (v == null) { dot.setAttribute("visibility", "hidden"); continue; }
      dot.setAttribute("visibility", "visible");
      dot.setAttribute("cx", xx); dot.setAttribute("cy", g.y(v));
      const sw = s.key === "observed" ? "key-ink" : s.key === "timestral" ? "key-solid" : "key-dash";
      let val = deg(v);
      if (s.key === "timestral" && cfg.band.low[idx] != null && cfg.band.low[idx] !== cfg.band.high[idx]) {
        val += ` <span class="muted">(${deg(cfg.band.low[idx])} to ${deg(cfg.band.high[idx])})</span>`;
      }
      rows.push(`<div class="t-row"><span class="${sw}"></span><span>${s.label}</span><span class="t-val">${val}</span></div>`);
    }
    const t = cfg.times[idx];
    tip.innerHTML = `<div class="t-time">${weekday(t)} ${monthDay(t)}, ${clock(t)}</div>${rows.join("")}`;
    tip.hidden = false;
    const box = svg.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const px = box.left + xx * (box.width / svg.viewBox.baseVal.width);
    let left = px + 14;
    if (left + tw > window.innerWidth - 8) left = px - tw - 14;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.min(Math.max(8, box.top + 8), window.innerHeight - th - 8)}px`;
  };
  const hide = () => { layer.setAttribute("visibility", "hidden"); tip.hidden = true; idx = null; };
  const fromEvent = (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * svg.viewBox.baseVal.width;
    return Math.round(((px - g.m.left) / g.iw) * (g.n - 1));
  };
  svg.addEventListener("pointermove", (e) => show(fromEvent(e)));
  svg.addEventListener("pointerdown", (e) => show(fromEvent(e)));
  svg.addEventListener("pointerleave", hide);
  el.onblur = hide;
  el.onkeydown = (e) => {
    const stepH = e.shiftKey ? 24 : 1;
    if (e.key === "ArrowRight") { show((idx ?? cfg.nowIdx ?? 0) + stepH); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { show((idx ?? cfg.nowIdx ?? 0) - stepH); e.preventDefault(); }
    else if (e.key === "Escape") hide();
  };
}

let resizeTimer;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => charts.forEach((_, el) => drawChart(el)), 80);
}).observe(document.body);
window.addEventListener("scroll", () => { $("tooltip").hidden = true; }, { passive: true });

// ---- search -------------------------------------------------------------------------------------
const input = $("q"), list = $("results");
let options = [], active = -1, searchTimer, searchId = 0;

function openList(items) {
  options = items;
  active = items.length ? 0 : -1;
  list.innerHTML = items.map((p, i) =>
    `<li id="opt-${i}" role="option" aria-selected="${i === active}" data-i="${i}">${escapeHtml(p.label)}` +
    (p.sub ? `<span class="sub">${escapeHtml(p.sub)}</span>` : "") + `</li>`).join("");
  list.hidden = !items.length;
  input.setAttribute("aria-expanded", String(!list.hidden));
  input.setAttribute("aria-activedescendant", active >= 0 ? `opt-${active}` : "");
}
function closeList() { list.hidden = true; input.setAttribute("aria-expanded", "false"); options = []; }
function highlight(i) {
  active = (i + options.length) % options.length;
  [...list.children].forEach((li, j) => li.setAttribute("aria-selected", String(j === active)));
  input.setAttribute("aria-activedescendant", `opt-${active}`);
  list.children[active]?.scrollIntoView({ block: "nearest" });
}
function choose(i) {
  const o = options[i];
  if (!o) return;
  closeList();
  input.value = "";
  input.blur();
  o.pick();
}
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const myLocation = {
  label: "Use my current location",
  pick() {
    if (!navigator.geolocation) return;
    $("summary").textContent = "Finding your location…";
    navigator.geolocation.getCurrentPosition(
      (p) => loadPlace({ name: "Your location", region: "", lat: +p.coords.latitude.toFixed(4), lon: +p.coords.longitude.toFixed(4) }, { push: true }),
      () => { $("summary").textContent = "Location access was blocked. Search for a city instead."; },
      { timeout: 10000 });
  },
};

input.addEventListener("focus", () => { if (!input.value.trim() && navigator.geolocation) openList([myLocation]); });
input.addEventListener("input", () => {
  const q = input.value.trim();
  clearTimeout(searchTimer);
  if (q.length < 2) { navigator.geolocation ? openList([myLocation]) : closeList(); return; }
  searchTimer = setTimeout(async () => {
    const id = ++searchId;
    try {
      const r = await fetch(`api/geocode?q=${encodeURIComponent(q)}`);
      const { results = [] } = await r.json();
      if (id !== searchId) return;
      openList(results.length ? results.map((p) => {
        const region = [p.region, p.country].filter(Boolean).join(", ");
        return { label: p.name, sub: region, pick: () => loadPlace({ name: p.name, region, lat: p.lat, lon: p.lon }, { push: true }) };
      }) : [{ label: `No places match “${q}”`, sub: "Check the spelling, or try a nearby larger city.", pick() {} }]);
    } catch {
      if (id === searchId) openList([{ label: "Search is unavailable right now", sub: "Try again in a moment.", pick() {} }]);
    }
  }, 220);
});
input.addEventListener("keydown", (e) => {
  if (list.hidden) return;
  if (e.key === "ArrowDown") { highlight(active + 1); e.preventDefault(); }
  else if (e.key === "ArrowUp") { highlight(active - 1); e.preventDefault(); }
  else if (e.key === "Enter") { choose(active); e.preventDefault(); }
  else if (e.key === "Escape") { closeList(); }
});
list.addEventListener("pointerdown", (e) => {
  const li = e.target.closest("li");
  if (li) { e.preventDefault(); choose(Number(li.dataset.i)); }
});
input.addEventListener("blur", () => setTimeout(closeList, 120));

// ---- units --------------------------------------------------------------------------------------
document.querySelectorAll(".units button").forEach((b) => {
  b.setAttribute("aria-pressed", String(b.dataset.unit === state.unit));
  b.addEventListener("click", () => {
    state.unit = b.dataset.unit;
    store.set("unit", state.unit);
    document.querySelectorAll(".units button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    render();
  });
});

$("retry").addEventListener("click", () => loadPlace(state.place));
loadPlace(initialPlace());
