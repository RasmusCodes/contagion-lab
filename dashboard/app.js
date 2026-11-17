/* Contagion Lab dashboard. Reads window.CL_DATA (data.js) and recomputes every interactive number in the browser. */
(() => {
  "use strict";
  const D = window.CL_DATA;
  const $ = (s, el = document) => el.querySelector(s);
  if (!D || !window.d3) {
    const m = document.querySelector("main");
    if (m) m.textContent = !D ? "The data file (data.js) did not load. Run scripts/build_dashboard.py and reopen this page."
      : "The charting library (d3) did not load. Check the internet connection and reload.";
    return;
  }
  const tip = $("#tip");
  let C = {};
  const TOK = ["ink", "ink-2", "muted", "rule", "surface", "surface-2", "bg", "accent", "s1", "s2", "s3", "s4", "s5", "s6",
    "s7", "s8", "pos", "pos-2", "neg", "neg-2", "mid", "context", "grid", "axis", "shade", "good", "warn", "crit"];
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    TOK.forEach(k => { C[k] = cs.getPropertyValue("--" + k).trim(); });
  }
  readColors();

  // ------------------------------------------------------------------------------------------- data
  const DATES = D.dates, ND = DATES.length;
  const TIME = DATES.map(s => new Date(s + "T00:00:00Z"));
  const U = new Map(D.universe.map(u => [u.id, u]));
  const cache = new Map();
  function ser(id) {
    if (!cache.has(id)) {
      const a = D.ret[id], s = D.scale[id], out = new Float64Array(a.length);
      for (let i = 0; i < a.length; i++) out[i] = a[i] === null ? NaN : a[i] / s;
      cache.set(id, out);
    }
    return cache.get(id);
  }
  function dec(a, s) { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] === null ? NaN : a[i] / s; return o; }
  function idxOf(dateStr) { // last index with date <= dateStr
    let lo = 0, hi = ND - 1;
    if (dateStr < DATES[0]) return 0;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (DATES[m] <= dateStr) lo = m; else hi = m - 1; }
    return lo;
  }
  function rangeIdx(v) {
    if (v === "all") return [0, ND - 1];
    if (v === "last3") return [Math.max(0, ND - 756), ND - 1];
    const [a, b] = v.split("|");
    return [idxOf(a), idxOf(b)];
  }
  const label = id => (U.get(id) || {}).name || (D.stress.labels || {})[id] || id;

  // ------------------------------------------------------------------------------------------- math
  function finiteCols(cols, i0, i1) {
    const rows = [];
    for (let i = i0; i <= i1; i++) { let ok = true; for (const c of cols) if (!Number.isFinite(c[i])) { ok = false; break; } if (ok) rows.push(i); }
    return rows;
  }
  function covOf(cols, rows) {
    const k = cols.length, n = rows.length, mu = new Array(k).fill(0);
    for (let j = 0; j < k; j++) { let s = 0; for (const i of rows) s += cols[j][i]; mu[j] = s / n; }
    const S = Array.from({ length: k }, () => new Array(k).fill(0));
    for (const i of rows) for (let a = 0; a < k; a++) { const da = cols[a][i] - mu[a]; for (let b = a; b < k; b++) S[a][b] += da * (cols[b][i] - mu[b]); }
    for (let a = 0; a < k; a++) for (let b = a; b < k; b++) { S[a][b] /= (n - 1); S[b][a] = S[a][b]; }
    return S;
  }
  function corrOf(S) { const k = S.length, R = S.map(r => r.slice()); for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) R[a][b] = S[a][b] / Math.sqrt(S[a][a] * S[b][b]); return R; }
  function inv(M) {
    const n = M.length, A = M.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      if (Math.abs(A[p][c]) < 1e-300) return null;
      [A[c], A[p]] = [A[p], A[c]];
      const d = A[c][c]; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
      for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; if (f) for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
    }
    return A.map(r => r.slice(n));
  }
  const mv = (M, v) => M.map(r => r.reduce((s, x, j) => s + x * v[j], 0));
  function erf(x) { // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
  const phi = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  function PhiInv(p) { // Acklam
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
    const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
    const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
    const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    const pl = 0.02425;
    if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  function quantile(sorted, q) { const p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p); return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo); }
  function pearson(x, y, idxs) {
    let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (const i of idxs) { const a = x[i], b = y[i]; n++; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; }
    const cv = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
    return cv / Math.sqrt(vx * vy);
  }
  function ranks(arr) { const o = arr.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]); const r = new Array(arr.length); o.forEach(([, i], k) => { r[i] = k + 1; }); return r; }

  // ------------------------------------------------------------------------------------------- formatting
  const f1 = x => (Number.isFinite(x) ? x.toFixed(1) : "n/a");
  const f2 = x => (Number.isFinite(x) ? x.toFixed(2) : "n/a");
  const sgn = (x, d = 1) => (Number.isFinite(x) ? (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d) : "n/a");
  const pct = (x, d = 1) => sgn(x, d) + "%";
  const fmtDate = s => { const t = new Date(s + "T00:00:00Z"); return t.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" }); };
  const fmtMonth = s => { const t = new Date(s + "T00:00:00Z"); return t.toLocaleDateString("en-US", { year: "numeric", month: "short", timeZone: "UTC" }); };
  function el(tag, attrs = {}, text) { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; }

  // ------------------------------------------------------------------------------------------- tooltip
  function showTip(evt, title, rows) {
    tip.replaceChildren();
    tip.appendChild(el("div", { class: "tt" }, title));
    for (const r of rows) {
      const row = el("div", { class: "row" });
      if (r.color) { const i = el("i"); i.style.setProperty("--c", r.color); row.appendChild(i); }
      row.appendChild(el("b", {}, r.value));
      row.appendChild(el("span", {}, r.label));
      tip.appendChild(row);
    }
    tip.hidden = false;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  function legend(container, items) {
    const L = typeof container === "string" ? $(container) : container;
    L.replaceChildren();
    for (const it of items) {
      const s = el("span"); const i = el("i", { class: it.kind || "" }); i.style.setProperty("--c", it.color);
      s.appendChild(i); s.appendChild(document.createTextNode(it.label)); L.appendChild(s);
    }
  }

  // ------------------------------------------------------------------------------------------- charts
  function shadeCrises(g, x, h, i0, i1) {
    for (const c of D.crises) {
      const a = Math.max(i0, idxOf(c.peak)), b = Math.min(i1, idxOf(c.trough));
      if (b <= a) continue;
      g.append("rect").attr("x", x(TIME[a])).attr("y", 0).attr("width", Math.max(1, x(TIME[b]) - x(TIME[a]))).attr("height", h)
        .attr("fill", C.shade);
    }
  }
  function axisStyle(sel) { sel.selectAll("path,line").attr("stroke", C.axis); sel.selectAll("text").attr("fill", C["ink-2"]).style("font-family", "var(--font-data)").style("font-size", "10.5px"); }

  /* Multi-panel time chart with one shared x axis and one crosshair. panels: [{height, series:[{name,color,values,width,fmt}],
     yDomain, yLabel, zero, log, area (stack of {values,color,name}), bands (per-index color), refLines}] */
  function timePanels(container, { panels, i0 = 0, i1 = ND - 1, marker = null, onClick = null, crises = true }) {
    const root = typeof container === "string" ? $(container) : container;
    root.replaceChildren();
    const W = Math.max(320, root.clientWidth || 600), m = { l: 48, r: 14, t: 6, b: 22 }, gap = 14;
    const H = panels.reduce((s, p) => s + p.height, 0) + gap * (panels.length - 1) + m.t + m.b;
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img");
    const x = d3.scaleUtc().domain([TIME[i0], TIME[i1]]).range([m.l, W - m.r]);
    let y0 = m.t;
    const step = Math.max(1, Math.floor((i1 - i0) / (W * 1.5)));
    const panelInfo = [];
    for (const p of panels) {
      const g = svg.append("g");
      const yr = [y0 + p.height, y0];
      let dom = p.yDomain;
      if (!dom) {
        let lo = Infinity, hi = -Infinity;
        for (const s of p.series || []) for (let i = i0; i <= i1; i++) { const v = s.values[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
        if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
        const pad = (hi - lo) * 0.06 || 1; dom = [lo - (p.log ? 0 : pad), hi + (p.log ? 0 : pad)];
        if (p.zero) { dom[0] = Math.min(dom[0], 0); dom[1] = Math.max(dom[1], 0); }
      }
      const y = p.log ? d3.scaleLog().domain(dom).range(yr) : d3.scaleLinear().domain(dom).range(yr).nice();
      if (p.bands) {
        let a = i0;
        for (let i = i0 + 1; i <= i1 + 1; i++) {
          if (i > i1 || p.bands[i] !== p.bands[a]) {
            if (p.bands[a]) g.append("rect").attr("x", x(TIME[a])).attr("y", yr[1]).attr("width", Math.max(0.5, x(TIME[Math.min(i, i1)]) - x(TIME[a]))).attr("height", p.height).attr("fill", p.bands[a]).attr("opacity", 0.22);
            a = i;
          }
        }
      } else if (crises) {
        const sg = g.append("g").attr("transform", `translate(0,${yr[1]})`);
        shadeCrises(sg, x, p.height, i0, i1);
      }
      const ax = g.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(Math.max(2, Math.round(p.height / 38)), p.tickFmt || undefined).tickSize(-(W - m.l - m.r)));
      axisStyle(ax); ax.selectAll("line").attr("stroke", C.grid); ax.select(".domain").remove();
      if (p.yLabel) g.append("text").attr("x", m.l + 4).attr("y", yr[1] + 11).text(p.yLabel).attr("fill", C.muted).style("font-size", "10.5px").style("font-family", "var(--font-data)");
      if (p.zero && dom[0] < 0 && dom[1] > 0) g.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", C.axis);
      for (const rl of p.refLines || []) g.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(rl)).attr("y2", y(rl)).attr("stroke", C.axis).attr("stroke-width", 0.8);
      if (p.area) {
        const cum = new Float64Array(ND);
        for (const s of p.area) {
          const lower = cum.slice();
          for (let i = i0; i <= i1; i++) cum[i] += Number.isFinite(s.values[i]) ? s.values[i] : 0;
          const pts = [];
          for (let i = i0; i <= i1; i += step) if (Number.isFinite(s.values[i])) pts.push(i);
          g.append("path").attr("d", d3.area().x(i => x(TIME[i])).y0(i => y(lower[i])).y1(i => y(cum[i]))(pts)).attr("fill", s.color).attr("opacity", 0.85);
        }
      }
      for (const s of p.series || []) {
        const pts = [];
        for (let i = i0; i <= i1; i += step) pts.push(i);
        if (pts[pts.length - 1] !== i1) pts.push(i1);
        const line = d3.line().defined(i => Number.isFinite(s.values[i]) && (!p.log || s.values[i] > 0)).x(i => x(TIME[i])).y(i => y(s.values[i]));
        g.append("path").attr("d", line(pts)).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", s.width || 1.6)
          .attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
      }
      panelInfo.push({ p, y, yr });
      y0 += p.height + gap;
    }
    const xa = svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.round(W / 110))));
    axisStyle(xa);
    if (marker !== null) svg.append("line").attr("x1", x(TIME[marker])).attr("x2", x(TIME[marker])).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.ink).attr("stroke-width", 1.2);
    const cross = svg.append("line").attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.muted).attr("stroke-width", 1).attr("visibility", "hidden");
    const bis = d3.bisector(d => d).left;
    function idxAt(evt) {
      const [px] = d3.pointer(evt, svg.node());
      const t = x.invert(px); let i = bis(TIME, t, i0, i1); if (i > i0 && (TIME[i] - t) > (t - TIME[i - 1])) i--;
      return Math.max(i0, Math.min(i1, i));
    }
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
      .style("cursor", onClick ? "pointer" : "crosshair")
      .on("pointermove", evt => {
        const i = idxAt(evt); cross.attr("x1", x(TIME[i])).attr("x2", x(TIME[i])).attr("visibility", "visible");
        const rows = [];
        for (const { p } of panelInfo) {
          for (const s of p.series || []) rows.push({ color: s.color, label: s.name, value: s.fmt ? s.fmt(s.values[i]) : f2(s.values[i]) });
          for (const s of p.area || []) rows.push({ color: s.color, label: s.name, value: s.fmt ? s.fmt(s.values[i]) : f2(s.values[i]) });
          if (p.extraTip) rows.push(...p.extraTip(i));
        }
        showTip(evt, fmtDate(DATES[i]), rows);
      })
      .on("pointerleave", () => { cross.attr("visibility", "hidden"); hideTip(); })
      .on("click", evt => { if (onClick) onClick(idxAt(evt)); });
    return { svg, x };
  }

  /* Horizontal bars from a zero baseline. items: [{label, value, lo, hi, color, tip}] */
  function barsH(container, items, { fmt = f2, unit = "", barH = 16, labelW = 150 } = {}) {
    const root = typeof container === "string" ? $(container) : container;
    root.replaceChildren();
    if (!items.length) { root.appendChild(el("p", { class: "note" }, "Nothing to show for this selection.")); return; }
    const W = Math.max(300, root.clientWidth || 500), step = barH + 6, m = { l: Math.min(labelW, W * 0.38), r: 52, t: 4, b: 20 };
    const H = m.t + m.b + items.length * step;
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    let lo = d3.min(items, d => Math.min(d.value, d.lo ?? d.value, 0)), hi = d3.max(items, d => Math.max(d.value, d.hi ?? d.value, 0));
    if (lo === hi) { lo -= 1; hi += 1; }
    const x = d3.scaleLinear().domain([lo, hi]).nice().range([m.l, W - m.r]);
    const ax = svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSize(-(H - m.t - m.b)));
    axisStyle(ax); ax.selectAll("line").attr("stroke", C.grid); ax.select(".domain").remove();
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.axis);
    items.forEach((d, k) => {
      const yc = m.t + k * step + step / 2, x0 = x(0), x1 = x(d.value);
      const w = Math.abs(x1 - x0), r = Math.min(4, w / 2, barH / 2);
      const left = Math.min(x0, x1);
      const path = d.value >= 0
        ? `M${left},${yc - barH / 2}h${Math.max(0, w - r)}a${r},${r} 0 0 1 ${r},${r}v${barH - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${Math.max(0, w - r)}z`
        : `M${left + w},${yc - barH / 2}h-${Math.max(0, w - r)}a${r},${r} 0 0 0 -${r},${r}v${barH - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${Math.max(0, w - r)}z`;
      const g = svg.append("g");
      g.append("path").attr("d", path).attr("fill", d.color || C.s1);
      if (d.lo !== undefined && d.hi !== undefined) g.append("line").attr("x1", x(d.lo)).attr("x2", x(d.hi)).attr("y1", yc).attr("y2", yc).attr("stroke", C["ink-2"]).attr("stroke-width", 1.2);
      g.append("text").attr("x", m.l - 8).attr("y", yc + 4).attr("text-anchor", "end").text(d.label).attr("fill", C.ink).style("font-size", "11.5px");
      // value label always right of the bar, its whisker and the zero line, so it never meets the category label
      const tx = Math.max(x0, x1, d.hi !== undefined ? x(d.hi) : x1) + 5;
      g.append("text").attr("x", tx).attr("y", yc + 4).attr("text-anchor", "start").text(fmt(d.value) + unit)
        .attr("fill", C["ink-2"]).style("font-size", "10.5px").style("font-family", "var(--font-data)");
      g.append("rect").attr("x", 0).attr("y", yc - step / 2).attr("width", W).attr("height", step).attr("fill", "transparent")
        .on("pointermove", evt => showTip(evt, d.label, d.tip || [{ value: fmt(d.value) + unit, label: "" }])).on("pointerleave", hideTip);
    });
  }

  /* Diverging stacked bars: rows [{label, parts:{key:value}, total}] */
  function stackH(container, rows, keys, { fmt = v => pct(v, 1), labelW = 170, barH = 16 } = {}) {
    const root = typeof container === "string" ? $(container) : container;
    root.replaceChildren();
    if (!rows.length) { root.appendChild(el("p", { class: "note" }, "Nothing to show for this selection.")); return; }
    const W = Math.max(320, root.clientWidth || 600), step = barH + 8, m = { l: Math.min(labelW, W * 0.4), r: 60, t: 4, b: 20 };
    const H = m.t + m.b + rows.length * step;
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    let lo = 0, hi = 0;
    for (const r of rows) { let p = 0, n = 0; for (const k of keys) { const v = r.parts[k.key] || 0; if (v > 0) p += v; else n += v; } lo = Math.min(lo, n, r.total ?? 0); hi = Math.max(hi, p, r.total ?? 0); }
    if (lo === hi) hi = lo + 1;
    const x = d3.scaleLinear().domain([lo, hi]).nice().range([m.l, W - m.r]);
    const ax = svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickSize(-(H - m.t - m.b)));
    axisStyle(ax); ax.selectAll("line").attr("stroke", C.grid); ax.select(".domain").remove();
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.axis);
    rows.forEach((r, k) => {
      const yc = m.t + k * step + step / 2; let pAcc = 0, nAcc = 0;
      for (const kk of keys) {
        const v = r.parts[kk.key] || 0; if (!v) continue;
        const a = v > 0 ? pAcc : nAcc + v, b = v > 0 ? pAcc + v : nAcc;
        if (v > 0) pAcc += v; else nAcc += v;
        const xa = x(a), xb = x(b);
        svg.append("rect").attr("x", xa + 1).attr("y", yc - barH / 2).attr("width", Math.max(0.5, xb - xa - 2)).attr("height", barH).attr("rx", 2).attr("fill", kk.color);
      }
      if (r.total !== undefined) svg.append("line").attr("x1", x(r.total)).attr("x2", x(r.total)).attr("y1", yc - barH / 2 - 3).attr("y2", yc + barH / 2 + 3).attr("stroke", C.ink).attr("stroke-width", 2);
      svg.append("text").attr("x", m.l - 8).attr("y", yc + 4).attr("text-anchor", "end").text(r.label).attr("fill", C.ink).style("font-size", "11.5px");
      if (r.total !== undefined) svg.append("text").attr("x", W - m.r + 6).attr("y", yc + 4).text(fmt(r.total)).attr("fill", C["ink-2"]).style("font-size", "10.5px").style("font-family", "var(--font-data)");
      svg.append("rect").attr("x", 0).attr("y", yc - step / 2).attr("width", W).attr("height", step).attr("fill", "transparent")
        .on("pointermove", evt => showTip(evt, r.label, [...(r.total !== undefined ? [{ value: fmt(r.total), label: "total" }] : []),
          ...keys.map(kk => ({ color: kk.color, value: fmt(r.parts[kk.key] || 0), label: kk.label }))])).on("pointerleave", hideTip);
    });
  }

  function heatmap(container, names, M, { fmt = f2, domain = [-1, 1], labels = names } = {}) {
    const root = typeof container === "string" ? $(container) : container;
    root.replaceChildren();
    const n = names.length, W = Math.max(300, root.clientWidth || 500), lw = 62, cell = Math.max(12, Math.min(34, (W - lw - 8) / n));
    const Hh = lw + cell * n + 4, Ww = lw + cell * n + 4;
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${Ww} ${Hh}`).style("max-width", Ww + "px");
    const col = d3.scaleLinear().domain([domain[0], 0, domain[1]]).range([C.neg, C.mid, C.pos]).clamp(true);
    for (let i = 0; i < n; i++) {
      svg.append("text").attr("x", lw - 5).attr("y", lw + i * cell + cell / 2 + 4).attr("text-anchor", "end").text(names[i]).attr("fill", C.ink).style("font-size", "10px").style("font-family", "var(--font-data)");
      svg.append("text").attr("transform", `translate(${lw + i * cell + cell / 2 + 3},${lw - 5}) rotate(-60)`).text(names[i]).attr("fill", C.ink).style("font-size", "10px").style("font-family", "var(--font-data)");
      for (let j = 0; j < n; j++) {
        const v = M[i][j];
        svg.append("rect").attr("x", lw + j * cell + 1).attr("y", lw + i * cell + 1).attr("width", cell - 2).attr("height", cell - 2).attr("rx", 2)
          .attr("fill", i === j ? C["surface-2"] : (Number.isFinite(v) ? col(v) : C["surface-2"]))
          .on("pointermove", evt => showTip(evt, `${labels[i]} / ${labels[j]}`, [{ value: fmt(v), label: "correlation" }])).on("pointerleave", hideTip);
      }
    }
  }

  // ------------------------------------------------------------------------------------------- claims
  function badgeEl(level) { return el("span", { class: "badge " + level }, level[0].toUpperCase() + level.slice(1)); }

  // =========================================================================================== NETWORK
  const NET = D.network;
  const NN = NET.names.length;
  const CLASS_ORDER = ["equity", "rates", "credit", "fx", "commodity", "volatility"];
  const CLASS_LABEL = { equity: "Equities", rates: "Rates", credit: "Credit", fx: "FX", commodity: "Commodities", volatility: "Volatility" };
  const netOrder = NET.names.map((n, i) => i).sort((a, b) => CLASS_ORDER.indexOf(NET.groups[NET.names[a]]) - CLASS_ORDER.indexOf(NET.groups[NET.names[b]]) || a - b);
  const MEASURES = {
    dy: { title: "Net pairwise spillovers", claim: "descriptive", note: "Mixes same-day correlation (descriptive, symmetric) with lagged spillovers (predictive). NET mostly measures how central a market is, not who moves first." },
    dyl: { title: "Net lagged spillovers", claim: "predictive", note: "Only the lagged part of the variance decomposition: direction in time, not mechanism. Early closes and stale prices create much of it (see the report, Q2)." },
    corr: { title: "Correlation network", claim: "descriptive", note: "Same-day co-movement over 63 trading days. No direction and no mechanism." },
    pcorr: { title: "Partial correlation network", claim: "descriptive", note: "Co-movement holding every other market fixed (precision matrix, 250 days). Still no direction." },
    granger: { title: "Granger network", claim: "predictive", note: "Lags of the source improve forecasts of the target (HC-robust, Benjamini-Hochberg 5%, 500-day window). Predictive only; asynchronous closes create many links." },
  };
  const netState = { k: NET.dates.length - 1, measure: "dy", topn: 22, timer: null };

  function netData(k, measure) {
    const N = NN, th = NET.theta[k], tc = NET.theta_c[k];
    const out = { edges: [], node: new Array(N).fill(0), size: new Array(N).fill(0), directed: true, rank: [] };
    if (measure === "dy" || measure === "dyl") {
      const T = (i, j) => (measure === "dy" ? th[i * N + j] : th[i * N + j] - tc[i * N + j]) / 1000;
      const to = new Array(N).fill(0), from = new Array(N).fill(0);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) { to[j] += T(i, j); from[i] += T(i, j); }
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const net = T(i, j) - T(j, i); // >0: j -> i
        if (net > 0) out.edges.push({ s: j, t: i, w: net }); else if (net < 0) out.edges.push({ s: i, t: j, w: -net });
      }
      for (let i = 0; i < N; i++) { out.node[i] = 100 * (to[i] - from[i]); out.size[i] = (to[i] + from[i]) / 2; }
      out.rank = NET.names.map((n, i) => ({ n, net: 100 * (to[i] - from[i]), to: 100 * to[i], from: 100 * from[i] }));
    } else if (measure === "corr" || measure === "pcorr") {
      out.directed = false;
      const end = idxOf(NET.dates[k]), win = measure === "corr" ? 63 : 250;
      const cols = NET.names.map(ser), rows = finiteCols(cols, Math.max(0, end - win + 1), end);
      let M = corrOf(covOf(cols, rows));
      if (measure === "pcorr") {
        const P = inv(covOf(cols, rows));
        if (P) M = P.map((r, i) => r.map((v, j) => (i === j ? 1 : -v / Math.sqrt(P[i][i] * P[j][j]))));
      }
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) out.edges.push({ s: i, t: j, w: Math.abs(M[i][j]), sign: Math.sign(M[i][j]), v: M[i][j] });
      for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) if (j !== i) s += Math.abs(M[i][j]); out.size[i] = s / (N - 1); out.node[i] = 0; }
      out.rank = NET.names.map((n, i) => ({ n, strength: out.size[i] }));
      out.M = M;
    } else {
      const g = NET.granger[k];
      const indeg = new Array(N).fill(0), outd = new Array(N).fill(0);
      if (g) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (g[i * N + j] === "1") { out.edges.push({ s: j, t: i, w: 1 }); outd[j]++; indeg[i]++; }
      for (let i = 0; i < N; i++) { out.node[i] = outd[i] - indeg[i]; out.size[i] = outd[i] + indeg[i]; }
      out.rank = NET.names.map((n, i) => ({ n, out: outd[i], in: indeg[i], net: outd[i] - indeg[i] }));
    }
    return out;
  }

  function drawNetwork() {
    const k = netState.k, ms = netState.measure, info = MEASURES[ms];
    const nd = netData(k, ms);
    $("#net-date-lbl").textContent = fmtMonth(NET.dates[k]);
    $("#net-title").textContent = info.title;
    $("#net-sub").textContent = `${fmtMonth(NET.dates[k])}. The ${Math.min(netState.topn, nd.edges.length)} strongest links. Markets run clockwise from the top: equities, rates, credit, FX, commodities, volatility.`;
    $("#net-topn-lbl").textContent = String(netState.topn);
    $("#net-badge").replaceChildren(badgeEl(info.claim));
    $("#net-claimnote").textContent = info.note;
    // tiles
    const di = idxOf(NET.dates[k]);
    const tci = NET.tci_roll[di] / 100, tvp = NET.tci_tvp[di] === null ? NaN : NET.tci_tvp[di] / 100, dyn = NET.tci_dyn[di] / 100;
    const tiles = $("#net-tiles"); tiles.replaceChildren();
    const tile = (lbl, val, note) => { const t = el("div", { class: "tile" }); t.appendChild(el("span", { class: "lbl" }, lbl)); t.appendChild(el("div", { class: "val" }, val)); if (note) t.appendChild(el("div", { class: "note" }, note)); tiles.appendChild(t); };
    tile("Total connectedness", f1(tci) + "%", "rolling 500-day VAR");
    tile("TVP-VAR connectedness", Number.isFinite(tvp) ? f1(tvp) + "%" : "n/a", "time-varying, filtered");
    tile("Lagged share", f1(100 * dyn / tci) + "%", "of total connectedness");
    if (ms === "granger") tile("Granger links", String(nd.edges.length), `of ${NN * (NN - 1)} possible`);
    // graph
    const root = $("#net-graph"); root.replaceChildren();
    const W = Math.max(320, root.clientWidth || 600), H = Math.min(560, Math.max(340, W * 0.82));
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", info.title);
    const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 70;
    const nClasses = new Set(NET.names.map(n => NET.groups[n])).size, gapA = 0.22, totalA = 2 * Math.PI - gapA * nClasses;
    const pos = new Array(NN); let a = -Math.PI / 2, prev = null;
    const per = totalA / NN;
    for (const i of netOrder) {
      const cl = NET.groups[NET.names[i]];
      if (prev !== null && cl !== prev) a += gapA;
      pos[i] = { x: cx + R * Math.cos(a + per / 2), y: cy + R * Math.sin(a + per / 2), a: a + per / 2 };
      a += per; prev = cl;
    }
    const edges = nd.edges.slice().sort((p, q) => q.w - p.w).slice(0, netState.topn);
    const wmax = d3.max(edges, e => e.w) || 1;
    const maxSize = d3.max(nd.size) || 1;
    const rad = i => 7 + 11 * Math.sqrt(Math.max(0, nd.size[i]) / maxSize);
    const defs = svg.append("defs");
    const mk = defs.append("marker").attr("id", "arr").attr("viewBox", "0 -5 10 10").attr("refX", 9).attr("refY", 0)
      .attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 9).attr("markerHeight", 9).attr("orient", "auto");
    mk.append("path").attr("d", "M0,-4L10,0L0,4").attr("fill", C["ink-2"]);
    const eg = svg.append("g");
    for (const e of edges) {
      const s = pos[e.s], t = pos[e.t];
      const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2, qx = mx + (cx - mx) * 0.35, qy = my + (cy - my) * 0.35;
      // shorten the end by the target radius along the curve's end tangent
      const dx = t.x - qx, dy = t.y - qy, L = Math.hypot(dx, dy) || 1, rt = rad(e.t) + 3;
      const ex = t.x - dx / L * rt, ey = t.y - dy / L * rt;
      const color = nd.directed ? C["ink-2"] : (e.sign >= 0 ? C.pos : C.neg);
      const sw = nd.directed && ms === "granger" ? 1.1 : 0.6 + 4.2 * (e.w / wmax);
      const path = eg.append("path").attr("d", `M${s.x},${s.y}Q${qx},${qy} ${nd.directed ? ex : t.x},${nd.directed ? ey : t.y}`)
        .attr("fill", "none").attr("stroke", color).attr("stroke-width", sw).attr("opacity", nd.directed ? 0.55 : 0.75);
      if (nd.directed) path.attr("marker-end", "url(#arr)");
      const tipRows = ms === "granger" ? [{ value: "significant", label: "lags improve the forecast" }]
        : ms === "corr" || ms === "pcorr" ? [{ value: f2(e.v), label: ms === "corr" ? "correlation" : "partial correlation" }]
          : [{ value: f2(100 * e.w) + " pp", label: "net spillover (share of forecast-error variance)" }];
      eg.append("path").attr("d", `M${s.x},${s.y}Q${qx},${qy} ${t.x},${t.y}`).attr("fill", "none").attr("stroke", "transparent").attr("stroke-width", 12)
        .on("pointermove", evt => showTip(evt, nd.directed ? `${NET.names[e.s]} → ${NET.names[e.t]}` : `${NET.names[e.s]} – ${NET.names[e.t]}`, tipRows))
        .on("pointerleave", hideTip);
    }
    const lim = d3.max(nd.node, v => Math.abs(v)) || 1;
    const nodeCol = v => (nd.directed ? d3.interpolateRgb(C.mid, v >= 0 ? C.pos : C.neg)(Math.min(1, Math.abs(v) / lim)) : C.context);
    const ng = svg.append("g");
    for (let i = 0; i < NN; i++) {
      const p = pos[i], r = rad(i);
      ng.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", r + 2).attr("fill", C.surface);
      ng.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", r).attr("fill", nodeCol(nd.node[i])).attr("stroke", C["ink-2"]).attr("stroke-width", 0.8);
      const lx = cx + (R + r + 12) * Math.cos(p.a), ly = cy + (R + r + 12) * Math.sin(p.a);
      const anchor = Math.cos(p.a) > 0.3 ? "start" : Math.cos(p.a) < -0.3 ? "end" : "middle";
      ng.append("text").attr("x", lx).attr("y", ly + 4).attr("text-anchor", anchor).text(NET.names[i]).attr("fill", C.ink).style("font-size", "11.5px").style("font-family", "var(--font-data)");
      const rows = nd.directed ? (ms === "granger" ? [{ value: String(nd.rank[i].out), label: "outgoing links" }, { value: String(nd.rank[i].in), label: "incoming links" }]
        : [{ value: sgn(nd.node[i]) + " pp", label: "NET" }, { value: f1(nd.rank[i].to), label: "TO others" }, { value: f1(nd.rank[i].from), label: "FROM others" }])
        : [{ value: f2(nd.size[i]), label: "mean |correlation| with others" }];
      const cls = CLASS_LABEL[NET.groups[NET.names[i]]] || NET.groups[NET.names[i]];
      ng.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", Math.max(14, r + 4)).attr("fill", "transparent")
        .on("pointermove", evt => showTip(evt, `${NET.names[i]} · ${NET.labels[NET.names[i]]} (${cls})`, rows)).on("pointerleave", hideTip);
    }
    if (nd.directed) legend("#net-legend", [{ color: C.pos, label: ms === "granger" ? "more outgoing than incoming links" : "net transmitter", kind: "dot" },
      { color: C.neg, label: ms === "granger" ? "more incoming links" : "net receiver", kind: "dot" }, { color: C["ink-2"], label: "arrow: direction; width: size" }]);
    else legend("#net-legend", [{ color: C.pos, label: "positive" }, { color: C.neg, label: "negative" }, { color: C.context, label: "node size: average |correlation|", kind: "dot" }]);
    // rank table
    const box = $("#net-rank"); box.replaceChildren();
    const tbl = el("table", { class: "t" }), thead = el("thead"), tr = el("tr");
    let cols, rows = nd.rank.slice();
    if (ms === "dy" || ms === "dyl") { cols = [["Market", r => `${r.n}`], ["NET", r => sgn(r.net)], ["TO", r => f1(r.to)], ["FROM", r => f1(r.from)]]; rows.sort((p, q) => q.net - p.net); $("#net-rank-sub").textContent = "Percentage points of 10-day forecast-error variance; sorted by NET."; }
    else if (ms === "granger") { cols = [["Market", r => r.n], ["Out", r => String(r.out)], ["In", r => String(r.in)], ["Net", r => sgn(r.net, 0)]]; rows.sort((p, q) => q.net - p.net); $("#net-rank-sub").textContent = "Significant directed links in the 500-day window ending this month."; }
    else { cols = [["Market", r => r.n], ["Mean |corr|", r => f2(r.strength)]]; rows.sort((p, q) => q.strength - p.strength); $("#net-rank-sub").textContent = "Average absolute (partial) correlation with the other 14 markets."; }
    cols.forEach(c => tr.appendChild(el("th", {}, c[0]))); thead.appendChild(tr); tbl.appendChild(thead);
    const tb = el("tbody");
    for (const r of rows) { const row = el("tr"); cols.forEach(c => row.appendChild(el("td", {}, c[1](r)))); tb.appendChild(row); }
    tbl.appendChild(tb); box.appendChild(tbl);
    drawTci();
  }

  function drawTci() {
    const roll = dec(NET.tci_roll, 100), tvp = dec(NET.tci_tvp, 100);
    let i0 = 0; while (i0 < ND && !Number.isFinite(roll[i0]) && !Number.isFinite(tvp[i0])) i0++;
    legend("#tci-legend", [{ color: C.s1, label: "rolling 500-day VAR" }, { color: C.s2, label: "TVP-VAR (daily, filtered)" }]);
    timePanels("#net-tci", {
      i0, panels: [{ height: 190, yLabel: "%", series: [{ name: "TVP-VAR", color: C.s2, values: tvp, width: 1, fmt: v => f1(v) + "%" }, { name: "rolling 500d", color: C.s1, values: roll, width: 2, fmt: v => f1(v) + "%" }] }],
      marker: idxOf(NET.dates[netState.k]),
      onClick: i => { const d = DATES[i]; let k = 0; while (k < NET.dates.length - 1 && NET.dates[k + 1] <= d) k++; netState.k = k; $("#net-date").value = k; drawNetwork(); },
    });
  }

  function initNetwork() {
    const s = $("#net-date"); s.max = NET.dates.length - 1; s.value = netState.k;
    s.addEventListener("input", () => { netState.k = +s.value; drawNetwork(); });
    $("#net-measure").addEventListener("change", e => { netState.measure = e.target.value; drawNetwork(); });
    $("#net-topn").addEventListener("input", e => { netState.topn = +e.target.value; drawNetwork(); });
    $("#net-play").addEventListener("click", () => {
      const b = $("#net-play");
      if (netState.timer) { clearInterval(netState.timer); netState.timer = null; b.textContent = "Play"; return; }
      if (netState.k >= NET.dates.length - 1) netState.k = 0;
      b.textContent = "Pause";
      netState.timer = setInterval(() => {
        netState.k++; $("#net-date").value = netState.k; drawNetwork();
        if (netState.k >= NET.dates.length - 1) { clearInterval(netState.timer); netState.timer = null; b.textContent = "Play"; }
      }, 320);
    });
  }

  // =========================================================================================== STRESS
  const ST = D.stress;
  const SN = ST.names, SNI = new Map(SN.map((n, i) => [n, i]));
  const METHOD_LABEL = { full: "Full history (static)", rolling: "Last 252 days", ewma: "EWMA (lambda 0.97)", stressed: "High-VIX days only", regime: "HMM regime mixture (today)", crisis: "HMM crisis state", glasso: "Graphical lasso (sparse)" };
  const ALIAS = { GLD: "GOLD", USO: "OIL" };
  const DRIVERS = ["SPY", "UST2Y", "UST10Y", "BAA10Y", "OIL", "GOLD", "COPPER_F", "DXY", "USDJPY", "VIX", "MOVE", "XLF", "XLK", "XLE", "IYR", "EEM"];
  const stState = { scen: 0, method: "full", asof: ST.latest, port: Object.keys(ST.portfolios)[0], custom: null };
  function covAt(method, asof) { const a = ST.cov[method] && ST.cov[method][asof]; if (!a) return null; const n = SN.length; return Array.from({ length: n }, (_, i) => a.slice(i * n, (i + 1) * n)); }
  function mapWeights(w) { const o = {}; for (const [k, v] of Object.entries(w)) { const kk = ALIAS[k] || k; if (SNI.has(kk)) o[kk] = (o[kk] || 0) + v; } return o; }
  /* conditional Gaussian + network decomposition (mirrors contagion.stress.propagation) */
  function propagate(S, shocks, weights, horizon) {
    const n = SN.length, Dx = [], s = [];
    for (const [k, v] of Object.entries(shocks)) if (SNI.has(k)) { Dx.push(SNI.get(k)); s.push(v); }
    if (!Dx.length) return null;
    const O = []; for (let i = 0; i < n; i++) if (!Dx.includes(i)) O.push(i);
    const SDD = Dx.map(i => Dx.map(j => S[i][j])), SDDi = inv(SDD);
    if (!SDDi) return null;
    const B = O.map(i => { const row = Dx.map(j => S[i][j]); return Dx.map((_, c) => row.reduce((acc, x, r) => acc + x * SDDi[r][c], 0)); });
    const mean = new Array(n).fill(0), sd = new Array(n).fill(0);
    Dx.forEach((i, k) => { mean[i] = s[k]; });
    O.forEach((i, k) => { mean[i] = B[k].reduce((acc, b, c) => acc + b * s[c], 0); });
    const condCov = O.map((i, a) => O.map((j, b) => S[i][j] - B[a].reduce((acc, x, c) => acc + x * S[Dx[c]][j], 0)));
    O.forEach((i, a) => { sd[i] = Math.sqrt(Math.max(0, condCov[a][a]) * horizon); });
    const Th = inv(S); let first = new Array(n).fill(0);
    if (Th) {
      O.forEach(i => { let acc = 0; Dx.forEach((j, c) => { acc += (-Th[i][j] / Th[i][i]) * s[c]; }); first[i] = acc; });
    }
    let direct = 0, f1v = 0, tot = 0;
    const per = [];
    for (const [k, w] of Object.entries(weights || {})) {
      const i = SNI.get(k); if (i === undefined) continue;
      const total = Math.expm1(mean[i]);
      if (Dx.includes(i)) { direct += w * total; per.push({ k, direct: w * total, first: 0, higher: 0, total: w * total }); }
      else { const fo = Math.expm1(first[i]); f1v += w * fo; tot += w * total; per.push({ k, direct: 0, first: w * fo, higher: w * (total - fo), total: w * total }); }
    }
    let pnlVar = 0;
    const wO = O.map(i => { const k = SN[i]; return weights && weights[k] ? weights[k] : 0; });
    for (let a = 0; a < O.length; a++) for (let b = 0; b < O.length; b++) pnlVar += wO[a] * wO[b] * condCov[a][b];
    return { mean, sd, Dx, pnl: { direct, first: f1v, higher: tot - f1v, total: direct + tot, sd: Math.sqrt(Math.max(0, pnlVar) * horizon) }, per };
  }
  function toDisplay(name, v) { return ST.units[name] === "bp" ? v : 100 * Math.expm1(v); }
  function fromDisplay(name, v) { return ST.units[name] === "bp" ? v : Math.log1p(v / 100); }

  function currentScenario() {
    if (stState.custom) return stState.custom;
    return ST.scenarios[stState.scen];
  }
  function drawShockEditor() {
    const sc = currentScenario(), box = $("#st-shocks"); box.replaceChildren();
    const hist = sc.kind === "historical";
    const names = hist ? DRIVERS.filter(n => n in sc.shocks) : Array.from(new Set([...Object.keys(sc.shocks), ...DRIVERS]));
    for (const n of names) {
      if (!SNI.has(n)) continue;
      const lab = el("label", { for: "shock-" + n }); lab.appendChild(document.createTextNode(n + (ST.units[n] === "bp" ? " (bp)" : " (%)")));
      const inp = el("input", { type: "number", id: "shock-" + n, step: ST.units[n] === "bp" ? "5" : "1" });
      if (n in sc.shocks) inp.value = (Math.round(toDisplay(n, sc.shocks[n]) * 10) / 10).toString();
      if (hist) inp.disabled = true;
      inp.addEventListener("change", () => {
        const base = currentScenario();
        const shocks = Object.assign({}, base.kind === "historical" ? {} : base.shocks);
        const v = parseFloat(inp.value);
        if (Number.isFinite(v) && v !== 0) shocks[n] = fromDisplay(n, v); else delete shocks[n];
        stState.custom = { name: "Custom scenario", family: "custom", desc: "Edited shocks.", kind: "hypothetical", horizon: base.horizon || 20, shocks };
        const sel = $("#st-scen"); if (![...sel.options].some(o => o.value === "custom")) sel.appendChild(el("option", { value: "custom" }, "Custom scenario"));
        sel.value = "custom"; drawStress(false);
      });
      lab.appendChild(inp); box.appendChild(lab);
    }
  }
  function drawStress(rebuildEditor = true) {
    const sc = currentScenario();
    if (rebuildEditor) drawShockEditor();
    const weights = mapWeights(ST.portfolios[stState.port]);
    const tiles = $("#st-tiles"); tiles.replaceChildren();
    const tile = (lbl, val, note) => { const t = el("div", { class: "tile" }); t.appendChild(el("span", { class: "lbl" }, lbl)); t.appendChild(el("div", { class: "val" }, val)); if (note) t.appendChild(el("div", { class: "note" }, note)); tiles.appendChild(t); };
    const hist = sc.kind === "historical";
    let res = null, moves;
    if (hist) {
      moves = SN.map(n => (n in sc.shocks ? sc.shocks[n] : NaN));
      let pnl = 0; for (const [k, w] of Object.entries(weights)) { const v = sc.shocks[k]; if (Number.isFinite(v)) pnl += w * Math.expm1(v); }
      tile("Portfolio P&L, realized", pct(100 * pnl), `${fmtDate(sc.window[0])} close to ${fmtDate(sc.window[1])}`);
      tile("Trading days", String(sc.horizon), "in the episode");
      $("#st-sub1").textContent = "Realized cumulative moves (historical replay).";
    } else {
      const S = covAt(stState.method, stState.asof);
      if (!S) { tile("Not available", "n/a", "This covariance view exists only for the latest date."); return; }
      res = propagate(S, sc.shocks, weights, sc.horizon || 20);
      if (!res) { tile("No shock", "n/a", "Enter at least one non-zero shock."); return; }
      moves = res.mean;
      tile("Expected portfolio P&L", pct(100 * res.pnl.total), `${stState.port}`);
      tile("Uncertainty, ±1 sd", "±" + f1(100 * res.pnl.sd) + "%", `over ${sc.horizon || 20} trading days`);
      tile("Direct", pct(100 * res.pnl.direct), "holdings that are shocked");
      tile("First-order network", pct(100 * res.pnl.first), "direct links to the shock");
      tile("Higher-order network", pct(100 * res.pnl.higher), "through intermediaries");
      $("#st-sub1").textContent = `${METHOD_LABEL[stState.method]}, estimated ${fmtDate(stState.asof)}. Orange: shocked drivers.`;
    }
    const itemsPct = [], itemsBp = [], itemsVol = [];
    SN.forEach((n, i) => {
      if (!Number.isFinite(moves[i])) return;
      const shocked = res ? res.Dx.includes(i) : false;
      const v = toDisplay(n, moves[i]);
      const it = { label: n, value: v, color: shocked ? C.s2 : C.s1, tip: [{ value: (ST.units[n] === "bp" ? sgn(v, 1) + " bp" : pct(v)), label: ST.labels[n] }] };
      if (res && !shocked && res.sd[i] > 0) { it.lo = toDisplay(n, moves[i] - res.sd[i]); it.hi = toDisplay(n, moves[i] + res.sd[i]); it.tip.push({ value: "±" + (ST.units[n] === "bp" ? f1(res.sd[i]) + " bp" : f1(100 * res.sd[i]) + "%"), label: "1 conditional sd" }); }
      (ST.units[n] === "bp" ? itemsBp : ST.classes[n] === "volatility" ? itemsVol : itemsPct).push(it);
    });
    itemsPct.sort((p, q) => p.value - q.value);
    barsH("#st-bars-pct", itemsPct, { fmt: v => sgn(v, 1), unit: "%", labelW: 90, barH: 12 });
    barsH("#st-bars-bp", itemsBp, { fmt: v => sgn(v, 0), unit: " bp", labelW: 70 });
    barsH("#st-bars-vol", itemsVol, { fmt: v => sgn(v, 0), unit: "%", labelW: 70 });
    // methods comparison
    if (!hist) {
      const items = [];
      for (const m of Object.keys(METHOD_LABEL)) {
        const S = covAt(m, stState.asof); if (!S) continue;
        const r = propagate(S, sc.shocks, weights, sc.horizon || 20); if (!r) continue;
        items.push({ label: METHOD_LABEL[m], value: 100 * r.pnl.total, color: m === stState.method ? C.s2 : C.s1, tip: [{ value: pct(100 * r.pnl.total), label: "expected P&L" }, { value: "±" + f1(100 * r.pnl.sd) + "%", label: "1 sd" }] });
      }
      barsH("#st-methods", items, { fmt: v => sgn(v, 1), unit: "%", labelW: 170, barH: 12 });
      const keys = [{ key: "direct", label: "direct", color: C.s1 }, { key: "first", label: "first-order network", color: C.s2 }, { key: "higher", label: "higher-order network", color: C.s3 }];
      stackH("#st-decomp", res.per.map(p => ({ label: `${p.k} (${f1(100 * (weights[p.k] || 0))}%)`, parts: { direct: 100 * p.direct, first: 100 * p.first, higher: 100 * p.higher }, total: 100 * p.total })), keys, { fmt: v => pct(v, 2) });
      legend("#st-decomp-legend", keys.map(k => ({ color: k.color, label: k.label, kind: "box" })).concat([{ color: C.ink, label: "total contribution" }]));
    } else {
      $("#st-methods").replaceChildren(el("p", { class: "note" }, "Historical replays use realized moves, so there is no model to compare."));
      const rows = Object.entries(weights).map(([k, w]) => ({ label: `${k} (${f1(100 * w)}%)`, parts: { direct: 100 * w * Math.expm1(sc.shocks[k] ?? NaN) || 0 }, total: 100 * w * Math.expm1(sc.shocks[k] ?? 0) }));
      stackH("#st-decomp", rows, [{ key: "direct", label: "realized", color: C.s1 }], { fmt: v => pct(v, 2) });
      legend("#st-decomp-legend", [{ color: C.s1, label: "realized contribution", kind: "box" }]);
    }
  }
  function fillMethodSelect(sel, asof, cur) {
    sel.replaceChildren();
    for (const [m, l] of Object.entries(METHOD_LABEL)) if (ST.cov[m] && ST.cov[m][asof]) sel.appendChild(el("option", { value: m }, l));
    sel.value = [...sel.options].some(o => o.value === cur) ? cur : "full";
    return sel.value;
  }
  function initStress() {
    const sel = $("#st-scen");
    const og1 = el("optgroup", { label: "Hypothetical" }), og2 = el("optgroup", { label: "Historical replay" });
    ST.scenarios.forEach((s, i) => (s.kind === "historical" ? og2 : og1).appendChild(el("option", { value: String(i) }, s.name)));
    sel.appendChild(og1); sel.appendChild(og2);
    sel.addEventListener("change", () => { if (sel.value === "custom") return; stState.custom = null; stState.scen = +sel.value; drawStress(); });
    const as = $("#st-asof"); ST.asof.slice().reverse().forEach(a => as.appendChild(el("option", { value: a }, a === ST.latest ? `${fmtDate(a)} (latest)` : fmtDate(a))));
    as.value = ST.latest;
    as.addEventListener("change", () => { stState.asof = as.value; stState.method = fillMethodSelect($("#st-method"), stState.asof, stState.method); drawStress(false); });
    stState.method = fillMethodSelect($("#st-method"), stState.asof, "full");
    $("#st-method").addEventListener("change", e => { stState.method = e.target.value; drawStress(false); });
    const ps = $("#st-port"); Object.keys(ST.portfolios).forEach(p => ps.appendChild(el("option", { value: p }, p)));
    ps.addEventListener("change", () => { stState.port = ps.value; drawStress(false); });
  }

  // =========================================================================================== CORRELATIONS
  const SETS = {
    macro: NET.names,
    sectors: ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR", "KRE"],
    havens: ["SPY", "SHY", "IEF", "TLT", "LQD", "HYG", "GLD", "UUP", "FXY", "USO", "DBC", "VIX"],
  };
  const coState = { a: "SPY", b: "IEF", win: 63, freq: 1, range: "all", set: "macro", end: ND - 1 };
  function aggregated(id, freq) {
    const x = ser(id); if (freq === 1) return x;
    const o = new Float64Array(ND).fill(NaN);
    for (let e = ND - 1; e - freq + 1 >= 0; e -= freq) { let s = 0, ok = true; for (let i = e - freq + 1; i <= e; i++) { if (!Number.isFinite(x[i])) { ok = false; break; } s += x[i]; } if (ok) o[e] = s; }
    return o;
  }
  function rollingCorr(x, y, win) {
    const out = new Float64Array(ND).fill(NaN), q = [];
    let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < ND; i++) {
      if (Number.isFinite(x[i]) && Number.isFinite(y[i])) {
        q.push(i); n++; sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; syy += y[i] * y[i]; sxy += x[i] * y[i];
        if (n > win) { const j = q.shift(); n--; sx -= x[j]; sy -= y[j]; sxx -= x[j] * x[j]; syy -= y[j] * y[j]; sxy -= x[j] * y[j]; }
        if (n === win) { const cv = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n; out[i] = cv / Math.sqrt(vx * vy); }
      }
    }
    return out;
  }
  function ewmaCorr(x, y, lam = 0.97, warm = 60) {
    const out = new Float64Array(ND).fill(NaN); let vx = 0, vy = 0, cxy = 0, n = 0;
    for (let i = 0; i < ND; i++) {
      if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) continue;
      if (n < warm) { vx += x[i] * x[i] / warm; vy += y[i] * y[i] / warm; cxy += x[i] * y[i] / warm; n++; continue; }
      vx = lam * vx + (1 - lam) * x[i] * x[i]; vy = lam * vy + (1 - lam) * y[i] * y[i]; cxy = lam * cxy + (1 - lam) * x[i] * y[i];
      out[i] = cxy / Math.sqrt(vx * vy);
    }
    return out;
  }
  function dccCorr(a, b) {
    const Z = D.dcc; if (!(a in Z.z) || !(b in Z.z)) return null;
    const za = dec(Z.z[a], 1000), zb = dec(Z.z[b], 1000);
    const rows = []; for (let i = 0; i < ND; i++) if (Number.isFinite(za[i]) && Number.isFinite(zb[i])) rows.push(i);
    if (rows.length < 100) return null;
    const qbar = pearson(za, zb, rows);
    const out = new Float64Array(ND).fill(NaN);
    let q11 = 1, q22 = 1, q12 = qbar, prev = null;
    for (const i of rows) {
      if (prev !== null) {
        q11 = (1 - Z.a - Z.b) + Z.a * za[prev] * za[prev] + Z.b * q11;
        q22 = (1 - Z.a - Z.b) + Z.a * zb[prev] * zb[prev] + Z.b * q22;
        q12 = (1 - Z.a - Z.b) * qbar + Z.a * za[prev] * zb[prev] + Z.b * q12;
      }
      out[i] = q12 / Math.sqrt(q11 * q22); prev = i;
    }
    return out;
  }
  /* Correlation breakdown (mirrors contagion.portfolio.breakdown.corr_breakdown): short-window correlation vs the
     long window that ends where the short one starts, Forbes-Rigobon adjusted for A's variance ratio, Fisher z. */
  function breakdown(x, y, short, long) {
    const idx = []; for (let i = 0; i < ND; i++) if (Number.isFinite(x[i]) && Number.isFinite(y[i])) idx.push(i);
    const n = idx.length, z = new Float64Array(ND).fill(NaN);
    if (n < short + long + 5) return z;
    const pre = f => { const a = new Float64Array(n + 1); for (let k = 0; k < n; k++) a[k + 1] = a[k] + f(idx[k]); return a; };
    const Sx = pre(i => x[i]), Sy = pre(i => y[i]), Sxx = pre(i => x[i] * x[i]), Syy = pre(i => y[i] * y[i]), Sxy = pre(i => x[i] * y[i]);
    const st = (a, b) => { const m = b - a, sx = Sx[b] - Sx[a], sy = Sy[b] - Sy[a]; const vx = Sxx[b] - Sxx[a] - sx * sx / m, vy = Syy[b] - Syy[a] - sy * sy / m, c = Sxy[b] - Sxy[a] - sx * sy / m; return { r: c / Math.sqrt(vx * vy), vx: vx / (m - 1) }; };
    const at = v => Math.atanh(Math.max(-0.999, Math.min(0.999, v)));
    const se = Math.sqrt(1 / (short - 3) + 1 / (long - 3));
    for (let k = short + long - 1; k < n; k++) {
      const s = st(k - short + 1, k + 1), l = st(k - short - long + 1, k - short + 1);
      const delta = s.vx / l.vx - 1, fr = s.r / Math.sqrt(Math.max(1e-12, 1 + delta * (1 - s.r * s.r)));
      z[idx[k]] = (at(fr) - at(l.r)) / se;
    }
    return z;
  }
  function episodes(z, thr = 3, gap = 21) {
    const out = []; let cur = null;
    for (let i = 0; i < ND; i++) {
      if (!Number.isFinite(z[i]) || Math.abs(z[i]) <= thr) continue;
      if (cur && i - cur.end <= gap) { cur.end = i; if (Math.abs(z[i]) > Math.abs(cur.peak)) cur.peak = z[i]; cur.days++; }
      else { if (cur) out.push(cur); cur = { start: i, end: i, peak: z[i], days: 1 }; }
    }
    if (cur) out.push(cur);
    return out;
  }
  function tailProfile(x, y, i0, i1) {
    const idx = []; for (let i = i0; i <= i1; i++) if (Number.isFinite(x[i]) && Number.isFinite(y[i])) idx.push(i);
    const xs = idx.map(i => x[i]), ys = idx.map(i => y[i]);
    const n = xs.length; if (n < 200) return null;
    const rx = ranks(xs), ry = ranks(ys);
    const zx = rx.map(r => PhiInv(r / (n + 1))), zy = ry.map(r => PhiInv(r / (n + 1)));
    const rho = pearson(zx, zy, zx.map((_, i) => i));
    const sorted = xs.slice().sort((p, q) => p - q);
    const qs = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98];
    return qs.map(q => {
      const thr = quantile(sorted, q), sel = [];
      for (let k = 0; k < n; k++) if (q < 0.5 ? xs[k] <= thr : xs[k] >= thr) sel.push(k);
      const emp = pearson(xs, ys, sel);
      const z = PhiInv(q); let vt;
      if (q < 0.5) { const l = phi(z) / Phi(z); vt = 1 - z * l - l * l; } else { const l = phi(z) / (1 - Phi(z)); vt = 1 + z * l - l * l; }
      const g = rho * Math.sqrt(vt) / Math.sqrt(rho * rho * vt + 1 - rho * rho);
      return { q, emp, g, n: sel.length };
    });
  }
  function drawCorrelations() {
    const { a, b, win, freq } = coState;
    const ua = U.get(a), ub = U.get(b);
    const warn = $("#co-warn"); warn.replaceChildren();
    if (freq === 1 && Math.abs(ua.close - ub.close) > 0.75) {
      const hh = h => { const H = Math.floor(((h % 24) + 24) % 24), M = Math.round((h - Math.floor(h)) * 60); return `${String(H).padStart(2, "0")}:${String(M).padStart(2, "0")}`; };
      warn.appendChild(el("div", { class: "warnline" }, `${a} closes about ${hh(ua.close)} ET and ${b} about ${hh(ub.close)} ET, so same-day returns cover different hours and daily correlation understates co-movement. Switch Returns to Weekly.`));
    }
    const x = aggregated(a, freq), y = aggregated(b, freq);
    const w = freq === 1 ? win : Math.max(8, Math.round(win / freq));
    const rc = rollingCorr(x, y, w), ew = ewmaCorr(x, y, freq === 1 ? 0.97 : 0.86, freq === 1 ? 60 : 12);
    const dc = freq === 1 ? dccCorr(a, b) : null;
    const [i0r, i1r] = rangeIdx(coState.range);
    let i0 = i0r; while (i0 < i1r && !Number.isFinite(rc[i0]) && !Number.isFinite(ew[i0])) i0++;
    $("#co-title").textContent = `${a} vs ${b}: rolling correlation`;
    const series = [{ name: `${w}-${freq === 1 ? "day" : "week"} window`, color: C.s1, values: rc, width: 1.8, fmt: f2 }, { name: "EWMA", color: C.s2, values: ew, width: 1.2, fmt: f2 }];
    if (dc) series.push({ name: "DCC-GARCH", color: C.s3, values: dc, width: 1.2, fmt: f2 });
    legend("#co-legend", series.map(s => ({ color: s.color, label: s.name })));
    // fill weekly gaps for drawing
    if (freq !== 1) for (const s of series) { let last = NaN; const v = s.values; for (let i = 0; i < ND; i++) { if (Number.isFinite(v[i])) last = v[i]; else v[i] = last; } }
    timePanels("#co-roll", { i0, i1: i1r, panels: [{ height: 220, yDomain: [-1, 1], zero: true, series }] });
    // breakdown monitor
    const shortN = freq === 1 ? win : Math.max(8, Math.round(win / freq)), longN = freq === 1 ? 756 : 156;
    const bz = breakdown(x, y, shortN, longN);
    if (freq !== 1) { let last = NaN; for (let i = 0; i < ND; i++) { if (Number.isFinite(bz[i])) last = bz[i]; else if (Number.isFinite(last) && i > 0) bz[i] = last; } }
    let bi0 = i0r; while (bi0 < i1r && !Number.isFinite(bz[bi0])) bi0++;
    if (bi0 < i1r) {
      const zmax = Math.max(4, d3.max(Array.from(bz.slice(bi0, i1r + 1)).filter(Number.isFinite).map(Math.abs)) || 4);
      timePanels("#co-break", { i0: bi0, i1: i1r, panels: [{ height: 130, yDomain: [-zmax, zmax], zero: true, refLines: [-3, 3], yLabel: "z", series: [{ name: "breakdown z (FR-adjusted)", color: C.s4, values: bz, width: 1.3, fmt: f2 }] }] });
      const eps = episodes(bz).filter(e => e.end >= bi0 && e.start <= i1r).sort((p, q) => Math.abs(q.peak) - Math.abs(p.peak)).slice(0, 8);
      const box = $("#co-break-tbl"); box.replaceChildren();
      if (eps.length) {
        const t = el("table", { class: "t" }), hr = el("tr");
        ["Alert episode", "Days beyond ±3", "Peak z", "Direction"].forEach(h => hr.appendChild(el("th", {}, h)));
        const th = el("thead"); th.appendChild(hr); t.appendChild(th); const tb = el("tbody");
        eps.sort((p, q) => q.start - p.start).forEach(e => { const tr = el("tr"); [`${DATES[e.start]} to ${DATES[e.end]}`, String(e.days), f2(e.peak), e.peak > 0 ? "correlation rose" : "correlation fell"].forEach(v => tr.appendChild(el("td", {}, v))); tb.appendChild(tr); });
        t.appendChild(tb); box.appendChild(t);
      } else box.appendChild(el("p", { class: "note" }, "No breakdown beyond ±3 in this period."));
    } else { $("#co-break").replaceChildren(el("p", { class: "note" }, "Not enough history for the three-year comparison window.")); $("#co-break-tbl").replaceChildren(); }
    const tp = tailProfile(x, y, i0r, i1r);
    const root = $("#co-tail"); root.replaceChildren();
    legend("#co-tail-legend", [{ color: C.s1, label: "empirical" }, { color: C.context, label: "Gaussian benchmark" }]);
    if (tp) {
      const W = Math.max(300, root.clientWidth || 500), H = 252, m = { l: 44, r: 16, t: 8, b: 42 };
      const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const xs = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
      const lo = Math.min(0, d3.min(tp, d => Math.min(d.emp, d.g))), hi = Math.max(0, d3.max(tp, d => Math.max(d.emp, d.g)));
      const ys = d3.scaleLinear().domain([lo, hi]).nice().range([H - m.b, m.t]);
      const ay = svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(ys).ticks(5).tickSize(-(W - m.l - m.r))); axisStyle(ay); ay.selectAll("line").attr("stroke", C.grid); ay.select(".domain").remove();
      const ax = svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(xs).tickValues([0.02, 0.2, 0.5, 0.8, 0.98]).tickFormat(d => Math.round(d * 100) + "%")); axisStyle(ax);
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", ys(0)).attr("y2", ys(0)).attr("stroke", C.axis);
      svg.append("line").attr("x1", xs(0.5)).attr("x2", xs(0.5)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.axis);
      const lower = tp.filter(d => d.q < 0.5), upper = tp.filter(d => d.q >= 0.5);
      for (const part of [lower, upper]) {
        svg.append("path").attr("d", d3.line().x(d => xs(d.q)).y(d => ys(d.g))(part)).attr("fill", "none").attr("stroke", C.context).attr("stroke-width", 2);
        svg.append("path").attr("d", d3.line().x(d => xs(d.q)).y(d => ys(d.emp))(part)).attr("fill", "none").attr("stroke", C.s1).attr("stroke-width", 2);
      }
      for (const d of tp) {
        svg.append("circle").attr("cx", xs(d.q)).attr("cy", ys(d.emp)).attr("r", 4.5).attr("fill", C.s1).attr("stroke", C.surface).attr("stroke-width", 2);
        svg.append("circle").attr("cx", xs(d.q)).attr("cy", ys(d.emp)).attr("r", 12).attr("fill", "transparent")
          .on("pointermove", evt => showTip(evt, `${a} ${d.q < 0.5 ? "at or below" : "at or above"} its ${Math.round(d.q * 100)}% quantile`, [{ color: C.s1, value: f2(d.emp), label: "empirical" }, { color: C.context, value: f2(d.g), label: "Gaussian" }, { value: String(d.n), label: "days" }]))
          .on("pointerleave", hideTip);
      }
      svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text(`quantile of ${a} conditioned on`).attr("fill", C.muted).style("font-size", "10.5px").style("font-family", "var(--font-data)");
    } else root.appendChild(el("p", { class: "note" }, "Not enough overlapping observations in this period."));
    drawHeat();
  }
  function drawHeat() {
    const names = SETS[coState.set].filter(n => U.has(n));
    const cols = names.map(n => aggregated(n, coState.freq));
    const s = $("#co-date");
    const end = Math.min(+s.value, ND - 1);
    const w = coState.freq === 1 ? coState.win : coState.win; // window in trading days, weekly uses every 5th day
    const rows = finiteCols(cols, Math.max(0, end - w + 1), end);
    $("#co-date-lbl").textContent = fmtDate(DATES[end]);
    if (rows.length < Math.max(8, (coState.freq === 1 ? coState.win : coState.win / coState.freq) * 0.6)) { $("#co-heat").replaceChildren(el("p", { class: "note" }, "Not enough data for this universe on this date.")); return; }
    const M = corrOf(covOf(cols, rows));
    $("#co-heat-sub").textContent = `${rows.length} ${coState.freq === 1 ? "daily" : "weekly"} observations ending ${fmtDate(DATES[end])}. Blue positive, red negative.`;
    heatmap("#co-heat", names, M, { labels: names.map(n => `${n} (${label(n)})`) });
  }
  function initCorrelations() {
    for (const id of ["#co-a", "#co-b"]) {
      const sel = $(id), groups = {};
      for (const u of D.universe) (groups[u.cls] = groups[u.cls] || []).push(u);
      for (const [g, us] of Object.entries(groups)) { const og = el("optgroup", { label: g }); us.forEach(u => og.appendChild(el("option", { value: u.id }, `${u.id} · ${u.name}`))); sel.appendChild(og); }
    }
    $("#co-a").value = coState.a; $("#co-b").value = coState.b;
    $("#co-a").addEventListener("change", e => { coState.a = e.target.value; drawCorrelations(); });
    $("#co-b").addEventListener("change", e => { coState.b = e.target.value; drawCorrelations(); });
    $("#co-win").addEventListener("change", e => { coState.win = +e.target.value; drawCorrelations(); });
    $("#co-freq").addEventListener("change", e => { coState.freq = +e.target.value; drawCorrelations(); });
    $("#co-range").addEventListener("change", e => { coState.range = e.target.value; drawCorrelations(); });
    $("#co-set").addEventListener("change", e => { coState.set = e.target.value; drawHeat(); });
    const s = $("#co-date"); s.min = 300; s.max = ND - 1; s.value = ND - 1;
    s.addEventListener("input", drawHeat);
  }

  // =========================================================================================== REGIMES
  const RG = D.regimes;
  const STATE_NAMES = ["calm", "normal", "stressed"];
  const stateColor = k => [C.pos, C.context, C.neg][k];
  const rgState = { kind: RG.realtime ? "realtime" : "filtered", range: "all" };
  let spyIndex = null;
  function spyLevel() {
    if (spyIndex) return spyIndex;
    const r = ser("SPY"), o = new Float64Array(ND).fill(NaN); let lv = 100, started = false;
    for (let i = 0; i < ND; i++) { if (Number.isFinite(r[i])) { lv *= Math.exp(r[i]); started = true; } o[i] = started ? lv : NaN; }
    spyIndex = o; return o;
  }
  function drawRegimes() {
    const probs = (RG[rgState.kind] || RG.filtered).map(a => dec(a, 1000));
    const [i0r, i1r] = rangeIdx(rgState.range);
    let i0 = i0r; while (i0 < i1r && !Number.isFinite(probs[0][i0])) i0++;
    const bands = new Array(ND).fill(null);
    for (let i = 0; i < ND; i++) { if (!Number.isFinite(probs[0][i])) continue; let best = 0; for (let k = 1; k < 3; k++) if (probs[k][i] > probs[best][i]) best = k; bands[i] = stateColor(best); }
    const vix = dec(D.vix, 100), sb = dec(RG.sb_corr, 1000), cp = dec(RG.bocpd_spy, 1000), lvl = spyLevel();
    legend("#rg-legend", STATE_NAMES.map((s, k) => ({ color: stateColor(k), label: `${s} regime`, kind: "box" })));
    const common = { i0, i1: i1r, crises: false };
    timePanels("#rg-price", Object.assign({}, common, { panels: [{ height: 200, log: true, yLabel: "S&P 500 total return (log)", bands, series: [{ name: "S&P 500 TR index", color: C.ink, values: lvl, width: 1.4, fmt: v => f1(v) }], tickFmt: "~s" }] }));
    timePanels("#rg-probs", Object.assign({}, common, { panels: [{ height: 110, yDomain: [0, 1], yLabel: "regime probabilities", area: STATE_NAMES.map((s, k) => ({ name: s, color: stateColor(k), values: probs[k], fmt: v => f2(v) })) }] }));
    timePanels("#rg-vix", Object.assign({}, common, { crises: true, panels: [{ height: 110, yLabel: "VIX", refLines: [15, 25, 35], series: [{ name: "VIX", color: C.s1, values: vix, width: 1.2, fmt: f1 }] }] }));
    timePanels("#rg-sb", Object.assign({}, common, { crises: true, panels: [{ height: 110, yDomain: [-1, 1], zero: true, yLabel: "stock-bond correlation (SPY/IEF, 126d)", series: [{ name: "SPY-IEF correlation", color: C.s2, values: sb, width: 1.4, fmt: f2 }] }] }));
    timePanels("#rg-cp", Object.assign({}, common, { crises: true, panels: [{ height: 100, yDomain: [0, 1], yLabel: "BOCPD: P(change in last 10 days), S&P 500", series: [{ name: "P(recent change)", color: C.s3, values: cp, width: 1.1, fmt: f2 }] }] }));
    // tiles
    let last = ND - 1; while (last > 0 && !Number.isFinite(probs[0][last])) last--;
    const tiles = $("#rg-tiles"); tiles.replaceChildren();
    const tile = (lbl, val, note) => { const t = el("div", { class: "tile" }); t.appendChild(el("span", { class: "lbl" }, lbl)); t.appendChild(el("div", { class: "val" }, val)); if (note) t.appendChild(el("div", { class: "note" }, note)); tiles.appendChild(t); };
    STATE_NAMES.forEach((s, k) => tile(`P(${s})`, f1(100 * probs[k][last]) + "%", fmtDate(DATES[last])));
    let lv = ND - 1; while (lv > 0 && !Number.isFinite(vix[lv])) lv--;
    tile("VIX", f1(vix[lv]), fmtDate(DATES[lv]));
    let ls = ND - 1; while (ls > 0 && !Number.isFinite(sb[ls])) ls--;
    tile("Stock-bond correlation", f2(sb[ls]), "SPY vs IEF, 126 days");
    // parameter table
    const info = RG.hmm_info, box = $("#rg-table"); box.replaceChildren();
    const tbl = el("table", { class: "t" }), th = el("tr");
    ["Regime", "S&P 500 vol", "Duration", "Share of days"].forEach(h => th.appendChild(el("th", {}, h)));
    const thead = el("thead"); thead.appendChild(th); tbl.appendChild(thead);
    const sm = RG.smoothed.map(a => dec(a, 1000)), tb = el("tbody");
    STATE_NAMES.forEach((s, k) => {
      let c = 0, n = 0; for (let i = 0; i < ND; i++) if (Number.isFinite(sm[0][i])) { n++; let b = 0; for (let j = 1; j < 3; j++) if (sm[j][i] > sm[b][i]) b = j; if (b === k) c++; }
      const tr = el("tr"); [s, f1(100 * info.state_spy_ann_vol[k]) + "%", f1(info.expected_duration_days[k]) + " days", f1(100 * c / n) + "%"].forEach(v => tr.appendChild(el("td", {}, v))); tb.appendChild(tr);
    });
    tbl.appendChild(tb); box.appendChild(tbl);
    if (RG.bayes) {
      const t2 = el("table", { class: "t" }), h2 = el("tr");
      ["Bayesian HMM (weekly)", "Vol, posterior mean", "90% interval", "Duration (weeks)", "90% interval"].forEach(h => h2.appendChild(el("th", {}, h)));
      const th2 = el("thead"); th2.appendChild(h2); t2.appendChild(th2); const b2 = el("tbody");
      RG.bayes.summary.forEach(r => { const tr = el("tr"); [r.state === 0 ? "calm" : "turbulent", f1(100 * r.vol_mean) + "%", `${f1(100 * r.vol_5)}-${f1(100 * r.vol_95)}%`, f1(r.duration_mean), `${f1(r.duration_5)}-${f1(r.duration_95)}`].forEach(v => tr.appendChild(el("td", {}, v))); b2.appendChild(tr); });
      t2.appendChild(b2); box.appendChild(el("div", { style: "height:10px" })); box.appendChild(t2);
    }
    const bic = info.bic; $("#rg-bic").textContent = `BIC by number of states: ${Object.entries(bic).map(([k, v]) => `${k} states ${Math.round(v).toLocaleString("en-US")}`).join(", ")} (lower fits better). BIC keeps falling with more states on fat-tailed returns; three states are kept for interpretability.`;
    const pb = $("#rg-pelt"); pb.replaceChildren();
    const t3 = el("table", { class: "t" }), h3 = el("tr");
    ["Segment", "Days", "S&P 500 vol", "Corr(SPY, 10Y)"].forEach(h => h3.appendChild(el("th", {}, h)));
    const th3 = el("thead"); th3.appendChild(h3); t3.appendChild(th3); const b3 = el("tbody");
    RG.pelt.slice().reverse().forEach(sg => { const tr = el("tr"); const days = idxOf(sg.end) - idxOf(sg.start) + 1; [`${sg.start} to ${sg.end}`, String(days), f1(100 * sg.vol) + "%", f2(sg.corr)].forEach(v => tr.appendChild(el("td", {}, v))); b3.appendChild(tr); });
    t3.appendChild(b3); pb.appendChild(t3);
  }
  function initRegimes() {
    if (!RG.realtime) $("#rg-kind").querySelector('option[value="realtime"]').disabled = true;
    $("#rg-kind").value = rgState.kind;
    $("#rg-kind").addEventListener("change", e => { rgState.kind = e.target.value; drawRegimes(); });
    $("#rg-range").addEventListener("change", e => { rgState.range = e.target.value; drawRegimes(); });
  }

  // =========================================================================================== EXPOSURE
  const HOLDABLE = ["SPY", "IWM", "QQQ", "EFA", "EEM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR", "KRE", "IWF", "IWD", "SHY", "IEF", "TLT", "LQD", "HYG", "GOLD", "OIL", "DBC", "UUP", "FXY"];
  const FACTORS = [["SPY", "S&P 500 (per 1%)", 1], ["UST10Y", "10Y yield (per +100bp)", 0.01], ["BAA10Y", "Baa spread (per +100bp)", 0.01], ["OIL", "Oil (per 1%)", 1], ["DXY", "Dollar index (per 1%)", 1], ["VIX", "VIX (per 1% change)", 1]];
  const exState = { port: Object.keys(ST.portfolios)[0], win: "756", method: "full", w: {} };
  function setWeights(p) { exState.w = {}; for (const [k, v] of Object.entries(ST.portfolios[p])) exState.w[k] = 100 * v; }
  function drawWeights() {
    const box = $("#ex-weights"); box.replaceChildren();
    for (const n of HOLDABLE) {
      if (!U.has(n)) continue;
      const lab = el("label", { for: "w-" + n }); lab.appendChild(document.createTextNode(n));
      const inp = el("input", { type: "number", id: "w-" + n, min: "0", max: "100", step: "1" });
      inp.value = exState.w[n] ? String(Math.round(exState.w[n] * 10) / 10) : "";
      inp.addEventListener("change", () => { const v = parseFloat(inp.value); if (Number.isFinite(v) && v > 0) exState.w[n] = v; else delete exState.w[n]; drawExposure(false); });
      lab.appendChild(inp); box.appendChild(lab);
    }
  }
  function drawExposure(rebuild = true) {
    if (rebuild) drawWeights();
    const tot = Object.values(exState.w).reduce((s, v) => s + v, 0);
    const tiles = $("#ex-tiles"); tiles.replaceChildren();
    const tile = (lbl, val, note) => { const t = el("div", { class: "tile" }); t.appendChild(el("span", { class: "lbl" }, lbl)); t.appendChild(el("div", { class: "val" }, val)); if (note) t.appendChild(el("div", { class: "note" }, note)); tiles.appendChild(t); };
    if (tot <= 0) { tile("No holdings", "n/a", "Enter at least one weight."); return; }
    const names = Object.keys(exState.w).filter(n => U.has(n)), w = names.map(n => exState.w[n] / tot);
    const cols = names.map(ser);
    const winN = exState.win === "all" ? ND : +exState.win;
    const rows = finiteCols(cols, Math.max(idxOf("2007-04-12"), ND - winN), ND - 1);
    if (rows.length < 60) { tile("Not enough data", "n/a", "Too few days where every holding has a price."); return; }
    const S = covOf(cols, rows);
    const port = rows.map(i => names.reduce((s, n, k) => s + w[k] * Math.expm1(cols[k][i]), 0));
    const mu = port.reduce((s, v) => s + v, 0) / port.length;
    const sdp = Math.sqrt(port.reduce((s, v) => s + (v - mu) * (v - mu), 0) / (port.length - 1));
    const srt = port.slice().sort((p, q) => p - q);
    const var99 = -quantile(srt, 0.01), q975 = quantile(srt, 0.025), es975 = -srt.filter(v => v <= q975).reduce((s, v, _, a) => s + v / a.length, 0);
    const Sw = S.map(r => r.reduce((s, x, j) => s + x * w[j], 0)), vp = w.reduce((s, x, i) => s + x * Sw[i], 0);
    const divr = w.reduce((s, x, i) => s + x * Math.sqrt(S[i][i]), 0) / Math.sqrt(vp);
    tile("Volatility (annualized)", f1(100 * sdp * Math.sqrt(252)) + "%", `${rows.length} days to ${fmtDate(DATES[rows[rows.length - 1]])}`);
    tile("1-day VaR 99%", f2(100 * var99) + "%", "historical");
    tile("1-day ES 97.5%", f2(100 * es975) + "%", "historical");
    tile("Diversification ratio", f2(divr), "weighted vol / portfolio vol");
    // risk contributions
    const rc = names.map((n, i) => ({ n, weight: 100 * w[i], rc: 100 * w[i] * Sw[i] / vp }));
    rc.sort((p, q) => q.rc - p.rc);
    legend("#ex-rc-legend", [{ color: C.s1, label: "share of risk", kind: "box" }, { color: C.s2, label: "weight", kind: "box" }]);
    const root = $("#ex-rc"); root.replaceChildren();
    const W = Math.max(300, root.clientWidth || 500), step = 26, m = { l: 60, r: 50, t: 4, b: 20 }, H = m.t + m.b + rc.length * step;
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([Math.min(0, d3.min(rc, d => d.rc)), Math.max(d3.max(rc, d => Math.max(d.rc, d.weight)), 1)]).nice().range([m.l, W - m.r]);
    const ax = svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(d => d + "%").tickSize(-(H - m.t - m.b))); axisStyle(ax); ax.selectAll("line").attr("stroke", C.grid); ax.select(".domain").remove();
    rc.forEach((d, k) => {
      const y = m.t + k * step;
      svg.append("rect").attr("x", Math.min(x(0), x(d.rc))).attr("y", y + 3).attr("width", Math.abs(x(d.rc) - x(0))).attr("height", 9).attr("rx", 2).attr("fill", C.s1);
      svg.append("rect").attr("x", x(0)).attr("y", y + 14).attr("width", Math.abs(x(d.weight) - x(0))).attr("height", 7).attr("rx", 2).attr("fill", C.s2);
      svg.append("text").attr("x", m.l - 8).attr("y", y + 15).attr("text-anchor", "end").text(d.n).attr("fill", C.ink).style("font-size", "11.5px").style("font-family", "var(--font-data)");
      svg.append("text").attr("x", Math.max(x(d.rc), x(d.weight)) + 5).attr("y", y + 12).text(f1(d.rc) + "%").attr("fill", C["ink-2"]).style("font-size", "10.5px").style("font-family", "var(--font-data)");
      svg.append("rect").attr("x", 0).attr("y", y).attr("width", W).attr("height", step).attr("fill", "transparent")
        .on("pointermove", evt => showTip(evt, `${d.n} · ${label(d.n)}`, [{ color: C.s1, value: f1(d.rc) + "%", label: "share of portfolio risk" }, { color: C.s2, value: f1(d.weight) + "%", label: "weight" }])).on("pointerleave", hideTip);
    });
    // factor regression
    const fcols = FACTORS.map(f => ser(f[0]));
    const frows = rows.filter(i => fcols.every(c => Number.isFinite(c[i])));
    const fbox = $("#ex-factors"); fbox.replaceChildren();
    if (frows.length > 60) {
      const k = FACTORS.length + 1, XtX = Array.from({ length: k }, () => new Array(k).fill(0)), Xty = new Array(k).fill(0);
      const yv = frows.map(i => names.reduce((s, n, j) => s + w[j] * Math.expm1(cols[j][i]), 0));
      const Xr = frows.map(i => [1, ...FACTORS.map((f, j) => fcols[j][i] * f[2])]);
      Xr.forEach((r, t) => { for (let a = 0; a < k; a++) { Xty[a] += r[a] * yv[t]; for (let b = 0; b < k; b++) XtX[a][b] += r[a] * r[b]; } });
      const XtXi = inv(XtX);
      if (XtXi) {
        const beta = mv(XtXi, Xty), res = yv.map((v, t) => v - Xr[t].reduce((s, x, j) => s + x * beta[j], 0));
        const s2 = res.reduce((s, v) => s + v * v, 0) / (yv.length - k);
        const ym = yv.reduce((s, v) => s + v, 0) / yv.length, vy = yv.reduce((s, v) => s + (v - ym) * (v - ym), 0) / (yv.length - 1);
        const Fm = FACTORS.map((_, j) => Xr.reduce((s, r) => s + r[j + 1], 0) / Xr.length);
        const SF = FACTORS.map((_, a) => FACTORS.map((_, b) => Xr.reduce((s, r) => s + (r[a + 1] - Fm[a]) * (r[b + 1] - Fm[b]), 0) / (Xr.length - 1)));
        const bf = beta.slice(1), SFb = mv(SF, bf);
        const tbl = el("table", { class: "t" }), hr = el("tr");
        ["Factor", "Portfolio return", "t-stat", "Share of variance"].forEach(h => hr.appendChild(el("th", {}, h)));
        const thd = el("thead"); thd.appendChild(hr); tbl.appendChild(thd); const tb = el("tbody");
        // displayed as % portfolio return per stated factor move: return factors are decimal per decimal (x1);
        // yield and spread factors are decimal per percentage point, i.e. per +100bp, so x100 gives %
        FACTORS.forEach((f, j) => { const se = Math.sqrt(s2 * XtXi[j + 1][j + 1]); const shown = bf[j] * (f[2] === 0.01 ? 100 : 1); const tr = el("tr"); [f[1], pct(shown, 2), f1(bf[j] / se), f1(100 * bf[j] * SFb[j] / vy) + "%"].forEach(v => tr.appendChild(el("td", {}, v))); tb.appendChild(tr); });
        const r2 = bf.reduce((s, b, j) => s + b * SFb[j], 0) / vy;
        const tr = el("tr"); ["Explained (R²)", "", "", f1(100 * r2) + "%"].forEach(v => tr.appendChild(el("td", {}, v))); tb.appendChild(tr);
        tbl.appendChild(tb); fbox.appendChild(tbl);
        fbox.appendChild(el("p", { class: "note" }, "Portfolio return: % move of the portfolio for the factor move in each row, holding the other factors fixed. Shares use the Euler split of explained variance and can be negative for offsetting factors."));
      }
    }
    // scenarios
    const wmap = mapWeights(Object.fromEntries(names.map((n, i) => [n, w[i]])));
    const missing = names.filter(n => !SNI.has(ALIAS[n] || n));
    const S2 = covAt(exState.method, ST.latest);
    const keys = [{ key: "direct", label: "direct", color: C.s1 }, { key: "first", label: "first-order network", color: C.s2 }, { key: "higher", label: "higher-order network", color: C.s3 }];
    const rowsS = [];
    for (const sc of ST.scenarios.filter(s => s.kind === "hypothetical")) { const r = propagate(S2, sc.shocks, wmap, sc.horizon || 20); if (r) rowsS.push({ label: sc.name, parts: { direct: 100 * r.pnl.direct, first: 100 * r.pnl.first, higher: 100 * r.pnl.higher }, total: 100 * r.pnl.total }); }
    stackH("#ex-sc", rowsS, keys, { fmt: v => pct(v, 1), labelW: 200 });
    legend("#ex-sc-legend", keys.map(k => ({ color: k.color, label: k.label, kind: "box" })).concat([{ color: C.ink, label: "total" }]).concat(missing.length ? [{ color: C.muted, label: `not in the stress universe: ${missing.join(", ")}` }] : []));
  }
  function initExposure() {
    const ps = $("#ex-port"); Object.keys(ST.portfolios).forEach(p => ps.appendChild(el("option", { value: p }, p)));
    setWeights(exState.port);
    ps.addEventListener("change", () => { exState.port = ps.value; setWeights(ps.value); drawExposure(); });
    $("#ex-win").addEventListener("change", e => { exState.win = e.target.value; drawExposure(false); });
    exState.method = fillMethodSelect($("#ex-method"), ST.latest, "full");
    $("#ex-method").addEventListener("change", e => { exState.method = e.target.value; drawExposure(false); });
  }

  // =========================================================================================== FINDINGS
  function drawFindings() {
    const box = $("#fd-cards"); box.replaceChildren();
    for (const [k, a] of Object.entries(D.answers || {})) {
      const card = el("article", { class: "card" });
      const head = el("div", { class: "view-head" }); head.appendChild(el("h2", {}, a.title));
      const bd = el("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }); a.claims.forEach(c => bd.appendChild(badgeEl(c)));
      card.appendChild(head); card.appendChild(bd); card.appendChild(el("p", {}, a.answer)); box.appendChild(card);
    }
    const cl = $("#fd-claims"); cl.replaceChildren();
    for (const [k, t] of Object.entries(D.meta.claims)) { const li = el("li"); const s1 = el("span"); s1.appendChild(badgeEl(k)); li.appendChild(s1); li.appendChild(el("span", { style: "font-family:var(--font-body);text-align:right" }, t)); cl.appendChild(li); }
  }

  // =========================================================================================== router
  const VIEWS = { network: drawNetwork, stress: () => drawStress(true), correlations: drawCorrelations, regimes: drawRegimes, exposure: () => drawExposure(true), findings: drawFindings };
  let current = null;
  function route() {
    const h = (location.hash || "#network").slice(1);
    const v = VIEWS[h] ? h : "network";
    document.querySelectorAll("section[data-view]").forEach(s => { s.hidden = s.dataset.view !== v; });
    document.querySelectorAll("nav.tabs a").forEach(a => { if (a.dataset.v === v) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    current = v; hideTip(); VIEWS[v]();
  }
  function rerender() { readColors(); if (current) VIEWS[current](); }
  $("#stamp").textContent = `data to ${D.meta.last}`;
  initNetwork(); initStress(); initCorrelations(); initRegimes(); initExposure();
  window.addEventListener("hashchange", route);
  let rt = null; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(rerender, 200); });
  try { window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender); } catch (e) { /* older browsers */ }
  new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  route();
})();
