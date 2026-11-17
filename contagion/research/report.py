"""Build the reproducible research report (HTML + Markdown) from the cached research sections.

    outputs/report/report.html           full document, open locally
    outputs/report/report_fragment.html  same page without the document skeleton (for publishing)
    outputs/report/report.md             Markdown twin (figures linked from outputs/figures)
"""
from __future__ import annotations

import html
import json
import pickle
import platform
import re
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd

from ..claims import CLAIM_TEXT, Claim
from ..paths import OUTPUTS, REPORT, TABLES
from .common import Section, img_b64

ORDER = ["q1", "q2", "q3", "q4", "q5", "q6"]
CLAIM_LABEL = {Claim.DESCRIPTIVE: "Descriptive", Claim.PREDICTIVE: "Predictive", Claim.IDENTIFIED: "Identified"}

CSS = """
/* Layout: one reading column (~72ch) inside a wider frame; figures and tables may use the full frame width. */
:root {
  --bg: #f3f4f1; --surface: #fcfcfb; --ink: #121417; --ink-2: #4b4f55; --muted: #7c8087; --rule: #dadcd7;
  --accent: #1f5fae; --desc: #2a78d6; --pred: #eb6834; --ident: #1baf7a; --code-bg: #eceee9;
  --fig-light: block; --fig-dark: none;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", "Segoe UI", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  --measure: 72ch; --frame: 1120px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d0d0d; --surface: #1a1a19; --ink: #efeeea; --ink-2: #c3c2b7; --muted: #8f8d86; --rule: #2e2e2b;
    --accent: #6da7ec; --desc: #3987e5; --pred: #d95926; --ident: #199e70; --code-bg: #232321;
    --fig-light: none; --fig-dark: block; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #efeeea; --ink-2: #c3c2b7; --muted: #8f8d86; --rule: #2e2e2b;
  --accent: #6da7ec; --desc: #3987e5; --pred: #d95926; --ident: #199e70; --code-bg: #232321;
  --fig-light: none; --fig-dark: block; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15.5px; line-height: 1.6;
  margin: 0; }
.frame { max-width: var(--frame); margin: 0 auto; padding-block: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.masthead { border-bottom: 1px solid var(--rule); padding-block: 40px 28px; }
.eyebrow { font-family: var(--font-data); font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 12px; }
h1, h2, h3 { font-family: var(--font-display); font-weight: 600; text-wrap: balance; line-height: 1.2; }
h1 { font-size: clamp(28px, 4.2vw, 42px); margin: 0 0 14px; max-width: 26ch; }
h2 { font-size: 26px; margin: 0 0 6px; }
h3 { font-size: 17px; margin: 22px 0 6px; }
p, li { max-width: var(--measure); }
.lede { font-size: 17px; color: var(--ink-2); margin: 0 0 20px; }
.claims { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 8px; }
.claim-def { font-size: 13.5px; color: var(--ink-2); display: flex; gap: 10px; align-items: baseline; min-width: 0; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-data); font-size: 11.5px;
  letter-spacing: 0.03em; text-transform: uppercase; color: var(--ink); border: 1px solid var(--rule);
  border-radius: 999px; padding: 2px 9px 2px 7px; white-space: nowrap; background: var(--surface); }
.badge::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--sw); }
.badge.descriptive { --sw: var(--desc); } .badge.predictive { --sw: var(--pred); } .badge.identified { --sw: var(--ident); }
nav.toc { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  border-bottom: 1px solid var(--rule); }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10;
  background: var(--bg); }
nav.toc .frame { display: flex; gap: 4px 18px; flex-wrap: wrap; padding-block: 10px; font-size: 13.5px; }
nav.toc a { color: var(--ink-2); text-decoration: none; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--accent); text-decoration: underline; }
main section { padding-block: 36px 8px; border-bottom: 1px solid var(--rule); }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin-top: 16px; }
.answer { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 18px;
  display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.answer h3 { margin: 0; font-size: 16px; }
.answer h3 a { color: var(--ink); text-decoration: none; }
.answer h3 a:hover { text-decoration: underline; }
.verdict { font-family: var(--font-display); font-size: 19px; font-weight: 600; margin: 0; }
.answer p { margin: 0; font-size: 14px; color: var(--ink-2); }
.badges { display: flex; gap: 6px; flex-wrap: wrap; }
.qhead { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: baseline; }
.callout { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 18px;
  margin: 14px 0; max-width: calc(var(--measure) + 40px); }
.callout p { margin: 0; }
.design { color: var(--ink-2); font-size: 14px; }
figure.fig { margin: 22px 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 12px; max-width: 100%; }
figure.fig img { width: 100%; height: auto; }
figure.fig img.fl { display: var(--fig-light); } figure.fig img.fd { display: var(--fig-dark); }
figcaption { font-size: 13.5px; color: var(--ink-2); margin-top: 8px; max-width: var(--measure); }
figcaption strong { color: var(--ink); }
.table-wrap { overflow-x: auto; margin: 14px 0 6px; border: 1px solid var(--rule); border-radius: 6px;
  background: var(--surface); }
table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
th { font-weight: 600; color: var(--ink-2); vertical-align: bottom; white-space: normal; min-width: 70px; }
td { font-family: var(--font-data); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-family: var(--font-body); }
tr:last-child td { border-bottom: none; }
.tcap { font-size: 13px; color: var(--muted); max-width: var(--measure); margin: 0 0 18px; }
.tcap strong { color: var(--ink-2); }
.findings { display: grid; gap: 10px; margin: 16px 0; }
.finding { border-left: 3px solid var(--sw); padding: 2px 0 2px 14px; }
.finding.descriptive { --sw: var(--desc); } .finding.predictive { --sw: var(--pred); } .finding.identified { --sw: var(--ident); }
.finding p { margin: 4px 0; font-size: 14px; }
.finding ul { margin: 4px 0 0; padding-left: 18px; font-size: 13px; color: var(--ink-2); }
code, pre { font-family: var(--font-data); font-size: 13px; background: var(--code-bg); border-radius: 4px; }
code { padding: 1px 5px; }
pre { padding: 12px 14px; overflow-x: auto; max-width: 100%; }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
footer { padding-block: 30px 50px; color: var(--muted); font-size: 13px; }
@media (max-width: 560px) { body { font-size: 15px; } h2 { font-size: 22px; } }
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(v, floatfmt="{:.2f}") -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return floatfmt.format(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    if v is None:
        return ""
    return str(v)


def table_html(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    idx_names = [n for n in (df.index.names or []) if n]
    head = "".join(f"<th>{_esc(n)}</th>" for n in (idx_names or [""])) + "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for ix, r in df.iterrows():
        ixs = ix if isinstance(ix, tuple) else (ix,)
        cells = "".join(f"<td>{_esc(_fmt(v))}</td>" for v in ixs)
        cells += "".join(f"<td>{_esc(_fmt(v, floatfmt))}</td>" for v in r.values)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def table_md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for ix, r in df.iterrows():
        ixs = " / ".join(map(str, ix)) if isinstance(ix, tuple) else str(ix)
        out.append("| " + " | ".join([ixs] + [_fmt(v, floatfmt) for v in r.values]) + " |")
    return "\n".join(out)


def badge(c: Claim) -> str:
    return f'<span class="badge {c.value}">{CLAIM_LABEL[c]}</span>'


def first_sentence(s: str) -> tuple[str, str]:
    m = re.match(r"^(.*?[.!?])\s+(.*)$", s, flags=re.S)
    return (m.group(1), m.group(2)) if m else (s, "")


def method_note_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..network.connectedness import connectedness

    rng = np.random.default_rng(0)
    S = np.array([[1.0, 0.6, 0.6], [0.6, 1.0, 0.0], [0.6, 0.0, 1.0]])
    Y = pd.DataFrame(rng.multivariate_normal(np.zeros(3), S, size=20000), columns=["hub", "a", "b"])
    c = connectedness(Y)
    hub = pd.DataFrame({"NET": c.net, "same-day part": c.net_contemporaneous(), "lagged part": c.net_dynamic()})
    Z = np.zeros((20000, 2))
    e = rng.standard_normal((20000, 2))
    for t in range(1, 20000):
        Z[t] = e[t] + np.array([0.0, 0.5 * Z[t - 1, 0]])
    c2 = connectedness(pd.DataFrame(Z, columns=["leader", "follower"]))
    ll = pd.DataFrame({"NET": c2.net, "same-day part": c2.net_contemporaneous(), "lagged part": c2.net_dynamic()})
    return hub, ll


def dfm_section(panel) -> tuple[str, str]:
    """Common-factor structure from the weekly dynamic factor model (descriptive)."""
    from ..paths import MODELS
    from .common import crisis_episodes, save_fig, shade_episodes

    if not (MODELS / "dfm_r2.csv").exists():
        return "", ""
    r2 = pd.read_csv(MODELS / "dfm_r2.csv", index_col=0)
    roll = pd.read_parquet(MODELS / "dfm_rolling_r2.parquet")
    groups = {"Equities": ["SPY", "IWM", "EFA", "EEM", "XLF"], "Rates and credit": ["UST2Y", "UST10Y", "BAA10Y"],
              "Commodities and FX": ["COPPER_F", "OIL", "GOLD", "DXY", "USDJPY"]}
    G = pd.DataFrame({g: roll[[c for c in cs if c in roll]].mean(axis=1) for g, cs in groups.items()})
    ep = crisis_episodes(panel, 0.15, 0.15, "2003-04-15")

    def draw(fig, P, mode):
        ax = fig.add_subplot(111)
        shade_episodes(ax, ep, P)
        for k, g in enumerate(G.columns):
            ax.plot(G.index, G[g], color=P["series"][k], lw=1.7, label=g)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of weekly variance")
        ax.legend(loc="upper left", ncols=3)
        ax.set_title("Variance explained by the global factor, rolling two years (group averages)")

    fg = save_fig("dfm_global_share", draw, (8.6, 3.2), "Global-factor share over time",
                  "Weekly dynamic factor model on the 15-asset macro network (one global factor plus asset-class "
                  "blocks, AR(1) factors, EM estimation). The factor is Kalman-smoothed over the full sample, so the "
                  "rolling shares describe history and are not a real-time signal. Shading: bear markets.")
    T = r2[["global_r2", "global_plus_block_r2", "block"]].copy()
    T.columns = ["global factor R2", "global + block R2", "block"]
    T["block"] = T["block"].replace({"Global": "none (only series in its class)"})
    T = T.sort_values("global factor R2", ascending=False)
    html_ = ("<h3>Common-factor structure (dynamic factor model)</h3>"
             "<p>A weekly dynamic factor model with one global factor and asset-class block factors summarises how "
             "integrated the network is. Over the full sample the global factor is an equity risk factor: it explains "
             f"{100 * T.loc['EFA', 'global factor R2']:.0f}% of EFA's and {100 * T.loc['SPY', 'global factor R2']:.0f}% "
             f"of SPY's weekly variance but {100 * T.loc['UST10Y', 'global factor R2']:.0f}% of the 10-year yield's, "
             "because the stock-bond correlation changes sign over the sample and averages out. Block factors with two "
             "or three members can reproduce one member almost exactly, so the block R2 overstates real structure.</p>"
             + table_html(T, "{:.2f}")
             + f'<figure class="fig"><img class="fl" alt="{_esc(fg.title)}" src="data:image/png;base64,{img_b64(fg.light)}">'
             f'<img class="fd" alt="{_esc(fg.title)}" src="data:image/png;base64,{img_b64(fg.dark)}">'
             f"<figcaption><strong>{_esc(fg.title)}.</strong> {_esc(fg.caption)}</figcaption></figure>")
    md = "\n".join(["### Common-factor structure (dynamic factor model)", "", table_md(T, "{:.2f}"), "",
                    f"![{fg.title}](../figures/{fg.light.name})", ""])
    return html_, md


def data_section(panel) -> tuple[str, str]:
    m = panel.meta
    U = panel.universe
    sets = [("macro", "Cross-asset network: US/intl equities, financials, Treasury 2Y/10Y, Baa spread, oil, gold, copper, dollar, yen, VIX, MOVE"),
            ("sync", "Robustness network of instruments that all price at the NYSE close (ETFs + VIX)"),
            ("sectors", "Sector network: 10 SPDR/iShares sectors with rates, credit, oil and VIX"),
            ("fin", "Financial institutions: 15 large banks, brokers and insurers with XLF, KRE and SPY")]
    rows = []
    for s, desc in sets:
        R = panel.returns_for(s)
        rows.append({"universe": s, "series": R.shape[1], "from": R.index[0].date(), "to": R.index[-1].date(),
                     "days": len(R), "purpose": desc})
    T = pd.DataFrame(rows).set_index("universe")
    asy = pd.read_csv(TABLES / "asynchrony_vs_spy.csv", index_col=0) if (TABLES / "asynchrony_vs_spy.csv").exists() else None
    pick = ["N225", "HSI", "FTSE", "AUDUSD", "USDJPY", "COPPER_F", "GOLD_F", "WTI_F", "BAA10Y", "MOVE", "DXY", "VIX"]
    A = None
    if asy is not None:
        A = asy.reindex([p for p in pick if p in asy.index])[["close_et", "lag-1", "lag+0", "lag+1", "timing_signal"]]
        A.columns = ["close (ET hour)", "corr(SPY t, x t-1)", "corr(SPY t, x t)", "corr(SPY t, x t+1)", "late reaction"]
    n_series = panel.returns.shape[1]
    html_parts = [
        '<section id="data"><h2>Data and methods</h2>',
        f"<p>{n_series} daily series from three public sources, aligned to the NYSE calendar "
        f"({_esc(m['master_calendar']['first'])} to {_esc(m['master_calendar']['last'])}, "
        f"{m['master_calendar']['n_days']:,} trading days): Yahoo Finance (ETFs, stocks, equity indices, futures), FRED "
        "(Treasury constant-maturity yields, Moody's spreads, Fed H.10 noon FX rates, Cboe volatility indices, financial "
        "stress indices) and Cboe (VIX family). FOMC announcement dates come from federalreserve.gov. Prices are "
        "dividend-adjusted; returns are log returns, yields and spreads are changes in basis points, volatility "
        "indices are log changes.</p>",
        table_html(T, "{:.0f}"),
        "<h3>Calendars and timestamps</h3>",
        "<p>Each series keeps its own trading dates. On each NYSE day the last available observation is used; a "
        "market holiday shows a zero return and the whole move lands on the next trading day, so cumulative returns "
        "are conserved. Gaps longer than three days are left missing. Every series carries its closing time, because "
        "same-date returns from markets that close hours apart cover different news windows. The table measures it: "
        "a series that absorbs part of each day's S&P 500 news on the next day shows a large "
        "<code>corr(SPY t, x t+1)</code>.</p>",
        table_html(A, "{:.2f}") if A is not None else "",
        '<p class="tcap">2010-2026. Late reaction = |corr at t+1| - |corr at t-1|. Tokyo and Hong Kong close before '
        "New York opens; H.10 FX rates are noon fixings; metals settle in the early afternoon; Moody's Baa yields "
        "update a day late. Every lead-lag statistic in this report is read against this table.</p>",
        "<h3>Data problems found and handled</h3><ul>"
        "<li>Yahoo's daily FX bars changed convention around 2011: since then the 'close' is a snapshot at 00:00 London "
        "(close equals open; on Brexit night GBP/USD shows a 1.32 low but a 1.456 close). No single date shift fixes "
        "both periods, so FX uses the Fed's H.10 noon rates instead.</li>"
        "<li>FRED now serves only the last three years of ICE BofA credit spreads; long-history credit uses Moody's "
        "Baa minus 10Y, plus LQD/HYG prices.</li>"
        "<li>Oil and gold splice front-month futures (2000 onward) into the USO and GLD ETFs from their inception, "
        "which removes futures roll jumps and the 2020-04-20 negative WTI print from the core sample.</li>"
        "<li>Yahoo's BK ticker now points to an unrelated 2026 listing; BNY Mellon is BNY.</li>"
        f"<li>{len(m.get('outlier_flags', []))} robust-outlier flags were reviewed; the large moves are real events "
        "(Aug-2007 bill squeeze, Oct-2008 peso, March 2020, the July-2025 copper tariff reversal, the Jan-2026 silver "
        "crash confirmed by SLV) except futures roll artefacts outside the core sample (corn, 2013-07-15).</li></ul>",
        "<h3>Models</h3><ul>"
        "<li>Networks: rolling and full-sample correlation, partial correlation (precision / graphical lasso), PCA "
        "absorption ratio, a dynamic factor model, pairwise and conditional Granger networks (HC1 robust, "
        "Benjamini-Hochberg), VAR and Diebold-Yilmaz connectedness (rolling 500/250-day and a TVP-VAR with forgetting "
        "factors), empirical tail dependence, Delta-CoVaR, MES and copulas.</li>"
        "<li>Regimes: 3-state Gaussian HMM (smoothed, filtered and real-time), Markov-switching variance, PELT and "
        "Bayesian online change points, GARCH volatility regimes, DCC-GARCH correlations, a Bayesian sticky HMM.</li>"
        "<li>Stress: conditional-Gaussian scenario propagation decomposed along the partial-correlation network "
        "(direct, first-order, higher-order exposure), VAR propagation, tail betas and a Student-t copula.</li>"
        "<li>Everything used in an out-of-sample test is estimated on data available at the time.</li></ul></section>",
    ]
    md = ["## Data and methods", "", table_md(T, "{:.0f}"), ""]
    if A is not None:
        md += ["### Calendars and timestamps", "", table_md(A), ""]
    return "".join(html_parts), "\n".join(md)


def build(panel) -> dict:
    REPORT.mkdir(parents=True, exist_ok=True)
    secs: dict[str, Section] = {}
    for k in ORDER:
        p = OUTPUTS / "research" / f"{k}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                secs[k] = pickle.load(f)
    last = panel.meta["master_calendar"]["last"]
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    claim_defs = "".join(
        f'<div class="claim-def">{badge(c)}<span>{_esc(CLAIM_TEXT[c])}</span></div>' for c in Claim)
    cards = []
    for k, s in secs.items():
        verdict, rest = first_sentence(s.answer)
        if s.numbers.get("verdict"):
            verdict, rest = s.numbers["verdict"], s.answer
        cl = sorted({f.claim for f in s.findings}, key=lambda c: list(Claim).index(c))
        cards.append(f'<article class="answer"><h3><a href="#{k}">{_esc(s.title)}</a></h3>'
                     f'<p class="verdict">{_esc(verdict)}</p><div class="badges">{"".join(badge(c) for c in cl)}</div>'
                     f'<p>{_esc(rest[:330].rsplit(" ", 1)[0] + "...") if len(rest) > 330 else _esc(rest)}</p></article>')
    body = [f'<title>Cross-Asset Contagion Report</title><meta name="claude:viewport-fit-sides" content="cover">',
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
            'family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">',
            f"<style>{CSS}</style>",
            '<header class="masthead"><div class="frame">',
            f'<p class="eyebrow">Contagion Lab - research report - data through {_esc(last)} - built {built}</p>',
            "<h1>How shocks travel across US markets, 2003-2026</h1>",
            '<p class="lede">Six questions about connectedness, transmitters, rate sensitivity, stress testing, tail '
            "dependence and regime-aware hedging, answered with placebo and out-of-sample tests where the question "
            "allows it. Every result carries a label for what it can support.</p>",
            f'<div class="claims">{claim_defs}</div></div></header>',
            '<nav class="toc" aria-label="Sections"><div class="frame"><a href="#answers">Answers</a><a href="#data">Data</a>'
            + "".join(f'<a href="#{k}">{k.upper()}</a>' for k in secs)
            + '<a href="#notes">Method notes</a><a href="#reproduce">Reproduce</a></div></nav>',
            '<main class="frame">',
            f'<section id="answers"><h2>Answers</h2><div class="answers">{"".join(cards)}</div></section>']
    dhtml, dmd = data_section(panel)
    body.append(dhtml)
    md = ["# How shocks travel across US markets, 2003-2026", "",
          f"Contagion Lab research report. Data through {last}; built {built}.", "",
          "Claim levels:", ""] + [f"- **{CLAIM_LABEL[c]}.** {CLAIM_TEXT[c]}" for c in Claim] + ["", "## Answers", ""]
    for k, s in secs.items():
        md.append(f"- **{s.title}** {s.answer}")
    md += ["", dmd, ""]
    for k, s in secs.items():
        cl = sorted({f.claim for f in s.findings}, key=lambda c: list(Claim).index(c))
        parts = [f'<section id="{k}"><div class="qhead"><h2>{_esc(s.title)}</h2>{"".join(badge(c) for c in cl)}</div>',
                 f'<div class="callout"><p>{_esc(s.answer)}</p></div>',
                 f'<p class="design"><strong>Design.</strong> {_esc(s.design)}</p>']
        md += [f"## {s.title}", "", s.answer, "", f"*Design.* {s.design}", ""]
        for fg in s.figures:
            parts.append(f'<figure class="fig"><img class="fl" alt="{_esc(fg.title)}" src="data:image/png;base64,{img_b64(fg.light)}">'
                         f'<img class="fd" alt="{_esc(fg.title)}" src="data:image/png;base64,{img_b64(fg.dark)}">'
                         f"<figcaption><strong>{_esc(fg.title)}.</strong> {_esc(fg.caption)}</figcaption></figure>")
            md += [f"![{fg.title}](../figures/{fg.light.name})", "", f"*{fg.title}.* {fg.caption}", ""]
        parts.append('<div class="findings">')
        for f in s.findings:
            cav = "".join(f"<li>{_esc(c)}</li>" for c in f.caveats)
            parts.append(f'<div class="finding {f.claim.value}"><div class="qhead"><strong>{_esc(f.headline)}</strong>'
                         f"{badge(f.claim)}</div><p>{_esc(f.detail)}</p>" + (f"<ul>{cav}</ul>" if cav else "") + "</div>")
            md += [f"- **{f.headline}** [{CLAIM_LABEL[f.claim]}] {f.detail}"] + [f"  - {c}" for c in f.caveats]
        parts.append("</div>")
        md.append("")
        for t in s.tables:
            parts.append(f"<h3>{_esc(t.title)}</h3>" + table_html(t.df, t.floatfmt)
                         + (f'<p class="tcap">{_esc(t.caption)}</p>' if t.caption else ""))
            md += [f"### {t.title}", "", table_md(t.df, t.floatfmt), "", t.caption, ""]
        if s.robustness:
            parts.append("<h3>Robustness</h3><ul>" + "".join(f"<li>{_esc(r)}</li>" for r in s.robustness) + "</ul>")
            md += ["**Robustness**", ""] + [f"- {r}" for r in s.robustness] + [""]
        parts.append("</section>")
        body.append("".join(parts))
    hub, ll = method_note_tables()
    dfm_html, dfm_md = dfm_section(panel)
    body.append(
        '<section id="notes"><h2>Method notes</h2>' + dfm_html +
        "<h3>Why a Diebold-Yilmaz 'transmitter' is not a cause</h3>"
        "<p>The same-day term of the variance decomposition is a squared correlation, identical in both directions. "
        "After the rows are normalised, an asset correlated with many others still ends up with a positive NET. "
        "Simulated data with no dynamics at all (a hub correlated 0.6 with two otherwise unrelated series) shows it:</p>"
        + table_html(hub, "{:+.1f}") +
        '<p class="tcap">20,000 simulated days, VAR(1), H = 10. The hub is a "net transmitter" although nothing in '
        "the data generating process moves first; the lagged part is zero.</p>"
        "<p>With a genuine one-day lead (y1 = 0.5 y0 yesterday + noise) the lagged part picks up the direction:</p>"
        + table_html(ll, "{:+.1f}") +
        "<p>This report therefore splits every NET into its same-day and lagged parts and checks the lagged part against "
        "closing times. Even the lagged part is a forecasting relation: asynchronous closes, stale prices and common "
        "drivers observed with delay all produce it without any contagion.</p>"
        "<h3>Look-ahead controls</h3><ul>"
        "<li>Rolling windows end on the date they are stamped with; the TVP-VAR is a filter, not a smoother.</li>"
        "<li>Real-time regime probabilities re-estimate the HMM on past data only and report filtered probabilities.</li>"
        "<li>Stress-test and hedging backtests use covariances estimated up to the previous close; thresholds that "
        "define stress days are expanding quantiles.</li>"
        "<li>Crisis dates are mechanical (a fixed peak/trough rule); named events are used only to annotate charts.</li>"
        "</ul></section>")
    md += ["## Method notes", "", dfm_md, "", table_md(hub, "{:+.1f}"), "", table_md(ll, "{:+.1f}"), ""]
    vers = {p: metadata.version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "arch",
                                             "hmmlearn", "ruptures", "networkx", "matplotlib")}
    body.append(
        '<section id="reproduce"><h2>Reproduce</h2>'
        "<p>From the <code>contagion-lab</code> folder, with the project's virtual environment:</p>"
        "<pre>python scripts/build_data.py          # fetch (or reuse the raw cache) and build data/processed\n"
        "python scripts/run_models.py          # networks and regimes -> outputs/models (~25 min)\n"
        "python scripts/run_research.py        # the six questions -> outputs/research, figures, tables\n"
        "python scripts/build_report.py        # this report\n"
        "python scripts/build_dashboard.py     # dashboard/ data export\n"
        "python -m pytest                      # estimator tests on simulated data</pre>"
        f"<p>Python {platform.python_version()}; " + ", ".join(f"{k} {v}" for k, v in vers.items()) +
        f". Panel built {_esc(panel.meta['built_utc'])}. Raw downloads are cached in <code>data/raw</code> with their "
        "fetch time, so a rebuild without <code>--refresh</code> reproduces these numbers exactly.</p></section>")
    body.append(f'</main><footer><div class="frame">Contagion Lab. Market data from Yahoo Finance, FRED (Federal Reserve '
                "Bank of St. Louis), Cboe and the Federal Reserve Board. Research use only; nothing here is investment "
                "advice.</div></footer>")
    fragment = "\n".join(body)
    (REPORT / "report_fragment.html").write_text(fragment, encoding="utf-8")
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
           "</head><body>" + fragment + "</body></html>")
    (REPORT / "report.html").write_text(doc, encoding="utf-8")
    (REPORT / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary = {k: s.answer for k, s in secs.items()}
    (REPORT / "answers.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"html": REPORT / "report.html", "fragment": REPORT / "report_fragment.html", "md": REPORT / "report.md",
            "size_mb": (REPORT / "report.html").stat().st_size / 1e6}
