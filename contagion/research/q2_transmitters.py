"""Q2. Which assets act as shock transmitters?

Three different notions of "transmitter" are measured separately, because they are routinely conflated:
  1. Diebold-Yilmaz NET connectedness (the standard answer). Split into its contemporaneous part (h = 0:
     symmetric squared correlations reweighted by row normalisation = CENTRALITY) and its dynamic part (h >= 1:
     lagged, PREDICTIVE direction).
  2. Granger out-degree (HC-robust, Benjamini-Hochberg FDR 5%), rolling 500-day windows. PREDICTIVE.
  3. Tail transmitters: Delta-CoVaR out-strength on risk-oriented returns. DESCRIPTIVE.
Timing check: dynamic NET is compared with each series' close time, and the whole exercise is repeated on the
synchronous universe (every series priced at the NYSE close), where timing artefacts cannot arise.
Financial institutions: MES, Delta-CoVaR and DY TO-connectedness in the 18-node financial network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..claims import Claim
from ..models import RISK_OFF_SIGN, load_dynamic, load_granger
from ..network.connectedness import connectedness
from ..network.tail import delta_covar, mes
from ..paths import MODELS
from .common import Finding, Section, Table, crisis_episodes, diverging_cmap, save_fig, save_table


def static_table(R: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    c = connectedness(R, p=1, H=10)
    df = pd.DataFrame({"TO": c.to, "FROM": c.from_, "NET": c.net, "NET same-day": c.net_contemporaneous(),
                       "NET lagged": c.net_dynamic()})
    return df.sort_values("NET", ascending=False), c


def granger_rank(key: str) -> pd.DataFrame:
    out = {}
    for kind in ("pairwise", "conditional"):
        g = load_granger(key, kind)
        out[f"out-degree ({kind})"] = g["out_degree"].mean()
        out[f"in-degree ({kind})"] = g["in_degree"].mean()
    return pd.DataFrame(out)


def covar_strength(R: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """Tail out-strength of i: mean over j of Delta-CoVaR_{j|i} / sd_j (risk-oriented returns, so negative = more
    downside for j when i is in its own 5% tail). Also in-strength."""
    Ro = R * pd.Series(RISK_OFF_SIGN)[R.columns]
    cols = list(Ro.columns)
    sd = Ro.std()
    M = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i != j:
                M.loc[j, i] = delta_covar(Ro[j], Ro[i], q)["delta_covar"] / sd[j]
    return pd.DataFrame({"tail out-strength": M.mean(axis=0), "tail in-strength": M.mean(axis=1)}), M


def fin_institutions(panel) -> pd.DataFrame:
    R = pd.read_parquet(MODELS / "R_fin.parquet")
    names = [c for c in R.columns if c not in ("SPY",)]
    c = connectedness(R, p=1, H=10)
    rows = {}
    gfc = R.loc["2007-07-01":"2009-06-30"]
    for n in names:
        rows[n] = {"MES 5% (full)": -100 * mes(R[n], R["SPY"], 0.05),
                   "MES 5% (GFC 2007-09)": -100 * mes(gfc[n], gfc["SPY"], 0.05),
                   "Delta-CoVaR of SPY | i (bp)": 1e4 * delta_covar(R["SPY"], R[n], 0.05)["delta_covar"],
                   "DY TO (fin network)": float(c.to[n]), "DY NET lagged": float(c.net_dynamic()[n])}
    df = pd.DataFrame(rows).T
    df.index = [f"{n} ({panel.label(n)})" if n not in ("XLF", "KRE") else f"{n} ({panel.label(n)})" for n in df.index]
    return df.sort_values("MES 5% (full)", ascending=False)


def run(panel) -> Section:
    Rm = pd.read_parquet(MODELS / "R_macro.parquet")
    Rs = pd.read_parquet(MODELS / "R_sync.parquet")
    tm, cm = static_table(Rm)
    ts, cs = static_table(Rs)
    d500 = load_dynamic("dy_macro_w500")
    net, netd = d500.net, d500.net_dynamic
    share_tx = (net > 0).mean()
    # crisis-specific top transmitters (NET averaged over each bear market, peak to trough)
    ep = crisis_episodes(panel, 0.15, 0.15, "2005-04-01")
    crisis_top = []
    for _, e in ep.iterrows():
        w = net.loc[e["peak"]: e["trough"]]
        wd = netd.loc[e["peak"]: e["trough"]]
        if len(w) == 0:
            continue
        m, md = w.mean().sort_values(ascending=False), wd.mean().sort_values(ascending=False)
        crisis_top.append({"bear market": f"{e['peak'].date()} to {e['trough'].date()}",
                           "top NET (total)": f"{m.index[0]} {m.iloc[0]:+.0f}, {m.index[1]} {m.iloc[1]:+.0f}",
                           "top NET (lagged only)": f"{md.index[0]} {md.iloc[0]:+.1f}, {md.index[1]} {md.iloc[1]:+.1f}",
                           "largest receiver (lagged)": f"{md.index[-1]} {md.iloc[-1]:+.1f}"})
    crisis_top = pd.DataFrame(crisis_top).set_index("bear market")
    # rank stability across non-overlapping 3-year blocks
    blocks = net.groupby(net.index.year // 3).mean()
    rk = blocks.rank(axis=1)
    stab = [stats.spearmanr(rk.iloc[i], rk.iloc[i + 1])[0] for i in range(len(rk) - 1)]
    blocks_d = netd.groupby(netd.index.year // 3).mean().rank(axis=1)
    stab_d = [stats.spearmanr(blocks_d.iloc[i], blocks_d.iloc[i + 1])[0] for i in range(len(blocks_d) - 1)]
    # timing check: how late does each series absorb S&P 500 news? |corr(SPY_t, x_t+1)| - |corr(SPY_t, x_t-1)|
    # (> 0: the series reacts a day late - earlier close or stale marks). Close hour alone misses stale pricing.
    from ..data.calendar import lead_lag_profile

    ll = lead_lag_profile(Rm, "SPY")
    timing = ll["timing_signal"].reindex(tm.index).fillna(0.0)
    close = panel.universe.loc[tm.index, "close_et"].astype(float)
    others = [n for n in tm.index if n != "SPY"]
    rho_close, p_close = stats.spearmanr(timing[others], tm.loc[others, "NET lagged"])
    rho_hour, _ = stats.spearmanr(close[others], tm.loc[others, "NET lagged"])
    lag_vs_spy = None
    gm, gs = granger_rank("macro"), granger_rank("sync")
    tail_tab, tailM = covar_strength(Rm)
    fin = fin_institutions(panel)

    tm_out = tm.join(gm[["out-degree (pairwise)", "out-degree (conditional)"]]).join(tail_tab)
    tm_out["close (ET hour)"] = close
    tm_out["late reaction to SPY"] = timing
    tm_out["share of windows net transmitter"] = share_tx.reindex(tm_out.index)
    t1 = Table("q2_macro", "Transmitter measures, macro network (2003-2026)", tm_out,
               "TO/FROM/NET in percentage points of 10-day forecast-error variance (VAR(1), full sample). "
               "NET splits exactly into a same-day part (centrality) and a lagged part (predictive direction). "
               "Granger degrees: average number of significant out-links per 500-day window (FDR 5%). Tail strength: "
               "mean Delta-CoVaR of other assets (in their own SDs) when the asset is in its worst 5% of days.")
    t2 = Table("q2_sync", "Same measures, synchronous universe (all prices at the NYSE close, 2007-2026)",
               ts.join(gs[["out-degree (pairwise)", "out-degree (conditional)"]]),
               "With every series priced at 16:00 ET, lagged spillovers cannot come from different closing times.")
    t3 = Table("q2_crises", "Largest transmitters inside each bear market (rolling 500-day NET)", crisis_top, "")
    t4 = Table("q2_fin", "Financial institutions: systemic-risk measures", fin,
               "MES: average loss on the S&P 500's worst 5% of days (percent). Delta-CoVaR: change in the S&P 500's 5% "
               "quantile when the institution moves from its median to its own 5% quantile (basis points). DY TO: "
               "contribution to other institutions' forecast-error variance in the 18-node financial network.")
    for t in (t1, t2, t3, t4):
        save_table(t)

    # ---------------------------------------------------------------- figures
    order = tm.sort_values("NET").index

    def draw_split(fig, P, mode):
        a1, a2 = fig.subplots(1, 2, sharey=True)
        y = np.arange(len(order))
        for ax, col, ttl in ((a1, "NET same-day", "Same-day part (centrality)"),
                             (a2, "NET lagged", "Lagged part (predictive direction)")):
            v = tm.loc[order, col]
            ax.barh(y, v, height=0.62, color=[P["series"][0] if x >= 0 else P["series"][7] for x in v])
            ax.axvline(0, color=P["base"], lw=0.8)
            ax.set_title(ttl)
            ax.grid(axis="y", visible=False)
        a1.set_yticks(y, order)
        a1.set_xlabel("NET, percentage points")
        a2.set_xlabel("NET, percentage points (note the smaller scale)")

    f1 = save_fig("q2_net_split", draw_split, (8.6, 4.4), "NET connectedness, split into its two parts",
                  "Full-sample VAR(1), H = 10. Blue: net transmitter, red: net receiver. The two panels add up to the "
                  "standard DY NET. Same-day correlations contribute symmetrically, so the left panel ranks how "
                  "central an asset is, not who moves first.")
    cmap = {m: diverging_cmap(m) for m in ("light", "dark")}
    netw = net.resample("ME").mean()
    ordh = net.mean().sort_values(ascending=False).index

    def draw_heat(fig, P, mode):
        ax = fig.add_subplot(111)
        M = netw[ordh].T.to_numpy()
        lim = np.nanpercentile(np.abs(M), 98)
        im = ax.imshow(M, aspect="auto", cmap=cmap[mode], vmin=-lim, vmax=lim, interpolation="nearest",
                       extent=[0, M.shape[1], len(ordh), 0])
        ax.set_yticks(np.arange(len(ordh)) + 0.5, ordh)
        yrs = netw.index.year
        ticks = [i for i in range(len(yrs)) if i == 0 or yrs[i] != yrs[i - 1]]
        ticks = [t for t in ticks if yrs[t] % 2 == 0]
        ax.set_xticks(ticks, [str(yrs[t]) for t in ticks])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("NET (pp)")
        cb.outline.set_visible(False)
        ax.set_title("Rolling 500-day NET connectedness, monthly average")

    f2 = save_fig("q2_net_heat", draw_heat, (9.0, 4.2), "Who transmits, over time",
                  "Blue = net transmitter, red = net receiver. Equity nodes transmit in almost every window; the sign "
                  "of most other nodes is stable too, because NET is dominated by centrality.")

    def draw_timing(fig, P, mode):
        ax = fig.add_subplot(111)
        x, yv = timing[others].to_numpy(), tm.loc[others, "NET lagged"].to_numpy()
        ax.scatter(x, yv, s=36, color=P["series"][0], edgecolor=P["surface"], linewidth=1.5, zorder=3)
        for n, a, b in zip(others, x, yv):
            ax.annotate(n, (a, b), xytext=(4, 3), textcoords="offset points", fontsize=8, color=P["ink2"])
        ax.axhline(0, color=P["base"], lw=0.8)
        ax.axvline(0, color=P["base"], lw=0.8)
        ax.set_xlabel("late reaction to S&P 500 news: |corr(SPY today, x tomorrow)| - |corr(SPY today, x yesterday)|")
        ax.set_ylabel("NET from lagged spillovers, pp")
        ax.set_title(f"Lagged receivers are the series that absorb equity news a day late (Spearman {rho_close:.2f})")

    f3 = save_fig("q2_timing", draw_timing, (7.8, 3.9), "Lagged NET vs late reaction to equity news",
                  "Series that close before the NYSE (H.10 FX at noon, metals and oil settlements) or carry stale "
                  "marks (Moody's Baa spread) absorb part of each day's equity news the next day, so a VAR records a "
                  "lagged link from equities to them. That is a timing artefact, not contagion.")
    sec = Section("q2", "Q2. Which assets act as shock transmitters?", "Which assets act as shock transmitters?",
                  "", "")
    sec.tables = [t1, t2, t3, t4]
    sec.figures = [f1, f2, f3]
    eq = ["SPY", "EFA", "EEM", "IWM", "XLF"]
    share_cont = float(tm.loc[eq, "NET same-day"].sum() / tm.loc[eq, "NET"].sum())
    top = tm.index[:4].tolist()
    sync_dyn_max = float(ts["NET lagged"].abs().max())
    tail_top = tail_tab.sort_values("tail out-strength").index[:3].tolist()
    sec.design = ("Full-sample and rolling Diebold-Yilmaz tables (VAR(1), H = 10), with NET split exactly into its "
                  "same-day and lagged parts; Granger networks with HC1 Wald tests and Benjamini-Hochberg control; "
                  "Delta-CoVaR tail networks; repeated on a synchronous universe in which every series is priced at "
                  "the NYSE close.")
    sec.answer = (
        f"In the standard Diebold-Yilmaz sense the transmitters are equities: {', '.join(top)} have the largest NET "
        f"(+{tm.loc[top[-1], 'NET']:.0f} to +{tm.loc[top[0], 'NET']:.0f} points) and are net transmitters in "
        f"{100 * share_tx[top].min():.0f}-{100 * share_tx[top].max():.0f}% of rolling windows. But "
        f"{100 * share_cont:.0f}% of the equity NET comes from same-day correlation, which is symmetric and only "
        "says these assets are central. Lagged spillovers carry the directional (predictive) content, and they are "
        f"small: {100 * cm.dynamic_share():.1f}% of total connectedness in the macro network and "
        f"{100 * cs.dynamic_share():.1f}% in the synchronous one. Their direction tracks how late each series "
        f"absorbs equity news (Spearman {rho_close:.2f} between that lag and lagged NET; the close hour alone gives "
        f"{rho_hour:.2f} because it misses stale marks), and with synchronous prices no asset's lagged NET exceeds "
        f"{sync_dyn_max:.1f} points. In the tails, {', '.join(tail_top)} have the largest "
        "Delta-CoVaR out-strength (descriptive). Which asset 'transmits' therefore depends on the definition, and "
        "none of these measures identifies a causal transmitter.")
    sec.findings = [
        Finding("Q2", Claim.DESCRIPTIVE, "Equities are the most central nodes",
                f"Same-day NET: " + ", ".join(f"{n} {tm.loc[n, 'NET same-day']:+.1f}" for n in top) + ".",
                ["Centrality depends on universe composition: adding more equity series raises equities' TO."]),
        Finding("Q2", Claim.PREDICTIVE, "Lagged spillovers are small and timing-driven",
                f"Lagged share of connectedness {100 * cm.dynamic_share():.1f}% (macro) vs {100 * cs.dynamic_share():.1f}% "
                f"(synchronous). Largest lagged receivers in the macro network: "
                + ", ".join(f"{n} {v:+.1f}" for n, v in tm['NET lagged'].sort_values().head(3).items()) + ".",
                ["Predictive links from asynchronous or stale prices are real forecasting relations but not contagion."]),
        Finding("Q2", Claim.DESCRIPTIVE, "Rankings are stable in sign, not because of crisis-specific transmission",
                f"Spearman rank correlation of NET between consecutive 3-year blocks: median {np.median(stab):.2f} "
                f"(total NET) vs {np.median(stab_d):.2f} (lagged NET only).", []),
    ]
    sec.numbers = {"verdict": f"Equities, mostly through centrality: lagged spillovers are only "
                              f"{100 * cm.dynamic_share():.1f}% of connectedness.",
                   "dynamic_share_macro": cm.dynamic_share(), "dynamic_share_sync": cs.dynamic_share(),
                   "rho_timing_lagged_net": rho_close, "p_timing": p_close, "rho_close_hour": rho_hour,
                   "rank_stability_total": stab,
                   "rank_stability_lagged": stab_d, "share_contemporaneous_equity_net": share_cont}
    sec.robustness = [
        "Rolling 250-day windows, VAR(2) and H = 20 versions are stored in outputs/models (dy_macro_w250, "
        "dy_macro_w500_p2, dy_macro_w500_h20).",
        "Granger out-degrees are averaged over rolling windows; conditional (full-VAR) tests control for every "
        "other series' lags.",
        f"Fraction of rolling windows in which each asset is a net transmitter is in the first table "
        f"(SPY {100 * share_tx['SPY']:.0f}%, VIX {100 * share_tx['VIX']:.0f}%, UST10Y {100 * share_tx['UST10Y']:.0f}%).",
    ]
    if lag_vs_spy is not None:
        sec.numbers["lead_lag_table"] = "outputs/tables/asynchrony_vs_spy.csv"
    return sec
