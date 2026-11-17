"""Q6. Can regime detection improve hedging decisions?

Out-of-sample hedging backtest, 2010-2026, weekly rebalancing, 2bp cost per unit of hedge notional traded.
  Experiment A: hedge the S&P 500 (SPY) with Treasuries (IEF), gold (GLD) and the dollar (UUP).
  Experiment B: hedge SPY with Treasuries only (the classic stock-bond hedge, which failed in 2022).
Minimum-variance hedge ratios h = Sigma_HH^-1 Sigma_HT from:
  static (expanding), rolling 252 days, EWMA 0.97, DCC-GARCH          - no regime model
  regime mixture: real-time 3-state HMM (SPY, VIX, 10Y, Baa spread), state covariances mixed with FILTERED
                  probabilities; parameters re-estimated twice a year on past data only
  regime switch:  covariance of the single most likely state
  change-point window: estimation window restarts at the last BOCPD change point in SPY returns
Evaluation: hedged volatility / variance reduction, expected shortfall, drawdown, bear-market P&L, turnover;
Diebold-Mariano tests on squared hedged returns. Claim: PREDICTIVE (out-of-sample decision performance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..claims import Claim
from ..portfolio.hedging import (BocpdCov, DccCov, EwmaCov, RegimeCov, RollingCov, StaticCov, dm_test_squared,
                                 hedge_backtest, hedge_metrics)
from ..paths import MODELS
from .common import Finding, Section, Table, crisis_episodes, save_fig, save_table, shade_episodes

FEATURES = ["VIX", "UST10Y", "BAA10Y"]
START = 750
REGIME_METHODS = ("regime mixture (HMM)", "regime switch (HMM)", "change-point window (BOCPD)")


def experiment(panel, target: str, hedges: list[str]) -> dict:
    R = panel.returns_for([target] + hedges + FEATURES)
    k = 1 + len(hedges)
    Xs = np.expm1(R[[target] + hedges].to_numpy(dtype=float))
    F = R[FEATURES].to_numpy(dtype=float)
    Xp = np.column_stack([Xs, F])
    feat = [0] + list(range(k, k + len(FEATURES)))
    provs = {"static": StaticCov(Xp), "rolling 252d": RollingCov(Xp, 252), "ewma 0.97": EwmaCov(Xp, 0.97),
             "dcc": DccCov(Xs, min_train=START, refit_every=504),
             "regime mixture (HMM)": RegimeCov(Xp, feat, n_states=3, min_train=START, refit_every=126),
             "regime switch (HMM)": RegimeCov(Xp, feat, n_states=3, min_train=START, refit_every=126,
                                              use="crisis_if_likely"),
             "change-point window (BOCPD)": BocpdCov(Xp, signal_col=0)}
    bts = {m: hedge_backtest(R[[target] + hedges], target, hedges, p, START, rebalance=5, cost_bps=2.0)
           for m, p in provs.items()}
    return {"R": R, "bt": bts, "regime_probs": provs["regime mixture (HMM)"].probs}


def summarize(res: dict, crises: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cr = {}, {}
    any_bt = next(iter(res["bt"].values()))
    rows["unhedged"] = {"ann. vol (%)": 100 * any_bt["unhedged"].std() * np.sqrt(252), "variance reduction (%)": 0.0,
                        "ES 5% daily (%)": 100 * hedge_metrics(any_bt)["es5_unhedged"],
                        "max drawdown (%)": 100 * hedge_metrics(any_bt)["maxdd_unhedged"],
                        "ann. return (%)": 100 * any_bt["unhedged"].mean() * 252, "annual turnover": 0.0}
    for m, bt in res["bt"].items():
        h = hedge_metrics(bt, crises)
        rows[m] = {"ann. vol (%)": 100 * h["ann_vol_hedged"], "variance reduction (%)": 100 * h["variance_reduction"],
                   "ES 5% daily (%)": 100 * h["es5_hedged"], "max drawdown (%)": 100 * h["maxdd_hedged"],
                   "ann. return (%)": 100 * h["ann_ret_hedged"], "annual turnover": h["annual_turnover"]}
        for c in h.get("crisis", []):
            cr.setdefault(c["window"], {})["unhedged"] = 100 * c["unhedged"]
            cr[c["window"]][m] = 100 * c["hedged"]
    T = pd.DataFrame(rows).T
    base = res["bt"]["rolling 252d"]["hedged"]
    stat = res["bt"]["static"]["hedged"]
    for m, bt in res["bt"].items():
        T.loc[m, "DM t vs rolling 252d"] = dm_test_squared(bt["hedged"], base)[0] if m != "rolling 252d" else np.nan
        T.loc[m, "DM t vs static"] = dm_test_squared(bt["hedged"], stat)[0] if m != "static" else np.nan
    return T, pd.DataFrame(cr).T


def run(panel) -> Section:
    """Backtests cached in outputs/research/cache/q6.pkl (keyed by sample end and settings)."""
    import pickle

    from ..paths import OUTPUTS

    sig = (str(panel.returns.index[-1].date()), START, tuple(FEATURES), "v1")
    cp = OUTPUTS / "research" / "cache" / "q6.pkl"
    cached = None
    if cp.exists():
        with open(cp, "rb") as f:
            cached = pickle.load(f)
        if cached.get("sig") != sig:
            cached = None
    if cached is None:
        cached = {"sig": sig, "A": experiment(panel, "SPY", ["IEF", "GLD", "UUP"]),
                  "B": experiment(panel, "SPY", ["IEF"])}
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump(cached, f)
    A, B = cached["A"], cached["B"]
    start_date = A["bt"]["static"].index[0]
    ep = crisis_episodes(panel, 0.10, 0.10, str(start_date.date()))
    crises = [(e["peak"], e["trough"]) for _, e in ep.iterrows()] + [(pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-12"))]
    TA, CA = summarize(A, crises)
    TB, CB = summarize(B, crises)
    t1 = Table("q6_multi", "Hedging SPY with Treasuries, gold and the dollar (out of sample, 2010-2026)", TA,
               "Weekly rebalancing, 2bp cost per unit of notional traded. DM t: Diebold-Mariano statistic on squared "
               "hedged returns; negative = lower variance than the comparison method.", "{:.2f}")
    t2 = Table("q6_bond", "Hedging SPY with Treasuries only", TB, "", "{:.2f}")
    t3 = Table("q6_crises", "Cumulative return in each sell-off (Treasuries + gold + dollar hedge), %", CA,
               "Sell-offs dated with the 10%/10% rule after the evaluation starts, plus the 2022 rate shock.", "{:.1f}")
    for t in (t1, t2, t3):
        save_table(t)

    # hedge ratio paths in the bond-only experiment
    paths = {m: B["bt"][m]["h_IEF"] for m in ("static", "ewma 0.97", "regime mixture (HMM)")}
    hmm = pd.read_parquet(MODELS / "regime_hmm.parquet") if (MODELS / "regime_hmm.parquet").exists() else None
    epa = crisis_episodes(panel, 0.15, 0.15, str(start_date.date()))

    def draw_h(fig, P, mode):
        ax = fig.add_subplot(111)
        shade_episodes(ax, epa, P)
        for k, (m, s) in enumerate(paths.items()):
            ax.plot(s.index, -s.values, lw=1.5 if k else 1.8, color=P["series"][k], label=m)
        ax.axhline(0, color=P["base"], lw=0.8)
        ax.set_ylabel("IEF held per $1 of SPY (long > 0)")
        ax.legend(loc="upper left", ncols=3)
        ax.set_title("Minimum-variance Treasury hedge ratio by estimation method")

    f1 = save_fig("q6_hedge_ratio", draw_h, (8.6, 3.4), "Treasury hedge ratio over time",
                  "Positive: long Treasuries against equities (a hedge when stocks and bonds are negatively "
                  "correlated). Shading: bear markets.")

    def draw_vr(fig, P, mode):
        a1, a2 = fig.subplots(1, 2, sharex=False)
        for ax, T, ttl in ((a1, TA, "Treasuries + gold + dollar"), (a2, TB, "Treasuries only")):
            v = T.drop(index="unhedged")["variance reduction (%)"].sort_values()
            y = np.arange(len(v))
            ax.barh(y, v.to_numpy(), height=0.55,
                    color=[P["series"][1] if m in REGIME_METHODS else P["series"][0] for m in v.index])
            ax.set_yticks(y, v.index)
            ax.grid(axis="y", visible=False)
            ax.set_title(ttl)
            ax.set_xlabel("variance reduction vs unhedged, %")
            for yy, val in zip(y, v.to_numpy()):
                ax.annotate(f"{val:.1f}", (val, yy), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=8, color=P["ink2"])

    f2 = save_fig("q6_variance", draw_vr, (9.6, 3.4), "Out-of-sample variance reduction by method",
                  "Orange: methods that use regime detection (HMM or change points); blue: methods that do not.")
    figs = [f1, f2]
    if hmm is not None and "realtime_2" in hmm:
        rt = hmm["realtime_2"]

        def draw_rt(fig, P, mode):
            ax = fig.add_subplot(111)
            shade_episodes(ax, crisis_episodes(panel, 0.15, 0.15, "2006-01-01"), P)
            ax.plot(rt.index, rt.values, color=P["series"][0], lw=1.0)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("probability")
            ax.set_title("Real-time probability of the high-volatility regime (3-state HMM, filtered, re-estimated on "
                         "past data only)")

        figs.append(save_fig("q6_regime_rt", draw_rt, (8.6, 2.8), "Real-time crisis-regime probability",
                             "The probability uses only information available on each day (parameters re-estimated "
                             "every six months on past data). Shading: bear markets."))
    sec = Section("q6", "Q6. Can regime detection improve hedging decisions?",
                  "Can regime detection improve hedging decisions?", "", "")
    sec.tables, sec.figures = [t1, t2, t3], figs

    def verdict(T):
        nonreg = [m for m in T.index if m not in REGIME_METHODS and m != "unhedged"]
        reg = [m for m in T.index if m in REGIME_METHODS]
        bn = T.loc[nonreg, "variance reduction (%)"].idxmax()
        br = T.loc[reg, "variance reduction (%)"].idxmax()
        return bn, br

    bnA, brA = verdict(TA)
    bnB, brB = verdict(TB)
    gapA = TA.loc[brA, "variance reduction (%)"] - TA.loc[bnA, "variance reduction (%)"]
    gapB = TB.loc[brB, "variance reduction (%)"] - TB.loc[bnB, "variance reduction (%)"]
    dmA = dm_test_squared(A["bt"][brA]["hedged"], A["bt"][bnA]["hedged"])
    dmB = dm_test_squared(B["bt"][brB]["hedged"], B["bt"][bnB]["hedged"])
    win22 = "2022-01-03..2022-10-12"
    c22 = CB.loc[win22] if win22 in CB.index else None
    k20 = [k for k in CA.index if k.startswith("2020-02")]
    c20 = CA.loc[k20[0]] if k20 else None
    better_regime = (gapA > 0 and dmA[1] < 0.05) or (gapB > 0 and dmB[1] < 0.05)
    worse_regime = (gapA < 0 and dmA[1] < 0.05) or (gapB < 0 and dmB[1] < 0.05)
    verdict = "Yes, modestly." if better_regime else ("No." if worse_regime else
                                                      "Not beyond what simple adaptive estimators deliver.")
    hm, hs = "regime mixture (HMM)", "regime switch (HMM)"
    vr = lambda T, m: T.loc[m, "variance reduction (%)"]  # noqa: E731
    sec.design = (f"Out-of-sample from {start_date.date()} (first 750 days used for estimation only); weekly "
                  "rebalancing; 2bp per unit traded; hedge ratios capped at 3x.")
    sec.answer = (
        f"{verdict} Hedges that adapt to recent data reduce the S&P 500's variance the most: {bnA} "
        f"{vr(TA, bnA):.1f}% with Treasuries, gold and the dollar, and {bnB} {vr(TB, bnB):.1f}% with Treasuries alone, "
        f"against {vr(TA, 'static'):.1f}% and {vr(TB, 'static'):.1f}% for a static hedge. The HMM regime hedges "
        f"reduce it by {vr(TA, hm):.1f}% and {vr(TA, hs):.1f}% (multi-asset), no better than static, and the best "
        f"regime-based method ({brA}, {vr(TA, brA):.1f}%) trails {bnA} by {abs(gapA):.1f} points "
        f"(Diebold-Mariano p = {dmA[1]:.3f}); with Treasuries alone the gap is {gapB:+.1f} points (p = {dmB[1]:.3f}). "
        + (f"The HMM's value is episodic. In the 2020 crash its hedge returned {c20[hm]:.1f}% against "
           f"{c20['unhedged']:.1f}% unhedged and {c20[bnA]:.1f}% for {bnA}" if c20 is not None else "")
        + (f"; in the 2022 rate shock it stayed long Treasuries while stocks and bonds fell together "
           f"({c22[hm]:.1f}% with a Treasury hedge vs {c22['unhedged']:.1f}% unhedged and {c22['ewma 0.97']:.1f}% "
           "for EWMA). " if c22 is not None else ". ")
        + "That pattern is consistent with its high-volatility state having been estimated on flight-to-quality "
        "crises: regime detection flags when markets are stressed, but a hedge needs the current correlation, and "
        "the regime model supplies the correlation of past crises.")
    sec.robustness = [
        "Hedge ratios are capped at 3x, rebalanced weekly and charged 2bp per unit of notional traded; other "
        "rebalancing frequencies and cost levels were not tested.",
        "The change-point window counts as regime detection (online Bayesian change points); it performs like a "
        "plain rolling window.",
        f"Turnover (annual, multiples of notional): " + ", ".join(
            f"{m} {TA.loc[m, 'annual turnover']:.1f}" for m in TA.index if m != "unhedged") + ".",
    ]
    sec.findings = [
        Finding("Q6", Claim.PREDICTIVE, "Regime-based vs adaptive hedges",
                f"Variance reduction (multi-hedge): " + ", ".join(
                    f"{m} {TA.loc[m, 'variance reduction (%)']:.1f}%" for m in TA.index if m != "unhedged") + ".",
                ["One 16-year path; differences of a few points of variance reduction are within sampling noise.",
                 "HMM states are estimated on four features; other feature sets would give other regimes."]),
        Finding("Q6", Claim.PREDICTIVE, "The 2022 test case",
                ("Cumulative return Jan-Oct 2022 with a Treasury hedge: " + ", ".join(
                    f"{m} {v:.1f}%" for m, v in c22.items()) + ".") if c22 is not None else "n/a",
                ["A single episode; it shows the mechanism (static hedges keep buying bonds as both fall), not a "
                 "statistically general result."]),
    ]
    sec.numbers = {"multi": TA.to_dict(), "bond": TB.to_dict(), "dmA": dmA, "dmB": dmB}
    return sec
