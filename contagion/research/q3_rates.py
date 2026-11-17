"""Q3. Which sectors are most vulnerable to rate shocks?

Three answers with three evidential levels:
  DESCRIPTIVE  daily betas of sector returns on 10Y yield changes - total, and partial (holding the S&P 500
               fixed: does the sector fall MORE than the market when yields rise?) - full sample and split by the
               stock-bond correlation regime known the day before (rolling 252-day SPY/IEF correlation).
  DESCRIPTIVE  average sector moves on large yield days and in the 2013 and 2022 rate-shock episodes.
  IDENTIFIED   response to monetary-policy-driven 2-year yield moves, identified through heteroskedasticity on
               scheduled FOMC announcement days (Rigobon & Sack 2004): with D = FOMC days and C = the trading day
               before each one,
                   beta = [Cov_D(r, di) - Cov_C(r, di)] / [Var_D(di) - Var_C(di)]
               Assumptions: (1) policy-news variance is higher on FOMC days, (2) the variance of every other shock
               is the same on D and C days, (3) responses are stable across days. Caveat: FOMC surprises also
               carry the Fed's information about the economy, so the "policy shock" is partly growth news.
Sector returns are used relative to the S&P 500 for the ranking (market-neutral vulnerability).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..claims import Claim
from ..paths import PROCESSED
from .common import Finding, Section, Table, save_fig, save_table

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR", "XLRE", "XLC", "KRE", "ITB",
           "IWF", "IWD", "IWM", "QQQ"]
LABELS = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
          "XLP": "Staples", "XLU": "Utilities", "XLV": "Health care", "XLY": "Discretionary", "IYR": "Real estate",
          "XLRE": "Real estate (XLRE)", "XLC": "Comm. services", "KRE": "Regional banks", "ITB": "Homebuilders",
          "IWF": "Growth (R1000G)", "IWD": "Value (R1000V)", "IWM": "Small caps", "QQQ": "Nasdaq-100"}


def ols_hac(y, X, lags=5):
    import statsmodels.api as sm

    res = sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res


def rigobon_sack(r: np.ndarray, di: np.ndarray, is_d: np.ndarray, pair: np.ndarray | None = None,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Identification through heteroskedasticity. Bootstrap: FOMC and control days resampled independently, or
    by meeting when ``pair`` is given (day-before design)."""
    def est(ix):
        d, c = ix[is_d[ix]], ix[~is_d[ix]]
        if len(d) < 5 or len(c) < 5:
            return np.nan, np.nan
        vd, vc = np.var(di[d], ddof=1), np.var(di[c], ddof=1)
        cd = np.cov(r[d], di[d])[0, 1]
        cc = np.cov(r[c], di[c])[0, 1]
        return (cd - cc) / (vd - vc), vd / vc

    allix = np.arange(len(r))
    b, vr = est(allix)
    rng = np.random.default_rng(seed)
    bs, vrs = [], []
    if pair is not None:
        meetings = np.unique(pair)
        groups = {m: allix[pair == m] for m in meetings}
    D, C = allix[is_d], allix[~is_d]
    for _ in range(n_boot):
        if pair is not None:
            ix = np.concatenate([groups[m] for m in rng.choice(meetings, len(meetings), replace=True)])
        else:
            ix = np.concatenate([rng.choice(D, len(D), replace=True), rng.choice(C, len(C), replace=True)])
        bb, vv = est(ix)
        bs.append(bb)
        vrs.append(vv)
    bs = np.array(bs)
    return {"beta": b, "ci90": (np.nanquantile(bs, 0.05), np.nanquantile(bs, 0.95)), "var_ratio": vr,
            "var_ratio_ci90": (np.nanquantile(vrs, 0.05), np.nanquantile(vrs, 0.95)),
            "share_boot_negative": float(np.nanmean(bs < 0))}


def fomc_sample(panel, R: pd.DataFrame, exclude_zlb: bool = False, controls: str = "weekday") -> pd.DataFrame:
    """FOMC announcement days plus control days.

    controls="weekday":    every non-FOMC Tuesday / Wednesday more than two trading days from an announcement
                           (announcements fall on Tuesdays or Wednesdays; matching the weekday keeps the
                           day-of-week pattern of data releases comparable)
    controls="day_before": the trading day before each announcement (Rigobon & Sack's original design)
    """
    fomc = pd.read_csv(PROCESSED / "fomc_dates.csv", parse_dates=["date"])
    sched = fomc.loc[fomc["kind"] == "scheduled", "date"]
    cal = R.index
    pos = cal.searchsorted(pd.DatetimeIndex(sched))
    pos = pos[(pos > 0) & (pos < len(cal))]
    dset = set(pos)
    rows = [(cal[p], True, k) for k, p in enumerate(pos)]
    if controls == "day_before":
        rows += [(cal[p - 1], False, k) for k, p in enumerate(pos) if p - 1 not in dset]
    else:
        near = np.zeros(len(cal), dtype=bool)
        for p in pos:
            near[max(0, p - 2): p + 3] = True
        ok = (~near) & np.isin(cal.dayofweek, [1, 2]) & (cal >= cal[pos[0]] - pd.Timedelta(days=30))
        rows += [(d, False, -1) for d in cal[ok]]
    S = pd.DataFrame(rows, columns=["date", "is_fomc", "meeting"]).set_index("date").sort_index()
    if exclude_zlb:
        ff = panel.levels["FEDFUNDS"].reindex(S.index)
        S = S[ff >= 0.25]
    return S


def run(panel) -> Section:
    ids = [s for s in SECTORS if s in panel.returns.columns]
    base = panel.returns[["SPY", "UST2Y", "UST10Y", "IEF"] + ids]
    R = base.loc["2000-08-24":]
    sbc = R["SPY"].rolling(252).corr(R["IEF"]).shift(1)
    regime = pd.Series(np.where(sbc < 0, "neg", np.where(sbc > 0, "pos", "na")), index=R.index)
    rows = {}
    for s in ids:
        d = R[[s, "SPY", "UST10Y"]].dropna()
        tot = ols_hac(d[s], d[["UST10Y"]])
        par = ols_hac(d[s], d[["UST10Y", "SPY"]])
        row = {"total beta (%/100bp)": 1e4 * tot.params["UST10Y"], "total t": tot.tvalues["UST10Y"],
               "partial beta (%/100bp)": 1e4 * par.params["UST10Y"], "partial t": par.tvalues["UST10Y"]}
        for reg in ("neg", "pos"):
            m = (regime.reindex(d.index) == reg)
            if m.sum() > 250:
                pr = ols_hac(d.loc[m, s], d.loc[m, ["UST10Y", "SPY"]])
                row[f"partial beta, {'negative' if reg == 'neg' else 'positive'} stock-bond corr"] = 1e4 * pr.params["UST10Y"]
        big = d["UST10Y"] >= d["UST10Y"].quantile(0.95)
        row["rel. return on top-5% yield-rise days (%)"] = 100 * (d.loc[big, s] - d.loc[big, "SPY"]).mean()
        for name, a, b in (("taper 2013", "2013-05-22", "2013-06-24"), ("2022 rate shock", "2022-01-04", "2022-10-12")):
            w = R.loc[a:b, [s, "SPY"]].dropna()
            row[f"rel. return {name} (%)"] = 100 * (np.expm1(w[s].sum()) - np.expm1(w["SPY"].sum())) if len(w) > 5 else np.nan
        rows[s] = row
    D = pd.DataFrame(rows).T

    # identified responses (relative to the S&P 500): weekday-matched controls (main), excluding the zero lower
    # bound, and the original day-before design (robustness)
    ident = {}
    for tag, zlb, ctl in (("all", False, "weekday"), ("ex-ZLB", True, "weekday"), ("day-before", False, "day_before")):
        S = fomc_sample(panel, R, exclude_zlb=zlb, controls=ctl)
        for s in ids + ["SPY"]:
            cols = [s, "UST2Y"] if s == "SPY" else [s, "SPY", "UST2Y"]
            d = R.loc[R.index.intersection(S.index), cols].join(S).dropna()
            y = (d[s] - d["SPY"]).to_numpy() if s != "SPY" else d["SPY"].to_numpy()
            ident[(s, tag)] = rigobon_sack(y, d["UST2Y"].to_numpy(), d["is_fomc"].to_numpy(),
                                           d["meeting"].to_numpy() if ctl == "day_before" else None,
                                           n_boot=1000, seed=1)
        ident[("_n", tag)] = {"fomc_days": int(S["is_fomc"].sum()), "control_days": int((~S["is_fomc"]).sum())}
    # per +10bp policy-driven 2Y move, in percent
    I = pd.DataFrame({s: {"identified rel. response, %/10bp": 1e3 * ident[(s, "all")]["beta"],
                          "90% CI low": 1e3 * ident[(s, "all")]["ci90"][0],
                          "90% CI high": 1e3 * ident[(s, "all")]["ci90"][1],
                          "ex-ZLB response, %/10bp": 1e3 * ident[(s, "ex-ZLB")]["beta"],
                          "ex-ZLB CI low": 1e3 * ident[(s, "ex-ZLB")]["ci90"][0],
                          "ex-ZLB CI high": 1e3 * ident[(s, "ex-ZLB")]["ci90"][1],
                          "day-before design, %/10bp": 1e3 * ident[(s, "day-before")]["beta"]} for s in ids}).T
    vr = ident[(ids[0], "all")]["var_ratio"]
    vr_ci = ident[(ids[0], "all")]["var_ratio_ci90"]
    vr_x = ident[(ids[0], "ex-ZLB")]["var_ratio"]
    vr_db = ident[(ids[0], "day-before")]["var_ratio"]
    spy_b = 1e3 * ident[("SPY", "all")]["beta"]
    spy_ci = [1e3 * x for x in ident[("SPY", "all")]["ci90"]]
    sig_mask = (I["90% CI low"] > 0) | (I["90% CI high"] < 0)
    n_excl0 = int(sig_mask.sum())
    sig_names = [f"{LABELS[i]} {I.loc[i, 'identified rel. response, %/10bp']:+.2f}%" for i in I.index[sig_mask.to_numpy()]]
    rank_corr = float(pd.Series(I["identified rel. response, %/10bp"]).corr(pd.Series(I["day-before design, %/10bp"]),
                                                                           method="spearman"))
    D.index = [f"{s} {LABELS[s]}" for s in D.index]
    I.index = [f"{s} {LABELS[s]}" for s in I.index]
    T_all = D.join(I)
    T_all = T_all.sort_values("identified rel. response, %/10bp")
    t1 = Table("q3_sectors", "Sector rate sensitivity: descriptive betas and FOMC-identified responses", T_all,
               "Betas: percent sector return per +100bp in the 10Y yield (daily, Newey-West t). Partial betas hold the "
               "S&P 500 fixed. Identified response: percent return RELATIVE TO THE S&P 500 per +10bp move in the "
               "2Y yield driven by FOMC news (identification through heteroskedasticity, meeting-paired bootstrap). "
               "Sorted from most to least vulnerable to policy tightening.")
    save_table(t1)

    # ---------------------------------------------------------------- figures
    order = T_all.index[::-1]

    def draw_dots(fig, P, mode):
        a1, a2 = fig.subplots(1, 2, sharey=True)
        y = np.arange(len(order))
        v = T_all.loc[order]
        a1.errorbar(v["identified rel. response, %/10bp"], y,
                    xerr=[v["identified rel. response, %/10bp"] - v["90% CI low"],
                          v["90% CI high"] - v["identified rel. response, %/10bp"]],
                    fmt="o", ms=5, color=P["series"][0], ecolor=P["series"][0], elinewidth=1.2, capsize=0)
        a1.axvline(0, color=P["base"], lw=0.8)
        a1.set_title("Identified: FOMC-driven 2Y move (relative to S&P 500)")
        a1.set_xlabel("% per +10bp, 90% CI")
        a1.set_yticks(y, order)
        a1.grid(axis="y", visible=False)
        a2.scatter(v["partial beta, negative stock-bond corr"], y, s=30, color=P["series"][0],
                   edgecolor=P["surface"], linewidth=1.2, zorder=3, label="negative stock-bond corr. regime")
        a2.scatter(v["partial beta, positive stock-bond corr"], y, s=30, color=P["series"][1],
                   edgecolor=P["surface"], linewidth=1.2, zorder=3, label="positive stock-bond corr. regime")
        for yy, a, b in zip(y, v["partial beta, negative stock-bond corr"], v["partial beta, positive stock-bond corr"]):
            a2.plot([a, b], [yy, yy], color=P["context"], lw=1.0, zorder=2)
        a2.axvline(0, color=P["base"], lw=0.8)
        a2.set_title("Descriptive: partial beta to 10Y by regime")
        a2.set_xlabel("% per +100bp, S&P 500 held fixed")
        a2.legend(loc="lower right", fontsize=7.5)
        a2.grid(axis="y", visible=False)

    f1 = save_fig("q3_sectors", draw_dots, (9.6, 5.4), "Sector vulnerability to rate shocks",
                  "Left: causal under the stated identification assumptions. Right: co-movement only; the regime is "
                  "the sign of the rolling one-year stock-bond correlation known the day before.")
    roll = {}
    for s in ("XLU", "IYR", "XLF", "XLK"):
        d = R[[s, "SPY", "UST10Y"]].dropna()
        cv = d.rolling(252).cov()
        b = []
        for t in d.index[251::5]:
            C = cv.loc[t]
            S2 = C.loc[["UST10Y", "SPY"], ["UST10Y", "SPY"]].to_numpy()
            sy = C.loc[["UST10Y", "SPY"], s].to_numpy()
            b.append((t, 1e4 * np.linalg.solve(S2, sy)[0]))
        roll[s] = pd.Series(dict(b))

    def draw_roll(fig, P, mode):
        ax = fig.add_subplot(111)
        for k, s in enumerate(roll):
            ax.plot(roll[s].index, roll[s].values, color=P["series"][k], lw=1.4, label=f"{s} {LABELS[s]}")
            ax.annotate(s, (roll[s].index[-1], roll[s].values[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=P["ink2"], va="center")
        ax.axhline(0, color=P["base"], lw=0.8)
        ax.set_ylabel("% per +100bp (S&P 500 held fixed)")
        ax.legend(loc="lower left", ncols=4)
        ax.set_title("Rolling one-year partial beta of sector returns to the 10Y yield")

    f2 = save_fig("q3_rolling", draw_roll, (8.6, 3.4), "Rate betas move with the macro regime",
                  "Descriptive. Utilities and real estate lose when yields rise in most windows; financials gain; "
                  "technology's sign flips with the stock-bond regime.")
    sec = Section("q3", "Q3. Which sectors are most vulnerable to rate shocks?",
                  "Which sectors are most vulnerable to rate shocks?", "", "")
    sec.tables = [t1]
    sec.figures = [f1, f2]
    most = T_all.index[:3].tolist()
    least = T_all.index[-3:].tolist()
    nF, nC = ident[("_n", "all")]["fomc_days"], ident[("_n", "all")]["control_days"]
    desc = D.sort_values("partial beta (%/100bp)")
    desc_most, desc_least = desc.index[:4].tolist(), desc.index[-3:].tolist()
    pos_reg = D.sort_values("partial beta, positive stock-bond corr").index[:3].tolist()
    sec.design = (f"Daily sector ETF returns 2000-2026; {nF} scheduled FOMC announcement days and {nC} control days "
                  "(non-FOMC Tuesdays and Wednesdays more than two trading days from an announcement). Robustness: "
                  "the original day-before control design and an ex-zero-lower-bound sample.")
    kre_rel = ("Regional banks illustrate the gap between the two notions: they rise with yields on ordinary days "
               "(steeper curve, margins) but their policy-identified response is negative."
               if "KRE" in ids and D.loc[f"KRE {LABELS['KRE']}", "partial beta (%/100bp)"] > 0
               and I.loc[f"KRE {LABELS['KRE']}", "identified rel. response, %/10bp"] < 0 else "")
    sec.answer = (
        f"On the descriptive evidence, {', '.join(desc_most)} have the most negative rate betas holding the market "
        f"fixed, and the lowest betas in the positive stock-bond-correlation (inflation) regime are "
        f"{', '.join(pos_reg)}. At the other end, {', '.join(desc_least)} gain relative to the market when yields "
        "rise. The causal question, what a policy-driven rate rise does, is answered for the market as a whole but "
        f"only coarsely by sector. FOMC days raise the variance of the 2Y yield {vr:.1f}x relative to control days "
        f"(90% CI {vr_ci[0]:.1f}-{vr_ci[1]:.1f}; {vr_x:.1f}x outside the zero lower bound). That is enough to identify "
        f"the S&P 500's response, {spy_b:+.2f}% per +10bp policy-driven 2Y move (90% CI {spy_ci[0]:.2f} to "
        f"{spy_ci[1]:.2f}), but only {n_excl0} of {len(ids)} sector responses relative to the market have a 90% "
        f"interval excluding zero ({', '.join(sig_names) if sig_names else 'none'}). The point estimates rank "
        f"{', '.join(most)} as most hurt by hawkish surprises. " + kre_rel)
    sec.findings = [
        Finding("Q3", Claim.IDENTIFIED, "Rate-sensitive sectors under FOMC identification",
                "Relative responses per +10bp: " + ", ".join(
                    f"{i.split()[0]} {T_all.loc[i, 'identified rel. response, %/10bp']:+.2f}% "
                    f"[{T_all.loc[i, '90% CI low']:+.2f}, {T_all.loc[i, '90% CI high']:+.2f}]" for i in most) + ".",
                ["Valid only if non-policy shocks have the same variance on FOMC and pre-FOMC days.",
                 "FOMC surprises mix policy and the Fed's information about the economy.",
                 "Zero-lower-bound years contribute little identifying variation."]),
        Finding("Q3", Claim.DESCRIPTIVE, "Regime dependence of descriptive rate betas",
                "Partial 10Y beta (negative vs positive stock-bond correlation regime): " + ", ".join(
                    f"{i.split()[0]} {D.loc[i, 'partial beta, negative stock-bond corr']:+.1f} vs "
                    f"{D.loc[i, 'partial beta, positive stock-bond corr']:+.1f}" for i in D.index
                    if i.split()[0] in ("XLK", "IWF", "XLU", "XLF")) + " (% per 100bp).",
                ["Daily yield changes mix growth, inflation and term-premium news; betas are not causal."]),
    ]
    short = {"XLRE": "real estate", "IYR": "real estate", "ITB": "homebuilders", "XLU": "utilities", "KRE": "regional banks",
             "XLF": "financials", "XLE": "energy", "XLC": "communication services", "XLY": "discretionary",
             "XLP": "staples", "XLV": "health care", "XLK": "technology", "XLB": "materials", "XLI": "industrials",
             "IWF": "growth", "IWD": "value", "IWM": "small caps", "QQQ": "Nasdaq-100"}
    top_short = list(dict.fromkeys(short.get(i.split()[0], i) for i in desc_most))[:3]
    verdict_q3 = (", ".join(top_short).capitalize() + " on daily betas; FOMC identification is precise for the "
                  "market, not for most sectors.")
    sec.numbers = {"verdict": verdict_q3,
                   "var_ratio": vr, "var_ratio_ci": vr_ci, "var_ratio_exzlb": vr_x, "var_ratio_day_before": vr_db,
                   "spy_identified": spy_b, "spy_ci": spy_ci, "n_fomc": nF, "n_control": nC,
                   "n_sectors_ci_excludes_zero": n_excl0, "rank_corr_weekday_vs_daybefore": rank_corr}
    sec.robustness = [
        f"Excluding zero-lower-bound meetings (fed funds < 0.25%): variance ratio {vr_x:.1f}x; estimates in the "
        "ex-ZLB columns.",
        f"Day-before controls (Rigobon & Sack's design): variance ratio {vr_db:.1f}x; Spearman rank correlation "
        f"with the main estimates {rank_corr:.2f}.",
        "3-month bill yields are not a usable policy indicator here: their variance is not higher on FOMC days.",
        "The descriptive top-5% yield-rise-day returns and the 2013 / 2022 episode returns are in the table; in 2022 "
        "defensives outperformed despite negative rate betas because the shock was inflation, not rates alone.",
    ]
    return sec
