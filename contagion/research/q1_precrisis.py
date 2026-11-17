"""Q1. Does cross-asset connectedness increase before crises?

Design (fixed before looking at outcomes)
* Crisis onsets: mechanical bear-market peaks of SPY total return (decline >= 15% confirms a peak, rally >= 15%
  confirms a trough), keeping only peaks >= 125 trading days after the previous trough so that bear-market
  rallies inside one crisis are not counted as new crises. Robustness: 10% / 10% dating.
* Indicators (all computed from data up to t only): DY total connectedness from 500- and 250-day rolling VARs,
  TVP-VAR connectedness, the dynamic (lagged) part of connectedness, 63-day average correlation, absorption ratio,
  Granger-network density.
* Event study: change in each indicator from 250 / 63 trading days before the peak to the peak, averaged over
  episodes; p-value from 5,000 placebo sets of random non-crisis dates.
* Prediction: out-of-sample logistic models for "SPY falls >= 10% from today's close within 63 trading days",
  expanding window, refit monthly, AUC with a block bootstrap: baseline (VIX, realized vol, momentum) vs baseline +
  connectedness.
Claims: DESCRIPTIVE (event study) and PREDICTIVE (out-of-sample classification); nothing here is causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..claims import Claim
from ..models import RISK_OFF_SIGN, load_dynamic, load_granger
from ..network.correlation import average_correlation, cov_to_corr, rolling_cov
from ..paths import MODELS
from .common import Finding, Section, Table, crisis_episodes, save_fig, save_table, shade_episodes

PRE = 250
POST = 120


def onsets(panel, down=0.15, up=0.15, min_sep=125, start="2003-04-15") -> pd.DataFrame:
    ep = crisis_episodes(panel, down, up, start)
    cal = panel.returns["SPY"].dropna().loc[start:].index
    keep, last_trough = [], None
    for _, e in ep.iterrows():
        if last_trough is None or cal.get_loc(e["peak"]) - cal.get_loc(last_trough) >= min_sep:
            keep.append(e)
        last_trough = e["trough"]
    return pd.DataFrame(keep).reset_index(drop=True)


def indicators(panel) -> pd.DataFrame:
    R = pd.read_parquet(MODELS / "R_macro.parquet")
    idx = R.index
    d500, d250, tv = load_dynamic("dy_macro_w500"), load_dynamic("dy_macro_w250"), load_dynamic("tvp_macro")
    cs = pd.read_parquet(MODELS / "corrstats_macro.parquet")
    g = load_granger("macro", "pairwise")
    X = pd.DataFrame(index=idx)
    X["TCI rolling 500d"] = d500.total.reindex(idx, method="ffill")
    X["TCI rolling 250d"] = d250.total.reindex(idx, method="ffill")
    X["TCI TVP-VAR"] = tv.total.reindex(idx)
    X.loc[X.index[:250 + 1], "TCI TVP-VAR"] = np.nan  # TVP burn-in
    X["Dynamic TCI 500d"] = d500.total_dynamic.reindex(idx, method="ffill")
    # average pairwise correlation after orienting every series so that its risk-off move is a fall: a rise in
    # this number means assets move together in the risk-on / risk-off direction
    Ro = R * pd.Series(RISK_OFF_SIGN)[R.columns]
    X["Avg corr 63d"] = average_correlation(cov_to_corr(rolling_cov(Ro.to_numpy(), 63)))
    X["Absorption 250d"] = cs["absorption_250"].reindex(idx)
    X["Granger density"] = g["density"].reindex(idx, method="ffill")
    return X


def event_changes(X: pd.DataFrame, ons: pd.DataFrame, lag: int) -> pd.DataFrame:
    rows = {}
    for _, e in ons.iterrows():
        i = X.index.get_indexer([e["peak"]], method="ffill")[0]
        if i - lag < 0:
            continue
        rows[e["peak"].date()] = X.iloc[i] - X.iloc[i - lag]
    return pd.DataFrame(rows).T


def placebo(X: pd.DataFrame, ons: pd.DataFrame, lag: int, n_rep: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    T = len(X)
    excl = np.zeros(T, dtype=bool)
    for _, e in ons.iterrows():
        a = X.index.get_indexer([e["peak"]], method="ffill")[0]
        b = X.index.get_indexer([e["trough"]], method="ffill")[0]
        excl[max(0, a - PRE): min(T, b + 63)] = True
    out = {}
    for c in X.columns:
        v = X[c].to_numpy()
        ok = np.flatnonzero(~excl & np.isfinite(v) & np.isfinite(np.roll(v, lag)) & (np.arange(T) >= lag))
        actual = event_changes(X[[c]], ons, lag)[c].dropna()
        n = len(actual)
        if n == 0 or len(ok) < 10 * n:
            continue
        ch = v[ok] - v[ok - lag]
        sims = np.array([ch[rng.choice(len(ch), n, replace=False)].mean() for _ in range(n_rep)])
        out[c] = {"n_episodes": n, "mean_change": float(actual.mean()), "n_positive": int((actual > 0).sum()),
                  "placebo_mean": float(sims.mean()), "p_one_sided": float((sims >= actual.mean()).mean()),
                  "placebo_sd": float(sims.std())}
    return out


def placebo_forward(X: pd.DataFrame, ons: pd.DataFrame, h: int = 63, n_rep: int = 5000, seed: int = 2) -> dict:
    """Does the indicator rise MORE than at random times over the h days AFTER the peak?"""
    rng = np.random.default_rng(seed)
    T = len(X)
    excl = np.zeros(T, dtype=bool)
    peaks = []
    for _, e in ons.iterrows():
        a = X.index.get_indexer([e["peak"]], method="ffill")[0]
        b = X.index.get_indexer([e["trough"]], method="ffill")[0]
        excl[max(0, a - PRE): min(T, b + 63)] = True
        peaks.append(a)
    peaks = np.array(peaks)
    out = {}
    for c in X.columns:
        v = X[c].to_numpy()
        act = np.array([v[p + h] - v[p] for p in peaks if p + h < T and np.isfinite(v[p])])
        ok = np.flatnonzero(~excl & (np.arange(T) + h < T))
        ok = ok[np.isfinite(v[ok]) & np.isfinite(v[ok + h])]
        if len(act) == 0 or len(ok) < 10 * len(act):
            continue
        ch = v[ok + h] - v[ok]
        sims = np.array([ch[rng.choice(len(ch), len(act), replace=False)].mean() for _ in range(n_rep)])
        out[c] = {"mean_change": float(act.mean()), "n_positive": int((act > 0).sum()), "n": int(len(act)),
                  "p_one_sided": float((sims >= act.mean()).mean())}
    return out


def event_paths(X: pd.DataFrame, ons: pd.DataFrame) -> dict:
    ks = np.arange(-PRE, POST + 1)
    paths = {}
    for c in X.columns:
        mats = []
        v = X[c].to_numpy()
        for _, e in ons.iterrows():
            i = X.index.get_indexer([e["peak"]], method="ffill")[0]
            if i - PRE < 0 or not np.isfinite(v[i - PRE]):
                continue
            row = np.full(len(ks), np.nan)
            sel = (i + ks >= 0) & (i + ks < len(v))
            row[sel] = v[i + ks[sel]] - v[i - PRE]
            mats.append(row)
        paths[c] = np.array(mats)
    return {"k": ks, "paths": paths}


def placebo_band(X: pd.DataFrame, ons: pd.DataFrame, col: str, n_rep: int = 1000, seed: int = 1):
    rng = np.random.default_rng(seed)
    ks = np.arange(-PRE, POST + 1)
    v = X[col].to_numpy()
    T = len(v)
    excl = np.zeros(T, dtype=bool)
    for _, e in ons.iterrows():
        a = X.index.get_indexer([e["peak"]], method="ffill")[0]
        b = X.index.get_indexer([e["trough"]], method="ffill")[0]
        excl[max(0, a - PRE): min(T, b + 63)] = True
    ok = np.flatnonzero(~excl & (np.arange(T) - PRE >= 0) & (np.arange(T) + POST < T))
    ok = ok[np.isfinite(v[ok - PRE])]
    n = len(event_paths(X[[col]], ons)["paths"][col])
    sims = np.empty((n_rep, len(ks)))
    for r in range(n_rep):
        d = rng.choice(ok, n, replace=False)
        sims[r] = np.nanmean(v[d[:, None] + ks[None, :]] - v[d - PRE][:, None], axis=0)
    return np.nanquantile(sims, [0.05, 0.95], axis=0)


def prediction_test(panel, X: pd.DataFrame, horizon: int = 63, dd: float = 0.10, start_oos: str = "2009-01-01",
                    seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    spy = panel.returns["SPY"].reindex(X.index).fillna(0.0)
    cum = spy.cumsum().to_numpy()
    T = len(cum)
    fut_min = np.full(T, np.nan)
    for t in range(T - horizon):
        fut_min[t] = (cum[t + 1: t + 1 + horizon] - cum[t]).min()
    y = pd.Series((fut_min <= np.log(1 - dd)).astype(float), index=X.index).where(np.isfinite(fut_min))
    vix = panel.levels["VIX"].reindex(X.index)
    F = pd.DataFrame({"log_vix": np.log(vix), "log_rv21": np.log(spy.rolling(21).std() * np.sqrt(252)),
                      "mom63": spy.rolling(63).sum()}, index=X.index)
    conn = X[["TCI rolling 500d", "TCI TVP-VAR", "Avg corr 63d", "Absorption 250d"]]
    sets = {"baseline (VIX, RV, momentum)": list(F.columns),
            "baseline + connectedness": list(F.columns) + list(conn.columns),
            "connectedness only": list(conn.columns)}
    D = pd.concat([F, conn, y.rename("y")], axis=1).dropna()
    dates = D.index
    oos_start = dates.searchsorted(pd.Timestamp(start_oos))
    preds = {k: np.full(len(D), np.nan) for k in sets}
    month = dates.to_period("M")
    refits = [i for i in range(oos_start, len(D)) if i == oos_start or month[i] != month[i - 1]]
    for j, i in enumerate(refits):
        end = refits[j + 1] if j + 1 < len(refits) else len(D)
        train = D.iloc[: max(0, i - horizon)]  # labels need `horizon` days of future: drop the last ones
        if train["y"].nunique() < 2:
            continue
        for k, cols in sets.items():
            mu, sd = train[cols].mean(), train[cols].std()
            m = LogisticRegression(C=1.0, max_iter=1000).fit((train[cols] - mu) / sd, train["y"])
            preds[k][i:end] = m.predict_proba((D[cols].iloc[i:end] - mu) / sd)[:, 1]
    yo = D["y"].to_numpy()[oos_start:]
    out = {"n_oos": int(len(yo)), "positive_rate": float(np.nanmean(yo))}
    auc = {k: float(roc_auc_score(yo, v[oos_start:])) for k, v in preds.items()}
    out["auc"] = auc
    rng = np.random.default_rng(seed)
    n, B = len(yo), 126
    diffs = []
    pa, pb = preds["baseline + connectedness"][oos_start:], preds["baseline (VIX, RV, momentum)"][oos_start:]
    for _ in range(1000):
        starts = rng.integers(0, n - B, size=n // B + 1)
        ii = np.concatenate([np.arange(s, s + B) for s in starts])[:n]
        if yo[ii].min() == yo[ii].max():
            continue
        diffs.append(roc_auc_score(yo[ii], pa[ii]) - roc_auc_score(yo[ii], pb[ii]))
    out["auc_diff_ci90"] = [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))]
    out["auc_diff"] = auc["baseline + connectedness"] - auc["baseline (VIX, RV, momentum)"]
    return out


def vol_regression(panel, X: pd.DataFrame) -> pd.DataFrame:
    """log future 63d realized vol on today's connectedness, controlling for today's vol (Newey-West, 63 lags)."""
    import statsmodels.api as sm

    spy = panel.returns["SPY"].reindex(X.index)
    rv_f = np.log(spy[::-1].rolling(63).std()[::-1].shift(-1) * np.sqrt(252))
    rows = {}
    for c in ["TCI rolling 500d", "TCI TVP-VAR", "Avg corr 63d", "Absorption 250d"]:
        D = pd.DataFrame({"y": rv_f, "x": (X[c] - X[c].mean()) / X[c].std(),
                          "log_rv21": np.log(spy.rolling(21).std() * np.sqrt(252)),
                          "log_vix": np.log(panel.levels["VIX"].reindex(X.index))}).dropna()
        for spec, cols in (("alone", ["x"]), ("with vol controls", ["x", "log_rv21", "log_vix"])):
            res = sm.OLS(D["y"], sm.add_constant(D[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 63})
            rows[(c, spec)] = {"coef_per_sd": res.params["x"], "t_NW": res.tvalues["x"], "R2": res.rsquared}
    return pd.DataFrame(rows).T


def run(panel) -> Section:
    X = indicators(panel)
    ons = onsets(panel)
    ons10 = onsets(panel, 0.10, 0.10)
    ons = ons[ons["peak"] >= X["TCI rolling 500d"].first_valid_index() + pd.Timedelta(days=365)].reset_index(drop=True)
    ons10 = ons10[ons10["peak"] >= X["TCI rolling 500d"].first_valid_index() + pd.Timedelta(days=365)].reset_index(drop=True)
    pl250, pl63 = placebo(X, ons, 250), placebo(X, ons, 63)
    pl250_10 = placebo(X, ons10, 250)
    fwd = placebo_forward(X, ons, 63)
    # Benjamini-Hochberg across all 14 pre-peak tests (7 indicators x 2 windows)
    from ..network.granger import bh_fdr

    keys = [(c, w) for c in X.columns for w in (250, 63)]
    pv = np.array([(pl250 if w == 250 else pl63)[c]["p_one_sided"] for c, w in keys])
    m = len(pv)
    order = np.argsort(pv)
    q_adj = np.empty(m)
    q_adj[order] = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    bh = {f"{c} ({w}d)": float(q) for (c, w), q in zip(keys, np.minimum(q_adj, 1.0))}
    any_bh = bool(bh_fdr(pv, 0.10).any())
    ev = event_changes(X, ons, 250)
    paths = event_paths(X, ons)
    pred = prediction_test(panel, X)
    vr = vol_regression(panel, X)
    # contemporaneous co-movement of connectedness with volatility
    spy = panel.returns["SPY"].reindex(X.index)
    rv63 = spy.rolling(63).std()
    corr_vol = {c: float(X[c].corr(rv63)) for c in X.columns}

    sec = Section("q1", "Q1. Does cross-asset connectedness increase before crises?",
                  "Does cross-asset connectedness increase before crises?", "", "")
    sec.design = (f"{len(ons)} crisis onsets dated mechanically from SPY total return (15% decline confirms a peak, "
                  f"15% rally a trough, peaks at least 125 trading days after the previous trough): "
                  + ", ".join(str(d.date()) for d in ons["peak"]) +
                  f". Robustness: 10%/10% dating gives {len(ons10)} onsets. Indicators use data up to each date only. "
                  "Placebo p-values compare the average pre-peak change with 5,000 random sets of non-crisis dates.")
    tab = []
    for c in X.columns:
        a, b, d = pl250.get(c, {}), pl63.get(c, {}), pl250_10.get(c, {})
        f = fwd.get(c, {})
        tab.append({"indicator": c, "episodes": a.get("n_episodes"), "mean change 250d->peak": a.get("mean_change"),
                    "episodes up": a.get("n_positive"), "placebo p (250d)": a.get("p_one_sided"),
                    "BH q (250d)": bh[f"{c} (250d)"],
                    "mean change 63d->peak": b.get("mean_change"), "placebo p (63d)": b.get("p_one_sided"),
                    "BH q (63d)": bh[f"{c} (63d)"],
                    "placebo p (250d, 10% dating)": d.get("p_one_sided"),
                    "mean change peak->+63d": f.get("mean_change"), "placebo p (after peak)": f.get("p_one_sided"),
                    "corr with 63d realized vol": corr_vol[c]})
    T1 = pd.DataFrame(tab).set_index("indicator")
    t1 = Table("q1_event_study", "Pre-crisis change in connectedness vs placebo", T1,
               "Change in each indicator from 250 (63) trading days before the bear-market peak to the peak, averaged "
               "over onsets. One-sided placebo p-value: share of 5,000 random non-crisis date sets with an equal or "
               "larger average rise. BH q: Benjamini-Hochberg adjusted across the 14 pre-peak tests. The 'after peak' "
               "columns test the rise over the 63 trading days after the peak the same way.")
    ev_tab = ev[["TCI rolling 500d", "TCI TVP-VAR", "Avg corr 63d", "Absorption 250d"]].copy()
    t2 = Table("q1_episodes", "Change from 250 trading days before each peak to the peak", ev_tab,
               "Points of TCI (percent), correlation units, absorption-ratio units.")
    auc = pd.DataFrame({"AUC (out of sample)": pred["auc"]})
    t3 = Table("q1_prediction", "Out-of-sample prediction of a >=10% fall within 63 trading days", auc,
               f"Expanding-window logistic regressions refit monthly, OOS {pred['n_oos']} days from 2009 "
               f"(base rate {100 * pred['positive_rate']:.1f}%). AUC difference (with - without connectedness) "
               f"{pred['auc_diff']:+.3f}, 90% block-bootstrap interval [{pred['auc_diff_ci90'][0]:+.3f}, "
               f"{pred['auc_diff_ci90'][1]:+.3f}].", "{:.3f}")
    t4 = Table("q1_vol_regression", "Connectedness and future 63-day volatility (Newey-West t-stats)", vr,
               "Dependent variable: log realized SPY volatility over the next 63 days. Coefficient per one standard "
               "deviation of the indicator.", "{:.3f}")
    for t in (t1, t2, t3, t4):
        save_table(t)
    sec.tables = [t1, t2, t3, t4]

    # ---------------------------------------------------------------- figures
    def draw_ts(fig, P, mode):
        ax = fig.add_subplot(111)
        shade_episodes(ax, ons, P)
        ax.plot(X.index, X["TCI TVP-VAR"], color=P["series"][1], lw=1.0, alpha=0.9, label="TVP-VAR (daily, filtered)")
        ax.plot(X.index, X["TCI rolling 500d"], color=P["series"][0], lw=1.8, label="Rolling 500-day VAR")
        ax.set_ylabel("Total connectedness, %")
        ax.legend(loc="upper left", ncols=2)
        ax.set_title("Total connectedness of the 15-asset macro network (shaded: bear markets, peak to trough)")

    f1 = save_fig("q1_tci", draw_ts, (8.6, 3.4), "Total connectedness over time",
                  "Diebold-Yilmaz total connectedness index (TCI): share of 10-day forecast-error variance due to "
                  "other assets. Shading marks the mechanically dated bear markets.")
    cols = ["TCI rolling 500d", "TCI TVP-VAR", "Avg corr 63d", "Absorption 250d"]
    bands = {c: placebo_band(X, ons, c) for c in cols}

    def draw_ev(fig, P, mode):
        axs = fig.subplots(1, 4, sharex=True)
        k = paths["k"]
        for ax, c in zip(axs, cols):
            M = paths["paths"][c]
            lo, hi = bands[c]
            ax.fill_between(k, lo, hi, color=P["context"], alpha=0.35, lw=0, label="placebo 90% band")
            for row in M:
                ax.plot(k, row, color=P["context"], lw=0.7, alpha=0.9)
            ax.plot(k, np.nanmean(M, axis=0), color=P["series"][0], lw=2.0, label="mean of onsets")
            ax.axvline(0, color=P["base"], lw=0.8)
            ax.axhline(0, color=P["base"], lw=0.8)
            ax.set_title(c, fontsize=9.5)
            ax.set_xlabel("trading days from peak")
        axs[0].set_ylabel("change since 250 days before peak")
        axs[0].legend(loc="upper left", fontsize=7.5)

    f2 = save_fig("q1_event", draw_ev, (10.5, 3.2), "Event study around bear-market peaks",
                  "Thin gray lines: individual onsets; blue: their mean; gray band: 90% range of the same mean over "
                  "1,000 random sets of non-crisis dates. Day 0 = bear-market peak.")
    sec.figures = [f1, f2]

    # ---------------------------------------------------------------- answer
    key = "TCI rolling 500d"
    p_main = pl250[key]["p_one_sided"]
    up_n, n = pl250[key]["n_positive"], pl250[key]["n_episodes"]
    tv_p = pl250["TCI TVP-VAR"]["p_one_sided"]
    best = min(((c, w, (pl250 if w == 250 else pl63)[c]["p_one_sided"]) for c, w in keys), key=lambda x: x[2])
    min_q = min(bh.values())
    fwd_sig = [c for c, v in fwd.items() if v["p_one_sided"] < 0.05]
    vr_alone = vr.xs("alone", level=1)["t_NW"]
    vr_ctrl = vr.xs("with vol controls", level=1)["t_NW"]
    sec.answer = (
        ("No, not reliably. " if not any_bh else "Only weakly. ")
        + f"Rolling-VAR connectedness rose over the year before {up_n} of {n} bear-market peaks (placebo p = "
        f"{p_main:.2f}; TVP-VAR p = {tv_p:.2f}). The smallest of the 14 pre-peak p-values is {best[2]:.2f} "
        f"({best[0]}, {best[1]}-day window), and after Benjamini-Hochberg correction the smallest q-value is "
        f"{min_q:.2f}: a hint of a rise in the last three months, not evidence of a systematic build-up. "
        "Connectedness does rise with the crisis itself: over the 63 days after the peak "
        + (", ".join(f"{c} ({fwd[c]['mean_change']:+.2f}, p = {fwd[c]['p_one_sided']:.2f})" for c in fwd_sig)
           if fwd_sig else "no indicator rises significantly more than at random times")
        + ". It is a volatility-state variable: alone it predicts next-quarter volatility (Newey-West t "
        f"{vr_alone.min():.1f} to {vr_alone.max():.1f}), but once today's realized volatility and VIX are controlled "
        f"for, the t-statistics fall to {vr_ctrl.min():.1f} to {vr_ctrl.max():.1f}. Out of sample, adding "
        "connectedness to a VIX / realized-vol / momentum model does not improve the prediction of a >=10% fall "
        f"within three months (AUC {pred['auc']['baseline (VIX, RV, momentum)']:.2f} -> "
        f"{pred['auc']['baseline + connectedness']:.2f}; difference 90% interval {pred['auc_diff_ci90'][0]:+.2f} to "
        f"{pred['auc_diff_ci90'][1]:+.2f}).")
    sec.findings = [
        Finding("Q1", Claim.DESCRIPTIVE, "No systematic pre-crisis rise in connectedness",
                f"Average change in rolling TCI from 250 days before the peak to the peak: "
                f"{pl250[key]['mean_change']:+.1f} points vs a placebo mean of {pl250[key]['placebo_mean']:+.1f} "
                f"(p = {p_main:.2f}); smallest BH-adjusted q across 14 pre-peak tests {min_q:.2f}.",
                [f"Only {n} independent onsets: low power; a modest true effect could be missed.",
                 "Rolling 500-day windows react slowly; the TVP-VAR and 63-day correlation are the faster checks."]),
        Finding("Q1", Claim.DESCRIPTIVE, "Connectedness rises during crises and tracks volatility",
                "Change over the 63 days after the peak (placebo p): "
                + ", ".join(f"{c} {v['mean_change']:+.2f} ({v['p_one_sided']:.2f})" for c, v in fwd.items()) + ".",
                ["Higher volatility mechanically raises measured correlation (Forbes-Rigobon), so part of the "
                 "co-movement with volatility is a measurement effect."]),
        Finding("Q1", Claim.PREDICTIVE, "No incremental out-of-sample value beyond volatility",
                f"OOS AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in pred["auc"].items()) + "; future-volatility "
                f"regressions lose significance once current volatility is controlled for.",
                ["Crash labels overlap heavily in time (63-day windows), so the effective sample is small.",
                 "An AUC below 0.5 for the connectedness model means the 2008-era relation did not carry forward."]),
    ]
    sec.numbers = {"onsets": [str(d.date()) for d in ons["peak"]], "placebo_250": pl250, "placebo_63": pl63,
                   "placebo_250_10pct": pl250_10, "placebo_after_peak": fwd, "bh_q": bh, "prediction": pred,
                   "corr_with_vol": corr_vol, "n_onsets_10pct": len(ons10)}
    sec.robustness = [
        f"10%/10% crisis dating ({len(ons10)} onsets): placebo p for rolling TCI = "
        f"{pl250_10.get(key, {}).get('p_one_sided', float('nan')):.2f}.",
        f"63-day pre-window: placebo p for rolling TCI = {pl63[key]['p_one_sided']:.2f}, for 63-day average "
        f"correlation = {pl63['Avg corr 63d']['p_one_sided']:.2f}.",
        "Granger-network density and the dynamic (lagged) part of TCI are included in the table; they are the "
        "only indicators that measure lead-lag structure rather than co-movement.",
    ]
    return sec
