"""Q4. Can dynamic networks improve portfolio stress testing?

Out-of-sample "conditional stress prediction" contest. On every stress day t - a day on which a driver series
(S&P 500, 10Y yield, oil, Baa spread, VIX or the dollar) moves more than its expanding 98th percentile of absolute
daily moves - each model takes the driver's realized move as the scenario and predicts the same-day move of the
other 30 series and of five portfolios, using only data up to t-1:
    E[x_O | x_d = s] = Sigma_Od / Sigma_dd * s,   Var[x_O | x_d] = Sigma_OO - Sigma_Od^2 / Sigma_dd
Models (the "network" is the covariance / partial-correlation structure they imply):
    static     expanding full-history covariance (a static network)
    rolling    252- and 63-day windows
    ewma       lambda 0.94 and 0.97
    dcc        GJR-GARCH + DCC, re-estimated every two years, filtered daily
    regime     real-time 3-state HMM mixture covariance (filtered probabilities)
    stressed   covariance of past high-VIX days
    tail beta  regression on the driver's own past 5% tail days
Scoring: squared errors scaled by each series' standard deviation, 90% interval coverage, portfolio P&L errors,
Diebold-Mariano tests against the static model. A second test asks whether the LAGGED network (a rolling VAR)
predicts the next day's moves after a stress day better than "no further move".
Claims: PREDICTIVE (out-of-sample forecast comparison). The conditional expectations are not causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..claims import Claim
from ..network.var import fit_var
from ..portfolio.hedging import (DccCov, EwmaCov, RegimeCov, RollingCov, StaticCov, StressedCov)
from ..portfolio.portfolios import PORTFOLIOS
from ..stress.engine import STRESS_UNIVERSE
from .common import Finding, Section, Table, save_fig, save_table

DRIVERS = ["SPY", "UST10Y", "OIL", "BAA10Y", "VIX", "DXY"]
START = 750


def find_events(X: np.ndarray, names: list[str], q: float = 0.98) -> list[tuple[int, int]]:
    ev = []
    for d in DRIVERS:
        j = names.index(d)
        a = np.abs(X[:, j])
        for t in range(START, len(X)):
            if a[t] >= np.quantile(a[:t], q):
                ev.append((t, j))
    return sorted(ev)


def dm_test(d: np.ndarray, lags: int = 5) -> tuple[float, float]:
    from scipy import stats

    d = d[np.isfinite(d)]
    n = len(d)
    dm = d - d.mean()
    s = dm @ dm / n + 2 * sum((1 - k / (lags + 1)) * (dm[k:] @ dm[:-k] / n) for k in range(1, lags + 1))
    t = d.mean() / np.sqrt(s / n)
    return float(t), float(2 * stats.norm.sf(abs(t)))


def run(panel) -> Section:
    """Heavy part cached in outputs/research/cache/q4.pkl (keyed by universe, sample end and settings)."""
    import pickle

    from ..paths import OUTPUTS

    R = panel.returns_for(STRESS_UNIVERSE)
    sig = (tuple(R.columns), str(R.index[-1].date()), len(R), START, tuple(DRIVERS))
    cp = OUTPUTS / "research" / "cache" / "q4.pkl"
    cached = None
    if cp.exists():
        with open(cp, "rb") as f:
            cached = pickle.load(f)
        if cached.get("sig") != sig:
            cached = None
    if cached is None:
        cached = {"sig": sig, **_compute(panel, R)}
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump(cached, f)
    return _narrate(R, cached["E"], cached["lag_tab"])


def _compute(panel, R: pd.DataFrame) -> dict:
    names = list(R.columns)
    X = R.to_numpy(dtype=float)
    T, N = X.shape
    vix = panel.levels["VIX"].reindex(R.index).to_numpy()
    sd = X.std(axis=0)
    feat = [names.index(c) for c in ("SPY", "VIX", "UST10Y", "BAA10Y")]
    providers = {
        "static": StaticCov(X), "rolling 252d": RollingCov(X, 252), "rolling 63d": RollingCov(X, 63),
        "ewma 0.97": EwmaCov(X, 0.97), "ewma 0.94": EwmaCov(X, 0.94),
        "dcc": DccCov(X, min_train=START, refit_every=504),
        "regime (HMM)": RegimeCov(X, feat, n_states=3, min_train=START, refit_every=126),
        "stressed (high VIX)": StressedCov(X, vix),
    }
    events = find_events(X, names)
    port = {k: pd.Series(w).reindex(names).fillna(0.0).to_numpy() for k, w in PORTFOLIOS.items()
            if all(a in names for a in w)}
    rows = []
    for t, j in events:
        s = X[t, j]
        O = np.array([k for k in range(N) if k != j])
        y = X[t]
        preds = {}
        for m, prov in providers.items():
            S = prov.cov_at(t - 1)
            if not np.all(np.isfinite(S)):
                continue
            beta = S[O, j] / S[j, j]
            cv = np.clip(np.diag(S)[O] - S[O, j] ** 2 / S[j, j], 1e-16, None)
            preds[m] = (beta * s, cv)
        # tail beta from the driver's own past tail (matching the sign of today's move)
        past = X[:t]
        tail = past[:, j] <= np.quantile(past[:, j], 0.05) if s < 0 else past[:, j] >= np.quantile(past[:, j], 0.95)
        tb = past[tail][:, O].mean(axis=0) / past[tail][:, j].mean()
        resid = past[tail][:, O] - np.outer(past[tail][:, j], tb)
        preds["tail beta"] = (tb * s, resid.var(axis=0) + 1e-16)
        for m, (p, cv) in preds.items():
            e = y[O] - p
            z = e / sd[O]
            row = {"t": t, "date": R.index[t], "driver": names[j], "model": m, "mse_scaled": float(np.mean(z ** 2)),
                   "coverage90": float(np.mean(np.abs(e) <= 1.645 * np.sqrt(cv))),
                   "r2_vs_zero": float(1 - np.sum(e ** 2 / sd[O] ** 2) / np.sum(y[O] ** 2 / sd[O] ** 2))}
            full = np.zeros(N)
            full[O] = p
            full[j] = s
            for k, w in port.items():
                row[f"pnl_err_{k}"] = float(abs(w @ (np.expm1(full) - np.expm1(y))))
            rows.append(row)
    E = pd.DataFrame(rows)
    # ---------------------------------------------------------------- lagged network: next-day propagation
    lag_rows = []
    spy = names.index("SPY")
    fits = {}
    for t, j in events:
        if j != spy or t + 1 >= T:
            continue
        key = t // 63
        if key not in fits:
            fits[key] = fit_var(X[max(0, t - 500):t], p=1, ridge=0.0)
        f = fits[key]
        pred = f.const + f.A[0] @ X[t]
        lag_rows.append({"t": t, **{n: (X[t + 1, k] - pred[k]) ** 2 - X[t + 1, k] ** 2 for k, n in enumerate(names)}})
    L = pd.DataFrame(lag_rows).set_index("t")
    lag_gain = (-L.mean() / pd.Series({n: np.mean(X[[r + 1 for r in L.index], k] ** 2) for k, n in enumerate(names)}))
    lag_tab = pd.DataFrame({"next-day MSE reduction vs no-change (%)": 100 * lag_gain.sort_values(ascending=False)})
    from ..data.calendar import lead_lag_profile

    lag_tab["late reaction to SPY"] = lead_lag_profile(R, "SPY")["timing_signal"].reindex(lag_tab.index)
    return {"E": E, "lag_tab": lag_tab}


def _narrate(R: pd.DataFrame, E: pd.DataFrame, lag_tab: pd.DataFrame) -> Section:
    # ---------------------------------------------------------------- scoring
    summ = E.groupby("model").agg(events=("t", "size"), mse_scaled=("mse_scaled", "mean"),
                                  coverage90=("coverage90", "mean"), r2_vs_zero=("r2_vs_zero", "mean"))
    base = E[E["model"] == "static"].set_index(["t", "driver"])["mse_scaled"]
    dm_rows = {}
    for m in summ.index:
        x = E[E["model"] == m].set_index(["t", "driver"])["mse_scaled"]
        d = (x - base).dropna().to_numpy()
        tstat, p = dm_test(d)
        dm_rows[m] = {"rel. MSE vs static": float(x.mean() / base.reindex(x.index).mean()), "DM t": tstat, "DM p": p,
                      "share of events better than static": float(np.mean(d < 0))}
    summ = summ.join(pd.DataFrame(dm_rows).T).sort_values("mse_scaled")
    by_driver = E.pivot_table(index="driver", columns="model", values="mse_scaled", aggfunc="mean")
    by_driver = by_driver.div(by_driver["static"], axis=0)
    pnl_cols = [c for c in E.columns if c.startswith("pnl_err_")]
    pnl = E.groupby("model")[pnl_cols].mean() * 100
    pnl.columns = [c.replace("pnl_err_", "") for c in pnl.columns]
    pnl = pnl.div(pnl.loc["static"], axis=1)
    # periods: where do dynamic models gain?
    E["year"] = pd.to_datetime(E["date"]).dt.year
    yearly = E.pivot_table(index="year", columns="model", values="mse_scaled", aggfunc="mean")
    yearly_rel = yearly.div(yearly["static"], axis=0)
    n_year = E[E["model"] == "static"].groupby("year")["t"].size()

    t1 = Table("q4_summary", "Out-of-sample conditional stress prediction, all stress days", summ,
               "Stress days: a driver's absolute move exceeds its expanding 98th percentile (data from t-1 only). "
               "mse_scaled: mean squared error of the 30 non-shocked series' same-day moves, each scaled by its "
               "standard deviation. coverage90: share of realized moves inside the model's 90% conditional interval "
               "(nominal 0.90). r2_vs_zero: share of the cross-section explained relative to predicting no move. "
               "DM: Diebold-Mariano test of the per-event loss difference against the static network.", "{:.3f}")
    t2 = Table("q4_by_driver", "Relative MSE by scenario driver (static = 1.00)", by_driver, "", "{:.2f}")
    t3 = Table("q4_portfolio", "Relative portfolio P&L error (static = 1.00)", pnl,
               "Mean absolute error of the predicted same-day portfolio P&L, relative to the static network.", "{:.2f}")
    t4 = Table("q4_lagged", "Lagged network: next-day propagation after S&P 500 stress days", lag_tab.head(12),
               "Rolling 500-day VAR(1). Positive = the VAR's prediction of the day after the shock beats "
               "'no further move'. Late reaction: |corr(SPY today, x tomorrow)| - |corr(SPY today, x yesterday)|.",
               "{:.2f}")
    for t in (t1, t2, t3, t4):
        save_table(t)

    models_order = summ.index.tolist()

    def draw_bars(fig, P, mode):
        ax = fig.add_subplot(111)
        v = summ["rel. MSE vs static"].reindex(models_order)
        y = np.arange(len(v))
        cols = [P["series"][0] if m != "static" else P["context"] for m in v.index]
        ax.barh(y, v.to_numpy() - 1.0, left=1.0, height=0.55, color=cols)
        ax.axvline(1.0, color=P["base"], lw=0.8)
        for yy, (m, val) in zip(y, v.items()):
            ax.annotate(f"{val:.2f}", (val, yy), xytext=(4 if val >= 1 else -4, 0), textcoords="offset points",
                        ha="left" if val >= 1 else "right", va="center", fontsize=8, color=P["ink2"])
        ax.set_yticks(y, v.index)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(0.75, v.min() - 0.06), max(1.05, v.max() + 0.06))
        ax.set_xlabel("prediction error relative to the static network (lower is better)")
        ax.set_title(f"Same-day stress prediction on {E['t'].nunique()} stress days, 2010-2026")

    f1 = save_fig("q4_models", draw_bars, (7.6, 3.6), "Dynamic vs static networks in out-of-sample stress tests",
                  "Each bar is the mean scaled squared error of the predicted cross-section of moves on stress days, "
                  "relative to the static (expanding full-history) network.")
    show = [m for m in ("ewma 0.97", "dcc", "regime (HMM)", "rolling 63d") if m in yearly_rel.columns]

    def draw_years(fig, P, mode):
        ax = fig.add_subplot(111)
        for k, m in enumerate(show):
            ax.plot(yearly_rel.index, yearly_rel[m], marker="o", ms=4, lw=1.6, color=P["series"][k], label=m)
        ax.axhline(1.0, color=P["base"], lw=0.8)
        ax.set_ylabel("MSE relative to static")
        ax.legend(loc="upper left", ncols=len(show))
        ax.set_title("Where dynamic networks help: relative error by year (below 1 = better than static)")

    f2 = save_fig("q4_years", draw_years, (8.4, 3.2), "Gains by year",
                  "Relative error of four time-varying networks against the static network, by calendar year of the "
                  "stress day. Below 1: the time-varying network predicted the cross-section better.")
    sec = Section("q4", "Q4. Can dynamic networks improve portfolio stress testing?",
                  "Can dynamic networks improve portfolio stress testing?", "", "")
    sec.tables, sec.figures = [t1, t2, t3, t4], [f1, f2]
    best = summ.index[0]
    stat = summ.loc["static"]
    nev = E["t"].nunique()
    n_drv = E.groupby("driver")["t"].nunique().to_dict()
    sec.design = (f"{nev} stress days (driver moves beyond the expanding 98th percentile) from 2010 to 2026 across "
                  f"six drivers ({', '.join(f'{k} {v}' for k, v in n_drv.items())}); 31-series stress universe; every "
                  "model uses data up to the previous close only.")
    rb = summ.loc[best]
    dyn = summ.drop(index=["static", "tail beta"], errors="ignore")
    sig_better = dyn[(dyn["rel. MSE vs static"] < 1) & (dyn["DM p"] < 0.05)].index.tolist()
    worse = dyn[dyn["rel. MSE vs static"] > 1].index.tolist()
    lag_pos = lag_tab[lag_tab.iloc[:, 0] > 0]
    lag_top = lag_pos.index[:3].tolist()
    yr = yearly_rel[best].dropna() if best in yearly_rel else pd.Series(dtype=float)
    yr = yr[n_year.reindex(yr.index).fillna(0) >= 10]
    best_years = yr.nsmallest(2)
    worst_year = yr.idxmax() if len(yr) else None
    cov_lo, cov_hi = summ["coverage90"].min(), summ["coverage90"].max()
    verdict = ("Yes." if best != "static" and rb["DM p"] < 0.05 else
               "Not clearly." if best != "static" else "No.")
    lag_timing = ", ".join(f"{n} {lag_tab.loc[n, 'late reaction to SPY']:+.2f}" for n in lag_top)
    sec.answer = (
        f"{verdict} {len(sig_better)} of {len(dyn)} time-varying networks ({', '.join(sig_better)}) beat the static "
        "(expanding full-history) network on stress days with Diebold-Mariano p < 0.05"
        + (f"; {' and '.join(worse)} do worse" if worse else "") + ". The best, "
        f"{best}, changes the scaled prediction error of the same-day cross-section by "
        f"{100 * (rb['rel. MSE vs static'] - 1):+.0f}% (DM t = {rb['DM t']:.1f}), beats static on "
        f"{100 * rb['share of events better than static']:.0f}% of stress days, and lowers the error of predicted "
        f"portfolio P&L by {100 * (1 - pnl.loc[best].max()):.0f}-{100 * (1 - pnl.loc[best].min()):.0f}% across the five "
        "portfolios. The winners all weight recent data heavily, so they follow the current dependence structure"
        + (f"; covariances conditioned on past stress ({' and '.join(worse)}) do worse, consistent with crises "
           "differing from one another (the stock-bond correlation had opposite signs in 2008-2020 and in 2022). "
           if worse else ". ")
        + f"No model is well calibrated: their nominal 90% intervals cover only {100 * cov_lo:.0f}-{100 * cov_hi:.0f}% "
        f"of realized stress-day moves ({100 * stat['coverage90']:.0f}% for static). "
        + (f"Among years with at least 10 stress days, {best} gains most in "
           f"{', '.join(str(y) for y in best_years.index)} and least in {worst_year}. " if len(best_years) else "")
        + "The lagged network adds little: a rolling VAR predicts the day after an S&P 500 shock better than 'no "
        f"further move' mainly for {', '.join(lag_top)}, series that absorb part of each day's S&P 500 news only the "
        f"next day (late-reaction scores {lag_timing}: earlier closes, stale marks or foreign underlying markets). "
        "That multi-day 'propagation' is a timing effect.")
    sec.findings = [
        Finding("Q4", Claim.PREDICTIVE, f"Best stress-day predictor: {best}",
                f"Relative MSE {rb['rel. MSE vs static']:.2f} (DM t {rb['DM t']:.1f}, p {rb['DM p']:.3f}); coverage "
                f"{100 * rb['coverage90']:.0f}% vs {100 * stat['coverage90']:.0f}% static.",
                ["Stress days cluster in time, so the effective number of independent events is smaller than the count.",
                 "Conditional expectations describe co-movement given the scenario; they do not identify causes."]),
        Finding("Q4", Claim.PREDICTIVE, "Lagged propagation is mostly timing",
                "Next-day MSE reduction from the VAR: " + ", ".join(
                    f"{n} {v:+.0f}%" for n, v in lag_tab['next-day MSE reduction vs no-change (%)'].head(4).items()) + ".",
                ["A predictable catch-up of stale or early-closing prices is useful for risk reporting but is not "
                 "contagion."]),
        Finding("Q4", Claim.PREDICTIVE, "Crisis-conditioned covariances do worse than static",
                ", ".join(f"{m}: relative MSE {summ.loc[m, 'rel. MSE vs static']:.2f} (DM p {summ.loc[m, 'DM p']:.2f})"
                          for m in worse) + "." if worse else "None of the time-varying models does worse than static.",
                ["Stressed and regime covariances import the correlations of past crises (flight to quality); "
                 "2022-type inflation shocks had different ones."]),
    ]
    sec.numbers = {"n_events": int(nev), "events_by_driver": n_drv, "summary": summ.to_dict(),
                   "yearly_rel": yearly_rel.round(3).to_dict()}
    sec.robustness = ["Per-driver results (table) show where the dynamic models gain: by construction the scenario "
                      "driver's own variance cancels, so the gains come from the correlation structure.",
                      "The tail-beta model uses only the driver's own past tail days; with few tail observations it "
                      "is noisy, which is visible in its error."]
    return sec
