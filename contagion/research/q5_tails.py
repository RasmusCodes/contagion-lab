"""Q5. Are correlations more informative during tail events?

"Informative" is tested in three ways, on synchronous instruments (all priced at the NYSE close, 2007-2026):
  1. DIFFERENT: correlation with the S&P 500 on its worst / best q% of days vs what a Gaussian with the same
     overall dependence implies (conditioning on a tail mechanically changes correlation even when dependence
     is constant, so the Gaussian benchmark is the right null); empirical tail dependence and copula fits.
  2. CONTAGION OR INTERDEPENDENCE: Forbes-Rigobon heteroskedasticity-adjusted crisis correlations vs the
     pre-crisis year, per bear market.
  3. PREDICTIVE: each year, which correlation matrix estimated on PAST data best predicts the correlations
     realized on the coming year's tail days (market's worst 10% of days): full sample, last 252 days, or past
     tail days only? And which best predicts a multi-asset portfolio's volatility on those days?
Claims: DESCRIPTIVE (1, 2) and PREDICTIVE (3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..claims import Claim
from ..network.copula import fit_bivariate
from ..network.correlation import fisher_z_test, forbes_rigobon_adjust
from ..network.tail import gaussian_tail_dependence
from .common import Finding, Section, Table, crisis_episodes, save_fig, save_table

PAIRS = ["EFA", "EEM", "XLF", "HYG", "LQD", "USO", "DBC", "IEF", "TLT", "GLD", "UUP", "FXY"]
SYNC = ["SPY", "IWM", "EFA", "EEM", "XLF", "SHY", "IEF", "TLT", "LQD", "HYG", "UUP", "FXY", "USO", "GLD", "DBC", "VIX"]
PORT = {"SPY": 0.35, "EFA": 0.15, "EEM": 0.05, "IEF": 0.20, "TLT": 0.05, "LQD": 0.07, "HYG": 0.05, "GLD": 0.05,
        "DBC": 0.03}


def conditional_corr(x, y, qs, n_sim=400_000, seed=11) -> pd.DataFrame:
    """corr(x, y | x in its lower q tail) for q < 0.5, upper tail for q > 0.5, with a Gaussian benchmark that has
    the sample's normal-scores correlation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rx = stats.norm.ppf(stats.rankdata(x) / (n + 1))
    ry = stats.norm.ppf(stats.rankdata(y) / (n + 1))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_sim)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_sim)
    rows = []
    for q in qs:
        m = x <= np.quantile(x, q) if q < 0.5 else x >= np.quantile(x, q)
        mm = z1 <= np.quantile(z1, q) if q < 0.5 else z1 >= np.quantile(z1, q)
        rows.append({"q": q, "n": int(m.sum()), "corr": float(np.corrcoef(x[m], y[m])[0, 1]),
                     "gauss": float(np.corrcoef(z1[mm], z2[mm])[0, 1])})
    out = pd.DataFrame(rows).set_index("q")
    out.attrs["rho_ns"] = rho
    return out


def boot_ci_condcorr(x, y, q, n_boot=500, block=21, seed=3):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        st = rng.integers(0, n - block, size=n // block + 1)
        ii = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        xb, yb = x[ii], y[ii]
        m = xb <= np.quantile(xb, q) if q < 0.5 else xb >= np.quantile(xb, q)
        vals.append(np.corrcoef(xb[m], yb[m])[0, 1])
    return np.quantile(vals, [0.05, 0.95])


def oos_tail_corr(R: pd.DataFrame, q: float = 0.10, min_tail: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, prow = [], []
    w = pd.Series(PORT).reindex(R.columns).fillna(0.0).to_numpy()
    iu = np.triu_indices(R.shape[1], 1)
    for y in range(2010, R.index[-1].year + 1):
        past, fut = R.loc[: f"{y - 1}-12-31"], R.loc[f"{y}-01-01": f"{y}-12-31"]
        thr = past["SPY"].quantile(q)
        ft = fut[fut["SPY"] <= thr]
        if len(ft) < min_tail or len(past) < 500:
            continue
        preds = {"full history": past.corr(), "last 252 days": past.iloc[-252:].corr(),
                 "past tail days": past[past["SPY"] <= thr].corr(), "past calm days": past[past["SPY"] > thr].corr()}
        tgt = ft.corr().to_numpy()
        tgt_all = fut.corr().to_numpy()
        sd_t = ft.std().to_numpy()
        real_vol = float((np.expm1(ft) * w).sum(axis=1).std())
        for k, C in preds.items():
            Cm = C.to_numpy()
            rows.append({"year": y, "predictor": k, "n_tail_days": len(ft),
                         "MAE vs tail-day corr": float(np.mean(np.abs(Cm[iu] - tgt[iu]))),
                         "MAE vs all-day corr": float(np.mean(np.abs(Cm[iu] - tgt_all[iu])))})
            pv = float(np.sqrt(w @ (Cm * np.outer(sd_t, sd_t)) @ w))
            prow.append({"year": y, "predictor": k, "abs. error, portfolio vol on tail days (%)":
                         100 * abs(pv - real_vol) / real_vol, "predicted/realized": pv / real_vol})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def forbes_rigobon_table(panel, R: pd.DataFrame) -> pd.DataFrame:
    ep = crisis_episodes(panel, 0.15, 0.15, str(R.index[0].date()))
    rows = []
    for _, e in ep.iterrows():
        a = R.index.get_indexer([e["peak"]], method="ffill")[0]
        b = R.index.get_indexer([e["trough"]], method="ffill")[0]
        if b - a < 15 or a < 250:
            continue
        calm, cr = R.iloc[a - 250:a], R.iloc[a:b + 1]
        vr = cr["SPY"].var() / calm["SPY"].var()
        for x in PAIRS:
            rc, rn = cr["SPY"].corr(cr[x]), calm["SPY"].corr(calm[x])
            ra = forbes_rigobon_adjust(rc, cr["SPY"].var(), calm["SPY"].var())
            zu, pu = fisher_z_test(rc, len(cr), rn, len(calm))
            za, pa = fisher_z_test(ra, len(cr), rn, len(calm))
            rows.append({"crisis": f"{e['peak'].date()}..{e['trough'].date()}", "pair": f"SPY-{x}",
                         "calm corr": rn, "crisis corr": rc, "FR-adjusted": ra, "SPY variance ratio": vr,
                         "rise (unadj.) p": pu / 2 if rc > rn else 1 - pu / 2,
                         "rise (adj.) p": pa / 2 if ra > rn else 1 - pa / 2})
    return pd.DataFrame(rows)


def run(panel) -> Section:
    R = panel.returns_for(SYNC)
    x = R["SPY"].to_numpy()
    qs = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98]
    cc, rows = {}, {}
    for p in PAIRS:
        y = R[p].to_numpy()
        c = conditional_corr(x, y, qs)
        cc[p] = c
        lo, hi = boot_ci_condcorr(x, y, 0.05)
        u, v = stats.rankdata(x) / (len(x) + 1), stats.rankdata(y) / (len(y) + 1)
        rho = c.attrs["rho_ns"]
        ll = float(np.mean((u <= 0.05) & (v <= 0.05)) / 0.05)
        lu = float(np.mean((u <= 0.05) & (v >= 0.95)) / 0.05)
        cop = fit_bivariate(x, y)
        tfit = cop[cop["family"] == "student_t"].iloc[0]
        rows[f"SPY-{p}"] = {
            "full-sample corr": float(np.corrcoef(x, y)[0, 1]),
            "corr | SPY worst 5%": c.loc[0.05, "corr"], "90% CI": f"[{lo:.2f}, {hi:.2f}]",
            "Gaussian benchmark": c.loc[0.05, "gauss"], "corr | SPY best 5%": c.loc[0.95, "corr"],
            "Gaussian (best 5%)": c.loc[0.95, "gauss"],
            "co-crash freq / q": ll, "Gaussian co-crash": gaussian_tail_dependence(rho, 0.05),
            "crash & rally freq / q": lu, "Gaussian crash & rally": gaussian_tail_dependence(-rho, 0.05),
            "best copula (AIC)": cop.loc[0, "family"] + (" (rotated)" if cop.attrs["rotated_for_one_sided_families"]
                                                         and cop.loc[0, "family"] in ("clayton", "gumbel", "survival_gumbel") else ""),
            "t-copula nu": float(tfit["params"][1])}
    T1 = pd.DataFrame(rows).T
    t1 = Table("q5_tails", "Dependence with the S&P 500 in the tails vs a Gaussian benchmark (2007-2026)", T1,
               "Conditional correlations on the S&P 500's worst / best 5% of days, with a block-bootstrap 90% interval, "
               "against a bivariate Gaussian with the same normal-scores correlation (which predicts tail "
               "correlations SHRINKING toward zero). 'freq / q': empirical joint-exceedance frequency at the 5% level "
               "divided by 5% (co-crash: both in their worst 5%; crash & rally: SPY worst 5% with the other asset in "
               "its best 5%).")
    fr = forbes_rigobon_table(panel, R)
    n_un = int((fr["rise (unadj.) p"] < 0.05).sum())
    n_ad = int((fr["rise (adj.) p"] < 0.05).sum())
    t2 = Table("q5_forbes_rigobon", "Contagion or interdependence? Crisis vs pre-crisis correlations", fr.set_index(["crisis", "pair"]),
               "Calm: the 250 trading days before the bear-market peak; crisis: peak to trough. FR-adjusted: "
               "Forbes-Rigobon correction for the higher S&P 500 variance in the crisis. One-sided p-values for a rise "
               "(Fisher z, iid approximation).", "{:.2f}")
    oos, pv = oos_tail_corr(R)
    agg = oos.groupby("predictor")[["MAE vs tail-day corr", "MAE vs all-day corr"]].mean()
    piv = oos.pivot(index="year", columns="predictor", values="MAE vs tail-day corr")
    wins = {k: float((piv[k] < piv["full history"]).mean()) for k in piv.columns if k != "full history"}
    agg["share of years beating full history (tail target)"] = pd.Series(wins)
    pagg = pv.groupby("predictor")[["abs. error, portfolio vol on tail days (%)", "predicted/realized"]].mean()
    agg = agg.join(pagg)
    from .q4_stress import dm_test

    dmt = {}
    for k in piv.columns:
        if k != "full history":
            dmt[k] = dm_test((piv[k] - piv["full history"]).to_numpy(), lags=1)
    agg["DM t vs full history"] = pd.Series({k: v[0] for k, v in dmt.items()})
    t3 = Table("q5_oos", "Which past correlations predict next year's tail-day correlations?", agg,
               f"Each year 2010-{R.index[-1].year}: correlation matrices of the 16 synchronous series estimated on data "
               "before the year; target: correlations on the year's days when the S&P 500 fell below its past 10% "
               "quantile. MAE over 120 pairs; portfolio: a multi-asset mix, volatility on those days using realized "
               "tail-day volatilities and each predicted correlation matrix.", "{:.3f}")
    for t in (t1, t2, t3):
        save_table(t)

    show = ["HYG", "EEM", "IEF", "GLD"]

    def draw_cc(fig, P, mode):
        axs = fig.subplots(1, 4, sharey=True)
        for ax, p in zip(axs, show):
            c = cc[p]
            ax.plot(c.index, c["gauss"], color=P["context"], lw=1.6, label="Gaussian benchmark")
            ax.plot(c.index, c["corr"], color=P["series"][0], lw=2.0, marker="o", ms=3.5, label="empirical")
            ax.axhline(0, color=P["base"], lw=0.8)
            ax.set_title(f"SPY vs {p}", fontsize=9.5)
            ax.set_xticks([0.02, 0.2, 0.5, 0.8, 0.98], ["2%", "20%", "50%", "80%", "98%"])
            ax.set_xlabel("S&P 500 quantile conditioned on")
        axs[0].set_ylabel("correlation on those days")
        axs[0].legend(loc="lower left", fontsize=7.5)

    f1 = save_fig("q5_condcorr", draw_cc, (10.4, 3.1), "Correlation on the S&P 500's tail days",
                  "Left of 50%: correlation on days when the S&P 500 is at or below that quantile; right: at or above. "
                  "Gray: what a bivariate Gaussian with the same normal-scores correlation implies (shrinking toward "
                  "zero in both tails). Blue above gray in the left tail = stronger co-movement on bad days than a "
                  "Gaussian allows.")

    def draw_oos(fig, P, mode):
        ax = fig.add_subplot(111)
        for k, (name, col) in enumerate([("full history", 2), ("last 252 days", 1), ("past tail days", 0)]):
            ax.plot(piv.index, piv[name], marker="o", ms=4, lw=1.6, color=P["series"][col], label=name)
        ax.set_ylabel("mean abs. error over 120 pairs")
        ax.legend(loc="upper left", ncols=3)
        ax.set_title("Predicting next year's tail-day correlations")

    f2 = save_fig("q5_oos", draw_oos, (8.4, 3.2), "Out-of-sample: which past correlations predict tail co-movement?",
                  "Lower is better. Each point predicts the correlations realized on that year's market-tail days.")
    sec = Section("q5", "Q5. Are correlations more informative during tail events?",
                  "Are correlations more informative during tail events?", "", "")
    sec.tables, sec.figures = [t1, t2, t3], [f1, f2]
    risky = [p for p in PAIRS if p in ("EFA", "EEM", "XLF", "HYG", "USO", "DBC")]
    above = [p for p in risky if T1.loc[f"SPY-{p}", "corr | SPY worst 5%"] > T1.loc[f"SPY-{p}", "Gaussian benchmark"]]
    best_pred = agg["MAE vs tail-day corr"].idxmin()
    tail_mae = agg.loc["past tail days", "MAE vs tail-day corr"]
    full_mae = agg.loc["full history", "MAE vs tail-day corr"]
    rec_mae = agg.loc["last 252 days", "MAE vs tail-day corr"]
    pv_err = agg["abs. error, portfolio vol on tail days (%)"]
    sec.design = ("Synchronous ETF universe (16 series, all priced at the NYSE close) from 2007-04; Gaussian "
                  "benchmarks use each pair's normal-scores correlation; bear markets dated as in Q1.")
    tail_wins = tail_mae < full_mae
    differ = len(above) >= max(1, len(risky) // 2)
    verdict = ("Yes." if (tail_wins and differ) else "Partly." if (tail_wins or differ) else "No.")
    fr_txt = (f"Most crisis-correlation 'contagion' disappears after the Forbes-Rigobon volatility adjustment: "
              f"{n_un} pair-crisis rises are significant unadjusted, {n_ad} after adjustment."
              if n_ad <= n_un / 2 else
              f"Crisis-correlation rises survive the Forbes-Rigobon volatility adjustment in {n_ad} of the {n_un} "
              "pair-crisis cases that are significant unadjusted.")
    havens = [p for p in ("IEF", "TLT", "FXY") if p in PAIRS]
    hv = {p: (T1.loc[f"SPY-{p}", "crash & rally freq / q"], T1.loc[f"SPY-{p}", "Gaussian crash & rally"]) for p in havens}
    hv_txt = ("Treasuries and the yen show the mirror image: they rally on the S&P 500's worst days more often than a "
              "Gaussian implies (crash-and-rally frequency " + ", ".join(f"{p} {a:.2f} vs {b:.2f}" for p, (a, b) in hv.items())
              + "). ") if hv else ""
    sec.answer = (
        f"{verdict} On the S&P 500's worst 5% of days, {len(above)} of {len(risky)} risky assets "
        f"({', '.join(above) if above else 'none'}) are more correlated with it than a Gaussian with the same overall "
        "dependence implies; under that Gaussian the correlation would shrink toward zero in the tail, so an excess "
        "is tail dependence rather than a conditioning artefact. " + hv_txt + "As forecasts, correlations measured on past tail "
        f"days predict the next year's tail-day correlations with a mean absolute error of {tail_mae:.3f}, against "
        f"{full_mae:.3f} for the full history and {rec_mae:.3f} for the last 252 days (lowest error: {best_pred}; "
        f"past tail days beat the full history in "
        f"{100 * agg.loc['past tail days', 'share of years beating full history (tail target)']:.0f}% of years, "
        f"DM t = {agg.loc['past tail days', 'DM t vs full history']:.1f}). For a multi-asset portfolio's volatility "
        f"on tail days the errors are {pv_err['past tail days']:.1f}% with tail correlations vs "
        f"{pv_err['full history']:.1f}% with the full history. " + fr_txt)
    sec.findings = [
        Finding("Q5", Claim.DESCRIPTIVE, "Tail correlation of risky assets vs the Gaussian benchmark",
                ", ".join(f"{p}: {T1.loc[f'SPY-{p}', 'corr | SPY worst 5%']:.2f} vs Gaussian "
                          f"{T1.loc[f'SPY-{p}', 'Gaussian benchmark']:.2f}" for p in risky) + ".",
                ["Conditional correlations on 5% of days use ~250 observations; intervals in the table."]),
        Finding("Q5", Claim.PREDICTIVE, "Forecasting future tail-day correlations",
                f"MAE: past tail days {tail_mae:.3f}, full history {full_mae:.3f}, last 252 days {rec_mae:.3f}.",
                ["Only ~16 forecast years; year-to-year variation is large (figure)."]),
        Finding("Q5", Claim.DESCRIPTIVE, "Crisis correlation rises before and after the volatility adjustment",
                f"Significant rises: {n_un} unadjusted vs {n_ad} Forbes-Rigobon adjusted (of {len(fr)} pair-crisis cases).",
                ["The FR adjustment assumes no common shocks and no feedback; it can over-correct."]),
    ]
    sec.numbers = {"fr_unadjusted": n_un, "fr_adjusted": n_ad, "fr_cases": int(len(fr)), "oos": agg.to_dict()}
    return sec
