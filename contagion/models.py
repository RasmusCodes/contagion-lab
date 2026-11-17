"""Standard model run: every network / regime output that the research questions and the dashboard reuse.

    run_all(panel)  -> writes outputs/models/*   (scripts/run_models.py)
    load_*()        -> read them back

Specifications are fixed here (not tuned on results): VAR(1) (BIC choice on the macro sample is reported),
H = 10-day forecast horizon, 500-day rolling windows stepped weekly, TVP-VAR forgetting factors 0.99 / 0.96
(Antonakakis et al. 2020 defaults), 3-state HMM on standardized SPY / VIX / 10Y / Baa-spread changes.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from .network.connectedness import DynamicConnectedness, rolling_connectedness, tvp_var_connectedness
from .paths import MODELS, ensure_dirs

NETWORKS = {
    "macro": "macro",
    "sectors": "sectors",
    "fin": "fin",
    "sync": "sync",
}

RISK_OFF_SIGN = {  # +1: the series FALLS in a typical risk-off move; -1: it RISES (used to orient tail measures)
    "SPY": 1, "IWM": 1, "EFA": 1, "EEM": 1, "XLF": 1, "UST2Y": 1, "UST10Y": 1, "BAA10Y": -1, "DXY": -1,
    "USDJPY": 1, "COPPER_F": 1, "OIL": 1, "GOLD": -1, "VIX": -1, "MOVE": -1,
}

REGIME_FEATURES = ["SPY", "VIX", "UST10Y", "BAA10Y"]


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def save_dynamic(dc: DynamicConnectedness, name: str) -> None:
    np.savez_compressed(MODELS / f"{name}.npz", dates=dc.dates.values.astype("datetime64[D]").astype(str),
                        names=np.array(dc.names), theta=dc.theta.astype(np.float32),
                        theta_contemp=dc.theta_contemp.astype(np.float32), method=dc.method,
                        params=json.dumps(dc.params))


def load_dynamic(name: str) -> DynamicConnectedness:
    z = np.load(MODELS / f"{name}.npz", allow_pickle=False)
    return DynamicConnectedness(list(z["names"]), pd.DatetimeIndex(pd.to_datetime(z["dates"])),
                                z["theta"].astype(float), z["theta_contemp"].astype(float), str(z["method"]),
                                json.loads(str(z["params"])))


def run_networks(panel, t0, quick=False) -> None:
    from .network.correlation import average_correlation, cov_to_corr, rolling_cov
    from .network.granger import rolling_granger
    from .network.pca import absorption_ratio, first_pc_share
    from .network.var import select_lag

    for key, setname in NETWORKS.items():
        R = panel.returns_for(setname)
        R.to_parquet(MODELS / f"R_{key}.parquet")
        if key == "macro":
            lag = {c: select_lag(R.to_numpy(), maxlag=5, criterion=c) for c in ("bic", "hqic", "aic")}
            (MODELS / "macro_lag_selection.json").write_text(json.dumps(lag))
            _log(f"macro lag selection {lag}", t0)
        for w in ((500, 250) if key == "macro" else (500,)):
            dc = rolling_connectedness(R, window=w, step=5, p=1, H=10)
            save_dynamic(dc, f"dy_{key}_w{w}")
            _log(f"rolling DY {key} window {w}: {len(dc.dates)} windows", t0)
        if key == "macro":
            dc = rolling_connectedness(R, window=500, step=5, p=2, H=10)
            save_dynamic(dc, "dy_macro_w500_p2")
            dc = rolling_connectedness(R, window=500, step=5, p=1, H=20)
            save_dynamic(dc, "dy_macro_w500_h20")
            _log("rolling DY macro robustness (p=2, H=20)", t0)
        if key in ("macro", "sectors", "sync") and not quick:
            tv = tvp_var_connectedness(R, p=1, H=10, kappa1=0.99, kappa2=0.96, prior_obs=250)
            save_dynamic(tv, f"tvp_{key}")
            _log(f"TVP-VAR {key}: {len(tv.dates)} dates", t0)
        # correlation / PCA summaries
        C63 = cov_to_corr(rolling_cov(R.to_numpy(), 63))
        C250 = cov_to_corr(rolling_cov(R.to_numpy(), 250))
        cs = pd.DataFrame({"avg_corr_63": average_correlation(C63), "avg_corr_250": average_correlation(C250),
                           "avg_abs_corr_250": (np.abs(C250).sum(axis=(1, 2)) - R.shape[1]) / (R.shape[1] * (R.shape[1] - 1))},
                          index=R.index)
        cs["absorption_250"] = absorption_ratio(R, 250)
        cs["pc1_250"] = first_pc_share(R, 250)
        cs.to_parquet(MODELS / f"corrstats_{key}.parquet")
        if key in ("macro", "sync"):
            for kind in ("pairwise", "conditional"):
                g = rolling_granger(R, window=500, step=21, p=1, q=0.05, kind=kind)
                np.savez_compressed(MODELS / f"granger_{key}_{kind}.npz",
                                    dates=g["dates"].values.astype("datetime64[D]").astype(str),
                                    names=np.array(list(R.columns), dtype=str),
                                    adjacency=g["adjacency"].astype(np.int8))
                _log(f"rolling Granger {key} {kind}: {len(g['dates'])} windows", t0)


def run_dfm(panel, t0) -> None:
    """Weekly dynamic factor model on the macro network: one global factor plus asset-class blocks (AR(1)).
    Weekly sums remove most of the timing mismatch between markets and keep the EM fast. Saves the R^2 table,
    the smoothed factors and the rolling 104-week share of each series' variance explained by the global factor
    (two-sided smoothed factor: descriptive only)."""
    from .network.factor import dynamic_factor_model, rolling_factor_r2

    R = panel.returns_for("macro")
    W = R.groupby(R.index.to_period("W-WED")).sum()
    W.index = W.index.to_timestamp(how="end").normalize()
    blocks = {c: panel.asset_class(c) for c in R.columns}
    res = dynamic_factor_model(W, blocks=blocks, maxiter=100)
    res["r2"].to_csv(MODELS / "dfm_r2.csv")
    res["factors"].to_parquet(MODELS / "dfm_factors.parquet")
    roll = rolling_factor_r2(W, res["factors"]["Global"], window=104)
    roll.to_parquet(MODELS / "dfm_rolling_r2.parquet")
    _log(f"weekly DFM: factors {res['factor_names']}", t0)


def load_granger(key: str, kind: str = "pairwise") -> dict:
    z = np.load(MODELS / f"granger_{key}_{kind}.npz", allow_pickle=True)  # own files; older runs stored object names
    A = z["adjacency"].astype(int)
    names = [str(n) for n in z["names"]]
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    N = len(names)
    return {"dates": dates, "names": names, "adjacency": A,
            "density": pd.Series(A.sum(axis=(1, 2)) / (N * (N - 1)), index=dates),
            "out_degree": pd.DataFrame(A.sum(axis=1), index=dates, columns=names),
            "in_degree": pd.DataFrame(A.sum(axis=2), index=dates, columns=names)}


def run_regimes(panel, t0, quick=False) -> None:
    from .regimes.bayesian import BayesianHMM
    from .regimes.changepoint import bocpd, pelt, segments_frame
    from .regimes.dcc import fit_dcc_from_returns
    from .regimes.hmm import GaussianHMM, expanding_filtered_probs
    from .regimes.volatility import garch_expanding, realized_vol, vix_regimes, vol_regimes

    R = panel.returns_for("macro")
    F = R[REGIME_FEATURES]
    Z = F / F.std()
    bic = {}
    models = {}
    for K in (2, 3, 4):
        m = GaussianHMM(K, n_init=4, seed=0).fit(Z.to_numpy())
        bic[K] = m.bic(Z.to_numpy())
        models[K] = m
    _log(f"HMM BIC by K: { {k: round(v) for k, v in bic.items()} }", t0)
    m = models[3]
    sm, fl = m.smooth(Z.to_numpy()), m.filter(Z.to_numpy())
    vit = m.viterbi(Z.to_numpy())
    hmm = pd.DataFrame(np.hstack([sm, fl]), index=R.index,
                       columns=[f"smoothed_{k}" for k in range(3)] + [f"filtered_{k}" for k in range(3)])
    hmm["viterbi"] = vit
    if not quick:
        rt = expanding_filtered_probs(Z, n_states=3, min_train=750, refit_every=126, n_init=2, seed=0)
        for k in range(3):
            hmm[f"realtime_{k}"] = rt[f"p_state{k}"]
        _log("HMM real-time (expanding) probabilities", t0)
    hmm.to_parquet(MODELS / "regime_hmm.parquet")
    info = {"bic": bic, "K": 3, "features": REGIME_FEATURES,
            "means": m.means_.tolist(), "covs": m.covs_.tolist(), "transmat": m.transmat_.tolist(),
            "expected_duration_days": m.expected_duration().tolist(),
            "feature_std": F.std().to_dict(),
            "state_spy_ann_vol": (np.sqrt(m.covs_[:, 0, 0]) * F["SPY"].std() * np.sqrt(252)).tolist()}
    (MODELS / "regime_hmm.json").write_text(json.dumps(info, indent=1))

    # Markov-switching variance model on SPY (Hamilton filter)
    import statsmodels.api as sm_api

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = sm_api.tsa.MarkovRegression(R["SPY"] * 100, k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
    msdf = pd.DataFrame({"filtered_high": ms.filtered_marginal_probabilities.iloc[:, int(np.argmax(ms.params.filter(like="sigma2").to_numpy()))],
                         "smoothed_high": ms.smoothed_marginal_probabilities.iloc[:, int(np.argmax(ms.params.filter(like="sigma2").to_numpy()))]},
                        index=R.index)
    msdf.to_parquet(MODELS / "regime_ms.parquet")
    _log("Markov-switching SPY", t0)

    # change points
    b1 = bocpd(R["SPY"], hazard=1 / 250)
    b2 = bocpd(R["UST10Y"], hazard=1 / 250)
    pd.DataFrame({"spy_p_recent": b1["p_recent"], "spy_map_run": b1["map_run"],
                  "ust10y_p_recent": b2["p_recent"], "ust10y_map_run": b2["map_run"]}).to_parquet(MODELS / "regime_bocpd.parquet")
    X2 = (R[["SPY", "UST10Y"]] / R[["SPY", "UST10Y"]].std()).to_numpy()
    bk = pelt(X2, min_size=63, jump=5)
    seg = segments_frame(R.index, bk, R[["SPY"]].to_numpy())
    seg["corr_spy_10y"] = [float(np.corrcoef(R["SPY"].loc[a:b], R["UST10Y"].loc[a:b])[0, 1]) for a, b in zip(seg["start"], seg["end"])]
    seg.to_csv(MODELS / "regime_pelt_spy_ust10y.csv", index=False)
    _log(f"BOCPD + PELT ({len(bk)} breaks)", t0)

    # volatility regimes
    ge = garch_expanding(R["SPY"], min_train=1000, refit_every=252)
    vol = pd.DataFrame({"garch_vol_ann": ge["vol"] * np.sqrt(252), "rv21_ann": realized_vol(R["SPY"], 21)}, index=R.index)
    vol["vol_regime"] = vol_regimes(vol["garch_vol_ann"].dropna()).reindex(R.index)
    vix = panel.levels["VIX"].reindex(R.index)
    vol["vix"] = vix
    vol["vix_regime"] = vix_regimes(vix)
    vol.to_parquet(MODELS / "regime_vol.parquet")
    _log("volatility regimes", t0)

    # stock-bond correlation regime
    sb = panel.returns_for(["SPY", "IEF"])
    sbc = pd.DataFrame({"corr_spy_ief_126": sb["SPY"].rolling(126).corr(sb["IEF"]),
                        "corr_spy_ief_252": sb["SPY"].rolling(252).corr(sb["IEF"])})
    sbc.to_parquet(MODELS / "stock_bond_corr.parquet")

    # DCC on the macro network
    if not quick:
        d = fit_dcc_from_returns(R)
        np.savez_compressed(MODELS / "dcc_macro.npz", dates=R.index.values.astype("datetime64[D]").astype(str),
                            names=np.array(list(R.columns), dtype=str), corr=d["corr"].astype(np.float32),
                            vol=d["vol"].to_numpy(np.float32), std_resid=d["std_resid"].to_numpy(np.float32),
                            a=d["a"], b=d["b"])
        _log(f"DCC macro a={d['a']:.4f} b={d['b']:.4f}", t0)

    # Bayesian sticky HMM on weekly S&P 500 returns
    if not quick:
        lv = panel.levels["SPY"].dropna()
        wk = np.log(lv.resample("W-WED").last()).diff().dropna().loc["2003-04-16":] * 100
        bh = BayesianHMM(n_states=2, n_iter=3000, burn=1000, thin=2, kappa=20.0, seed=0).fit(wk)
        bh.state_prob.to_parquet(MODELS / "regime_bayes_weekly.parquet")
        summ = bh.summary(scale=0.01, annualize=np.sqrt(52))
        lo = np.quantile(np.stack([bh.state_prob.iloc[:, 1].to_numpy()]), 0.5)
        summ.to_csv(MODELS / "regime_bayes_summary.csv")
        _log(f"Bayesian HMM weekly: {summ.round(3).to_dict()} {lo:.2f}", t0)


def load_dcc() -> dict:
    z = np.load(MODELS / "dcc_macro.npz", allow_pickle=True)
    return {"dates": pd.DatetimeIndex(pd.to_datetime(z["dates"])), "names": [str(n) for n in z["names"]],
            "corr": z["corr"], "vol": z["vol"], "std_resid": z["std_resid"], "a": float(z["a"]), "b": float(z["b"])}


def load_regimes() -> dict:
    out = {"hmm": pd.read_parquet(MODELS / "regime_hmm.parquet"),
           "hmm_info": json.loads((MODELS / "regime_hmm.json").read_text()),
           "ms": pd.read_parquet(MODELS / "regime_ms.parquet"),
           "bocpd": pd.read_parquet(MODELS / "regime_bocpd.parquet"),
           "pelt": pd.read_csv(MODELS / "regime_pelt_spy_ust10y.csv", parse_dates=["start", "end"]),
           "vol": pd.read_parquet(MODELS / "regime_vol.parquet"),
           "stock_bond": pd.read_parquet(MODELS / "stock_bond_corr.parquet")}
    if (MODELS / "regime_bayes_weekly.parquet").exists():
        out["bayes"] = pd.read_parquet(MODELS / "regime_bayes_weekly.parquet")
        out["bayes_summary"] = pd.read_csv(MODELS / "regime_bayes_summary.csv", index_col=0)
    return out


def run_all(panel, quick: bool = False, stages=("networks", "regimes")) -> None:
    ensure_dirs()
    t0 = time.time()
    if "networks" in stages:
        run_networks(panel, t0, quick)
    if "networks" in stages or "dfm" in stages:
        run_dfm(panel, t0)
    if "regimes" in stages:
        run_regimes(panel, t0, quick)
    _log("done", t0)
