"""Cross-asset hedging: minimum-variance hedge ratios from different covariance "views", and an out-of-sample
backtest that only ever uses information available at each rebalance date.

hedged return_t = r_target,t - h' r_hedges,t,   h = Sigma_HH^-1 Sigma_HT  (estimated at the last rebalance <= t-1)

Covariance providers (``cov_at(i)`` returns the covariance using rows <= i):
    StaticCov    expanding window (all history)
    RollingCov   last ``window`` rows
    EwmaCov      exponentially weighted (lambda)
    RegimeCov    real-time Gaussian-HMM: state covariances from a model re-estimated on data up to each refit
                 date, mixed with the FILTERED state probabilities of day i
    BocpdCov     window that restarts at the last change point detected online by BOCPD (min ``min_window`` rows)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contagion.regimes.changepoint import bocpd
from contagion.regimes.hmm import GaussianHMM


class StaticCov:
    name = "static"

    def __init__(self, X: np.ndarray, min_obs: int = 250):
        self.X = X
        self.S1 = np.cumsum(X, axis=0)
        self.S2 = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
        self.min_obs = min_obs

    def cov_at(self, i: int) -> np.ndarray:
        n = i + 1
        m = self.S1[i] / n
        return (self.S2[i] - n * np.outer(m, m)) / (n - 1)


class RollingCov:
    name = "rolling"

    def __init__(self, X: np.ndarray, window: int = 252):
        self.X, self.w = X, window

    def cov_at(self, i: int) -> np.ndarray:
        return np.cov(self.X[max(0, i - self.w + 1): i + 1], rowvar=False)


class EwmaCov:
    name = "ewma"

    def __init__(self, X: np.ndarray, lam: float = 0.97, init: int = 250):
        T, N = X.shape
        self.S = np.empty((T, N, N))
        S = np.cov(X[:init], rowvar=False)
        mu = X[:init].mean(axis=0)
        for t in range(T):
            if t >= init:
                d = X[t] - mu
                S = lam * S + (1 - lam) * np.outer(d, d)
            self.S[t] = S

    def cov_at(self, i: int) -> np.ndarray:
        return self.S[i]


class RegimeCov:
    """Real-time regime-mixture covariance. HMM features: standardized returns of ``feature_cols`` (indices
    into X), scaled with the training-sample standard deviation at each refit."""

    name = "regime"

    def __init__(self, X: np.ndarray, feature_cols: list[int], n_states: int = 2, min_train: int = 750,
                 refit_every: int = 63, seed: int = 0, use: str = "mixture"):
        T, N = X.shape
        self.covs = np.full((T, N, N), np.nan)
        self.probs = np.full((T, n_states), np.nan)
        prev = None
        for start in range(min_train, T, refit_every):
            F = X[:start, feature_cols]
            sd = F.std(axis=0)
            m = GaussianHMM(n_states, n_init=3, seed=seed).fit(F / sd, init_from=prev)
            prev = m
            post = m.smooth(F / sd)  # within the training sample only
            state_cov, state_mu = [], []
            for k in range(n_states):
                w = post[:, k] / post[:, k].sum()
                mu = w @ X[:start]
                Xc = X[:start] - mu
                state_cov.append((Xc * w[:, None]).T @ Xc)
                state_mu.append(mu)
            end = min(start + refit_every, T)
            filt = m.filter(X[:end, feature_cols] / sd)[start - 1:end]  # p(s_i | x_1..i), i = start-1 .. end-1
            for j, i in enumerate(range(start - 1, end)):
                p = filt[j]
                self.probs[i] = p
                if use == "crisis_if_likely":
                    k = int(np.argmax(p))
                    self.covs[i] = state_cov[k]
                else:
                    mbar = sum(pk * mu for pk, mu in zip(p, state_mu))
                    self.covs[i] = sum(pk * (S + np.outer(mu - mbar, mu - mbar))
                                       for pk, S, mu in zip(p, state_cov, state_mu))
        self.name = "regime" if use == "mixture" else "regime_switch"

    def cov_at(self, i: int) -> np.ndarray:
        return self.covs[i]


class DccCov:
    """GJR-GARCH(1,1)-t volatilities + DCC(1,1) correlations, all parameters re-estimated every ``refit_every`` rows
    on data before the block and filtered forward inside it. cov_at(i) = forecast for row i+1 from rows <= i."""

    name = "dcc"

    def __init__(self, X: np.ndarray, min_train: int = 750, refit_every: int = 504, o: int = 1):
        import warnings

        from arch import arch_model

        from contagion.regimes.dcc import DCC, _corr, _q_path

        T, N = X.shape
        self.S = np.full((T, N, N), np.nan)
        for start in range(min_train, T, refit_every):
            end = min(start + refit_every, T)
            sig, mu = np.empty((end, N)), np.empty(N)
            for j in range(N):
                sc = 1.0 / X[:start, j].std()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    am = arch_model(X[:end, j] * sc, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t")
                    res = am.fit(disp="off", last_obs=start)
                    fx = am.fix(res.params)
                sig[:, j] = np.asarray(fx.conditional_volatility) / sc
                mu[j] = res.params.iloc[0] / sc
            E = (X[:end] - mu) / sig
            m = DCC().fit(E[:start])
            Rt = _corr(_q_path(E, m.a, m.b, m.Qbar))  # R_t uses e up to t-1
            for t in range(max(start, 1), end):
                self.S[t - 1] = Rt[t] * np.outer(sig[t], sig[t])
            self.last_params = (m.a, m.b)

    def cov_at(self, i: int) -> np.ndarray:
        return self.S[i]


class StressedCov:
    """Covariance of past days on which VIX was in the top ``1-q`` of its own history up to that day."""

    name = "stressed"

    def __init__(self, X: np.ndarray, vix_level: np.ndarray, q: float = 0.9, min_obs: int = 60):
        self.X, self.v, self.q, self.min_obs = X, vix_level, q, min_obs

    def cov_at(self, i: int) -> np.ndarray:
        v = self.v[: i + 1]
        ok = np.isfinite(v)
        m = ok & (v >= np.nanquantile(v, self.q))
        if m.sum() < self.min_obs:
            return np.full((self.X.shape[1],) * 2, np.nan)
        return np.cov(self.X[: i + 1][m], rowvar=False)


class BocpdCov:
    name = "bocpd"

    def __init__(self, X: np.ndarray, signal_col: int = 0, min_window: int = 63, max_window: int = 1000,
                 hazard: float = 1 / 500):
        self.X, self.min_w, self.max_w = X, min_window, max_window
        res = bocpd(pd.Series(X[:, signal_col]), hazard=hazard)
        self.run = res["map_run"].to_numpy()

    def cov_at(self, i: int) -> np.ndarray:
        w = int(np.clip(self.run[i] + 1, self.min_w, self.max_w))
        return np.cov(self.X[max(0, i - w + 1): i + 1], rowvar=False)


def hedge_ratio(S: np.ndarray, t_idx: int, h_idx: list[int], cap: float = 3.0) -> np.ndarray:
    Shh = S[np.ix_(h_idx, h_idx)]
    Sht = S[h_idx, t_idx]
    h = np.linalg.lstsq(Shh, Sht, rcond=None)[0]
    return np.clip(h, -cap, cap)


def hedge_backtest(R: pd.DataFrame, target: str, hedges: list[str], provider, start: int, rebalance: int = 5,
                   cost_bps: float = 2.0, cap: float = 3.0) -> pd.DataFrame:
    """Simple-return backtest. Hedge ratios set at the close of rebalance day i are applied from day i+1."""
    cols = [target] + hedges
    X = np.expm1(R[cols].to_numpy(dtype=float))
    T = len(X)
    h = np.zeros(len(hedges))
    hs = np.zeros((T, len(hedges)))
    cost = np.zeros(T)
    for i in range(start - 1, T - 1):
        if (i - (start - 1)) % rebalance == 0:
            S = provider.cov_at(i)
            new = hedge_ratio(S, 0, list(range(1, len(cols))), cap) if np.all(np.isfinite(S)) else h
            cost[i + 1] = cost_bps * 1e-4 * np.abs(new - h).sum()
            h = new
        hs[i + 1] = h
    hedged = X[:, 0] - (X[:, 1:] * hs).sum(axis=1) - cost
    out = pd.DataFrame({"unhedged": X[:, 0], "hedged": hedged, "cost": cost}, index=R.index)
    for j, c in enumerate(hedges):
        out[f"h_{c}"] = hs[:, j]
    return out.iloc[start:]


def hedge_metrics(bt: pd.DataFrame, crisis_windows: list[tuple] | None = None) -> dict:
    u, h = bt["unhedged"], bt["hedged"]
    hcols = [c for c in bt.columns if c.startswith("h_")]
    turnover = bt[hcols].diff().abs().sum(axis=1).sum() / (len(bt) / 252)

    def es(x, a=0.05):
        q = x.quantile(a)
        return float(-x[x <= q].mean())

    def mdd(x):
        lv = (1 + x).cumprod()
        return float((lv / lv.cummax() - 1).min())

    out = {"ann_vol_unhedged": float(u.std() * np.sqrt(252)), "ann_vol_hedged": float(h.std() * np.sqrt(252)),
           "variance_reduction": float(1 - h.var() / u.var()), "es5_unhedged": es(u), "es5_hedged": es(h),
           "maxdd_unhedged": mdd(u), "maxdd_hedged": mdd(h), "ann_ret_unhedged": float(u.mean() * 252),
           "ann_ret_hedged": float(h.mean() * 252), "annual_turnover": float(turnover),
           "cost_drag_ann": float(bt["cost"].mean() * 252)}
    if crisis_windows:
        cr = []
        for a, b in crisis_windows:
            w = bt.loc[a:b]
            if len(w):
                cr.append({"window": f"{pd.Timestamp(a).date()}..{pd.Timestamp(b).date()}",
                           "unhedged": float((1 + w["unhedged"]).prod() - 1),
                           "hedged": float((1 + w["hedged"]).prod() - 1)})
        out["crisis"] = cr
    return out


def dm_test_squared(a: pd.Series, b: pd.Series, lags: int = 10) -> tuple[float, float]:
    """Diebold-Mariano style test that E[a^2] = E[b^2] (e.g. hedged returns under two methods), Newey-West SE.
    Negative statistic: a has lower variance."""
    from scipy import stats

    d = (a ** 2 - b ** 2).dropna().to_numpy()
    n = len(d)
    dm = d - d.mean()
    g0 = dm @ dm / n
    s = g0 + 2 * sum((1 - k / (lags + 1)) * (dm[k:] @ dm[:-k] / n) for k in range(1, lags + 1))
    t = d.mean() / np.sqrt(s / n)
    return float(t), float(2 * stats.norm.sf(abs(t)))
