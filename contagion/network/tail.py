"""Tail dependence, CoVaR and MES. All DESCRIPTIVE: they measure joint tail behaviour, not transmission.

* Empirical tail dependence  lambda_L(q) = P(U <= q, V <= q) / q  on rank pseudo-observations. Under a Gaussian
  copula this tends to zero as q -> 0; a persistently positive value indicates genuine tail dependence.
* MES (Acharya, Pedersen, Philippon & Richardson 2017): expected return of i on the market's worst q days.
* Delta-CoVaR (Adrian & Brunnermeier 2016): change in the system's q-quantile when i moves from its median to
  its own q-quantile, from a q-quantile regression of the system on i. The direction "i -> system" is a
  regression convention; reversing it gives a different number, and neither is causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pseudo_obs(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return stats.rankdata(X) / (len(X) + 1)
    return np.column_stack([stats.rankdata(X[:, j]) / (X.shape[0] + 1) for j in range(X.shape[1])])


def tail_dependence(x, y, q: float = 0.05, tail: str = "lower") -> float:
    u, v = pseudo_obs(x), pseudo_obs(y)
    if tail == "lower":
        return float(np.mean((u <= q) & (v <= q)) / q)
    return float(np.mean((u > 1 - q) & (v > 1 - q)) / q)


def tail_dependence_matrix(X: pd.DataFrame, q: float = 0.05, tail: str = "lower", signs=None) -> pd.DataFrame:
    """Matrix of empirical tail dependence. ``signs`` (+1/-1 per column) orients each series so that its
    "bad" tail is the lower one (e.g. -1 for VIX, credit spreads and yields if you want sell-off co-crashes)."""
    A = X.to_numpy(dtype=float)
    if signs is not None:
        A = A * np.asarray(signs, dtype=float)[None, :]
    U = pseudo_obs(A)
    E = (U <= q) if tail == "lower" else (U > 1 - q)
    M = (E.T.astype(float) @ E.astype(float)) / len(U) / q
    np.fill_diagonal(M, 1.0)
    return pd.DataFrame(M, index=X.columns, columns=X.columns)


def gaussian_tail_dependence(rho: float, q: float = 0.05) -> float:
    """Finite-q tail-dependence value implied by a Gaussian copula with correlation rho (benchmark)."""
    z = stats.norm.ppf(q)
    p = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z, z])
    return float(p / q)


def mes(r_i: pd.Series, r_m: pd.Series, q: float = 0.05) -> float:
    """Marginal expected shortfall: mean of r_i on days when r_m is at or below its q-quantile."""
    d = pd.concat([r_i, r_m], axis=1).dropna()
    thr = d.iloc[:, 1].quantile(q)
    return float(d.loc[d.iloc[:, 1] <= thr].iloc[:, 0].mean())


def delta_covar(r_sys: pd.Series, r_i: pd.Series, q: float = 0.05) -> dict:
    """Static Delta-CoVaR of the system conditional on institution / asset i (quantile regression)."""
    import statsmodels.api as sm

    d = pd.concat([r_sys, r_i], axis=1).dropna()
    y, x = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    X = sm.add_constant(x)
    res = sm.QuantReg(y, X).fit(q=q, max_iter=5000)
    a, b = res.params
    var_i, med_i = np.quantile(x, q), np.median(x)
    return {"beta": float(b), "covar": float(a + b * var_i), "delta_covar": float(b * (var_i - med_i)),
            "var_i": float(var_i)}


def covar_network(R: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """[i, j] = Delta-CoVaR of i conditional on j being at its q-quantile (in i's units). Diagonal NaN."""
    cols = list(R.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i != j:
                out.loc[i, j] = delta_covar(R[i], R[j], q)["delta_covar"]
    return out


def rolling_tail_dependence(x: pd.Series, y: pd.Series, window: int = 750, q: float = 0.1,
                            step: int = 21) -> pd.Series:
    vals, dates = [], []
    xa, ya = x.to_numpy(), y.to_numpy()
    for e in range(window, len(x) + 1, step):
        vals.append(tail_dependence(xa[e - window:e], ya[e - window:e], q))
        dates.append(x.index[e - 1])
    return pd.Series(vals, index=dates, name=f"lambdaL_{q}")
