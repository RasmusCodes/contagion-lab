"""Portfolio risk: VaR / ES (historical, Gaussian, Cornish-Fisher, filtered historical simulation, EVT),
Euler risk decomposition, factor decomposition and VaR backtests.

Conventions: ``pnl`` are simple portfolio returns; VaR and ES are reported as POSITIVE loss fractions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def simple_returns(R_log: pd.DataFrame) -> pd.DataFrame:
    return np.expm1(R_log)


def portfolio_returns(R_log: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily-rebalanced portfolio simple returns from log returns of the holdings."""
    w = pd.Series(weights, dtype=float)
    return (simple_returns(R_log[w.index]) * w).sum(axis=1).rename("portfolio")


def var_es(pnl, alpha: float = 0.05, method: str = "historical", **kw) -> tuple[float, float]:
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    if method == "historical":
        q = np.quantile(x, alpha)
        return float(-q), float(-x[x <= q].mean())
    mu, sd = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(alpha)
    if method == "gaussian":
        return float(-(mu + sd * z)), float(-(mu - sd * stats.norm.pdf(z) / alpha))
    if method == "cornish_fisher":
        s, k = stats.skew(x), stats.kurtosis(x)
        zc = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
        var = -(mu + sd * zc)
        tail = x[x <= -var]
        return float(var), float(-tail.mean()) if len(tail) else float(var)
    if method == "evt":
        return evt_var_es(x, alpha, **kw)
    raise ValueError(method)


def evt_var_es(pnl, alpha: float = 0.01, threshold_q: float = 0.90) -> tuple[float, float]:
    """Peaks-over-threshold: GPD fitted to losses above their ``threshold_q`` quantile (McNeil & Frey 2000)."""
    L = -np.asarray(pnl, dtype=float)
    u = np.quantile(L, threshold_q)
    exc = L[L > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    n, nu = len(L), len(exc)
    var = u + beta / xi * ((n / nu * alpha) ** (-xi) - 1) if abs(xi) > 1e-8 else u - beta * np.log(n / nu * alpha)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.inf
    return float(var), float(es)


def hill_tail_index(pnl, k: int | None = None) -> float:
    L = np.sort(-np.asarray(pnl, dtype=float))[::-1]
    L = L[L > 0]
    k = k or max(10, int(0.05 * len(L)))
    return float(1.0 / np.mean(np.log(L[:k] / L[k])))


def fhs_var_es(r: pd.Series, alpha: float = 0.01, vol: pd.Series | None = None) -> tuple[float, float]:
    """Filtered historical simulation: standardized residuals from a GARCH volatility, rescaled by the
    one-day-ahead volatility forecast."""
    from contagion.regimes.volatility import garch

    g = garch(r) if vol is None else None
    v = g["vol"] if vol is None else vol
    z = (r / v).dropna()
    res = g["result"] if g else None
    sig_next = float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])) / 100 if res is not None else float(v.iloc[-1])
    sims = z.to_numpy() * sig_next
    return var_es(sims, alpha, "historical")


# ----------------------------------------------------------------------------------- decomposition
def component_vol(weights: dict[str, float], Sigma: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility: contribution_i = w_i (Sigma w)_i / sigma_p (sums to sigma_p)."""
    w = pd.Series(weights, dtype=float)
    S = Sigma.loc[w.index, w.index].to_numpy()
    wv = w.to_numpy()
    sp = float(np.sqrt(wv @ S @ wv))
    mc = S @ wv / sp
    cc = wv * mc
    return pd.DataFrame({"weight": wv, "marginal": mc, "contribution": cc, "share": cc / sp}, index=w.index)


def component_es(R_log: pd.DataFrame, weights: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Historical ES contributions: -E[w_i r_i | portfolio loss beyond VaR] (sums to portfolio ES)."""
    w = pd.Series(weights, dtype=float)
    Rs = simple_returns(R_log[w.index])
    p = (Rs * w).sum(axis=1)
    q = p.quantile(alpha)
    tail = Rs.loc[p <= q]
    contrib = -(tail * w).mean()
    return pd.DataFrame({"weight": w, "es_contribution": contrib, "share": contrib / contrib.sum()})


def factor_decomposition(port: pd.Series, F: pd.DataFrame) -> dict:
    """OLS of portfolio returns on factors; variance split into factor contributions (Euler on the factor
    covariance) and residual."""
    d = pd.concat([port, F], axis=1).dropna()
    y, X = d.iloc[:, 0].to_numpy(), d.iloc[:, 1:].to_numpy()
    Xc = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(Xc, y, rcond=None)[0]
    beta = b[1:]
    SF = np.cov(X, rowvar=False)
    fvar = beta @ SF @ beta
    resid = y - Xc @ b
    tot = fvar + resid.var(ddof=1)
    contrib = beta * (SF @ beta) / tot
    out = pd.DataFrame({"beta": beta, "variance_share": contrib}, index=F.columns)
    return {"table": out, "residual_share": float(resid.var(ddof=1) / tot), "r2": float(fvar / tot)}


# ----------------------------------------------------------------------------------- backtests
def kupiec_pof(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int(hits.sum())
    p = x / n if n else 0
    if x in (0, n):
        lr = -2 * (n * np.log(1 - alpha) if x == 0 else n * np.log(alpha))
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha) - (n - x) * np.log(1 - p) - x * np.log(p))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(pp, k0, k1):
        return (k0 * np.log(1 - pp) if k0 else 0) + (k1 * np.log(pp) if k1 and pp > 0 else 0)

    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(lr), float(stats.chi2.sf(lr, 1))
