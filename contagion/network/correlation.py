"""Correlation-based dependence: rolling / EWMA covariance tensors, partial correlations, shrinkage, exceedance
correlations and tests for changes in correlation.

All measures in this module are DESCRIPTIVE (co-movement only).

Heteroskedasticity caveat (Forbes & Rigobon 2002; Boyer, Gibson & Loretan 1999): conditioning on high-volatility
or tail periods raises measured correlation even when the underlying dependence is unchanged. Use
``forbes_rigobon_adjust`` and the Gaussian benchmark in ``exceedance_correlation`` before calling a rise
"contagion".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------------- covariance tensors
def rolling_cov(X: np.ndarray, window: int) -> np.ndarray:
    """(T, N, N) sample covariance over [t-window+1, t]; entries before ``window-1`` are NaN."""
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    S1 = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])
    S2 = np.concatenate([np.zeros((1, N, N)), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    out = np.full((T, N, N), np.nan)
    s1 = S1[window:] - S1[:-window]
    s2 = S2[window:] - S2[:-window]
    out[window - 1:] = (s2 - s1[:, :, None] * s1[:, None, :] / window) / (window - 1)
    return out


def ewma_cov(X: np.ndarray, lam: float = 0.94, init: int = 60, demean: bool = False) -> np.ndarray:
    """RiskMetrics EWMA covariance using information up to and including t. Initialised with the sample
    covariance of the first ``init`` observations (entries before ``init-1`` are NaN)."""
    X = np.asarray(X, dtype=float)
    if demean:
        X = X - X[:init].mean(axis=0)
    T, N = X.shape
    out = np.full((T, N, N), np.nan)
    S = X[:init].T @ X[:init] / init
    out[init - 1] = S
    for t in range(init, T):
        x = X[t]
        S = lam * S + (1 - lam) * np.outer(x, x)
        out[t] = S
    return out


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    """Works on (N, N) or (T, N, N)."""
    d = np.sqrt(np.clip(np.diagonal(S, axis1=-2, axis2=-1), 1e-300, None))
    C = S / (d[..., :, None] * d[..., None, :])
    idx = np.arange(S.shape[-1])
    C[..., idx, idx] = 1.0
    return C


def average_correlation(C: np.ndarray) -> np.ndarray:
    """Mean off-diagonal correlation of (N, N) or (T, N, N)."""
    N = C.shape[-1]
    return (np.nansum(C, axis=(-2, -1)) - N) / (N * (N - 1))


def partial_correlation(S: np.ndarray) -> np.ndarray:
    """Partial correlations from a covariance matrix: -P_ij / sqrt(P_ii P_jj), P = S^-1."""
    P = np.linalg.pinv(S)
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(np.asarray(X, dtype=float)).covariance_


def glasso_partial_correlation(X: np.ndarray, alpha: float | None = None) -> tuple[np.ndarray, float]:
    """Sparse partial-correlation network via the graphical lasso on standardized data.
    ``alpha=None`` picks the penalty by 5-fold cross-validation. Returns (partial correlations, alpha)."""
    from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

    Z = np.asarray(X, dtype=float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    if alpha is None:
        m = GraphicalLassoCV(cv=5, max_iter=500).fit(Z)
        alpha = float(m.alpha_)
    else:
        m = GraphicalLasso(alpha=alpha, max_iter=500).fit(Z)
    P = m.precision_
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc, alpha


# ----------------------------------------------------------------------------------- frames
def rolling_corr_frame(R: pd.DataFrame, window: int = 250) -> dict:
    """Convenience: rolling correlation tensor + average correlation series for a DataFrame of returns."""
    C = cov_to_corr(rolling_cov(R.to_numpy(), window))
    avg = pd.Series(average_correlation(C), index=R.index, name=f"avg_corr_{window}")
    return {"corr": C, "avg": avg, "names": list(R.columns), "dates": R.index}


def pair_rolling_corr(x: pd.Series, y: pd.Series, window: int = 63) -> pd.Series:
    return x.rolling(window).corr(y)


# ----------------------------------------------------------------------------------- tails
def exceedance_correlation(x, y, quantiles=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95), benchmark: bool = True,
                           n_sim: int = 400_000, seed: int = 7) -> pd.DataFrame:
    """Longin-Solnik / Ang-Chen exceedance correlations.

    q < 0.5: corr(x, y | x <= Q_x(q) and y <= Q_y(q));  q >= 0.5: corr(x, y | x >= Q_x(q) and y >= Q_y(q)).
    ``gauss`` is the same statistic for a bivariate normal with the sample's normal-scores correlation: under
    normality exceedance correlations SHRINK toward zero in the tails, so an empirical value above the benchmark
    is evidence of genuine tail dependence (and not the conditioning bias).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    rx = stats.norm.ppf(stats.rankdata(x) / (n + 1))
    ry = stats.norm.ppf(stats.rankdata(y) / (n + 1))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if benchmark:
        rng = np.random.default_rng(seed)
        z1 = rng.standard_normal(n_sim)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_sim)
    rows = []
    for q in quantiles:
        if q < 0.5:
            m = (x <= np.quantile(x, q)) & (y <= np.quantile(y, q))
        else:
            m = (x >= np.quantile(x, q)) & (y >= np.quantile(y, q))
        c = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 5 else np.nan
        row = {"q": q, "n": int(m.sum()), "corr": c}
        if benchmark:
            if q < 0.5:
                mm = (z1 <= np.quantile(z1, q)) & (z2 <= np.quantile(z2, q))
            else:
                mm = (z1 >= np.quantile(z1, q)) & (z2 >= np.quantile(z2, q))
            row["gauss"] = float(np.corrcoef(z1[mm], z2[mm])[0, 1])
        rows.append(row)
    out = pd.DataFrame(rows).set_index("q")
    out.attrs["normal_scores_rho"] = rho
    return out


def forbes_rigobon_adjust(rho_crisis: float, var_crisis: float, var_calm: float) -> float:
    """Heteroskedasticity-adjusted crisis correlation rho* = rho / sqrt(1 + delta (1 - rho^2)),
    delta = var_crisis / var_calm - 1 of the *source* market. Assumes no omitted common shocks / endogeneity."""
    delta = var_crisis / var_calm - 1.0
    return float(rho_crisis / np.sqrt(1.0 + delta * (1.0 - rho_crisis ** 2)))


# ----------------------------------------------------------------------------------- tests
def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Two-sided test of equal correlations in two independent samples (iid normal approximation)."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def jennrich_test(X1: np.ndarray, X2: np.ndarray) -> tuple[float, int, float]:
    """Jennrich (1970) chi-square test that two samples share the same correlation MATRIX.

    chi2 = 0.5 tr(Z^2) - dg(Z)' S^-1 dg(Z),  Z = sqrt(c) Rbar^-1 (R1 - R2),  c = n1 n2 / (n1 + n2),
    S_ij = delta_ij + rbar_ij rbar^ij,  df = p (p - 1) / 2.  Assumes iid observations within each sample.
    """
    X1, X2 = np.asarray(X1, float), np.asarray(X2, float)
    n1, n2 = len(X1), len(X2)
    R1, R2 = np.corrcoef(X1, rowvar=False), np.corrcoef(X2, rowvar=False)
    p = R1.shape[0]
    Rb = (n1 * R1 + n2 * R2) / (n1 + n2)
    Rbi = np.linalg.inv(Rb)
    c = n1 * n2 / (n1 + n2)
    Z = np.sqrt(c) * Rbi @ (R1 - R2)
    S = np.eye(p) + Rb * Rbi
    dz = np.diag(Z)
    chi2 = 0.5 * np.trace(Z @ Z) - dz @ np.linalg.solve(S, dz)
    df = p * (p - 1) // 2
    return float(chi2), df, float(stats.chi2.sf(chi2, df))
