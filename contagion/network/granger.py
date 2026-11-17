"""Granger-causality networks with heteroskedasticity-robust Wald tests and false-discovery-rate control.

j "Granger-causes" i if lags of j improve the forecast of i given i's own lags (pairwise) or given all other
series' lags (conditional / VAR block exclusion). This is PREDICTIVE content, not causation. In daily asset data
the main sources of significant links are:
  * asynchronous closes (a market that settles at 13:30 ET "predicts" one that closes at 16:00 the next day,
    and vice versa for later closes) - see contagion.data.calendar.lead_lag_profile;
  * stale or smoothed prices (index marks of illiquid bonds lag liquid markets);
  * common drivers observed with different delays.
Volatility clustering makes classic F-tests over-reject; HC1 robust covariances are used throughout.
Edges are kept after Benjamini-Hochberg FDR control across all N(N-1) tests (Billio, Getmansky, Lo & Pelizzon
2012 use the share of significant links, "DGC", as a systemic-risk indicator).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .var import lagmat


def _hc1_wald(X: np.ndarray, y: np.ndarray, test_idx: np.ndarray) -> tuple[float, np.ndarray]:
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = (X * e[:, None] ** 2).T @ X
    V = XtX_inv @ meat @ XtX_inv * n / (n - k)
    bt = b[test_idx]
    Vt = V[np.ix_(test_idx, test_idx)]
    W = float(bt @ np.linalg.solve(Vt, bt))
    return W, b


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejections (boolean array of the same shape; NaNs never rejected)."""
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    ok = np.isfinite(flat)
    m = ok.sum()
    rej = np.zeros_like(flat, dtype=bool)
    if m == 0:
        return rej.reshape(p.shape)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(flat[ok])]
    thresh = q * np.arange(1, m + 1) / m
    passed = flat[order] <= thresh
    if passed.any():
        kmax = np.max(np.flatnonzero(passed))
        rej[order[: kmax + 1]] = True
    return rej.reshape(p.shape)


def pairwise_granger(Y, p: int = 1) -> dict:
    """Pairwise tests. Returns {'stat', 'pval'} as (N, N) arrays with [i, j] = test that j Granger-causes i."""
    names = list(Y.columns) if isinstance(Y, pd.DataFrame) else None
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    Yt, X = lagmat(Y, p)
    n = Yt.shape[0]
    stat = np.full((N, N), np.nan)
    for i in range(N):
        own = X[:, [i + l * N for l in range(p)]]
        for j in range(N):
            if i == j:
                continue
            oth = X[:, [j + l * N for l in range(p)]]
            D = np.hstack([np.ones((n, 1)), own, oth])
            W, _ = _hc1_wald(D, Yt[:, i], np.arange(1 + p, 1 + 2 * p))
            stat[i, j] = W
    pval = stats.chi2.sf(stat, p)
    return {"stat": stat, "pval": pval, "names": names, "p": p, "kind": "pairwise"}


def conditional_granger(Y, p: int = 1) -> dict:
    """Block-exclusion tests inside the full VAR(p): does j help predict i given ALL series' lags?"""
    names = list(Y.columns) if isinstance(Y, pd.DataFrame) else None
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    Yt, X = lagmat(Y, p)
    n = Yt.shape[0]
    D = np.hstack([np.ones((n, 1)), X])
    k = D.shape[1]
    XtX_inv = np.linalg.pinv(D.T @ D)
    B = XtX_inv @ D.T @ Yt
    E = Yt - D @ B
    stat = np.full((N, N), np.nan)
    for i in range(N):
        meat = (D * E[:, i][:, None] ** 2).T @ D
        V = XtX_inv @ meat @ XtX_inv * n / (n - k)
        for j in range(N):
            if i == j:
                continue
            idx = np.array([1 + j + l * N for l in range(p)])
            bt = B[idx, i]
            stat[i, j] = float(bt @ np.linalg.solve(V[np.ix_(idx, idx)], bt))
    pval = stats.chi2.sf(stat, p)
    return {"stat": stat, "pval": pval, "names": names, "p": p, "kind": "conditional"}


def granger_network(res: dict, q: float = 0.05) -> pd.DataFrame:
    """Adjacency [i, j] = 1 if j -> i survives BH-FDR at level q."""
    A = bh_fdr(res["pval"], q).astype(int)
    np.fill_diagonal(A, 0)
    return pd.DataFrame(A, index=res["names"], columns=res["names"])


def degree_table(adj: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"out_degree": adj.sum(axis=0), "in_degree": adj.sum(axis=1),
                         "net_degree": adj.sum(axis=0) - adj.sum(axis=1)})


def rolling_granger(R: pd.DataFrame, window: int = 500, step: int = 21, p: int = 1, q: float = 0.05,
                    kind: str = "pairwise") -> dict:
    """Granger network re-estimated on moving windows. Returns density (share of significant directed links)
    and out/in-degree time series."""
    X = R.to_numpy(dtype=float)
    T, N = X.shape
    ends = list(range(window, T + 1, step))
    f = pairwise_granger if kind == "pairwise" else conditional_granger
    dens, outd, ind, adjs = [], [], [], []
    for e in ends:
        res = f(X[e - window:e], p=p)
        A = bh_fdr(res["pval"], q).astype(int)
        np.fill_diagonal(A, 0)
        adjs.append(A)
        dens.append(A.sum() / (N * (N - 1)))
        outd.append(A.sum(axis=0))
        ind.append(A.sum(axis=1))
    dates = R.index[[e - 1 for e in ends]]
    return {"dates": dates, "density": pd.Series(dens, index=dates, name="granger_density"),
            "out_degree": pd.DataFrame(outd, index=dates, columns=R.columns),
            "in_degree": pd.DataFrame(ind, index=dates, columns=R.columns),
            "adjacency": np.array(adjs), "params": {"window": window, "step": step, "p": p, "q": q, "kind": kind}}
