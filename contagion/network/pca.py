"""Principal components of the correlation matrix: absorption ratio and market-mode loadings.

Absorption ratio (Kritzman, Li, Page & Rigobon 2011): share of total variance absorbed by the first k
eigenvectors. High AR = markets "tightly coupled", so shocks spread more broadly. Computed on the CORRELATION
matrix here because the universe mixes units (log returns, basis points). Descriptive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation import cov_to_corr, rolling_cov


def eigvals_desc(C: np.ndarray) -> np.ndarray:
    """Eigenvalues in descending order for (N, N) or (T, N, N) (NaN slices stay NaN)."""
    C = np.asarray(C, dtype=float)
    if C.ndim == 2:
        return np.linalg.eigvalsh(C)[::-1]
    out = np.full(C.shape[:2], np.nan)
    ok = np.isfinite(C).all(axis=(1, 2))
    out[ok] = np.linalg.eigvalsh(C[ok])[:, ::-1]
    return out


def absorption_ratio(R: pd.DataFrame, window: int = 250, k: int | None = None) -> pd.Series:
    N = R.shape[1]
    k = k or max(1, int(round(N / 5)))
    C = cov_to_corr(rolling_cov(R.to_numpy(), window))
    ev = eigvals_desc(C)
    ar = ev[:, :k].sum(axis=1) / ev.sum(axis=1)
    return pd.Series(ar, index=R.index, name=f"AR{k}_{window}")


def ar_shift(ar: pd.Series, short: int = 15, long: int = 252) -> pd.Series:
    """Standardized AR shift: (mean_short - mean_long) / std_long (Kritzman et al. 2011)."""
    return (ar.rolling(short).mean() - ar.rolling(long).mean()) / ar.rolling(long).std()


def first_pc_share(R: pd.DataFrame, window: int = 250) -> pd.Series:
    C = cov_to_corr(rolling_cov(R.to_numpy(), window))
    ev = eigvals_desc(C)
    return pd.Series(ev[:, 0] / ev.sum(axis=1), index=R.index, name=f"PC1_{window}")


def pca(R: pd.DataFrame, k: int = 3) -> dict:
    """Static PCA on standardized returns: loadings (N, k), factor scores (T, k), explained variance ratios."""
    Z = (R - R.mean()) / R.std()
    C = np.corrcoef(Z.to_numpy(), rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    V = V * np.sign(V.sum(axis=0))  # orientation: positive sum of loadings
    scores = Z.to_numpy() @ V[:, :k]
    return {"loadings": pd.DataFrame(V[:, :k], index=R.columns, columns=[f"PC{i+1}" for i in range(k)]),
            "scores": pd.DataFrame(scores, index=R.index, columns=[f"PC{i+1}" for i in range(k)]),
            "explained": pd.Series(w / w.sum(), index=[f"PC{i+1}" for i in range(len(w))])}
