"""Vector autoregressions: OLS / ridge estimation, lag selection, MA representation, impulse responses.

    y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   E[u_t u_t'] = Sigma

Written in numpy (rather than statsmodels) because the connectedness code refits thousands of rolling windows;
``tests/test_var.py`` checks the estimates against statsmodels.

Interpretation: a VAR is a reduced-form forecasting model. Lag coefficients and generalized impulse responses
describe *predictive* dynamics; they become causal only under an identification scheme (e.g. a Cholesky order,
which is an assumption about who moves first within the day and is rarely defensible for daily asset returns).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VARFit:
    names: list[str]
    p: int
    const: np.ndarray  # (N,)
    A: np.ndarray  # (p, N, N)
    Sigma: np.ndarray  # (N, N)
    resid: np.ndarray  # (T-p, N)
    nobs: int
    ridge: float = 0.0

    @property
    def N(self) -> int:
        return self.A.shape[1]

    def companion(self) -> np.ndarray:
        N, p = self.N, self.p
        C = np.zeros((N * p, N * p))
        C[:N, :] = np.hstack(list(self.A))
        if p > 1:
            C[N:, :-N] = np.eye(N * (p - 1))
        return C

    def is_stable(self) -> bool:
        return bool(np.max(np.abs(np.linalg.eigvals(self.companion()))) < 1.0)

    def ma(self, H: int) -> np.ndarray:
        return ma_coefficients(self.A, H)

    def girf(self, H: int, shock: int, size: float | None = None) -> np.ndarray:
        return girf(self.A, self.Sigma, H, shock, size)

    def cholesky_irf(self, H: int) -> np.ndarray:
        P = np.linalg.cholesky(self.Sigma)
        return ma_coefficients(self.A, H) @ P


def lagmat(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y_t, X_t) with X_t = [y_{t-1}', ..., y_{t-p}'] for t = p..T-1."""
    T, N = Y.shape
    X = np.hstack([Y[p - l: T - l] for l in range(1, p + 1)])
    return Y[p:], X


def fit_var(Y, p: int = 1, ridge: float = 0.0, const: bool = True, names: list[str] | None = None) -> VARFit:
    """OLS (ridge=0) or ridge VAR.

    Ridge is applied to standardized data and mapped back, so the penalty does not depend on the units of each
    series (log returns vs basis points). The intercept is never penalized.
    """
    if isinstance(Y, pd.DataFrame):
        names = names or list(Y.columns)
        Y = Y.to_numpy(dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    names = names or [f"y{i}" for i in range(N)]
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Y - mu) / sd if ridge > 0 else Y
    Yt, X = lagmat(Z, p)
    n = Yt.shape[0]
    Xc = np.hstack([np.ones((n, 1)), X]) if const else X
    if ridge > 0:
        k = Xc.shape[1]
        D = np.eye(k)
        if const:
            D[0, 0] = 0.0
        B = np.linalg.solve(Xc.T @ Xc + ridge * n * D, Xc.T @ Yt)
    else:
        B, *_ = np.linalg.lstsq(Xc, Yt, rcond=None)
    U = Yt - Xc @ B
    c = B[0] if const else np.zeros(N)
    Ablk = B[1:] if const else B  # (N*p, N), rows = regressors
    A = np.stack([Ablk[l * N:(l + 1) * N].T for l in range(p)])  # A[l][i, j]: effect of y_j,t-l-1 on y_i,t
    if ridge > 0:  # map back to original units: y = mu + S z
        S, Sinv = np.diag(sd), np.diag(1.0 / sd)
        A = np.stack([S @ A[l] @ Sinv for l in range(p)])
        c = mu + sd * c - sum(A[l] @ mu for l in range(p))
        U = U * sd
    k_eff = N * p + (1 if const else 0)
    Sigma = U.T @ U / max(n - k_eff, 1)
    return VARFit(names=names, p=p, const=c, A=A, Sigma=Sigma, resid=U, nobs=n, ridge=ridge)


def select_lag(Y, maxlag: int = 5, criterion: str = "bic") -> int:
    """Information-criterion lag choice on a common estimation sample (the first ``maxlag`` rows dropped)."""
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    best, best_p = np.inf, 1
    for p in range(1, maxlag + 1):
        f = fit_var(Y[maxlag - p:], p)
        n = f.nobs
        Sig = f.resid.T @ f.resid / n
        ld = np.linalg.slogdet(Sig)[1]
        k = N * (N * p + 1)
        ic = {"aic": ld + 2 * k / n, "bic": ld + np.log(n) * k / n, "hqic": ld + 2 * np.log(np.log(n)) * k / n}[criterion]
        if ic < best:
            best, best_p = ic, p
    return best_p


def ma_coefficients(A: np.ndarray, H: int) -> np.ndarray:
    """Psi_0..Psi_{H-1} of the MA(inf) representation: Psi_h = sum_{l=1..min(h,p)} A_l Psi_{h-l}."""
    p, N, _ = A.shape
    Psi = np.zeros((H, N, N))
    Psi[0] = np.eye(N)
    for h in range(1, H):
        acc = np.zeros((N, N))
        for l in range(1, min(h, p) + 1):
            acc += A[l - 1] @ Psi[h - l]
        Psi[h] = acc
    return Psi


def girf(A: np.ndarray, Sigma: np.ndarray, H: int, shock: int, size: float | None = None) -> np.ndarray:
    """Generalized impulse response (Pesaran & Shin 1998) to a shock in variable ``shock``.

    Returns (H, N) responses. ``size`` is the shock in the variable's own units (default: one standard deviation).
    The h=0 row equals the conditional expectation of the other variables given the shocked one: it is the
    regression of everything on the shocked variable, not a structural effect.
    """
    Psi = ma_coefficients(A, H)
    s = Sigma[:, shock] / Sigma[shock, shock]
    size = np.sqrt(Sigma[shock, shock]) if size is None else size
    return np.einsum("hij,j->hi", Psi, s) * size
