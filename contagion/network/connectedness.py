"""Diebold-Yilmaz connectedness from generalized forecast-error variance decompositions (GFEVD).

    theta_ij(H) = sigma_jj^-1 sum_{h<H} (e_i' Psi_h Sigma e_j)^2 / sum_{h<H} (e_i' Psi_h Sigma Psi_h' e_i)

rows normalised to one (Diebold & Yilmaz 2012). From the normalised table:
    FROM_i = sum_{j!=i} theta_ij   (share of i's forecast-error variance attributed to others)
    TO_j   = sum_{i!=j} theta_ij   (j's contribution to everybody else)
    NET_j  = TO_j - FROM_j          (> 0: "net transmitter")
    TCI    = 100 * sum_{i!=j} theta_ij / N

What the numbers can and cannot say
-----------------------------------
* The h = 0 term is (Sigma_ij)^2 / (sigma_ii sigma_jj) = rho_ij^2: pure contemporaneous correlation, symmetric
  in i and j. ``split_contemporaneous`` separates it from the h >= 1 (lagged, predictive) part.
* Row normalisation makes NET non-zero even when there are NO dynamics at all: a node correlated with many others
  has a large row sum, gets scaled down as a receiver and therefore looks like a net transmitter. NET from the
  contemporaneous part measures *centrality*, not direction. Only the dynamic part carries (predictive)
  direction, and even that is Granger-type direction in time, not causation. tests/test_connectedness.py
  demonstrates both facts on simulated data.
* GFEVD shares are invariant to rescaling any series (checked in tests).

Dynamic versions
----------------
``rolling_connectedness``  re-estimates a (ridge) VAR on a moving window.
``tvp_var_connectedness``  time-varying-parameter VAR with Kalman-filter forgetting factors (Koop & Korobilis 2014;
                           Antonakakis, Chatziantoniou & Gabauer 2020). Filtered estimates only: no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .var import fit_var, lagmat, ma_coefficients


def gfevd(A: np.ndarray, Sigma: np.ndarray, H: int = 10, split: bool = False):
    """Row-normalised GFEVD table. With ``split=True`` also return its contemporaneous (h=0) and dynamic parts,
    which add up to the full table."""
    Psi = ma_coefficients(A, H)
    PS = Psi @ Sigma  # (H, N, N): e_i' Psi_h Sigma e_j
    sjj = np.diag(Sigma)
    num_h = PS ** 2 / sjj[None, None, :]
    num = num_h.sum(axis=0)
    den = np.einsum("hij,jk,hik->i", Psi, Sigma, Psi)
    raw = num / den[:, None]
    rs = raw.sum(axis=1, keepdims=True)
    theta = raw / rs
    if not split:
        return theta
    contemporaneous = (num_h[0] / den[:, None]) / rs
    return theta, contemporaneous, theta - contemporaneous


@dataclass
class Connectedness:
    names: list[str]
    theta: np.ndarray  # normalised GFEVD, rows sum to 1
    theta_contemp: np.ndarray | None = None
    theta_dynamic: np.ndarray | None = None

    @property
    def N(self) -> int:
        return len(self.names)

    def _off(self, M: np.ndarray) -> np.ndarray:
        M = M.copy()
        np.fill_diagonal(M, 0.0)
        return M

    @property
    def total(self) -> float:
        return 100.0 * self._off(self.theta).sum() / self.N

    @property
    def to(self) -> pd.Series:
        return pd.Series(100.0 * self._off(self.theta).sum(axis=0), index=self.names)

    @property
    def from_(self) -> pd.Series:
        return pd.Series(100.0 * self._off(self.theta).sum(axis=1), index=self.names)

    @property
    def net(self) -> pd.Series:
        return self.to - self.from_

    @property
    def pairwise_net(self) -> pd.DataFrame:
        """[i, j] = net flow from j to i ( theta_ij - theta_ji ), in percent."""
        return pd.DataFrame(100.0 * (self.theta - self.theta.T), index=self.names, columns=self.names)

    def table(self) -> pd.DataFrame:
        """The classic DY table: rows = receivers, columns = transmitters, plus FROM / TO / NET margins."""
        df = pd.DataFrame(100.0 * self.theta, index=self.names, columns=self.names)
        df["FROM"] = self.from_
        to = self.to.reindex(df.columns[:-1])
        df.loc["TO"] = list(to) + [self.total]
        df.loc["NET"] = list(self.net.reindex(df.columns[:-1])) + [np.nan]
        return df

    def dynamic_share(self) -> float:
        """Share of off-diagonal connectedness that comes from lagged dynamics (h >= 1) rather than h = 0."""
        if self.theta_dynamic is None:
            return np.nan
        d = self._off(self.theta_dynamic).sum()
        return float(d / self._off(self.theta).sum())

    def net_dynamic(self) -> pd.Series:
        """NET computed from the dynamic part only: the (predictive) directional component."""
        D = self._off(self.theta_dynamic)
        return pd.Series(100.0 * (D.sum(axis=0) - D.sum(axis=1)), index=self.names)

    def net_contemporaneous(self) -> pd.Series:
        """NET from the h = 0 part only: driven by centrality + row normalisation, not by direction."""
        C = self._off(self.theta_contemp)
        return pd.Series(100.0 * (C.sum(axis=0) - C.sum(axis=1)), index=self.names)

    def block(self, groups: dict[str, str]) -> pd.DataFrame:
        """Aggregate pairwise spillovers to blocks (e.g. asset classes): [receiver block, transmitter block], in
        percent of the receiving block's total forecast-error variance."""
        labels = [groups[n] for n in self.names]
        blocks = sorted(set(labels), key=labels.index)
        idx = {b: [i for i, l in enumerate(labels) if l == b] for b in blocks}
        out = pd.DataFrame(0.0, index=blocks, columns=blocks)
        for r in blocks:
            for c in blocks:
                out.loc[r, c] = 100.0 * self.theta[np.ix_(idx[r], idx[c])].sum() / len(idx[r])
        return out


def connectedness(Y, p: int = 1, H: int = 10, ridge: float = 0.0) -> Connectedness:
    names = list(Y.columns) if isinstance(Y, pd.DataFrame) else None
    f = fit_var(Y, p=p, ridge=ridge, names=names)
    th, c, d = gfevd(f.A, f.Sigma, H, split=True)
    return Connectedness(f.names, th, c, d)


@dataclass
class DynamicConnectedness:
    """Time series of connectedness measures."""

    names: list[str]
    dates: pd.DatetimeIndex
    theta: np.ndarray  # (T, N, N)
    theta_contemp: np.ndarray  # (T, N, N)
    method: str = ""
    params: dict = field(default_factory=dict)

    def _offsum(self, M, axis):
        M = M.copy()
        idx = np.arange(M.shape[1])
        M[:, idx, idx] = 0.0
        return M.sum(axis=axis)

    @property
    def total(self) -> pd.Series:
        N = len(self.names)
        return pd.Series(100.0 * self._offsum(self.theta, (1, 2)) / N, index=self.dates, name="TCI")

    @property
    def total_dynamic(self) -> pd.Series:
        N = len(self.names)
        return pd.Series(100.0 * self._offsum(self.theta - self.theta_contemp, (1, 2)) / N, index=self.dates,
                         name="TCI_dynamic")

    @property
    def to(self) -> pd.DataFrame:
        return pd.DataFrame(100.0 * self._offsum(self.theta, 1), index=self.dates, columns=self.names)

    @property
    def from_(self) -> pd.DataFrame:
        return pd.DataFrame(100.0 * self._offsum(self.theta, 2), index=self.dates, columns=self.names)

    @property
    def net(self) -> pd.DataFrame:
        return self.to - self.from_

    @property
    def net_dynamic(self) -> pd.DataFrame:
        D = self.theta - self.theta_contemp
        return pd.DataFrame(100.0 * (self._offsum(D, 1) - self._offsum(D, 2)), index=self.dates, columns=self.names)

    def at(self, date) -> Connectedness:
        k = self.dates.get_indexer([pd.Timestamp(date)], method="ffill")[0]
        th = self.theta[k]
        return Connectedness(self.names, th, self.theta_contemp[k], th - self.theta_contemp[k])


def rolling_connectedness(Y: pd.DataFrame, window: int = 500, step: int = 5, p: int = 1, H: int = 10,
                          ridge: float = 0.0) -> DynamicConnectedness:
    """Re-estimate the VAR on [t-window+1, t] every ``step`` days. Stamped at the window's last date."""
    X = Y.to_numpy(dtype=float)
    T, N = X.shape
    ends = list(range(window, T + 1, step))
    if ends and ends[-1] != T:
        ends.append(T)
    th = np.empty((len(ends), N, N))
    tc = np.empty_like(th)
    for k, e in enumerate(ends):
        f = fit_var(X[e - window:e], p=p, ridge=ridge)
        th[k], tc[k], _ = gfevd(f.A, f.Sigma, H, split=True)
    dates = Y.index[[e - 1 for e in ends]]
    return DynamicConnectedness(list(Y.columns), pd.DatetimeIndex(dates), th, tc, "rolling",
                                {"window": window, "step": step, "p": p, "H": H, "ridge": ridge})


def tvp_var_filter(Y: np.ndarray, p: int = 1, kappa1: float = 0.99, kappa2: float = 0.96, prior_obs: int = 250,
                   prior_scale: float = 1.0):
    """Kalman filter for a TVP-VAR with forgetting factors. Returns filtered (A_t, Sigma_t, c_t) for t >= p.

    State: beta_t = vec of [c, A_1..A_p] per equation (N * (N p + 1)), random walk with covariance inflated by
    1/kappa1 each period. Measurement covariance: EWMA of one-step prediction errors with decay kappa2.
    The prior comes from OLS on the first ``prior_obs`` observations (those dates are part of the burn-in and
    their estimates should not be interpreted).
    """
    T, N = Y.shape
    k = N * p + 1
    Yt, Xl = lagmat(Y, p)
    Z = np.hstack([np.ones((Yt.shape[0], 1)), Xl])  # (n, k)
    n = Yt.shape[0]
    m = min(prior_obs, n)
    B0, *_ = np.linalg.lstsq(Z[:m], Yt[:m], rcond=None)  # (k, N)
    U0 = Yt[:m] - Z[:m] @ B0
    Sig = U0.T @ U0 / max(m - k, 1)
    beta = B0.T.reshape(-1).copy()  # equation-major: [eq0 coeffs (k), eq1 coeffs (k), ...]
    ZZinv = np.linalg.pinv(Z[:m].T @ Z[:m])
    P = prior_scale * np.kron(Sig, ZZinv)
    A_out = np.empty((n, p, N, N))
    S_out = np.empty((n, N, N))
    c_out = np.empty((n, N))
    for t in range(n):
        z = Z[t]
        P = P / kappa1
        Bmat = beta.reshape(N, k)
        e = Yt[t] - Bmat @ z
        Sig = kappa2 * Sig + (1 - kappa2) * np.outer(e, e)
        # X_t = I_N kron z' ; XP = X_t P (N x Nk)
        Pr = P.reshape(N, k, N * k)
        XP = np.einsum("j,ijm->im", z, Pr)
        S = np.einsum("inj,j->in", XP.reshape(N, N, k), z) + Sig
        Sinv = np.linalg.inv(S)
        K = XP.T @ Sinv  # (Nk, N)
        beta = beta + K @ e
        P = P - K @ XP
        P = (P + P.T) / 2
        Bmat = beta.reshape(N, k)
        c_out[t] = Bmat[:, 0]
        A_out[t] = np.stack([Bmat[:, 1 + l * N: 1 + (l + 1) * N] for l in range(p)])
        S_out[t] = Sig
    return A_out, S_out, c_out


def tvp_var_connectedness(Y: pd.DataFrame, p: int = 1, H: int = 10, kappa1: float = 0.99, kappa2: float = 0.96,
                          prior_obs: int = 250, standardize: bool = True) -> DynamicConnectedness:
    """TVP-VAR connectedness at every date (filtered). Series are scaled by their full-sample standard deviation
    for numerical conditioning only; GFEVD shares do not depend on scale."""
    X = Y.to_numpy(dtype=float)
    if standardize:
        X = X / X.std(axis=0)
    A, S, _ = tvp_var_filter(X, p=p, kappa1=kappa1, kappa2=kappa2, prior_obs=prior_obs)
    n, N = A.shape[0], X.shape[1]
    th = np.empty((n, N, N))
    tc = np.empty_like(th)
    for t in range(n):
        th[t], tc[t], _ = gfevd(A[t], S[t], H, split=True)
    dates = Y.index[p:]
    return DynamicConnectedness(list(Y.columns), pd.DatetimeIndex(dates), th, tc, "tvp_var",
                                {"p": p, "H": H, "kappa1": kappa1, "kappa2": kappa2, "prior_obs": prior_obs})
