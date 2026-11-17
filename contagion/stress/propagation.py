"""Shock-propagation engines.

1. ``conditional_gaussian``  E[x_O | x_D = s] = mu_O + Sigma_OD Sigma_DD^-1 (s - mu_D) and the conditional covariance.
   "What usually moves with the shocked series", given a covariance estimate. DESCRIPTIVE, not causal: shocking
   SPY does not mean SPY *causes* the other moves; the same regression links SPY to whatever drives it.

2. ``network_decomposition`` splits that conditional expectation along the partial-correlation (precision)
   network. With nodewise regression weights B = I - diag(Theta)^-1 Theta (Theta = Sigma^-1, zero diagonal):
       total        = (I - B_OO)^-1 B_OD s        (identical to 1.)
       first order  = B_OD s                      (through direct conditional links of the shocked nodes)
       higher order = total - first order         (through intermediate, unshocked nodes)
   A portfolio's exposure is then: direct (it holds the shocked series) + first-order + higher-order network.

3. ``var_propagation`` adds time: the day-0 innovation implied by the shock (as in 1., applied to VAR residuals)
   is pushed through the VAR's MA coefficients; impact (h=0) vs subsequent (h>=1) propagation. PREDICTIVE.

4. ``tail_betas``: the same conditional move estimated only from days in the driver's tail (empirical) or from a
   fitted Student-t copula with empirical margins (Monte Carlo). Captures tail dependence the covariance misses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _split(names: list[str], shocks: dict[str, float]):
    D = [names.index(k) for k in shocks]
    O = [i for i in range(len(names)) if i not in D]
    s = np.array([shocks[names[i]] for i in D], dtype=float)
    return D, O, s


def conditional_gaussian(Sigma: pd.DataFrame, shocks: dict[str, float], mu: pd.Series | None = None) -> pd.DataFrame:
    names = list(Sigma.index)
    S = Sigma.to_numpy(dtype=float)
    D, O, s = _split(names, shocks)
    m = np.zeros(len(names)) if mu is None else mu.reindex(names).to_numpy(dtype=float)
    SDD_inv = np.linalg.pinv(S[np.ix_(D, D)])
    B = S[np.ix_(O, D)] @ SDD_inv
    mean = np.empty(len(names))
    sd = np.zeros(len(names))
    mean[D] = s
    mean[O] = m[O] + B @ (s - m[D])
    cond = S[np.ix_(O, O)] - B @ S[np.ix_(D, O)]
    sd[O] = np.sqrt(np.clip(np.diag(cond), 0, None))
    out = pd.DataFrame({"expected": mean, "cond_sd": sd, "shocked": [i in D for i in range(len(names))]}, index=names)
    out.attrs["cond_cov_O"] = pd.DataFrame(cond, index=[names[i] for i in O], columns=[names[i] for i in O])
    return out


def network_decomposition(Sigma: pd.DataFrame, shocks: dict[str, float]) -> pd.DataFrame:
    names = list(Sigma.index)
    Theta = np.linalg.pinv(Sigma.to_numpy(dtype=float))
    d = np.diag(Theta)
    Bfull = np.eye(len(names)) - Theta / d[:, None]
    np.fill_diagonal(Bfull, 0.0)
    D, O, s = _split(names, shocks)
    B_OO, B_OD = Bfull[np.ix_(O, O)], Bfull[np.ix_(O, D)]
    first = B_OD @ s
    total = np.linalg.solve(np.eye(len(O)) - B_OO, first)
    out = pd.DataFrame(0.0, index=names, columns=["total", "first_order", "higher_order"])
    out.iloc[D, 0] = s
    out.iloc[O, 0] = total
    out.iloc[O, 1] = first
    out.iloc[O, 2] = total - first
    return out


def portfolio_exposure(decomp: pd.DataFrame, weights: dict[str, float], shocks: dict[str, float],
                       simple: pd.Series | None = None) -> dict:
    """Portfolio P&L split into direct / first-order network / higher-order network (in return units of the
    portfolio, using exp(r)-1 for log-return assets)."""
    w = pd.Series(weights, dtype=float)
    held = [a for a in w.index if a in decomp.index]

    def to_simple(x):
        return np.expm1(x)

    direct = sum(w[a] * to_simple(decomp.loc[a, "total"]) for a in held if a in shocks)
    total_ind = sum(w[a] * to_simple(decomp.loc[a, "total"]) for a in held if a not in shocks)
    first = sum(w[a] * to_simple(decomp.loc[a, "first_order"]) for a in held if a not in shocks)
    higher = total_ind - first
    return {"direct": float(direct), "first_order": float(first), "higher_order": float(higher),
            "total": float(direct + total_ind)}


def var_propagation(fit, shocks: dict[str, float], H: int = 10) -> dict:
    """Responses over h = 0..H-1 to day-0 innovations consistent with the shock (generalized, Pesaran-Shin
    style). Returns per-horizon and cumulative responses (DataFrames, horizon x series)."""
    from contagion.network.var import ma_coefficients

    names = fit.names
    D, O, s = _split(names, shocks)
    S = fit.Sigma
    u0 = S[:, D] @ np.linalg.pinv(S[np.ix_(D, D)]) @ s  # E[u | u_D = s]
    Psi = ma_coefficients(fit.A, H)
    path = np.einsum("hij,j->hi", Psi, u0)
    per = pd.DataFrame(path, columns=names)
    cum = per.cumsum()
    return {"per_horizon": per, "cumulative": cum, "impact": per.iloc[0], "after_impact": cum.iloc[-1] - per.iloc[0]}


def tail_betas(R: pd.DataFrame, driver: str, direction: int = -1, q: float = 0.05) -> pd.Series:
    """Empirical tail regression: E[x | driver in its q tail] / E[driver | driver in its q tail].
    ``direction`` -1 uses the lower tail (falls), +1 the upper tail."""
    x = R[driver]
    m = x <= x.quantile(q) if direction < 0 else x >= x.quantile(1 - q)
    sub = R.loc[m]
    return sub.mean() / sub[driver].mean()


def tail_betas_copula(R: pd.DataFrame, driver: str, direction: int = -1, q: float = 0.05, n_sim: int = 200_000,
                      seed: int = 0) -> pd.Series:
    """Tail betas from a Student-t copula fitted to the ranks, with each margin mapped back through its own
    empirical quantiles. Smoother than the empirical version and uses the whole sample's dependence shape."""
    from contagion.network.copula import fit_t_copula, simulate_t_copula

    f = fit_t_copula(R.to_numpy())
    U = simulate_t_copula(f["R"], f["nu"], n_sim, seed=seed)
    sims = np.column_stack([np.quantile(R.iloc[:, j].to_numpy(), U[:, j]) for j in range(R.shape[1])])
    S = pd.DataFrame(sims, columns=R.columns)
    out = tail_betas(S, driver, direction, q)
    out.attrs["nu"] = f["nu"]
    return out
