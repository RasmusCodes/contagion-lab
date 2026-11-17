"""Copulas: bivariate family fits (Gaussian, Student-t, Clayton, Gumbel, survival Gumbel, Frank) and a
multivariate Student-t copula for simulation. Inputs are rank pseudo-observations (margins handled separately).

Tail-dependence coefficients implied by the fitted family:
    Gaussian / Frank: 0 / 0     Student-t: 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))) in both tails
    Clayton: lambda_L = 2^(-1/theta)     Gumbel: lambda_U = 2 - 2^(1/theta)     survival Gumbel: lambda_L = 2 - 2^(1/theta)
Negatively dependent pairs are rotated (v -> 1 - v) before fitting the one-sided families; the rotation is
reported. DESCRIPTIVE: a copula describes the shape of joint co-movement, not a mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .tail import pseudo_obs

EPS = 1e-10


def _clip(u):
    return np.clip(u, EPS, 1 - EPS)


# ------------------------------------------------------------------ log densities
def ll_gaussian(u, v, rho):
    a, b = stats.norm.ppf(_clip(u)), stats.norm.ppf(_clip(v))
    r2 = 1 - rho ** 2
    return np.sum(-0.5 * np.log(r2) - (rho ** 2 * (a ** 2 + b ** 2) - 2 * rho * a * b) / (2 * r2))


def ll_student(u, v, rho, nu):
    a, b = stats.t.ppf(_clip(u), nu), stats.t.ppf(_clip(v), nu)
    r2 = 1 - rho ** 2
    lg = (special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
          - 0.5 * np.log(r2))
    q = (a ** 2 - 2 * rho * a * b + b ** 2) / (nu * r2)
    return np.sum(lg - (nu + 2) / 2 * np.log1p(q) + (nu + 1) / 2 * (np.log1p(a ** 2 / nu) + np.log1p(b ** 2 / nu)))


def ll_clayton(u, v, th):
    u, v = _clip(u), _clip(v)
    return np.sum(np.log1p(th) - (th + 1) * (np.log(u) + np.log(v))
                  - (2 + 1 / th) * np.log(u ** -th + v ** -th - 1))


def ll_gumbel(u, v, th):
    u, v = _clip(u), _clip(v)
    x, y = -np.log(u), -np.log(v)
    s = x ** th + y ** th
    A = s ** (1 / th)
    return np.sum(-A - np.log(u) - np.log(v) + (th - 1) * (np.log(x) + np.log(y))
                  - (2 - 1 / th) * np.log(s) + np.log(A + th - 1))


def ll_frank(u, v, th):
    u, v = _clip(u), _clip(v)
    if abs(th) < 1e-6:
        return 0.0
    num = th * (1 - np.exp(-th)) * np.exp(-th * (u + v))
    den = ((1 - np.exp(-th)) - (1 - np.exp(-th * u)) * (1 - np.exp(-th * v))) ** 2
    return np.sum(np.log(np.abs(num)) - np.log(den))


def t_tail_dependence(rho: float, nu: float) -> float:
    return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))


# ------------------------------------------------------------------ fitting
def fit_bivariate(x, y) -> pd.DataFrame:
    """Fit all families by maximum likelihood on pseudo-observations; rank by AIC."""
    u, v = pseudo_obs(np.asarray(x)), pseudo_obs(np.asarray(y))
    tau = stats.kendalltau(u, v)[0]
    rotated = tau < 0
    vv = 1 - v if rotated else v
    tau_a = abs(tau)
    rows = []
    r0 = np.sin(np.pi * tau / 2)
    res = optimize.minimize_scalar(lambda r: -ll_gaussian(u, v, r), bounds=(-0.995, 0.995), method="bounded")
    rows.append({"family": "gaussian", "params": (res.x,), "loglik": -res.fun, "k": 1, "lambda_L": 0.0, "lambda_U": 0.0})

    def nll_t(z):
        return -ll_student(u, v, np.tanh(z[0]), 2.05 + np.exp(z[1]))

    res = optimize.minimize(nll_t, x0=[np.arctanh(np.clip(r0, -0.99, 0.99)), np.log(6.0)], method="Nelder-Mead",
                            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
    rho_t, nu_t = np.tanh(res.x[0]), 2.05 + np.exp(res.x[1])
    lt = t_tail_dependence(rho_t, nu_t)
    rows.append({"family": "student_t", "params": (rho_t, nu_t), "loglik": -res.fun, "k": 2, "lambda_L": lt, "lambda_U": lt})

    th0 = max(2 * tau_a / (1 - tau_a), 0.05)
    res = optimize.minimize_scalar(lambda t: -ll_clayton(u, vv, t), bounds=(1e-3, 30), method="bounded")
    rows.append({"family": "clayton", "params": (res.x,), "loglik": -res.fun, "k": 1,
                 "lambda_L": 2 ** (-1 / res.x), "lambda_U": 0.0})
    res = optimize.minimize_scalar(lambda t: -ll_gumbel(u, vv, t), bounds=(1.0001, 20), method="bounded")
    rows.append({"family": "gumbel", "params": (res.x,), "loglik": -res.fun, "k": 1,
                 "lambda_L": 0.0, "lambda_U": 2 - 2 ** (1 / res.x)})
    res = optimize.minimize_scalar(lambda t: -ll_gumbel(1 - u, 1 - vv, t), bounds=(1.0001, 20), method="bounded")
    rows.append({"family": "survival_gumbel", "params": (res.x,), "loglik": -res.fun, "k": 1,
                 "lambda_L": 2 - 2 ** (1 / res.x), "lambda_U": 0.0})
    res = optimize.minimize_scalar(lambda t: -ll_frank(u, v, t), bounds=(-40, 40), method="bounded")
    rows.append({"family": "frank", "params": (res.x,), "loglik": -res.fun, "k": 1, "lambda_L": 0.0, "lambda_U": 0.0})
    out = pd.DataFrame(rows)
    out["aic"] = 2 * out["k"] - 2 * out["loglik"]
    out = out.sort_values("aic").reset_index(drop=True)
    out.attrs.update({"kendall_tau": float(tau), "rotated_for_one_sided_families": bool(rotated), "n": len(u),
                      "unused_start": th0})
    return out


# ------------------------------------------------------------------ multivariate t copula
def _nearest_corr(R: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((R + R.T) / 2)
    w = np.clip(w, 1e-6, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def fit_t_copula(X, nu_bounds=(2.1, 60.0)) -> dict:
    """Correlation from Kendall's tau inversion (R_ij = sin(pi tau_ij / 2)), nu by profile maximum likelihood."""
    U = pseudo_obs(np.asarray(X, dtype=float))
    n, d = U.shape
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = stats.kendalltau(U[:, i], U[:, j])[0]
    R = _nearest_corr(np.sin(np.pi * tau / 2))
    Ri = np.linalg.inv(R)
    ld = np.linalg.slogdet(R)[1]

    def nll(nu):
        Z = stats.t.ppf(_clip(U), nu)
        q = np.einsum("ij,jk,ik->i", Z, Ri, Z)
        ll = (special.gammaln((nu + d) / 2) + (d - 1) * special.gammaln(nu / 2) - d * special.gammaln((nu + 1) / 2)
              - 0.5 * ld - (nu + d) / 2 * np.log1p(q / nu) + (nu + 1) / 2 * np.log1p(Z ** 2 / nu).sum(axis=1))
        return -ll.sum()

    res = optimize.minimize_scalar(nll, bounds=nu_bounds, method="bounded")
    return {"R": R, "nu": float(res.x), "loglik": float(-res.fun), "n": n}


def simulate_t_copula(R: np.ndarray, nu: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = R.shape[0]
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, d)) @ L.T
    W = rng.chisquare(nu, size=n) / nu
    return stats.t.cdf(Z / np.sqrt(W)[:, None], nu)
