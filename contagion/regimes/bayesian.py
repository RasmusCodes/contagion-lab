"""Bayesian regime model: sticky Gaussian HMM estimated by Gibbs sampling (forward-filtering backward-sampling).

Priors
    Sigma_k ~ Inverse-Wishart(nu0, S0),  mu_k | Sigma_k ~ N(m0, Sigma_k / k0)        (Normal-Inverse-Wishart)
    row k of the transition matrix ~ Dirichlet(alpha + kappa * e_k)                 (kappa = stickiness)
Identification: states are relabelled at every draw by the variance of the first feature (state 0 = calmest).

What it adds over the EM HMM: full posterior uncertainty, i.e. credible bands for regime probabilities, regime
volatilities and expected durations. The state probabilities reported are posterior (smoothed, full-sample):
a historical description, not a real-time signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _sample_iw(nu: float, S: np.ndarray, rng) -> np.ndarray:
    return stats.invwishart.rvs(df=nu, scale=S, random_state=rng)


class BayesianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 2000, burn: int = 500, thin: int = 2, kappa: float = 50.0,
                 alpha: float = 1.0, seed: int = 0):
        self.K, self.n_iter, self.burn, self.thin = n_states, n_iter, burn, thin
        self.kappa, self.alpha = kappa, alpha
        self.rng = np.random.default_rng(seed)

    def _loglik(self, X, mus, Sigmas):
        return np.column_stack([stats.multivariate_normal(mus[k], Sigmas[k], allow_singular=True).logpdf(X)
                                for k in range(self.K)]).reshape(len(X), self.K)

    def _ffbs(self, logB, P, pi0):
        T, K = logB.shape
        alpha = np.empty((T, K))
        a = pi0
        for t in range(T):
            pred = a if t == 0 else a @ P
            w = pred * np.exp(logB[t] - logB[t].max())
            a = w / w.sum()
            alpha[t] = a
        z = np.empty(T, dtype=int)
        u = self.rng.random(T)
        z[-1] = int(np.searchsorted(np.cumsum(alpha[-1]), u[-1] * alpha[-1].sum()))
        for t in range(T - 2, -1, -1):
            w = alpha[t] * P[:, z[t + 1]]
            c = np.cumsum(w)
            z[t] = min(int(np.searchsorted(c, u[t] * c[-1])), K - 1)
        return z, alpha

    def fit(self, X) -> "BayesianHMM":
        idx = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, d = X.shape
        K = self.K
        m0, k0 = X.mean(axis=0), 0.01
        nu0 = d + 3.0
        S0 = np.cov(X, rowvar=False).reshape(d, d) * (nu0 - d - 1)
        # initial states: quantiles of a rolling dispersion of the first feature
        disp = pd.Series(np.abs(X[:, 0])).rolling(10, min_periods=1).mean().to_numpy()
        z = np.clip(np.searchsorted(np.quantile(disp, np.linspace(0, 1, K + 1)[1:-1]), disp), 0, K - 1)
        P = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(P, 0.95)
        draws = {"mu": [], "Sigma": [], "P": [], "state_counts": np.zeros((T, K))}
        n_keep = 0
        for it in range(self.n_iter):
            # 1. emission parameters | states
            mus, Sigmas = np.empty((K, d)), np.empty((K, d, d))
            for k in range(K):
                Xk = X[z == k]
                n = len(Xk)
                xb = Xk.mean(axis=0) if n else m0
                Sk = ((Xk - xb).T @ (Xk - xb)) if n else np.zeros((d, d))
                kn, nun = k0 + n, nu0 + n
                mn = (k0 * m0 + n * xb) / kn
                Sn = S0 + Sk + (k0 * n / kn) * np.outer(xb - m0, xb - m0)
                Sig = np.atleast_2d(_sample_iw(nun, Sn, self.rng))
                mus[k] = self.rng.multivariate_normal(mn, Sig / kn)
                Sigmas[k] = Sig
            # 2. relabel by variance of the first feature
            order = np.argsort(Sigmas[:, 0, 0])
            mus, Sigmas = mus[order], Sigmas[order]
            inv = np.argsort(order)
            z = inv[z]
            # 3. transition matrix | states (sticky Dirichlet)
            counts = np.zeros((K, K))
            np.add.at(counts, (z[:-1], z[1:]), 1)
            P = np.vstack([self.rng.dirichlet(self.alpha + counts[k] + self.kappa * np.eye(K)[k]) for k in range(K)])
            # 4. states | parameters
            pi0 = np.full(K, 1.0 / K)
            z, _ = self._ffbs(self._loglik(X, mus, Sigmas), P, pi0)
            if it >= self.burn and (it - self.burn) % self.thin == 0:
                draws["mu"].append(mus)
                draws["Sigma"].append(Sigmas)
                draws["P"].append(P)
                draws["state_counts"][np.arange(T), z] += 1
                n_keep += 1
        self.draws = {k: (np.array(v) if k != "state_counts" else v / n_keep) for k, v in draws.items()}
        self.state_prob = pd.DataFrame(self.draws["state_counts"], index=idx,
                                       columns=[f"p_state{k}" for k in range(K)])
        return self

    def summary(self, scale: float = 1.0, annualize: float = np.sqrt(252)) -> pd.DataFrame:
        S = self.draws["Sigma"][:, :, 0, 0]
        vol = np.sqrt(S) * scale * annualize
        P = self.draws["P"]
        dur = 1.0 / (1.0 - np.stack([P[:, k, k] for k in range(self.K)], axis=1))
        rows = []
        for k in range(self.K):
            rows.append({"state": k, "vol_mean": vol[:, k].mean(), "vol_5": np.quantile(vol[:, k], 0.05),
                         "vol_95": np.quantile(vol[:, k], 0.95), "duration_mean": dur[:, k].mean(),
                         "duration_5": np.quantile(dur[:, k], 0.05), "duration_95": np.quantile(dur[:, k], 0.95),
                         "share_of_time": float(self.state_prob.iloc[:, k].mean())})
        return pd.DataFrame(rows).set_index("state")
