"""Gaussian hidden Markov models with strictly causal (filtered) state probabilities.

Estimation uses hmmlearn's EM with several restarts; states are relabelled in increasing order of the variance
of the first feature so that state 0 is always the calmest regime. Three kinds of state probability exist and
must not be mixed up:

    smoothed   P(s_t | x_1..x_T)   uses the future: fine for describing history, never for decisions
    filtered   P(s_t | x_1..x_t)   computed here by our own forward pass
    real-time  filtered probabilities from a model whose PARAMETERS were also estimated only on data up to t
               (``expanding_filtered_probs``): the only version allowed in backtests (Q4, Q6)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


class GaussianHMM:
    def __init__(self, n_states: int = 2, covariance_type: str = "full", n_init: int = 5, n_iter: int = 200,
                 tol: float = 1e-4, seed: int = 0, sticky: float = 0.0):
        self.K = n_states
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.sticky = sticky  # pseudo-counts added to the transition diagonal (encourages persistent regimes)
        self.model = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, init_from: "GaussianHMM | None" = None) -> "GaussianHMM":
        from hmmlearn.hmm import GaussianHMM as _H

        X = np.asarray(X, dtype=float)
        best, best_ll = None, -np.inf
        seeds = [self.seed + 17 * k for k in range(self.n_init)]
        for k, s in enumerate(seeds):
            m = _H(n_components=self.K, covariance_type=self.covariance_type, n_iter=self.n_iter, tol=self.tol,
                   random_state=s, init_params="stmc", params="stmc", min_covar=1e-6)
            if self.sticky > 0:
                m.transmat_prior = np.eye(self.K) * self.sticky + 1.0
            if init_from is not None and k == 0 and init_from.model is not None:  # warm start
                m.init_params = ""
                m.startprob_ = init_from.model.startprob_.copy()
                m.transmat_ = init_from.model.transmat_.copy()
                m.means_ = init_from.model.means_.copy()
                m.covars_ = init_from.model._covars_.copy() if hasattr(init_from.model, "_covars_") else init_from.model.covars_.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m.fit(X)
                    ll = m.score(X)
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError("HMM estimation failed for every restart")
        self.model = best
        self.loglik_ = best_ll
        self._order()
        return self

    def _order(self) -> None:
        m = self.model
        var0 = np.array([self.cov(k)[0, 0] for k in range(self.K)])
        order = np.argsort(var0)
        self.order_ = order
        self.means_ = m.means_[order]
        self.covs_ = np.array([self.cov(k) for k in order])
        self.transmat_ = m.transmat_[np.ix_(order, order)]
        self.startprob_ = m.startprob_[order]

    def cov(self, k: int) -> np.ndarray:
        c = self.model.covars_[k]
        return np.diag(c) if c.ndim == 1 else c

    # ------------------------------------------------------------------ inference
    def log_emission(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([stats.multivariate_normal(self.means_[k], self.covs_[k], allow_singular=True).logpdf(X)
                                for k in range(self.K)]).reshape(len(X), self.K)

    def filter(self, X: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """Filtered probabilities P(s_t | x_1..t) (T, K)."""
        logB = self.log_emission(X)
        T = len(logB)
        A = self.transmat_
        out = np.empty((T, self.K))
        a = self.startprob_ if prior is None else prior
        for t in range(T):
            pred = a if t == 0 and prior is None else a @ A
            lb = logB[t]
            w = pred * np.exp(lb - lb.max())
            s = w.sum()
            a = w / s if s > 0 else np.full(self.K, 1.0 / self.K)
            out[t] = a
        return out

    def smooth(self, X: np.ndarray) -> np.ndarray:
        """Smoothed probabilities P(s_t | x_1..T) (uses the future - descriptive only)."""
        post = self.model.predict_proba(np.asarray(X, dtype=float))
        return post[:, self.order_]

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        path = self.model.predict(np.asarray(X, dtype=float))
        inv = np.argsort(self.order_)
        return inv[path]

    def expected_duration(self) -> np.ndarray:
        return 1.0 / (1.0 - np.diag(self.transmat_))

    def bic(self, X: np.ndarray) -> float:
        n, d = np.asarray(X).shape
        k_cov = d * (d + 1) / 2 if self.covariance_type == "full" else d
        k = self.K * (self.K - 1) + (self.K - 1) + self.K * (d + k_cov)
        return float(-2 * self.loglik_ + k * np.log(n))


def expanding_filtered_probs(X: pd.DataFrame, n_states: int = 2, min_train: int = 750, refit_every: int = 63,
                             n_init: int = 3, seed: int = 0, sticky: float = 0.0) -> pd.DataFrame:
    """Real-time regime probabilities: at each refit date the HMM is re-estimated on data up to that date only;
    between refits the frozen model is run forward (filtered). Rows before ``min_train`` are NaN.

    Returned columns: p_state0 .. p_state{K-1} (state 0 = lowest variance of the first feature) and 'state'.
    """
    Xa = X.to_numpy(dtype=float)
    T = len(Xa)
    out = np.full((T, n_states), np.nan)
    prev = None
    a_last = None
    for start in range(min_train, T, refit_every):
        m = GaussianHMM(n_states=n_states, n_init=n_init, seed=seed, sticky=sticky).fit(Xa[:start], init_from=prev)
        # filter from the beginning with the new parameters (cheap) so the state at `start-1` is consistent
        filt = m.filter(Xa[: min(start + refit_every, T)])
        out[start: min(start + refit_every, T)] = filt[start: min(start + refit_every, T)]
        prev = m
        a_last = filt[-1]
    df = pd.DataFrame(out, index=X.index, columns=[f"p_state{k}" for k in range(n_states)])
    df["state"] = np.where(df.notna().all(axis=1), np.nanargmax(np.nan_to_num(out, nan=-1), axis=1), -1)
    df.attrs["last_model"] = prev
    df.attrs["last_filtered"] = a_last
    return df
