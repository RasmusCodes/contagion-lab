"""Stress-testing engine: covariance "views" of the network x scenarios x portfolios.

Covariance methods (all use data up to ``asof`` only)
    full       sample covariance of the whole history
    rolling    last ``window`` days (default 252)
    ewma       exponentially weighted (lambda 0.97)
    stressed   days on which VIX was in the top decile of its history (regulatory "stressed" calibration)
    regime     Gaussian-HMM mixture: state covariances weighted by today's filtered state probabilities
    crisis     covariance of the HMM's highest-volatility state
    glasso     graphical-lasso (sparse partial-correlation network) covariance

Every result carries both the asset-level conditional moves and a portfolio P&L split into
direct / first-order network / higher-order network exposure (see stress.propagation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .propagation import conditional_gaussian, network_decomposition, portfolio_exposure, tail_betas
from .scenarios import Scenario

STRESS_UNIVERSE = ["SPY", "IWM", "EFA", "EEM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR",
                   "KRE", "SHY", "IEF", "TLT", "LQD", "HYG", "UST2Y", "UST10Y", "BAA10Y", "OIL", "GOLD", "COPPER_F",
                   "DBC", "DXY", "USDJPY", "VIX", "MOVE"]

REGIME_FEATURES = ["SPY", "VIX", "UST10Y", "BAA10Y"]

METHODS = ["full", "rolling", "ewma", "stressed", "regime", "crisis", "glasso"]


def weighted_cov(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = w / w.sum()
    mu = w @ X
    Xc = X - mu
    return mu, (Xc * w[:, None]).T @ Xc


@dataclass
class StressEngine:
    R: pd.DataFrame
    vix_level: pd.Series | None = None
    window: int = 252
    lam: float = 0.97
    _cache: dict = field(default_factory=dict)

    @classmethod
    def from_panel(cls, panel, ids=None, asof=None, start=None, **kw) -> "StressEngine":
        R = panel.returns_for(ids or STRESS_UNIVERSE, start=start, end=asof)
        vix = panel.levels["VIX"].reindex(R.index) if "VIX" in panel.levels else None
        return cls(R=R, vix_level=vix, **kw)

    @property
    def asof(self) -> pd.Timestamp:
        return self.R.index[-1]

    @property
    def names(self) -> list[str]:
        return list(self.R.columns)

    # ---------------------------------------------------------------- covariance views
    def cov(self, method: str = "full") -> pd.DataFrame:
        if method in self._cache:
            return self._cache[method]
        X = self.R.to_numpy(dtype=float)
        if method == "full":
            S = np.cov(X, rowvar=False)
        elif method == "rolling":
            S = np.cov(X[-self.window:], rowvar=False)
        elif method == "ewma":
            w = self.lam ** np.arange(len(X))[::-1]
            S = weighted_cov(X, w)[1]
        elif method == "stressed":
            v = self.vix_level.to_numpy()
            m = v >= np.nanquantile(v, 0.9)
            S = np.cov(X[m], rowvar=False)
        elif method in ("regime", "crisis"):
            S = self._regime_cov(method)
        elif method == "glasso":
            from sklearn.covariance import GraphicalLasso

            sd = X.std(axis=0)
            Z = (X - X.mean(0)) / sd
            g = GraphicalLasso(alpha=0.02, max_iter=500).fit(Z)
            S = g.covariance_ * np.outer(sd, sd)
        else:
            raise ValueError(f"unknown covariance method {method!r}")
        out = pd.DataFrame(S, index=self.names, columns=self.names)
        self._cache[method] = out
        return out

    def _regime_fit(self):
        if "hmm" not in self._cache:
            from contagion.regimes.hmm import GaussianHMM

            F = self.R[[c for c in REGIME_FEATURES if c in self.R.columns]]
            Z = (F / F.std()).to_numpy()
            m = GaussianHMM(n_states=3, n_init=4, seed=0).fit(Z)
            self._cache["hmm"] = (m, m.smooth(Z), m.filter(Z)[-1])
        return self._cache["hmm"]

    def _regime_cov(self, method: str) -> np.ndarray:
        m, post, today = self._regime_fit()
        X = self.R.to_numpy(dtype=float)
        mus, covs = [], []
        for k in range(m.K):
            mu, S = weighted_cov(X, post[:, k])
            mus.append(mu)
            covs.append(S)
        if method == "crisis":
            return covs[-1]  # states are ordered by SPY variance: last = most volatile
        mbar = sum(p * mu for p, mu in zip(today, mus))
        return sum(p * (S + np.outer(mu - mbar, mu - mbar)) for p, S, mu in zip(today, covs, mus))

    def regime_state_today(self) -> dict:
        m, post, today = self._regime_fit()
        vols = np.sqrt([m.covs_[k][0, 0] for k in range(m.K)]) * self.R["SPY"].std() * np.sqrt(252)
        return {"filtered_probs": today.tolist(), "spy_ann_vol_by_state": vols.tolist()}

    # ---------------------------------------------------------------- scenarios
    def run(self, scenario: Scenario, method: str = "full", weights: dict[str, float] | None = None) -> dict:
        shocks = {k: v for k, v in scenario.shocks.items() if k in self.names}
        if not shocks:
            raise ValueError(f"no shocked series of {scenario.name!r} are in the engine universe")
        if scenario.kind == "historical":
            moves = pd.Series(scenario.shocks).reindex(self.names)
            res = {"moves": pd.DataFrame({"expected": moves, "cond_sd": 0.0, "shocked": True})}
            if weights:
                res["pnl"] = {"total": float(sum(w * np.expm1(moves.get(a, np.nan)) for a, w in weights.items()))}
            return res
        S = self.cov(method)
        cg = conditional_gaussian(S, shocks)
        # Conditional MEANS do not depend on the horizon (regression coefficients); the conditional dispersion of
        # an h-day cumulative move scales with sqrt(h) under serially uncorrelated daily returns.
        cg["cond_sd"] = cg["cond_sd"] * np.sqrt(scenario.horizon_days)
        dec = network_decomposition(S, shocks)
        res = {"moves": cg, "decomposition": dec}
        if weights:
            res["pnl"] = portfolio_exposure(dec, weights, shocks)
            res["pnl"]["sd"] = self.pnl_sd(cg, weights) * np.sqrt(scenario.horizon_days)
        return res

    def pnl_sd(self, cg: pd.DataFrame, weights: dict[str, float]) -> float:
        """Conditional standard deviation of portfolio P&L given the shock (linearised, log-return units)."""
        C = cg.attrs["cond_cov_O"]
        w = pd.Series(weights).reindex(C.index).fillna(0.0).to_numpy()
        return float(np.sqrt(max(w @ C.to_numpy() @ w, 0.0)))

    def compare(self, scenario: Scenario, methods=METHODS, weights: dict[str, float] | None = None) -> pd.DataFrame:
        cols = {}
        pnl = {}
        for m in methods:
            r = self.run(scenario, m, weights)
            cols[m] = r["moves"]["expected"]
            if weights:
                pnl[m] = r["pnl"]
        out = pd.DataFrame(cols)
        if weights:
            out.attrs["pnl"] = pd.DataFrame(pnl).T
        return out

    def tail(self, driver: str, shock: float, q: float = 0.05) -> pd.Series:
        """Single-driver stress using empirical tail betas (driver's tail matching the shock's sign)."""
        tb = tail_betas(self.R, driver, direction=-1 if shock < 0 else 1, q=q)
        return tb * shock
