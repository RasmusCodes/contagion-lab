"""Dynamic conditional correlation (Engle 2002), DCC(1,1) with correlation targeting.

    Q_t = (1 - a - b) Qbar + a e_{t-1} e_{t-1}' + b Q_{t-1},   R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2

e_t are GARCH-standardized residuals. (a, b) are estimated by maximising the second-stage (correlation) quasi
log-likelihood. The recursion is linear in Q, so it is run for all N^2 entries at once with an IIR filter.
R_t uses information up to t-1 (it is the conditional correlation for day t). DESCRIPTIVE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, signal


def _q_path(E: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    T, N = E.shape
    S = (E[:, :, None] * E[:, None, :]).reshape(T, N * N)
    u = (1 - a - b) * Qbar.reshape(-1)[None, :] + a * S[:-1]  # input for t = 1..T-1
    Q = np.empty((T, N * N))
    Q[0] = Qbar.reshape(-1)
    zi = (b * Q[0])[None, :]  # y_t = u_t + b y_{t-1}
    Q[1:], _ = signal.lfilter([1.0], [1.0, -b], u, axis=0, zi=zi)
    return Q.reshape(T, N, N)


def _corr(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diagonal(Q, axis1=1, axis2=2))
    return Q / (d[:, :, None] * d[:, None, :])


class DCC:
    def __init__(self):
        self.a = self.b = None
        self.Qbar = None

    def nll(self, params, E):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e12
        R = _corr(_q_path(E, a, b, self.Qbar))
        sign, ld = np.linalg.slogdet(R)
        if np.any(sign <= 0):
            return 1e12
        q = np.einsum("ti,ti->t", E, np.linalg.solve(R, E[:, :, None])[:, :, 0])
        return 0.5 * float(np.sum(ld + q - np.einsum("ti,ti->t", E, E)))

    def fit(self, E, x0=(0.03, 0.95)) -> "DCC":
        E = np.asarray(E, dtype=float)
        self.Qbar = np.corrcoef(E, rowvar=False)
        self.E = E
        res = optimize.minimize(self.nll, x0=np.asarray(x0), args=(E,), method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-4, "maxiter": 600})
        self.a, self.b = map(float, res.x)
        self.loglik_ = -float(res.fun)
        return self

    def correlations(self, E=None) -> np.ndarray:
        E = self.E if E is None else np.asarray(E, dtype=float)
        return _corr(_q_path(E, self.a, self.b, self.Qbar))

    def forecast_next(self, E=None) -> np.ndarray:
        """R_{T+1|T}: correlation for the day after the sample."""
        E = self.E if E is None else np.asarray(E, dtype=float)
        Q = _q_path(E, self.a, self.b, self.Qbar)
        Qn = (1 - self.a - self.b) * self.Qbar + self.a * np.outer(E[-1], E[-1]) + self.b * Q[-1]
        return _corr(Qn[None])[0]


def fit_dcc_from_returns(R: pd.DataFrame, o: int = 1) -> dict:
    """Univariate GJR-GARCH(1,1)-t per series, then DCC on the standardized residuals."""
    from .volatility import garch

    vols, stds = {}, {}
    for c in R.columns:
        g = garch(R[c], o=o)
        vols[c], stds[c] = g["vol"], g["std_resid"]
    V = pd.DataFrame(vols).loc[R.index]
    E = pd.DataFrame(stds).loc[R.index]
    m = DCC().fit(E.to_numpy())
    return {"model": m, "vol": V, "std_resid": E, "corr": m.correlations(), "a": m.a, "b": m.b}
