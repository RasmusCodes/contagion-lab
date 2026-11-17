"""Volatility models and volatility-regime classification.

``garch``             GJR-GARCH(1,1) with Student-t errors (arch package); returns conditional volatility and
                      standardized residuals. In-sample parameters: descriptive use.
``garch_expanding``   parameters re-estimated on expanding windows, volatility filtered forward: no look-ahead.
``vol_regimes``       labels from EXPANDING percentiles of a volatility series (a day is classified against the
                      history available at that day only).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

LABELS = ["calm", "normal", "elevated", "crisis"]


def garch(r: pd.Series, scale: float | None = None, o: int = 1, dist: str = "t") -> dict:
    """``scale=None`` rescales the series to unit standard deviation for the optimiser (log returns and basis
    points differ by orders of magnitude); results are mapped back to the original units."""
    from arch import arch_model

    scale = (1.0 / float(r.std())) if scale is None else scale
    y = r.dropna() * scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(y, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist).fit(disp="off")
    vol = res.conditional_volatility / scale
    std = res.std_resid
    return {"vol": vol, "std_resid": std, "params": res.params, "result": res}


def garch_expanding(r: pd.Series, min_train: int = 1000, refit_every: int = 252, scale: float = 100.0,
                    o: int = 1, dist: str = "t") -> pd.DataFrame:
    """One-step-ahead conditional volatility sigma_{t|t-1} and standardized residuals using parameters estimated
    on data before each refit block only."""
    from arch import arch_model

    y = r.dropna() * scale
    T = len(y)
    vol = pd.Series(np.nan, index=y.index)
    for start in range(min_train, T, refit_every):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(y.iloc[: min(start + refit_every, T)], mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
            res = am.fit(disp="off", last_obs=y.index[start])  # arch's last_obs is exclusive: fit on y[:start]
            fc = am.fix(res.params)  # filter the whole block with frozen parameters
        cv = fc.conditional_volatility
        sl = slice(start, min(start + refit_every, T))
        vol.iloc[sl] = cv.iloc[sl].to_numpy()
    mu = y.expanding().mean().shift(1)
    out = pd.DataFrame({"vol": vol / scale, "std_resid": (y - mu) / vol})
    return out


def realized_vol(r: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    v = r.rolling(window).std()
    return v * np.sqrt(252) if annualize else v


def vol_regimes(vol: pd.Series, qs=(0.5, 0.8, 0.95), min_history: int = 500) -> pd.Series:
    """Classify each date against the expanding distribution of past volatility (no look-ahead)."""
    v = vol.dropna()
    lab = pd.Series(pd.NA, index=v.index, dtype="object")
    arr = v.to_numpy()
    for t in range(min_history, len(v)):
        hist = arr[:t]
        cut = np.quantile(hist, qs)
        k = int(np.searchsorted(cut, arr[t], side="right"))
        lab.iloc[t] = LABELS[k]
    return lab


def vix_regimes(vix: pd.Series, bounds=(15.0, 25.0, 35.0)) -> pd.Series:
    """Fixed VIX-level buckets: calm < 15 <= normal < 25 <= elevated < 35 <= crisis."""
    k = np.searchsorted(np.asarray(bounds), vix.to_numpy(), side="right")
    return pd.Series(np.array(LABELS)[k], index=vix.index)
