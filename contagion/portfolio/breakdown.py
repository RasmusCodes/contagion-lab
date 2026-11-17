"""Correlation-breakdown detection.

A "breakdown" is a short-window correlation that departs from its longer-run level by more than sampling noise.
Two corrections matter before calling it contagion:
  * Forbes-Rigobon: in turbulent windows the variance of the source market rises, which mechanically raises the
    measured correlation. ``fr_corr`` reports the heteroskedasticity-adjusted short-window correlation.
  * overlapping windows: alerts on consecutive days are not independent; we report alert EPISODES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contagion.network.correlation import jennrich_test


def corr_breakdown(x: pd.Series, y: pd.Series, short: int = 63, long: int = 756, z_thresh: float = 3.0) -> pd.DataFrame:
    """Short-window correlation vs the long window that ENDS where the short window starts (no overlap)."""
    rs = x.rolling(short).corr(y)
    rl = x.shift(short).rolling(long).corr(y.shift(short))
    vs = x.rolling(short).var()
    vl = x.shift(short).rolling(long).var()
    delta = vs / vl - 1
    fr = rs / np.sqrt(1 + delta * (1 - rs ** 2))
    se = np.sqrt(1 / (short - 3) + 1 / (long - 3))
    z = (np.arctanh(rs.clip(-0.999, 0.999)) - np.arctanh(rl.clip(-0.999, 0.999))) / se
    zfr = (np.arctanh(fr.clip(-0.999, 0.999)) - np.arctanh(rl.clip(-0.999, 0.999))) / se
    out = pd.DataFrame({"corr_short": rs, "corr_long": rl, "fr_corr": fr, "z": z, "z_fr": zfr})
    out["alert"] = out["z"].abs() > z_thresh
    out["alert_fr"] = out["z_fr"].abs() > z_thresh
    return out


def alert_episodes(flags: pd.Series, min_gap: int = 21) -> pd.DataFrame:
    f = flags.fillna(False).astype(bool)
    idx = np.flatnonzero(f.to_numpy())
    if len(idx) == 0:
        return pd.DataFrame(columns=["start", "end", "days"])
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            eps.append((s, prev))
            s = i
        prev = i
    eps.append((s, prev))
    return pd.DataFrame([{"start": f.index[a], "end": f.index[b], "days": int(f.iloc[a:b + 1].sum())} for a, b in eps])


def rolling_jennrich(R: pd.DataFrame, short: int = 63, long: int = 504, step: int = 21) -> pd.Series:
    """p-value of Jennrich's test: correlation matrix of the last ``short`` days vs the preceding ``long`` days."""
    X = R.to_numpy(dtype=float)
    vals, dates = [], []
    for e in range(short + long, len(X) + 1, step):
        _, _, p = jennrich_test(X[e - short:e], X[e - short - long:e - short])
        vals.append(p)
        dates.append(R.index[e - 1])
    return pd.Series(vals, index=dates, name="jennrich_p")
