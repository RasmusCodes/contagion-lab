"""Trading calendars, timestamp alignment and asynchrony handling.

Alignment rule (``align_to_master``)
    Every series is indexed by its own local trading date. On each NYSE (master) date we take the last observation
    dated on or before that date. A master date is *fresh* for a series if the series printed at least one new
    observation since the previous master date (this also catches prints on days the NYSE was closed, e.g. FX on
    Good Friday, which are folded into the next NYSE day). Consecutive non-fresh days count up ``staleness``;
    beyond ``max_stale`` the level is set to NaN. Returns are differences of these aligned levels, so a market
    holiday shows a zero return and the whole move lands on the next fresh day: cumulative returns are conserved.

Asynchrony
    Series close at different clock times (NYSE 16:00 ET, Treasury CMT ~15:30, metals settle ~13:30, Yahoo FX bars
    at London midnight ~19:00, Tokyo at 01:00 ET). Same-date returns therefore cover different windows, which
    (i) biases same-day correlations toward zero and (ii) creates spurious lead-lag ("Granger") links from the
    later-closing to the earlier-closing market's next day. Tools here measure it (``lead_lag_profile``) and correct
    covariance for it (``sync_covariance``, Dimson / Newey-West style), or remove it by aggregation (``to_weekly``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def master_calendar(reference: pd.Series | pd.DataFrame, start: str | None = None,
                    end: str | None = None) -> pd.DatetimeIndex:
    """NYSE trading days taken from a reference NYSE-listed series' own trading dates (SPY by default)."""
    idx = reference.dropna().index if isinstance(reference, pd.Series) else reference.dropna(how="all").index
    idx = pd.DatetimeIndex(idx).normalize().unique().sort_values()
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    weekend = idx[idx.dayofweek >= 5]
    if len(weekend):
        raise ValueError(f"reference calendar contains weekend dates: {list(weekend[:5])}")
    return pd.DatetimeIndex(idx, name="date")


def align_to_master(s: pd.Series, master: pd.DatetimeIndex, max_stale: int = 3) -> tuple[pd.Series, pd.Series]:
    """Align a series to the master calendar. Returns (levels, staleness) indexed by ``master``.

    staleness = number of consecutive master days without a new print (0 = fresh). Levels with staleness above
    ``max_stale`` are NaN. Dates before the first observation are NaN with staleness -1.
    """
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return pd.Series(np.nan, index=master), pd.Series(-1, index=master, dtype="int64")
    obs_dates = s.index.values
    n_le = np.searchsorted(obs_dates, master.values, side="right")  # number of observations dated <= master day
    has_any = n_le > 0
    prev = np.concatenate([[0], n_le[:-1]])
    fresh = n_le > prev
    fresh[0] = has_any[0]
    stale = np.zeros(len(master), dtype="int64")
    run = 0
    for k in range(len(master)):
        if not has_any[k]:
            stale[k] = -1
            run = 0
            continue
        run = 0 if fresh[k] else run + 1
        stale[k] = run
    vals = np.where(has_any, s.to_numpy()[np.clip(n_le - 1, 0, None)], np.nan)
    vals = np.where((stale >= 0) & (stale <= max_stale), vals, np.nan)
    return pd.Series(vals, index=master), pd.Series(stale, index=master)


# --------------------------------------------------------------------------------------------- asynchrony
def lead_lag_profile(returns: pd.DataFrame, ref: str, lags=(-2, -1, 0, 1, 2)) -> pd.DataFrame:
    """corr(ref_t, x_{t+k}) for each series x and lag k.

    A large k=+1 value means x reacts to ref's day-t news on its own day t+1, i.e. x closes earlier than ref
    (or prices stale). A large k=-1 value means x closes later than ref (x's day-t return contains news that ref
    only absorbs on t+1). Either pattern produces a lead-lag link that is a timing artefact, not contagion.
    """
    r0 = returns[ref]
    out = {}
    for c in returns.columns:
        if c == ref:
            continue
        row = {}
        for k in lags:
            row[k] = r0.corr(returns[c].shift(-k))
        out[c] = row
    df = pd.DataFrame(out).T
    df.columns = [f"lag{k:+d}" for k in lags]
    if "lag+1" in df and "lag-1" in df:
        # > 0: x reacts a day late to ref (x closes earlier or prices stale); < 0: x closes later than ref
        df["timing_signal"] = df["lag+1"].abs() - df["lag-1"].abs()
    return df


def _autocov(X: np.ndarray, k: int) -> np.ndarray:
    """Gamma_k = E[x_t x_{t-k}'] for demeaned X (T x N)."""
    T = X.shape[0]
    return X[k:].T @ X[: T - k] / T


def nearest_psd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps * max(w.max(), 1.0), None)
    return (V * w) @ V.T


def sync_covariance(R: np.ndarray | pd.DataFrame, lags: int = 1, method: str = "dimson") -> np.ndarray:
    """Covariance corrected for non-synchronous closes.

    method="dimson":   Gamma_0 + sum_k (Gamma_k + Gamma_k')  (unbiased for the synchronous covariance when the
                       timing offset is below ``lags`` days; projected to the nearest PSD matrix)
    method="bartlett": Newey-West weights 1 - k/(lags+1); always PSD but only partially corrects.
    """
    X = np.asarray(R, dtype=float)
    X = X - np.nanmean(X, axis=0)
    X = np.nan_to_num(X)
    S = _autocov(X, 0)
    for k in range(1, lags + 1):
        G = _autocov(X, k)
        w = 1.0 if method == "dimson" else 1.0 - k / (lags + 1)
        S = S + w * (G + G.T)
    return nearest_psd(S) if method == "dimson" else (S + S.T) / 2


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def to_weekly(levels: pd.DataFrame, anchor: str = "W-WED") -> pd.DataFrame:
    """Sample aligned levels weekly (last available level in each week ending on ``anchor``).

    Weekly sampling removes most of the timing mismatch between markets that close hours apart, at the cost of
    ~5x fewer observations.
    """
    return levels.resample(anchor).last()


def calendar_coverage(staleness: pd.DataFrame) -> pd.DataFrame:
    """Per series: first date, share of master days that are fresh, longest stale run."""
    rows = {}
    for c in staleness.columns:
        st = staleness[c]
        live = st[st >= 0]
        if live.empty:
            rows[c] = {"first": pd.NaT, "fresh_share": np.nan, "max_stale_run": np.nan, "stale_days": 0}
            continue
        rows[c] = {"first": live.index[0], "fresh_share": float((live == 0).mean()),
                   "max_stale_run": int(live.max()), "stale_days": int((live > 0).sum())}
    return pd.DataFrame(rows).T
