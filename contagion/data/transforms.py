"""Level -> return transforms and data-quality checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRANSFORM_UNITS = {
    "logret": "log return (decimal)",
    "dlog": "log change (decimal)",
    "dbp": "change in basis points",
    "none": "level",
}


def clean_levels(s: pd.Series, kind: str, transform: str) -> tuple[pd.Series, list[dict]]:
    """Remove values that cannot be transformed (non-positive prices for log transforms). Returns (clean, issues)."""
    issues = []
    s = s.astype(float)
    if transform in ("logret", "dlog"):
        bad = s <= 0
        for d, v in s[bad].items():
            issues.append({"date": d, "issue": "non-positive level masked", "value": float(v)})
        s = s.mask(bad)
    return s, issues


def to_returns(levels: pd.Series, transform: str) -> pd.Series:
    if transform == "logret" or transform == "dlog":
        return np.log(levels).diff()
    if transform == "dbp":
        return levels.diff() * 100.0
    if transform == "none":
        return levels.copy()
    raise ValueError(f"unknown transform {transform!r}")


def robust_outliers(r: pd.Series, z: float = 12.0) -> pd.DataFrame:
    """Flag returns more than ``z`` robust standard deviations (MAD, rolling 1y) from the rolling median.

    Flags are reported for review, never silently removed: genuine crash days (2008, 2020) must survive.
    """
    r = r.dropna()
    if len(r) < 300:
        return pd.DataFrame(columns=["date", "value", "robust_z"])
    med = r.rolling(252, min_periods=120).median()
    mad = (r - med).abs().rolling(252, min_periods=120).median() * 1.4826
    rz = (r - med) / mad.replace(0, np.nan)
    hit = rz.abs() > z
    return pd.DataFrame({"date": r.index[hit], "value": r[hit].to_numpy(), "robust_z": rz[hit].to_numpy()})
