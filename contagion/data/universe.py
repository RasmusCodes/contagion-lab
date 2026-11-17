"""Asset universe: one row per series in config/universe.csv.

Columns
-------
id          internal identifier used everywhere downstream
source      yahoo | fred | cboe
symbol      the source's ticker / series id
kind        price (total-return adjusted) | index_level | futures | fx_rate | yield | spread | vol_index | stress_index
transform   logret (log return) | dbp (change in basis points) | dlog (log change) | none
close_et    approximate close of the series' own trading day, in hours after midnight US Eastern (standard time)
            on that same local date. 16.0 = NYSE close; 1.0 = Tokyo (15:00 JST); 19.0 = London-midnight FX bars.
sets        semicolon-separated sub-universes (macro, sectors, fin, global, sync, ...)
"""
from __future__ import annotations

import pandas as pd

from ..paths import CONFIG

SYNC_TOLERANCE_HOURS = 0.75  # closes within 45 minutes of 16:00 ET are treated as synchronous with NYSE


def load_universe(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or CONFIG / "universe.csv", dtype={"close_et": float}, keep_default_na=False,
                     encoding="utf-8-sig")
    df = df.set_index("id")
    if df.index.duplicated().any():
        raise ValueError(f"duplicate ids in universe: {df.index[df.index.duplicated()].tolist()}")
    df["sets"] = df["sets"].map(lambda s: tuple(x for x in str(s).split(";") if x))
    df["synchronous"] = (df["close_et"] - 16.0).abs() <= SYNC_TOLERANCE_HOURS
    return df


def members(universe: pd.DataFrame, set_name: str) -> list[str]:
    ids = [i for i, s in universe["sets"].items() if set_name in s]
    if not ids:
        raise KeyError(f"no series in set {set_name!r}")
    return ids


def all_sets(universe: pd.DataFrame) -> list[str]:
    return sorted({s for ss in universe["sets"] for s in ss})
