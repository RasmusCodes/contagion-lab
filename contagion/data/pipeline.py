"""Build and load the processed panel.

    raw cache --(clean, overrides)--> local-date levels --(align to NYSE calendar)--> levels / staleness
              --(transform)--> returns (log returns, bp changes, log vol changes) --> data/processed/*.parquet

``Panel`` is the object every analysis consumes. ``Panel.returns_for`` returns a balanced block for a
sub-universe: it starts when the last member starts and drops the (rare) rows with gaps longer than
``max_stale`` days.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..paths import CONFIG, PROCESSED, ensure_dirs
from .calendar import align_to_master, calendar_coverage, master_calendar
from .sources import FetchError, fetch
from .transforms import clean_levels, robust_outliers, to_returns
from .universe import load_universe, members

log = logging.getLogger(__name__)


def _load_overrides() -> pd.DataFrame:
    p = CONFIG / "data_overrides.csv"
    if not p.exists():
        return pd.DataFrame(columns=["id", "date", "action", "value", "reason"])
    df = pd.read_csv(p, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_panel(refresh: bool = False, start: str = "1999-01-04", max_stale: int = 3,
                verbose: bool = True) -> "Panel":
    ensure_dirs()
    U = load_universe()
    overrides = _load_overrides()
    raw_levels: dict[str, pd.Series] = {}
    vintage, issues, failed = {}, [], {}
    for sid, row in U.iterrows():
        if row["source"] == "splice":
            continue
        try:
            df = fetch(row["source"], row["symbol"], refresh=refresh)
        except FetchError as e:
            failed[sid] = str(e)
            if verbose:
                print(f"  ! {sid:10s} {row['source']}:{row['symbol']} failed: {e}")
            continue
        s, iss = clean_levels(df["value"], row["kind"], row["transform"])
        for o in overrides[overrides["id"] == sid].itertuples():
            if o.action == "mask":
                s.loc[s.index == o.date] = np.nan
            elif o.action == "set":
                s.loc[o.date] = float(o.value)
            iss.append({"date": o.date, "issue": f"override {o.action}: {o.reason}", "value": o.value})
        raw_levels[sid] = s.sort_index()
        vintage[sid] = {"source": row["source"], "symbol": row["symbol"], "fetched_utc": df.attrs.get("fetched_utc"),
                        "first_raw": str(s.dropna().index.min().date()) if s.notna().any() else None,
                        "last_raw": str(s.dropna().index.max().date()) if s.notna().any() else None}
        issues += [{"id": sid, **i} for i in iss]
        if verbose:
            print(f"  . {sid:10s} {row['source']:5s} {row['symbol']:12s} {vintage[sid]['first_raw']} -> "
                  f"{vintage[sid]['last_raw']}  n={s.notna().sum()}")

    if "SPY" not in raw_levels:
        raise RuntimeError("SPY is required to define the NYSE master calendar")
    master = master_calendar(raw_levels["SPY"], start=start)

    daily_ids = [i for i in raw_levels if U.loc[i, "transform"] != "none"]
    levels, stale = {}, {}
    for sid in daily_ids:
        lv, st = align_to_master(raw_levels[sid], master, max_stale=max_stale)
        levels[sid], stale[sid] = lv, st
    levels = pd.DataFrame(levels)
    stale = pd.DataFrame(stale)
    returns = pd.DataFrame({sid: to_returns(levels[sid], U.loc[sid, "transform"]) for sid in daily_ids})

    # Spliced series: returns of the first component until the second has a valid return, then the second.
    for sid, row in U[U["source"] == "splice"].iterrows():
        a, b = row["symbol"].split(">")
        if a not in returns or b not in returns:
            failed[sid] = f"splice components missing: {a}, {b}"
            continue
        switch = returns[b].first_valid_index()
        r = returns[a].where(returns.index < switch, returns[b])
        returns[sid] = r
        stale[sid] = stale[a].where(stale.index < switch, stale[b])
        lv = np.exp(r.fillna(0).cumsum())
        levels[sid] = (100 * lv).where(r.notna() | (r.index == r.first_valid_index()))
        vintage[sid] = {"source": "splice", "symbol": row["symbol"], "switch_date": str(switch.date())}
        daily_ids.append(sid)

    outliers = []
    for sid in daily_ids:
        o = robust_outliers(returns[sid])
        outliers += [{"id": sid, "date": str(d.date()), "value": float(v), "robust_z": float(z)}
                     for d, v, z in zip(o["date"], o["value"], o["robust_z"])]

    weekly_ids = [i for i in raw_levels if U.loc[i, "transform"] == "none"]
    weekly = pd.DataFrame({sid: raw_levels[sid] for sid in weekly_ids}) if weekly_ids else pd.DataFrame()

    cov = calendar_coverage(stale)
    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "master_calendar": {"first": str(master[0].date()), "last": str(master[-1].date()), "n_days": len(master),
                            "reference": "SPY trading dates (NYSE)"},
        "max_stale": max_stale,
        "vintage": vintage,
        "failed": failed,
        "issues": [{**i, "date": str(pd.Timestamp(i["date"]).date())} for i in issues],
        "outlier_flags": outliers,
        "coverage": {k: {"first": str(v["first"].date()) if pd.notna(v["first"]) else None,
                         "fresh_share": v["fresh_share"], "max_stale_run": v["max_stale_run"],
                         "stale_days": v["stale_days"]} for k, v in cov.iterrows()},
    }
    levels.to_parquet(PROCESSED / "levels.parquet")
    returns.to_parquet(PROCESSED / "returns.parquet")
    stale.to_parquet(PROCESSED / "staleness.parquet")
    if not weekly.empty:
        weekly.to_parquet(PROCESSED / "stress_weekly.parquet")
    (PROCESSED / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return load_panel()


@dataclass
class Panel:
    levels: pd.DataFrame
    returns: pd.DataFrame
    staleness: pd.DataFrame
    universe: pd.DataFrame
    meta: dict
    stress_weekly: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------ selection helpers
    def ids(self, which: str | list[str]) -> list[str]:
        if isinstance(which, str):
            if which in self.returns.columns:
                return [which]
            return [i for i in members(self.universe, which) if i in self.returns.columns]
        return [i for i in which if i in self.returns.columns]

    def common_start(self, ids: list[str]) -> pd.Timestamp:
        firsts = [self.returns[i].first_valid_index() for i in ids]
        return max(firsts)

    def returns_for(self, which: str | list[str], start: str | None = None, end: str | None = None,
                    balanced: bool = True, drop_stale_rows: bool = False,
                    trim_stale_tail: bool = True) -> pd.DataFrame:
        """Balanced return block. ``trim_stale_tail`` drops trailing days on which any member has not printed
        yet (publication lags, e.g. H.10 FX is released weekly), so the last row is a genuine joint close."""
        ids = self.ids(which)
        R = self.returns[ids]
        st = self.staleness[ids]
        if balanced:
            cs = self.common_start(ids)
            R, st = R.loc[cs:], st.loc[cs:]
            R = R.iloc[1:] if R.iloc[0].isna().any() else R
            st = st.loc[R.index]
        if start is not None:
            R, st = R.loc[pd.Timestamp(start):], st.loc[pd.Timestamp(start):]
        if end is not None:
            R, st = R.loc[: pd.Timestamp(end)], st.loc[: pd.Timestamp(end)]
        if balanced:
            keep = R.notna().all(axis=1)
            if drop_stale_rows:
                keep &= (st == 0).all(axis=1)
            R = R.loc[keep]
        if trim_stale_tail and len(R):
            fresh = (self.staleness.loc[R.index, ids] == 0).all(axis=1).to_numpy()
            if fresh.any():
                R = R.iloc[: int(np.flatnonzero(fresh)[-1]) + 1]
        return R

    def levels_for(self, which: str | list[str], start=None, end=None) -> pd.DataFrame:
        ids = self.ids(which)
        L = self.levels[ids]
        return L.loc[pd.Timestamp(start) if start else None: pd.Timestamp(end) if end else None]

    def label(self, sid: str) -> str:
        return str(self.universe.loc[sid, "name"]) if sid in self.universe.index else sid

    def asset_class(self, sid: str) -> str:
        return str(self.universe.loc[sid, "asset_class"])


def load_panel() -> Panel:
    p = PROCESSED
    if not (p / "returns.parquet").exists():
        raise FileNotFoundError("processed panel missing: run scripts/build_data.py first")
    sw = pd.read_parquet(p / "stress_weekly.parquet") if (p / "stress_weekly.parquet").exists() else pd.DataFrame()
    return Panel(levels=pd.read_parquet(p / "levels.parquet"), returns=pd.read_parquet(p / "returns.parquet"),
                 staleness=pd.read_parquet(p / "staleness.parquet"), universe=load_universe(),
                 meta=json.loads((p / "meta.json").read_text(encoding="utf-8")), stress_weekly=sw)
