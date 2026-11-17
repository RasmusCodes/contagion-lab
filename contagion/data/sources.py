"""Raw-data fetchers with an on-disk cache (data/raw/<source>/).

Each fetcher returns a DataFrame indexed by the series' own *local trading date* (tz-naive, normalised) with a
``value`` column, and stores the untouched payload so the whole pipeline can be rebuilt offline from the cache.

Sources
-------
yahoo   query1.finance.yahoo.com/v8/finance/chart  (ETFs, stocks, indices, futures, FX). ``value`` = adjusted close
        (dividends and splits) when Yahoo provides it, else close. Bar timestamps are converted to the exchange's
        own time zone (with daylight saving) before taking the date.
fred    fred.stlouisfed.org/graph/fredgraph.csv     (Treasury yields, credit spreads, vol indices, stress indices)
cboe    cdn.cboe.com delayed_quotes/charts/historical (VIX family). Duplicate dates are removed (last kept).
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..paths import RAW

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (contagion-lab research pipeline)"}


class FetchError(RuntimeError):
    pass


def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 45) -> requests.Response:
    err: Exception | None = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            err = FetchError(f"HTTP {r.status_code} for {r.url}")
            if r.status_code == 404:
                break
        except requests.RequestException as e:  # network hiccup: back off and retry
            err = e
        time.sleep(2.0 * (k + 1))
    raise FetchError(str(err))


def _safe(symbol: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in symbol)


def _cache_path(source: str, symbol: str, ext: str) -> Path:
    p = RAW / source
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{_safe(symbol)}.{ext}"


# --------------------------------------------------------------------------------------------- yahoo
def _yahoo_parse(payload: dict) -> pd.DataFrame:
    res = payload["chart"]["result"][0]
    meta = res["meta"]
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    ts = pd.to_datetime(np.asarray(res["timestamp"], dtype="int64"), unit="s", utc=True)
    local_date = ts.tz_convert(tz).tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    close = pd.to_numeric(pd.Series(q.get("close"), dtype="float64"), errors="coerce")
    adj = res["indicators"].get("adjclose")
    adjclose = (pd.to_numeric(pd.Series(adj[0].get("adjclose"), dtype="float64"), errors="coerce")
                if adj else close.copy())
    df = pd.DataFrame({"close": close.to_numpy(), "adjclose": adjclose.to_numpy()}, index=local_date)
    # Yahoo leaves the newest bar's adjclose empty; the adjustment factor of the newest bar is 1 by construction.
    trailing = df["adjclose"].isna() & df["close"].notna()
    if trailing.any():
        last_valid = df["adjclose"].last_valid_index()
        tail_mask = trailing & ((df.index > last_valid) if last_valid is not None else True)
        df.loc[tail_mask, "adjclose"] = df.loc[tail_mask, "close"]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["value"] = df["adjclose"]
    df.attrs["tz"] = tz
    df.attrs["currency"] = meta.get("currency")
    return df


def _yahoo_valid(payload: dict) -> bool:
    res = (payload.get("chart") or {}).get("result") or []
    return bool(res) and bool(res[0].get("timestamp"))


def fetch_yahoo(symbol: str, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path("yahoo", symbol, "json")
    cached_ok = path.exists() and _yahoo_valid(json.loads(path.read_text(encoding="utf-8")))
    if refresh or not cached_ok:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"period1": 0, "period2": int(time.time()), "interval": "1d",
                  "events": "div,split", "includeAdjustedClose": "true"}
        payload = {}
        for attempt in range(3):  # an empty result is sometimes transient throttling
            payload = _get(url, params=params).json()
            if _yahoo_valid(payload):
                break
            time.sleep(3.0 * (attempt + 1))
        if not _yahoo_valid(payload):
            meta = ((payload.get("chart") or {}).get("result") or [{}])[0].get("meta", {})
            first = meta.get("firstTradeDate")
            first = pd.to_datetime(first, unit="s").date() if first else None
            raise FetchError(f"yahoo returned no price history for {symbol} (firstTradeDate={first}); "
                             "the ticker may have been reassigned")
        payload["_fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = _yahoo_parse(payload)
    df.attrs["fetched_utc"] = payload.get("_fetched_utc")
    return df


# --------------------------------------------------------------------------------------------- fred
def fetch_fred(series_id: str, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path("fred", series_id, "csv")
    stamp = path.with_suffix(".fetched")
    if refresh or not path.exists():
        r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, timeout=60)
        path.write_text(r.text, encoding="utf-8")
        stamp.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"), encoding="utf-8")
    raw = pd.read_csv(path)
    date_col, val_col = raw.columns[0], raw.columns[1]
    df = pd.DataFrame({"value": pd.to_numeric(raw[val_col], errors="coerce").to_numpy()},
                      index=pd.to_datetime(raw[date_col]).dt.normalize())
    df = df[~df.index.duplicated(keep="last")].dropna().sort_index()
    df.attrs["tz"] = "America/New_York"
    df.attrs["fetched_utc"] = stamp.read_text(encoding="utf-8") if stamp.exists() else None
    return df


# --------------------------------------------------------------------------------------------- cboe
def fetch_cboe(symbol: str, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path("cboe", symbol, "json")
    if refresh or not path.exists():
        r = _get(f"https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json")
        payload = r.json()
        payload["_fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["data"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    out = pd.DataFrame({"value": pd.to_numeric(df["close"], errors="coerce")}).dropna()
    out = out[out["value"] > 0]
    out.attrs["tz"] = "America/New_York"
    out.attrs["fetched_utc"] = payload.get("_fetched_utc")
    return out


FETCHERS = {"yahoo": fetch_yahoo, "fred": fetch_fred, "cboe": fetch_cboe}


def fetch(source: str, symbol: str, refresh: bool = False) -> pd.DataFrame:
    try:
        f = FETCHERS[source]
    except KeyError as e:
        raise ValueError(f"unknown source {source!r}") from e
    return f(symbol, refresh=refresh)
