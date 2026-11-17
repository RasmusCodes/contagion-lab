"""FOMC announcement dates scraped from federalreserve.gov (cached in data/raw/fed/).

Historical years come from ``fomchistorical<YYYY>.htm`` (meeting headings such as "January 28-29 Meeting - 2003",
"March 15 (unscheduled) Meeting - 2020", "March 25 Conference Call - 2003"); recent years from
``fomccalendars.htm`` (statement links ``monetaryYYYYMMDDa.htm``).

Statement date of a scheduled meeting = last day of the meeting. Unscheduled actions are kept separately because
they respond to market turmoil (their yield moves are endogenous to the crisis) and are excluded from the
identification exercise by default.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from ..paths import RAW
from .sources import _get

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], start=1)}
HEAD = re.compile(r"^(?P<m1>[A-Za-z]+)\s+(?P<d1>\d{1,2})(?:-(?:(?P<m2>[A-Za-z]+)\s+)?(?P<d2>\d{1,2}))?\s+"
                  r"(?P<label>.*?)\s+-\s+(?P<yr>\d{4})$")


def _page(name: str, refresh: bool) -> str:
    d = RAW / "fed"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if refresh or not p.exists():
        p.write_text(_get(f"https://www.federalreserve.gov/monetarypolicy/{name}").text, encoding="utf-8")
    return p.read_text(encoding="utf-8")


def _historical_year(yr: int, refresh: bool) -> list[dict]:
    html = _page(f"fomchistorical{yr}.htm", refresh)
    link_dates = {pd.Timestamp(x) for x in re.findall(r"monetary/?(?:\d{4}/)?(\d{8})[a-z]?(?:\.htm|/default\.htm)", html)}
    out = []
    for h in re.findall(r"<h5[^>]*>(.*?)</h5>", html, flags=re.S):
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h)).strip()
        m = HEAD.match(text)
        if not m:
            continue
        label = m["label"].lower()
        y = int(m["yr"])
        m_end = MONTHS[(m["m2"] or m["m1"]).lower()]
        d_end = int(m["d2"] or m["d1"])
        dt = pd.Timestamp(date(y, m_end, d_end))
        if "cancel" in label or "notation" in label:
            continue
        if "unscheduled" in label or "conference call" in label:
            # only count calls that produced a policy press release that day or the next
            rel = [d for d in link_dates if 0 <= (d - dt).days <= 1]
            if rel:
                out.append({"date": min(rel), "kind": "unscheduled", "heading": text})
            continue
        if "meeting" in label:
            out.append({"date": dt, "kind": "scheduled", "heading": text,
                        "statement_link_found": dt in link_dates})
    return out


def _current_calendar(refresh: bool) -> list[dict]:
    html = _page("fomccalendars.htm", refresh)
    ds = sorted({pd.Timestamp(x) for x in re.findall(r"monetary(\d{8})a\.htm", html)})
    return [{"date": d, "kind": "scheduled", "heading": "fomccalendars.htm statement link",
             "statement_link_found": True} for d in ds]


def fomc_dates(first_year: int = 2000, last_hist_year: int = 2020, refresh: bool = False) -> pd.DataFrame:
    rows = []
    for yr in range(first_year, last_hist_year + 1):
        rows += _historical_year(yr, refresh)
    rows += [r for r in _current_calendar(refresh) if r["date"].year > last_hist_year]
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def announcement_trading_days(fomc: pd.DataFrame, calendar: pd.DatetimeIndex, kinds=("scheduled",)) -> pd.DatetimeIndex:
    """Map announcement dates to the trading day on which markets first react (same day, or next if closed)."""
    sel = fomc[fomc["kind"].isin(kinds)]["date"]
    pos = calendar.searchsorted(pd.DatetimeIndex(sel), side="left")
    pos = pos[pos < len(calendar)]
    return pd.DatetimeIndex(sorted(set(calendar[pos])))
