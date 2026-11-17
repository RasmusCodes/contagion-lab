"""Crisis and stress-event dating.

Two kinds of dates, kept strictly apart:

* ``bear_episodes`` - mechanical peak/trough dating (Lunde-Timmermann style state machine) applied to an equity
  total-return index. A peak is confirmed once the index falls ``down`` below it; a trough once the index rises
  ``up`` above it. These are the crisis dates used for statistical tests, so the test cannot be tuned by
  choosing which episodes to include.
* ``NAMED_EVENTS`` - a curated list of well-known stress dates, used only to annotate charts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bear_episodes(log_returns: pd.Series, down: float = 0.15, up: float = 0.15) -> pd.DataFrame:
    r = log_returns.dropna()
    level = np.exp(r.cumsum().to_numpy())
    idx = r.index
    n = len(level)
    rows = []
    state, pk, tr = "bull", 0, 0
    crossing = None
    for t in range(1, n):
        if state == "bull":
            if level[t] > level[pk]:
                pk = t
            elif level[t] <= level[pk] * (1 - down):
                state, tr, crossing = "bear", t, t
        else:
            if level[t] < level[tr]:
                tr = t
            elif level[t] >= level[tr] * (1 + up):
                rows.append((pk, crossing, tr, False))
                state, pk = "bull", t
    if state == "bear":
        rows.append((pk, crossing, tr, True))
    out = pd.DataFrame([{
        "peak": idx[p], "crossing": idx[c], "trough": idx[t],
        "depth": level[t] / level[p] - 1.0,
        "days_peak_to_trough": t - p, "provisional": prov,
    } for p, c, t, prov in rows])
    return out


def drawdown(log_returns: pd.Series) -> pd.Series:
    lv = np.exp(log_returns.fillna(0).cumsum())
    return lv / lv.cummax() - 1.0


# Annotation only (never used to select test samples).
NAMED_EVENTS = [
    ("2007-02-27", "Shanghai sell-off / subprime warnings"),
    ("2007-08-09", "BNP Paribas freezes funds"),
    ("2007-10-09", "S&P 500 pre-GFC high"),
    ("2008-03-14", "Bear Stearns rescue"),
    ("2008-09-15", "Lehman bankruptcy"),
    ("2009-03-09", "S&P 500 GFC low"),
    ("2010-05-06", "Flash crash"),
    ("2011-08-08", "US downgrade sell-off"),
    ("2013-05-22", "Taper tantrum begins"),
    ("2015-08-24", "China devaluation sell-off"),
    ("2016-02-11", "Oil / high-yield low"),
    ("2016-06-24", "Brexit vote"),
    ("2018-02-05", "Volmageddon"),
    ("2018-12-24", "Q4-2018 low"),
    ("2020-02-24", "COVID sell-off begins"),
    ("2020-03-16", "COVID crash / Fed to zero"),
    ("2020-03-23", "Fed unlimited QE / COVID low"),
    ("2022-01-03", "S&P 500 high before 2022 rate shock"),
    ("2022-06-13", "CPI shock"),
    ("2022-09-28", "UK gilt / LDI crisis"),
    ("2022-10-12", "2022 low"),
    ("2023-03-10", "SVB failure"),
    ("2023-10-19", "10Y Treasury reaches 5%"),
    ("2024-08-05", "Yen carry unwind"),
    ("2025-04-03", "Tariff shock"),
    ("2025-04-08", "Tariff-shock low"),
]


def named_events() -> pd.DataFrame:
    return pd.DataFrame([(pd.Timestamp(d), e) for d, e in NAMED_EVENTS], columns=["date", "event"])
