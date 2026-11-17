"""Scenario library.

Shocks are stated in each series' native return units (see config/universe.csv ``transform``):
    logret / dlog series: cumulative log change, e.g. log(0.8) for a 20% fall
    dbp series:           change in basis points, e.g. +100
Hypothetical scenarios shock a few "driver" series; the propagation engines fill in everything else.
Historical scenarios replay the realised cumulative moves of every series between two closes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ln = np.log


@dataclass
class Scenario:
    name: str
    shocks: dict[str, float]
    family: str
    description: str = ""
    kind: str = "hypothetical"
    window: tuple[str, str] | None = None  # historical: (close before the episode, last close of the episode)
    tags: list[str] = field(default_factory=list)
    horizon_days: int = 20  # trading days over which the shock unfolds; scales the conditional dispersion only


HYPOTHETICAL = [
    Scenario("Equity crash -20%", {"SPY": ln(0.80)}, "equity_crash", "S&P 500 falls 20%; everything else conditional."),
    Scenario("Equity crash -35%", {"SPY": ln(0.65)}, "equity_crash", "GFC/COVID-scale S&P 500 decline."),
    Scenario("Rates +100bp parallel", {"UST2Y": 100.0, "UST10Y": 100.0}, "rate_shock", "Parallel upward shift."),
    Scenario("Bear flattener", {"UST2Y": 150.0, "UST10Y": 75.0}, "rate_shock", "Hawkish policy repricing (2022-type)."),
    Scenario("Bull steepener", {"UST2Y": -150.0, "UST10Y": -50.0}, "rate_shock", "Emergency easing."),
    Scenario("Long-end selloff (term premium)", {"UST10Y": 100.0}, "rate_shock", "10Y +100bp, 2Y conditional (Oct-2023 type)."),
    Scenario("Oil -40%", {"OIL": ln(0.60)}, "commodity", "Demand collapse / supply glut (2014-16, 2020)."),
    Scenario("Oil +50%", {"OIL": ln(1.50)}, "commodity", "Supply shock (2022-type)."),
    Scenario("Copper -25%", {"COPPER_F": ln(0.75)}, "commodity", "Industrial-metals slump / China slowdown."),
    Scenario("Gold +20%", {"GOLD": ln(1.20)}, "commodity", "Safe-haven bid."),
    Scenario("Credit spreads +150bp", {"BAA10Y": 150.0}, "credit", "Investment-grade spread widening (Baa-Treasury)."),
    Scenario("USD +10%", {"DXY": ln(1.10)}, "fx", "Broad dollar squeeze."),
    Scenario("Yen carry unwind", {"USDJPY": ln(0.90)}, "fx", "Yen +10% vs USD (Aug-2024 type)."),
    Scenario("Volatility shock (VIX x2.5)", {"VIX": ln(2.5)}, "volatility", "Implied-volatility spike."),
    Scenario("Liquidity crisis", {"VIX": ln(3.0), "MOVE": ln(2.0), "BAA10Y": 200.0, "SPY": ln(0.80)}, "liquidity",
             "Joint shock: equity vol x3, rates vol x2, credit +200bp, equities -20%."),
    Scenario("Financials -30%", {"XLF": ln(0.70)}, "sector", "Banking-sector stress."),
    Scenario("Technology -30%", {"XLK": ln(0.70)}, "sector", "Tech de-rating."),
    Scenario("Energy -35%", {"XLE": ln(0.65)}, "sector", "Energy-sector collapse."),
    Scenario("Real estate -25%", {"IYR": ln(0.75)}, "sector", "Property / REIT stress."),
]

HISTORICAL = [
    ("GFC: Lehman to Nov-2008 low", "2008-09-12", "2008-11-20", "equity_crash"),
    ("GFC: peak to trough", "2007-10-09", "2009-03-09", "equity_crash"),
    ("US downgrade 2011", "2011-07-22", "2011-08-08", "equity_crash"),
    ("Taper tantrum 2013", "2013-05-21", "2013-06-24", "rate_shock"),
    ("China devaluation 2015", "2015-08-17", "2015-08-25", "equity_crash"),
    ("Oil and high-yield 2015-16", "2015-11-03", "2016-02-11", "commodity"),
    ("Volmageddon 2018", "2018-01-26", "2018-02-08", "volatility"),
    ("Q4 2018 sell-off", "2018-09-20", "2018-12-24", "equity_crash"),
    ("COVID crash 2020", "2020-02-19", "2020-03-23", "liquidity"),
    ("2022 rate shock", "2022-01-03", "2022-10-12", "rate_shock"),
    ("SVB 2023", "2023-03-08", "2023-03-17", "sector"),
    ("Yen carry unwind 2024", "2024-07-31", "2024-08-05", "fx"),
    ("Tariff shock 2025", "2025-04-02", "2025-04-08", "equity_crash"),
]


def historical_scenarios(returns: pd.DataFrame) -> list[Scenario]:
    """Realised cumulative moves (sum of daily log returns / bp changes) after the start close through the end
    close. Series without data over the whole window are omitted from that scenario."""
    out = []
    for name, a, b, fam in HISTORICAL:
        w = returns.loc[pd.Timestamp(a) + pd.Timedelta(days=1): pd.Timestamp(b)]
        if w.empty:
            continue
        ok = w.notna().all()
        shocks = w.loc[:, ok].sum().to_dict()
        out.append(Scenario(name, shocks, fam, f"Realised moves {a} close to {b} close.", kind="historical",
                            window=(a, b), horizon_days=len(w)))
    return out


def all_scenarios(returns: pd.DataFrame | None = None) -> list[Scenario]:
    return HYPOTHETICAL + (historical_scenarios(returns) if returns is not None else [])
