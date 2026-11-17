"""Fetch (or load from cache) every series in config/universe.csv and build data/processed/.

    python scripts/build_data.py            # use the raw cache where present
    python scripts/build_data.py --refresh  # re-download everything
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from contagion.data import build_panel  # noqa: E402
from contagion.data.calendar import lead_lag_profile  # noqa: E402
from contagion.data.fomc import fomc_dates  # noqa: E402
from contagion.paths import PROCESSED, TABLES, ensure_dirs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start", default="1999-01-04")
    a = ap.parse_args()
    ensure_dirs()
    print("building panel ...")
    panel = build_panel(refresh=a.refresh, start=a.start)
    m = panel.meta
    print(f"master calendar {m['master_calendar']['first']} -> {m['master_calendar']['last']} "
          f"({m['master_calendar']['n_days']} NYSE days); series: {panel.returns.shape[1]}; failed: {m['failed']}")
    print(f"issues logged: {len(m['issues'])}; robust-outlier flags: {len(m['outlier_flags'])}")

    fomc = fomc_dates(refresh=a.refresh)
    fomc.to_csv(PROCESSED / "fomc_dates.csv", index=False)
    print(f"FOMC dates: {len(fomc)} ({(fomc['kind'] == 'scheduled').sum()} scheduled); "
          f"{fomc['date'].min().date()} -> {fomc['date'].max().date()}")

    # Asynchrony diagnostic against the NYSE close (SPY) over a common recent window.
    R = panel.returns.loc["2010-01-01":]
    ll = lead_lag_profile(R, "SPY")
    ll["close_et"] = panel.universe.loc[ll.index, "close_et"]
    ll = ll.sort_values("timing_signal")
    ll.to_csv(TABLES / "asynchrony_vs_spy.csv")
    print("largest timing signals (corr at lag+1 minus lag-1 vs SPY):")
    print(ll[["lag-1", "lag+0", "lag+1", "timing_signal", "close_et"]].round(3).head(6).to_string())
    print(ll[["lag-1", "lag+0", "lag+1", "timing_signal", "close_et"]].round(3).tail(8).to_string())
    (PROCESSED / "build_summary.json").write_text(json.dumps({
        "n_series": int(panel.returns.shape[1]), "failed": m["failed"], "n_fomc": int(len(fomc))}, indent=1))


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
