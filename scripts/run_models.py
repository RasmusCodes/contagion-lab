"""Estimate every network and regime model on the processed panel and cache the results in outputs/models/.

    python scripts/run_models.py                 # full run (~10-20 min)
    python scripts/run_models.py --quick         # skip TVP-VAR, real-time HMM, DCC and the Bayesian HMM
    python scripts/run_models.py --only regimes  # one stage: networks | regimes
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from contagion.data import load_panel  # noqa: E402
from contagion.models import run_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["networks", "dfm", "regimes"], default=None)
    a = ap.parse_args()
    stages = (a.only,) if a.only else ("networks", "regimes")
    run_all(load_panel(), quick=a.quick, stages=stages)
