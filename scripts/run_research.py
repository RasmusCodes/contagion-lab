"""Run the six research questions (each caches its Section to outputs/research/<q>.pkl).

    python scripts/run_research.py              # all six
    python scripts/run_research.py --only q1 q3
"""
from __future__ import annotations

import argparse
import importlib
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from contagion.data import load_panel  # noqa: E402
from contagion.paths import OUTPUTS  # noqa: E402

MODULES = {"q1": "contagion.research.q1_precrisis", "q2": "contagion.research.q2_transmitters",
           "q3": "contagion.research.q3_rates", "q4": "contagion.research.q4_stress",
           "q5": "contagion.research.q5_tails", "q6": "contagion.research.q6_hedging"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = OUTPUTS / "research"
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    for key, mod in MODULES.items():
        if a.only and key not in a.only:
            continue
        t = time.time()
        sec = importlib.import_module(mod).run(panel)
        with open(out / f"{key}.pkl", "wb") as f:
            pickle.dump(sec, f)
        print(f"== {key} ({time.time() - t:.0f}s)\n{sec.answer}\n", flush=True)
