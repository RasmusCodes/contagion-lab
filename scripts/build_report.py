"""Assemble outputs/report/report.html (+ report.md) from the cached research sections."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from contagion.data import load_panel  # noqa: E402
from contagion.research.report import build  # noqa: E402

if __name__ == "__main__":
    out = build(load_panel())
    print({k: str(v) for k, v in out.items()})
