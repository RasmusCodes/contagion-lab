"""Project paths. Everything is relative to the repository root so the project can be moved."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
REPORT = OUTPUTS / "report"
DASHBOARD = ROOT / "dashboard"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, OUTPUTS, MODELS, FIGURES, TABLES, REPORT, DASHBOARD):
        p.mkdir(parents=True, exist_ok=True)
