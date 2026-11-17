"""Write dashboard/data.js from the panel, the cached models and the research outputs, and wrap
dashboard/page.html (the publishable page body) into dashboard/index.html for local use.

    python scripts/build_dashboard.py
    python -m http.server 8765 --directory dashboard     # then open http://localhost:8765
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from contagion.data import load_panel  # noqa: E402
from contagion.paths import DASHBOARD  # noqa: E402
from contagion.viz.export import build  # noqa: E402

SKELETON = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head><body>\n{body}\n</body></html>\n")

if __name__ == "__main__":
    if "--html-only" not in sys.argv:
        print(build(load_panel()))
    page = (DASHBOARD / "page.html").read_text(encoding="utf-8")
    # cache-busting stamps for local browsers (the publishable page.html keeps plain file names)
    stamp = str(int(time.time()))
    local = page.replace('src="app.js"', f'src="app.js?v={stamp}"').replace('src="data.js"', f'src="data.js?v={stamp}"')
    (DASHBOARD / "index.html").write_text(SKELETON.format(body=local), encoding="utf-8")
    print("wrote", DASHBOARD / "index.html")
