"""Shared research utilities: plotting style (light + dark renders of every figure), findings, crisis dates."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..claims import Claim  # noqa: E402
from ..data.events import bear_episodes  # noqa: E402
from ..paths import FIGURES, TABLES  # noqa: E402

PALETTE = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "base": "#c3c2b7", "mid": "#f0efec", "shade": "#0b0b0b",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
              "context": "#b9b7ae",
              "blue": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
              "red": ["#f9d4d3", "#f2a9a8", "#ea7d7c", "#e34948", "#c93a39", "#a82f2e", "#862524", "#661c1b"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "base": "#383835", "mid": "#383835", "shade": "#f0efec",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
             "context": "#5d5c57",
             "blue": ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4"],
             "red": ["#661c1b", "#862524", "#a82f2e", "#c93a39", "#e34948", "#e66767", "#ea7d7c", "#f2a9a8"]},
}


def _sans() -> list[str]:
    from matplotlib import font_manager as fm

    names = {f.name for f in fm.fontManager.ttflist}
    return ["Segoe UI"] if "Segoe UI" in names else ["DejaVu Sans"]


def style(mode: str = "light") -> dict:
    P = PALETTE[mode]
    plt.rcParams.update({
        "figure.facecolor": P["surface"], "axes.facecolor": P["surface"], "savefig.facecolor": P["surface"],
        "text.color": P["ink"], "axes.labelcolor": P["ink2"], "axes.edgecolor": P["base"],
        "xtick.color": P["muted"], "ytick.color": P["muted"], "xtick.labelcolor": P["ink2"],
        "ytick.labelcolor": P["ink2"], "axes.grid": True, "grid.color": P["grid"], "grid.linewidth": 0.7,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "font.family": _sans(), "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlecolor": P["ink"],
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "axes.prop_cycle": matplotlib.cycler(color=P["series"]),
        "text.parse_math": False, "figure.dpi": 100, "savefig.dpi": 160, "axes.axisbelow": True,
    })
    return P


def diverging_cmap(mode: str = "light") -> LinearSegmentedColormap:
    P = PALETTE[mode]
    blues = P["blue"][::-1] if mode == "light" else P["blue"][::-1]
    reds = P["red"]
    if mode == "light":
        cols = [P["blue"][6], P["blue"][4], P["blue"][1], P["mid"], P["red"][1], P["red"][3], P["red"][6]]
    else:
        cols = [P["blue"][7], P["blue"][5], P["blue"][2], P["mid"], P["red"][2], P["red"][5], P["red"][7]]
    del blues, reds
    return LinearSegmentedColormap.from_list(f"div_{mode}", cols)


def sequential_cmap(mode: str = "light") -> LinearSegmentedColormap:
    P = PALETTE[mode]
    cols = [P["surface"], P["blue"][1], P["blue"][3], P["blue"][5], P["blue"][7]] if mode == "light" else \
        [P["surface"], P["blue"][1], P["blue"][3], P["blue"][5], P["blue"][7]]
    return LinearSegmentedColormap.from_list(f"seq_{mode}", cols)


@dataclass
class Figure:
    id: str
    title: str
    caption: str
    light: Path
    dark: Path


@dataclass
class Table:
    id: str
    title: str
    df: pd.DataFrame
    caption: str = ""
    floatfmt: str = "{:.2f}"


@dataclass
class Finding:
    question: str
    claim: Claim
    headline: str
    detail: str
    caveats: list[str] = field(default_factory=list)


@dataclass
class Section:
    key: str
    title: str
    question: str
    answer: str
    design: str
    findings: list[Finding] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    robustness: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)


def save_fig(fid: str, draw, figsize=(8.0, 3.6), title: str = "", caption: str = "") -> Figure:
    """Render ``draw(fig, P, mode)`` in light and dark and save both PNGs."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {}
    for mode in ("light", "dark"):
        P = style(mode)
        fig = plt.figure(figsize=figsize, layout="constrained")
        draw(fig, P, mode)
        p = FIGURES / f"{fid}_{mode}.png"
        fig.savefig(p)
        plt.close(fig)
        paths[mode] = p
    return Figure(fid, title, caption, paths["light"], paths["dark"])


def save_table(t: Table) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    t.df.to_csv(TABLES / f"{t.id}.csv")


def shade_episodes(ax, episodes: pd.DataFrame, P: dict, alpha: float = 0.07) -> None:
    for _, e in episodes.iterrows():
        ax.axvspan(e["peak"], e["trough"], color=P["shade"], alpha=alpha, lw=0)


def crisis_episodes(panel, down: float = 0.15, up: float = 0.15, start: str | None = None) -> pd.DataFrame:
    r = panel.returns["SPY"].dropna()
    if start:
        r = r.loc[start:]
    return bear_episodes(r, down=down, up=up)


def img_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def fmt_pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def nw_tstat(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West t-statistics."""
    import statsmodels.api as sm

    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res.params, res.tvalues
