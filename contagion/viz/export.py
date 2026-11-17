"""Export everything the dashboard needs into dashboard/data.js (one global: window.CL_DATA).

Encoding: daily series are arrays aligned to one shared date list. Returns are scaled integers
(log returns x 1e5, basis-point changes x 100) with null for missing; probabilities x 1000. Covariances are
floats rounded to 5 significant digits. Everything else the dashboard shows is recomputed in the browser.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..claims import CLAIM_TEXT, Claim
from ..data.events import NAMED_EVENTS
from ..models import load_dcc, load_dynamic, load_granger, load_regimes
from ..paths import DASHBOARD, MODELS, OUTPUTS
from ..portfolio.portfolios import PORTFOLIOS
from ..research.common import crisis_episodes
from ..stress.engine import STRESS_UNIVERSE, StressEngine
from ..stress.scenarios import HYPOTHETICAL, historical_scenarios

DASH_IDS = ["SPY", "IWM", "EFA", "EEM", "XLF", "UST2Y", "UST10Y", "BAA10Y", "DXY", "USDJPY", "COPPER_F", "OIL", "GOLD",
            "VIX", "MOVE", "SHY", "IEF", "TLT", "LQD", "HYG", "UUP", "FXY", "GLD", "USO", "DBC", "XLB", "XLE", "XLI",
            "XLK", "XLP", "XLU", "XLV", "XLY", "IYR", "KRE", "QQQ", "IWF", "IWD", "N225", "FTSE", "GDAXI", "HSI",
            "EURUSD", "AUDUSD", "UST3M", "UST30Y", "BE10Y"]
START = "2000-08-24"


def _ints(a, scale, digits=0):
    a = np.asarray(a, dtype=float) * scale
    return [None if not np.isfinite(v) else int(round(v)) for v in a]


def _g5(x):
    return float(f"{x:.5g}")


def build(panel) -> dict:
    U = panel.universe
    cal = panel.returns.loc[START:].index
    dates = [d.strftime("%Y-%m-%d") for d in cal]
    pos = {d: i for i, d in enumerate(cal)}
    R = panel.returns.loc[START:, DASH_IDS]
    scale = {i: (100.0 if U.loc[i, "transform"] == "dbp" else 1e5) for i in DASH_IDS}
    uni = [{"id": i, "name": str(U.loc[i, "name"]), "cls": str(U.loc[i, "asset_class"]), "group": str(U.loc[i, "group"]),
            "close": float(U.loc[i, "close_et"]), "unit": "bp" if U.loc[i, "transform"] == "dbp" else "%",
            "sync": bool(U.loc[i, "synchronous"])} for i in DASH_IDS]
    out = {"meta": {"built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "last": dates[-1],
                    "claims": {c.value: CLAIM_TEXT[c] for c in Claim}},
           "dates": dates, "universe": uni,
           "ret": {i: _ints(R[i].to_numpy(), scale[i]) for i in DASH_IDS}, "scale": scale,
           "vix": _ints(panel.levels["VIX"].reindex(cal).to_numpy(), 100)}
    ep = crisis_episodes(panel, 0.15, 0.15, "2000-09-01")
    out["crises"] = [{"peak": str(e["peak"].date()), "trough": str(e["trough"].date()), "depth": round(float(e["depth"]), 3)}
                     for _, e in ep.iterrows()]
    out["events"] = [{"date": d, "label": t} for d, t in NAMED_EVENTS]

    # ---------------------------------------------------------------- network (macro)
    d500 = load_dynamic("dy_macro_w500")
    tv = load_dynamic("tvp_macro")
    names = d500.names
    month_end = pd.Series(np.arange(len(d500.dates)), index=d500.dates).groupby(d500.dates.to_period("M")).last()
    ix = month_end.to_numpy()
    groups = {n: str(U.loc[n, "asset_class"]) for n in names}
    g_pw = load_granger("macro", "pairwise")
    g_ix = [int(np.searchsorted(g_pw["dates"], d500.dates[k], side="right") - 1) for k in ix]
    out["network"] = {
        "names": names, "labels": {n: str(U.loc[n, "name"]) for n in names}, "groups": groups,
        "dates": [d500.dates[k].strftime("%Y-%m-%d") for k in ix],
        "theta": [_ints(d500.theta[k].ravel(), 1000) for k in ix],
        "theta_c": [_ints(d500.theta_contemp[k].ravel(), 1000) for k in ix],
        "granger": ["".join(str(int(v)) for v in g_pw["adjacency"][max(g, 0)].ravel()) if g >= 0 else None for g in g_ix],
        "tci_roll": _ints(d500.total.reindex(cal, method="ffill").to_numpy(), 100),
        "tci_tvp": _ints(tv.total.reindex(cal).where(lambda s: s.index > tv.dates[250]).to_numpy(), 100),
        "tci_dyn": _ints(d500.total_dynamic.reindex(cal, method="ffill").to_numpy(), 100),
        "params": d500.params,
    }

    # ---------------------------------------------------------------- stress
    hist = historical_scenarios(panel.returns)
    eng_latest = StressEngine.from_panel(panel)
    stress_names = eng_latest.names
    asofs = [pd.Timestamp(f"{y}-12-31") for y in range(2008, eng_latest.asof.year)] + [eng_latest.asof]
    covs = {m: {} for m in ("full", "rolling", "ewma", "stressed", "regime", "crisis", "glasso")}
    for a in asofs:
        e = eng_latest if a == eng_latest.asof else StressEngine.from_panel(panel, asof=a)
        key = e.asof.strftime("%Y-%m-%d")
        for m in ("full", "rolling", "ewma", "stressed"):
            covs[m][key] = [_g5(v) for v in e.cov(m).to_numpy().ravel()]
    lk = eng_latest.asof.strftime("%Y-%m-%d")
    for m in ("regime", "crisis", "glasso"):
        covs[m][lk] = [_g5(v) for v in eng_latest.cov(m).to_numpy().ravel()]
    scen = [{"name": s.name, "family": s.family, "desc": s.description, "kind": s.kind, "horizon": s.horizon_days,
             "shocks": {k: float(v) for k, v in s.shocks.items()}} for s in HYPOTHETICAL]
    scen += [{"name": s.name, "family": s.family, "desc": s.description, "kind": s.kind, "horizon": s.horizon_days,
              "window": list(s.window), "shocks": {k: float(v) for k, v in s.shocks.items() if k in stress_names}}
             for s in hist]
    out["stress"] = {"names": stress_names, "asof": list(covs["full"].keys()), "latest": lk, "cov": covs,
                     "scenarios": scen, "portfolios": PORTFOLIOS, "regime_today": eng_latest.regime_state_today(),
                     "units": {n: ("bp" if U.loc[n, "transform"] == "dbp" else "%") for n in stress_names},
                     "labels": {n: str(U.loc[n, "name"]) for n in stress_names},
                     "classes": {n: str(U.loc[n, "asset_class"]) for n in stress_names}}

    # ---------------------------------------------------------------- regimes
    rg = load_regimes()
    hmm = rg["hmm"].reindex(cal)
    reg = {"hmm_info": rg["hmm_info"],
           "smoothed": [_ints(hmm[f"smoothed_{k}"], 1000) for k in range(3)],
           "filtered": [_ints(hmm[f"filtered_{k}"], 1000) for k in range(3)],
           "realtime": [_ints(hmm[f"realtime_{k}"], 1000) for k in range(3)] if "realtime_0" in hmm else None,
           "ms_high": _ints(rg["ms"]["filtered_high"].reindex(cal), 1000),
           "bocpd_spy": _ints(rg["bocpd"]["spy_p_recent"].reindex(cal), 1000),
           "garch_vol": _ints(rg["vol"]["garch_vol_ann"].reindex(cal), 1000),
           "sb_corr": _ints(rg["stock_bond"]["corr_spy_ief_126"].reindex(cal), 1000),
           "pelt": [{"start": str(r.start.date()), "end": str(r.end.date()), "vol": round(float(r.ann_vol_first), 3),
                     "corr": round(float(r.corr_spy_10y), 3)} for r in rg["pelt"].itertuples()]}
    if "bayes" in rg:
        b = rg["bayes"]
        reg["bayes"] = {"dates": [d.strftime("%Y-%m-%d") for d in b.index], "p_turb": _ints(b.iloc[:, 1], 1000),
                        "summary": rg["bayes_summary"].round(4).reset_index().to_dict(orient="records")}
    out["regimes"] = reg

    # ---------------------------------------------------------------- DCC residuals (macro)
    dcc = load_dcc()
    sr = pd.DataFrame(dcc["std_resid"], index=dcc["dates"], columns=dcc["names"]).reindex(cal)
    out["dcc"] = {"names": dcc["names"], "a": dcc["a"], "b": dcc["b"],
                  "z": {n: _ints(sr[n].to_numpy(), 1000) for n in dcc["names"]}}

    # ---------------------------------------------------------------- research answers
    ans = {}
    for k in ("q1", "q2", "q3", "q4", "q5", "q6"):
        p = OUTPUTS / "research" / f"{k}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                s = pickle.load(f)
            ans[k] = {"title": s.title, "answer": s.answer, "claims": sorted({x.claim.value for x in s.findings})}
    out["answers"] = ans
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    js = "window.CL_DATA = " + json.dumps(out, separators=(",", ":"), allow_nan=False) + ";\n"
    (DASHBOARD / "data.js").write_text(js, encoding="utf-8")
    return {"path": str(DASHBOARD / "data.js"), "mb": len(js) / 1e6, "asofs": len(asofs)}
