"""Dynamic factor models.

``dynamic_factor_model`` fits a state-space DFM (statsmodels DynamicFactorMQ, EM algorithm) with one global
factor loading on every series plus block factors per asset class, all AR(1):
    x_it = lambda_ig f_t^global + lambda_ib f_t^block(i) + e_it
The Kalman-smoothed factors use the full sample (two-sided), so rolling R^2 on them is a DESCRIPTIVE integration
measure, not a real-time indicator. ``rolling_factor_r2`` gives the share of each series' variance explained by
the common factor in a moving window.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def dynamic_factor_model(R: pd.DataFrame, blocks: dict[str, str] | None = None, factor_order: int = 1,
                         maxiter: int = 200, tolerance: float = 1e-6) -> dict:
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    Z = (R - R.mean()) / R.std()
    if blocks:
        factors = {c: ["Global", blocks[c]] for c in Z.columns}
        names = ["Global"] + sorted(set(blocks.values()))
    else:
        factors = {c: ["Global"] for c in Z.columns}
        names = ["Global"]
    # single-member blocks are not identified separately from the idiosyncratic term: fold them into Global only
    if blocks:
        counts = pd.Series(blocks).value_counts()
        for c in Z.columns:
            if counts[blocks[c]] < 2:
                factors[c] = ["Global"]
        names = ["Global"] + sorted(b for b in set(blocks.values()) if counts[b] >= 2)
    factor_orders = {n: factor_order for n in names}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = DynamicFactorMQ(Z, factors=factors, factor_orders=factor_orders, idiosyncratic_ar1=False,
                              standardize=False)
        res = mod.fit(disp=False, maxiter=maxiter, tolerance=tolerance)
    F = res.factors.smoothed[names] if hasattr(res.factors, "smoothed") else pd.DataFrame(res.factors["smoothed"])
    F.index = Z.index
    r2 = {}
    for c in Z.columns:
        y = Z[c].to_numpy()
        fg = F["Global"].to_numpy()
        Xg = np.column_stack([np.ones(len(y)), fg])
        bg = np.linalg.lstsq(Xg, y, rcond=None)[0]
        r2g = 1 - np.var(y - Xg @ bg) / np.var(y)
        cols = [n for n in factors[c]]
        Xa = np.column_stack([np.ones(len(y))] + [F[n].to_numpy() for n in cols])
        ba = np.linalg.lstsq(Xa, y, rcond=None)[0]
        r2a = 1 - np.var(y - Xa @ ba) / np.var(y)
        r2[c] = {"global_r2": r2g, "global_plus_block_r2": r2a, "block": factors[c][-1]}
    return {"factors": F, "r2": pd.DataFrame(r2).T, "result": res, "factor_names": names}


def rolling_factor_r2(R: pd.DataFrame, f: pd.Series, window: int = 250) -> pd.DataFrame:
    """Rolling R^2 of each series on a factor (squared rolling correlation)."""
    return R.rolling(window).corr(f) ** 2
