"""Change-point detection.

``pelt``   offline, exact optimal segmentation (Killick et al. 2012) of a (multivariate) Gaussian mean/covariance
           model via ruptures. Uses the whole sample: a historical segmentation, not a real-time signal.
``bocpd``  Bayesian online change-point detection (Adams & MacKay 2007) with a Normal-Gamma prior (unknown mean and
           variance, Student-t predictive). Online: the run-length posterior at t uses x_1..x_t only.
           With a constant hazard P(r_t = 0) equals the hazard by construction, so the informative outputs are the
           run-length posterior summaries: P(r_t < k) ("a change in the last k days"), the MAP and the mean run length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special


def pelt(X, penalty: float | None = None, min_size: int = 20, model: str = "normal", jump: int = 5) -> list[int]:
    """Indices where new segments start (excluding 0). Default penalty: BIC-type 2 * d(d+3)/2 * log(n)... scaled
    to the number of parameters of a Gaussian segment."""
    import ruptures as rpt

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    if penalty is None:
        k = d + d * (d + 1) / 2  # mean + covariance parameters per segment
        penalty = k * np.log(n)
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(X)
    bkps = algo.predict(pen=penalty)
    return [b for b in bkps if b < n]


def segments_frame(index: pd.DatetimeIndex, bkps: list[int], X=None) -> pd.DataFrame:
    edges = [0] + list(bkps) + [len(index)]
    rows = []
    Xa = None if X is None else np.asarray(X, dtype=float).reshape(len(index), -1)
    for a, b in zip(edges[:-1], edges[1:]):
        row = {"start": index[a], "end": index[b - 1], "n": b - a}
        if Xa is not None:
            row["ann_vol_first"] = float(np.std(Xa[a:b, 0]) * np.sqrt(252))
            row["mean_first"] = float(np.mean(Xa[a:b, 0]))
        rows.append(row)
    return pd.DataFrame(rows)


def bocpd(x, hazard: float = 1 / 250, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
          beta0: float | None = None, r_max: int = 1500, k_recent: int = 10) -> pd.DataFrame:
    """Univariate BOCPD. Returns per-date: p_recent (= P(r_t < k_recent)), map_run, mean_run, pred_logp."""
    idx = x.index if isinstance(x, pd.Series) else None
    xa = np.asarray(x, dtype=float)
    T = len(xa)
    if beta0 is None:
        beta0 = alpha0 * np.nanvar(xa[: min(250, T)])  # prior scale anchored on the first year only (no look-ahead)
    logH, log1mH = np.log(hazard), np.log1p(-hazard)
    # run-length distribution (log) and sufficient statistics per run length
    logR = np.array([0.0])
    mu = np.array([mu0])
    kap = np.array([kappa0])
    alp = np.array([alpha0])
    bet = np.array([beta0])
    p_recent = np.empty(T)
    map_run = np.empty(T, dtype=int)
    mean_run = np.empty(T)
    pred_lp = np.empty(T)
    for t in range(T):
        xt = xa[t]
        # Student-t predictive for each run length
        scale2 = bet * (kap + 1) / (alp * kap)
        nu = 2 * alp
        z = (xt - mu) ** 2 / (nu * scale2)
        logpred = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * scale2)
                   - (nu + 1) / 2 * np.log1p(z))
        lg = logR + logpred + log1mH
        lc = special.logsumexp(logR + logpred + logH)
        new = np.concatenate([[lc], lg])
        lev = special.logsumexp(new)
        pred_lp[t] = lev
        logR = new - lev
        # update statistics (prepend prior for run length 0)
        mu_n = (kap * mu + xt) / (kap + 1)
        bet_n = bet + kap * (xt - mu) ** 2 / (2 * (kap + 1))
        mu = np.concatenate([[mu0], mu_n])
        kap = np.concatenate([[kappa0], kap + 1])
        alp = np.concatenate([[alpha0], alp + 0.5])
        bet = np.concatenate([[beta0], bet_n])
        if len(logR) > r_max:  # truncate the long tail, renormalise
            logR, mu, kap, alp, bet = logR[:r_max], mu[:r_max], kap[:r_max], alp[:r_max], bet[:r_max]
            logR = logR - special.logsumexp(logR)
        R = np.exp(logR)
        p_recent[t] = R[:k_recent].sum()
        map_run[t] = int(np.argmax(R))
        mean_run[t] = float((np.arange(len(R)) * R).sum())
    out = pd.DataFrame({"p_recent": p_recent, "map_run": map_run, "mean_run": mean_run, "pred_logp": pred_lp},
                       index=idx)
    out.attrs.update({"hazard": hazard, "k_recent": k_recent})
    return out


def bocpd_changes(res: pd.DataFrame, drop: float = 0.5, min_gap: int = 20) -> list:
    """Dates where the MAP run length collapses (falls below ``drop`` x its previous value), at least
    ``min_gap`` days apart."""
    mr = res["map_run"].to_numpy()
    hits, last = [], -10 ** 9
    for t in range(1, len(mr)):
        if mr[t - 1] >= 20 and mr[t] < drop * mr[t - 1] and t - last >= min_gap:
            hits.append(res.index[t - mr[t]] if res.index is not None else t - mr[t])
            last = t
    return hits
