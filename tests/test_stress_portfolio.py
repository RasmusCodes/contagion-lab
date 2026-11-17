import numpy as np
import pandas as pd

from contagion.portfolio.hedging import EwmaCov, RollingCov, StaticCov, hedge_backtest, hedge_ratio
from contagion.portfolio.risk import component_vol, evt_var_es, kupiec_pof, var_es
from contagion.stress.propagation import conditional_gaussian, network_decomposition, var_propagation
from contagion.network.var import fit_var


def _cov(rng, n=6):
    A = rng.standard_normal((n, n))
    S = A @ A.T + n * np.eye(n)
    names = [f"x{i}" for i in range(n)]
    return pd.DataFrame(S, index=names, columns=names)


def test_network_decomposition_equals_conditional_gaussian(rng):
    S = _cov(rng)
    shocks = {"x0": -2.0, "x3": 1.5}
    cg = conditional_gaussian(S, shocks)
    dec = network_decomposition(S, shocks)
    assert np.allclose(cg["expected"], dec["total"], atol=1e-10)
    assert np.allclose(dec["first_order"] + dec["higher_order"], dec["total"].where(~cg["shocked"], 0.0), atol=1e-10)


def test_conditional_gaussian_matches_regression(rng):
    X = rng.multivariate_normal(np.zeros(3), [[1, 0.6, 0.2], [0.6, 1, 0.3], [0.2, 0.3, 1]], size=200000)
    S = pd.DataFrame(np.cov(X, rowvar=False), index=list("abc"), columns=list("abc"))
    cg = conditional_gaussian(S, {"a": 1.0})
    beta_b = np.polyfit(X[:, 0], X[:, 1], 1)[0]
    assert abs(cg.loc["b", "expected"] - beta_b) < 0.01
    assert abs(cg.loc["b", "cond_sd"] - np.sqrt(1 - 0.36)) < 0.01


def test_var_propagation_impact_is_conditional_mean(rng):
    from conftest import simulate_var

    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    Y = simulate_var(np.array([[0.2, 0.0], [0.3, 0.1]]), S, 5000, rng)
    f = fit_var(pd.DataFrame(Y, columns=["a", "b"]), 1)
    pr = var_propagation(f, {"a": 1.0}, H=5)
    assert np.isclose(pr["impact"]["b"], f.Sigma[1, 0] / f.Sigma[0, 0])
    assert pr["after_impact"]["b"] > 0  # a leads b in the simulated system


def test_component_vol_sums_to_portfolio_vol(rng):
    S = _cov(rng, 4)
    w = {"x0": 0.4, "x1": 0.3, "x2": 0.2, "x3": 0.1}
    cv = component_vol(w, S)
    wv = np.array(list(w.values()))
    assert np.isclose(cv["contribution"].sum(), np.sqrt(wv @ S.to_numpy() @ wv))
    assert np.isclose(cv["share"].sum(), 1.0)


def test_var_es_gaussian_and_evt(rng):
    x = rng.standard_normal(200000) * 0.01
    v, e = var_es(x, 0.01, "historical")
    vg, eg = var_es(x, 0.01, "gaussian")
    assert abs(v - 0.02326) < 0.0005 and abs(vg - 0.02326) < 0.0005
    assert abs(e - 0.02665) < 0.0007 and abs(eg - 0.02665) < 0.0005
    t = rng.standard_t(3, 200000) * 0.01
    ve, ee = evt_var_es(t, 0.001)
    q = -np.quantile(t, 0.001)
    assert abs(ve - q) / q < 0.15 and ee > ve
    lr, p = kupiec_pof(rng.random(5000) < 0.01, 0.01)
    assert p > 0.001


def test_hedge_ratio_and_no_lookahead(rng):
    n = 3000
    h = rng.standard_normal(n) * 0.01
    tgt = 0.8 * h + rng.standard_normal(n) * 0.005
    R = pd.DataFrame({"T": tgt, "H": h}, index=pd.bdate_range("2010-01-01", periods=n))
    X = np.expm1(R.to_numpy())
    assert abs(hedge_ratio(np.cov(X, rowvar=False), 0, [1])[0] - 0.8) < 0.03
    bt = hedge_backtest(R, "T", ["H"], StaticCov(X), start=500, rebalance=5, cost_bps=0.0)
    assert bt["hedged"].var() < 0.4 * bt["unhedged"].var()
    # changing the future must not change hedge ratios already set
    R2 = R.copy()
    R2.iloc[2000:, 1] *= -3
    X2 = np.expm1(R2.to_numpy())
    for prov_a, prov_b in ((StaticCov(X), StaticCov(X2)), (RollingCov(X, 252), RollingCov(X2, 252)),
                           (EwmaCov(X, 0.97), EwmaCov(X2, 0.97))):
        a = hedge_backtest(R, "T", ["H"], prov_a, start=500)
        b = hedge_backtest(R2, "T", ["H"], prov_b, start=500)
        assert np.allclose(a["h_H"].iloc[:1400], b["h_H"].iloc[:1400])
