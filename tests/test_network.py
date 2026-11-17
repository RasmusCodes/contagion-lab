import numpy as np
import pandas as pd
import pytest
from conftest import simulate_var

from contagion.network.connectedness import connectedness, gfevd, rolling_connectedness, tvp_var_connectedness
from contagion.network.correlation import (exceedance_correlation, jennrich_test, partial_correlation,
                                           rolling_cov)
from contagion.network.granger import bh_fdr, conditional_granger, pairwise_granger
from contagion.network.var import fit_var, girf, ma_coefficients, select_lag


def test_var_matches_statsmodels(rng):
    from statsmodels.tsa.api import VAR

    A = np.array([[[0.3, 0.1, 0.0], [0.0, 0.2, 0.1], [0.05, 0.0, -0.1]], [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, 0.05]]])
    S = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    Y = simulate_var(A, S, 3000, rng)
    mine = fit_var(Y, p=2)
    sm = VAR(Y).fit(2, trend="c")
    assert np.allclose(mine.A, sm.coefs, atol=1e-10)
    assert np.allclose(mine.const, sm.intercept, atol=1e-10)
    assert np.allclose(mine.Sigma, sm.sigma_u, atol=1e-10)
    assert np.allclose(ma_coefficients(mine.A, 6), sm.ma_rep(5), atol=1e-10)
    sel = VAR(Y).select_order(4)
    assert select_lag(Y, maxlag=4, criterion="bic") == sel.bic
    assert select_lag(Y, maxlag=4, criterion="aic") == sel.aic


def test_ridge_var_is_unit_free(rng):
    A = np.array([[0.2, 0.1], [0.0, 0.1]])
    Y = simulate_var(A, np.eye(2), 2000, rng)
    f1 = fit_var(Y, 1, ridge=0.05)
    Y2 = Y * np.array([1.0, 1000.0])
    f2 = fit_var(Y2, 1, ridge=0.05)
    D = np.diag([1.0, 1000.0])
    assert np.allclose(f2.A[0], D @ f1.A[0] @ np.linalg.inv(D), atol=1e-10)


def test_gfevd_rows_sum_and_scale_invariance(rng):
    A = np.array([[0.4, 0.2, 0.0], [0.0, 0.1, 0.3], [0.1, 0.0, 0.2]])
    S = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 0.5]])
    th, c, d = gfevd(A[None], S, H=10, split=True)
    assert np.allclose(th.sum(axis=1), 1.0)
    assert np.allclose(c + d, th)
    D = np.diag([10.0, 0.01, 3.0])
    th2 = gfevd((D @ A @ np.linalg.inv(D))[None], D @ S @ D, H=10)
    assert np.allclose(th, th2, atol=1e-12)


def test_contemporaneous_net_is_centrality_not_direction(rng):
    """No dynamics at all, node 0 correlated with 1 and 2 (which are mutually uncorrelated): DY NET still
    labels node 0 a 'net transmitter'. The dynamic NET is exactly zero."""
    S = np.array([[1.0, 0.6, 0.6], [0.6, 1.0, 0.0], [0.6, 0.0, 1.0]])
    Y = rng.multivariate_normal(np.zeros(3), S, size=20000)
    con = connectedness(pd.DataFrame(Y, columns=["hub", "a", "b"]), p=1, H=10)
    assert con.net["hub"] > 5.0  # looks like a transmitter ...
    assert abs(con.net_dynamic()["hub"]) < 1.0  # ... but there is no directional (lagged) content
    assert con.dynamic_share() < 0.02


def test_lead_lag_shows_in_dynamic_net(rng):
    A = np.array([[0.0, 0.0], [0.5, 0.0]])  # y0 leads y1
    Y = simulate_var(A, np.eye(2), 20000, rng)
    con = connectedness(pd.DataFrame(Y, columns=["leader", "follower"]), p=1, H=10)
    assert con.net_dynamic()["leader"] > 10.0
    assert con.net["leader"] > 10.0


def test_girf_impact_is_regression(rng):
    S = np.array([[1.0, 0.5], [0.5, 2.0]])
    r = girf(np.zeros((1, 2, 2)), S, H=3, shock=0, size=1.0)
    assert np.allclose(r[0], [1.0, 0.5])
    assert np.allclose(r[1:], 0.0)


def test_rolling_and_tvp_connectedness_track_static(rng):
    A = np.array([[0.2, 0.3, 0.0], [0.0, 0.2, 0.2], [0.1, 0.0, 0.2]])
    S = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    Y = pd.DataFrame(simulate_var(A, S, 3000, rng), columns=list("abc"),
                     index=pd.bdate_range("2010-01-01", periods=3000))
    true = connectedness(Y).total
    rc = rolling_connectedness(Y, window=500, step=50)
    assert abs(rc.total.mean() - true) < 3.0
    tv = tvp_var_connectedness(Y, kappa1=0.995, kappa2=0.99, prior_obs=300)
    assert abs(tv.total.iloc[500:].mean() - true) < 4.0


def test_granger_power_and_size(rng):
    A = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.0, 0.0]])  # 0 -> 1 only
    Y = simulate_var(A, np.eye(3), 4000, rng)
    pw = pairwise_granger(Y, p=1)
    assert pw["pval"][1, 0] < 1e-6
    others = [pw["pval"][i, j] for i in range(3) for j in range(3) if i != j and (i, j) != (1, 0)]
    assert min(others) > 0.001
    cg = conditional_granger(Y, p=1)
    assert cg["pval"][1, 0] < 1e-6


def test_granger_size_under_garch_errors(rng):
    """With volatility clustering and no causality, the HC1 test should not grossly over-reject."""
    T, reps, rej = 1500, 60, 0
    for _ in range(reps):
        h = np.ones(T)
        e = rng.standard_normal((T, 2))
        x = np.zeros((T, 2))
        for t in range(1, T):
            h[t] = 0.05 + 0.1 * x[t - 1, 0] ** 2 + 0.85 * h[t - 1]
            x[t] = np.sqrt(h[t]) * e[t]
        x[:, 1] = x[:, 1] * np.sqrt(h)  # common volatility, independent signs
        p = pairwise_granger(x, p=1)["pval"][1, 0]
        rej += p < 0.05
    assert rej / reps < 0.15


def test_bh_fdr():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej = bh_fdr(p, 0.05)
    assert rej[:2].all() and not rej[2:].any()


def test_rolling_cov_matches_pandas(rng):
    X = rng.standard_normal((300, 3))
    C = rolling_cov(X, 60)
    pdc = pd.DataFrame(X).rolling(60).cov().to_numpy().reshape(300, 3, 3)
    assert np.allclose(C[59:], pdc[59:])


def test_partial_correlation_chain(rng):
    # x0 -> x1 -> x2 chain: x0 and x2 are conditionally independent given x1
    n = 50000
    x0 = rng.standard_normal(n)
    x1 = 0.7 * x0 + rng.standard_normal(n)
    x2 = 0.7 * x1 + rng.standard_normal(n)
    pc = partial_correlation(np.cov(np.column_stack([x0, x1, x2]), rowvar=False))
    assert abs(pc[0, 2]) < 0.02 and pc[0, 1] > 0.3


def test_jennrich_size_and_power(rng):
    R = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    ps = [jennrich_test(rng.multivariate_normal(np.zeros(3), R, 400), rng.multivariate_normal(np.zeros(3), R, 400))[2]
          for _ in range(150)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.12
    R2 = np.array([[1, 0.8, 0.3], [0.8, 1, 0.4], [0.3, 0.4, 1]])
    p = jennrich_test(rng.multivariate_normal(np.zeros(3), R, 400), rng.multivariate_normal(np.zeros(3), R2, 400))[2]
    assert p < 1e-4


def test_exceedance_gaussian_matches_benchmark(rng):
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=60000)
    ex = exceedance_correlation(z[:, 0], z[:, 1], quantiles=(0.1, 0.9), n_sim=300000)
    assert np.allclose(ex["corr"], ex["gauss"], atol=0.05)
