import numpy as np
from scipy import stats

from contagion.network.copula import fit_bivariate, fit_t_copula, simulate_t_copula, t_tail_dependence
from contagion.network.tail import gaussian_tail_dependence, tail_dependence


def _clayton(theta, n, rng):
    V = rng.gamma(1 / theta, 1.0, size=n)
    E = rng.exponential(size=(n, 2))
    return (1 + E / V[:, None]) ** (-1 / theta)


def test_clayton_is_selected_and_recovered(rng):
    U = _clayton(2.0, 8000, rng)
    fit = fit_bivariate(U[:, 0], U[:, 1])
    assert fit.loc[0, "family"] == "clayton"
    assert abs(fit.loc[0, "params"][0] - 2.0) < 0.2
    assert abs(fit.loc[0, "lambda_L"] - 2 ** -0.5) < 0.05


def test_t_copula_recovery_and_tail_dependence(rng):
    R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    U = simulate_t_copula(R, nu=4.0, n=20000, seed=3)
    f = fit_t_copula(stats.norm.ppf(U))  # margins irrelevant: ranks are used
    assert abs(f["nu"] - 4.0) < 0.8
    assert np.allclose(f["R"], R, atol=0.03)
    lam = t_tail_dependence(0.6, 4.0)
    emp = tail_dependence(U[:, 0], U[:, 1], q=0.02)
    assert abs(emp - lam) < 0.08


def test_gaussian_tail_dependence_is_small_and_benchmark_matches(rng):
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=200000)
    emp = tail_dependence(z[:, 0], z[:, 1], q=0.01)
    bench = gaussian_tail_dependence(0.5, 0.01)
    assert abs(emp - bench) < 0.04
    assert bench < 0.2
