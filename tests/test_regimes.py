import numpy as np
import pandas as pd

from contagion.regimes.bayesian import BayesianHMM
from contagion.regimes.changepoint import bocpd, pelt
from contagion.regimes.dcc import DCC, _corr, _q_path
from contagion.regimes.hmm import GaussianHMM, expanding_filtered_probs


def _two_state(rng, T=3000):
    P = np.array([[0.98, 0.02], [0.05, 0.95]])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.random() < P[s[t - 1], 1]
    vol = np.where(s == 0, 0.5, 2.0)
    x = rng.standard_normal(T) * vol
    return x, s, P


def test_hmm_recovers_states_and_filter_is_causal(rng):
    x, s, P = _two_state(rng)
    X = x[:, None]
    m = GaussianHMM(2, n_init=3).fit(X)
    assert np.allclose(np.sqrt([m.covs_[0, 0, 0], m.covs_[1, 0, 0]]), [0.5, 2.0], atol=0.15)
    assert np.allclose(np.diag(m.transmat_), np.diag(P), atol=0.03)
    filt = m.filter(X)
    acc = np.mean((filt[:, 1] > 0.5) == (s == 1))
    assert acc > 0.88
    # causality: filtered probabilities up to t do not change when the future changes
    X2 = X.copy()
    X2[2000:] *= 5
    assert np.allclose(m.filter(X2)[:2000], filt[:2000])


def test_expanding_probs_have_no_lookahead(rng):
    x, s, _ = _two_state(rng, 1600)
    df = pd.DataFrame({"x": x}, index=pd.bdate_range("2000-01-03", periods=1600))
    a = expanding_filtered_probs(df, min_train=800, refit_every=200, n_init=2)
    df2 = df.copy()
    df2.iloc[1300:] *= 4
    b = expanding_filtered_probs(df2, min_train=800, refit_every=200, n_init=2)
    assert np.allclose(a.iloc[:1200, :2].to_numpy(), b.iloc[:1200, :2].to_numpy(), equal_nan=True)


def test_bocpd_detects_variance_break(rng):
    x = np.concatenate([rng.standard_normal(600) * 0.5, rng.standard_normal(400) * 2.0])
    res = bocpd(pd.Series(x), hazard=1 / 300)
    assert res["map_run"].iloc[650] < 60  # run length restarted near t = 600
    assert res["map_run"].iloc[590] > 300
    assert res["p_recent"].iloc[600:615].max() > 0.5


def test_pelt_finds_break(rng):
    x = np.concatenate([rng.standard_normal(500) * 0.5, rng.standard_normal(500) * 2.0])
    b = pelt(x, min_size=20)
    assert any(abs(k - 500) <= 10 for k in b)


def _sim_dcc(rng, T, a, b, Qbar):
    E = np.zeros((T, Qbar.shape[0]))
    Q = Qbar.copy()
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(E[t - 1], E[t - 1]) + b * Q
        d = np.sqrt(np.diag(Q))
        E[t] = np.linalg.cholesky(Q / np.outer(d, d)) @ rng.standard_normal(len(d))
    return E


def test_dcc_recovers_parameters():
    a, b = 0.05, 0.90
    Qbar = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    est = []
    for seed in (1, 2, 3):
        E = _sim_dcc(np.random.default_rng(seed), 4000, a, b, Qbar)
        m = DCC().fit(E)
        est.append((m.a, m.b))
        # the optimiser reaches the likelihood maximum (checked against a coarse grid)
        grid = min(m.nll((aa, bb), E) for aa in np.arange(0.02, 0.10, 0.01) for bb in np.arange(0.80, 0.97, 0.01))
        assert m.nll((m.a, m.b), E) <= grid + 1e-6
    ea, eb = np.mean(est, axis=0)
    assert abs(ea - a) < 0.015 and abs(eb - b) < 0.03
    E = _sim_dcc(np.random.default_rng(9), 60, a, b, Qbar)
    # vectorised recursion equals the loop
    R1 = _corr(_q_path(E[:50], a, b, Qbar))
    Q = Qbar.copy()
    for t in range(1, 50):
        Q = (1 - a - b) * Qbar + a * np.outer(E[t - 1], E[t - 1]) + b * Q
    d = np.sqrt(np.diag(Q))
    assert np.allclose(R1[49], Q / np.outer(d, d))


def test_bayesian_hmm(rng):
    x, s, _ = _two_state(rng, 800)
    m = BayesianHMM(2, n_iter=400, burn=150, thin=2, seed=1).fit(x)
    acc = np.mean((m.state_prob["p_state1"].to_numpy() > 0.5) == (s == 1))
    assert acc > 0.85
    summ = m.summary(scale=1.0, annualize=1.0)
    assert abs(summ.loc[0, "vol_mean"] - 0.5) < 0.1 and abs(summ.loc[1, "vol_mean"] - 2.0) < 0.3
