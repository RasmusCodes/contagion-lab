import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def simulate_var(A, Sigma, T, rng, burn=200, c=None):
    A = np.asarray(A)
    if A.ndim == 2:
        A = A[None]
    p, N, _ = A.shape
    L = np.linalg.cholesky(Sigma)
    c = np.zeros(N) if c is None else c
    Y = np.zeros((T + burn, N))
    for t in range(p, T + burn):
        y = c + L @ rng.standard_normal(N)
        for l in range(p):
            y += A[l] @ Y[t - l - 1]
        Y[t] = y
    return Y[burn:]
