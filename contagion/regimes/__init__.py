"""Regime detection: Gaussian HMM (EM, filtered/real-time), Markov switching, change points (PELT, BOCPD),
volatility regimes (GARCH, expanding percentiles), DCC dynamic correlations, Bayesian sticky HMM (Gibbs)."""
from .hmm import GaussianHMM, expanding_filtered_probs  # noqa: F401
