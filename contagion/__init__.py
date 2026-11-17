"""contagion: a cross-asset financial contagion and systemic-risk research platform.

Sub-packages
------------
data        raw fetchers, calendar alignment, transforms, the processed panel
network     correlation / partial correlation / PCA / factor / Granger / VAR / connectedness / tail / copula
regimes     hidden Markov models, change points, volatility regimes, DCC, Bayesian regime models
stress      scenarios and shock-propagation engines
portfolio   risk decomposition, tail risk, hedging, correlation breakdown
research    the six research questions and the report builder
viz         dashboard export

Every estimator that produces a dependence measure is tagged with a ``claim`` level (see ``contagion.claims``)
so downstream code and the dashboard can keep descriptive, predictive and identified (causal-under-assumptions)
results apart.
"""

__version__ = "0.1.0"
