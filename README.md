# Contagion Lab

A research platform for how financial shocks propagate across US equities, sectors, financial institutions,
Treasury yields, credit, FX, commodities and volatility indices. It combines a timestamp-aware data pipeline,
a dynamic network-modelling library, regime models, a stress-testing engine, portfolio-risk tools, six research
studies with a reproducible report, and an interactive dashboard.

**Results:** the full report is [outputs/report/report.md](outputs/report/report.md) (renders here, with figures)
and `outputs/report/report.html` (self-contained, light and dark themes; download and open). Tables behind every
number are CSV files in [outputs/tables](outputs/tables).

Every dependence measure carries a claim level, and the report and dashboard keep them apart:

| Level | Meaning | Examples |
|---|---|---|
| Descriptive | co-movement; no direction, no mechanism | correlations, partial correlations, PCA, tail dependence, copulas, conditional stress expectations |
| Predictive | incremental forecasting content; direction in time only | Granger networks, VAR lag dynamics, the lagged part of Diebold-Yilmaz connectedness, out-of-sample tests |
| Identified | a causal effect under stated assumptions | FOMC-day identification through heteroskedasticity (Q3) |

## Quick start

Python 3.11 or newer. Commands shown for macOS / Linux; on Windows use `.venv\Scripts\python`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest                              # 33 estimator tests on simulated data (no download)
.venv/bin/python scripts/build_data.py                  # download ~90 series -> data/raw, data/processed
.venv/bin/python scripts/run_models.py                  # networks + regimes -> outputs/models (~25 min)
.venv/bin/python scripts/run_research.py                # Q1-Q6 -> figures, tables, research sections (~15 min)
.venv/bin/python scripts/build_report.py                # outputs/report/report.html and report.md
.venv/bin/python scripts/build_dashboard.py             # dashboard/data.js and dashboard/index.html
```

The repository ships code, figures, tables and the report, but no market data: `build_data.py` downloads it from
Yahoo Finance, FRED, Cboe and the Federal Reserve, whose terms apply to anything you download. The dashboard needs
the full pipeline to have run, because its data file is generated locally. Then serve this folder,
`.venv/bin/python -m http.server 8765`, and open `http://localhost:8765/dashboard/`.
Raw downloads are cached in `data/raw` with their fetch time; `build_data.py --refresh` re-downloads.
`run_models.py --only regimes` (or `networks`, `dfm`) and `run_research.py --only q4` rerun single stages; Q4 and Q6 cache their
backtests in `outputs/research/cache`, keyed by the sample end, so text-only changes rerun in seconds.

## Layout

```
config/universe.csv          93 series: source, symbol, transform, closing time, sub-universes
config/data_overrides.csv    manual masks / fixes applied before alignment (empty)
contagion/data/              fetchers (Yahoo, FRED, Cboe, Fed FOMC calendar), calendar alignment, panel
contagion/network/           correlation, partial correlation, PCA, dynamic factors, Granger, VAR,
                             Diebold-Yilmaz + TVP-VAR connectedness, tail dependence / CoVaR / MES, copulas, graphs
contagion/regimes/           Gaussian HMM (filtered + real-time), PELT and Bayesian online change points,
                             GARCH volatility regimes, DCC-GARCH, Bayesian sticky HMM (Gibbs)
contagion/stress/            scenario library, propagation engines, StressEngine (7 covariance views)
contagion/portfolio/         VaR / ES / EVT, Euler risk decomposition, factor decomposition, hedging backtests,
                             correlation-breakdown monitors
contagion/research/          q1..q6 studies and the report builder
contagion/viz/export.py      dashboard data export
dashboard/                   page.html (page body) and app.js; build_dashboard.py adds index.html and data.js
outputs/                     figures (light + dark), tables (CSV) and the report; models/ and research/ hold
                             local caches that the pipeline regenerates (not in the repository)
tests/                       estimator tests on simulated data
```

## Data

Sources: Yahoo Finance chart API (ETFs, stocks, equity indices, futures), FRED (Treasury CMT yields, Moody's
spreads, Fed H.10 noon FX rates, Cboe volatility indices, St. Louis and Chicago Fed stress indices), Cboe (VIX
family) and federalreserve.gov (FOMC dates). The master calendar is NYSE trading days (from SPY). Each series is
aligned by its own local trading date; holidays give a zero return and the move lands on the next trading day, so
cumulative returns are conserved; gaps over three days stay missing. `Panel.returns_for(set)` returns a balanced
block for a sub-universe (`macro`, `sync`, `sectors`, `fin`, `global`, ...).

Timing is treated as data, not noise: every series has a closing time, and `outputs/tables/asynchrony_vs_spy.csv`
measures how much of each day's S&P 500 news it absorbs the next day. Problems found while building the panel:

* Yahoo's daily FX bars switched convention around 2011 (the "close" became a 00:00-London snapshot equal to the
  open). FX therefore uses the Fed's H.10 noon rates.
* FRED serves only three years of ICE BofA credit spreads; long-history credit uses Moody's Baa minus 10Y (which
  updates a day late relative to equities) and LQD / HYG prices.
* Oil and gold splice front-month futures into USO / GLD at the ETFs' inception (removes roll jumps and the
  2020-04-20 negative WTI print from the core sample). Corn and natural-gas futures keep visible roll artefacts.
* Yahoo's `BK` now points to an unrelated listing; BNY Mellon is `BNY`.

## Research questions (answers in outputs/report/report.html)

1. **Does connectedness rise before crises?** No reliable pre-crisis build-up (placebo tests, BH correction);
   connectedness rises during crises and adds no out-of-sample crash prediction beyond VIX / realized volatility.
2. **Which assets transmit shocks?** Equities in the standard DY sense, but ~90% of that is same-day centrality;
   lagged spillovers are 1-3% of connectedness and line up with late-reacting (early-closing or stale) series.
3. **Which sectors are most rate-vulnerable?** Real estate, homebuilders and utilities on descriptive betas;
   FOMC-day identification pins down the market's response but only a few sector responses precisely.
4. **Do dynamic networks improve stress testing?** Yes: EWMA / DCC / rolling covariances beat a static network on
   out-of-sample stress days; crisis-conditioned covariances do worse; all under-cover the tails.
5. **Are correlations more informative in tail events?** Tail dependence is real (above Gaussian benchmarks), but
   past tail-day correlations do not forecast future tail-day correlations better than full-sample ones.
6. **Does regime detection improve hedging?** No: HMM-based hedges help in flight-to-quality crashes (2020) and
   fail when the crisis type changes (2022); adaptive estimators (EWMA, DCC) hedge better.

## Dashboard

Six views: network (five dependence measures by month, with claim badges), stress test (hypothetical and
historical scenarios, seven covariance views, direct / first-order / higher-order exposure), correlation explorer
(rolling, EWMA, DCC; a Forbes-Rigobon-adjusted breakdown monitor with alert episodes; tail-conditional
correlation against an exact Gaussian benchmark; matrix on any date),
regime timeline (smoothed / filtered / real-time HMM, VIX, stock-bond correlation, change points), portfolio
exposure (Euler risk, factor betas, VaR / ES, scenario P&L by channel) and findings.

## Extending

* Add a series: one row in `config/universe.csv` (source, symbol, transform, closing time, sets), then rerun
  `build_data.py`.
* Add a scenario: append to `HYPOTHETICAL` or `HISTORICAL` in `contagion/stress/scenarios.py`.
* Every out-of-sample routine takes data up to t-1 only; keep that property when adding models (the tests show
  how to check it by perturbing the future).

Research use only; nothing here is investment advice.
