"""Example portfolios used by the research questions and the dashboard (weights sum to 1, long-only)."""

PORTFOLIOS = {
    "60/40": {"SPY": 0.60, "IEF": 0.40},
    "Global multi-asset": {"SPY": 0.35, "EFA": 0.15, "EEM": 0.05, "IEF": 0.20, "TLT": 0.05, "LQD": 0.07,
                           "HYG": 0.05, "GOLD": 0.05, "DBC": 0.03},
    "Risk-balanced": {"SPY": 0.15, "EFA": 0.05, "EEM": 0.03, "IEF": 0.35, "TLT": 0.12, "LQD": 0.10, "GOLD": 0.10,
                      "DBC": 0.10},
    "US sectors equal-weight": {s: 0.1 for s in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR"]},
    "Bank-heavy equity": {"SPY": 0.50, "XLF": 0.30, "KRE": 0.20},
}
