"""Claim levels: the evidential status of every dependence measure the platform produces.

The platform never reports a network edge, a "transmitter" ranking or a stress-test response without saying which
of these it is. The dashboard renders the level as a badge, and the report groups findings by level.

DESCRIPTIVE   co-movement in the sample (correlations, partial correlations, PCA, tail dependence, copulas,
              contemporaneous conditional expectations). Says nothing about direction or mechanism.
PREDICTIVE    incremental forecasting content (Granger tests, VAR lag dynamics, the h>=1 part of forecast-error
              variance decompositions, out-of-sample forecast comparisons). Direction in time, not in mechanism:
              common drivers, asynchronous closes and stale prices all create predictive links without causation.
IDENTIFIED    a causal effect under explicitly stated identifying assumptions (here: identification through
              heteroskedasticity on FOMC announcement days). Valid only if the assumptions hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Claim(str, Enum):
    DESCRIPTIVE = "descriptive"
    PREDICTIVE = "predictive"
    IDENTIFIED = "identified"


CLAIM_TEXT = {
    Claim.DESCRIPTIVE: "Descriptive co-movement: no direction, no mechanism.",
    Claim.PREDICTIVE: "Predictive (lead-lag) content: direction in time only; not evidence of causation.",
    Claim.IDENTIFIED: "Causal effect under stated identifying assumptions; invalid if the assumptions fail.",
}


@dataclass(frozen=True)
class Tagged:
    """A result plus its claim level and the assumptions a reader needs to interpret it."""

    value: object
    claim: Claim
    method: str
    assumptions: tuple[str, ...] = ()

    def describe(self) -> str:
        a = "; ".join(self.assumptions) if self.assumptions else "none stated"
        return f"[{self.claim.value}] {self.method} - {CLAIM_TEXT[self.claim]} Assumptions: {a}"


# Claim level of each measure family, used by the dashboard and the report.
MEASURE_CLAIMS = {
    "rolling_correlation": Claim.DESCRIPTIVE,
    "ewma_correlation": Claim.DESCRIPTIVE,
    "dcc_correlation": Claim.DESCRIPTIVE,
    "partial_correlation": Claim.DESCRIPTIVE,
    "pca_absorption": Claim.DESCRIPTIVE,
    "dynamic_factor": Claim.DESCRIPTIVE,
    "tail_dependence": Claim.DESCRIPTIVE,
    "copula": Claim.DESCRIPTIVE,
    "covar": Claim.DESCRIPTIVE,
    "conditional_stress": Claim.DESCRIPTIVE,
    "granger": Claim.PREDICTIVE,
    "var_irf": Claim.PREDICTIVE,
    "dy_connectedness": Claim.DESCRIPTIVE,  # mixes contemporaneous (descriptive) and lagged (predictive) parts
    "dy_dynamic_part": Claim.PREDICTIVE,
    "fomc_heteroskedasticity": Claim.IDENTIFIED,
}
