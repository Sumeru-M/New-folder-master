"""
constraints.py — Institutional Portfolio Constraints

Modular constraint system for portfolio optimization.
Each constraint is a self-contained class that produces cvxpy-compatible
constraint expressions. Add new constraints by subclassing BaseConstraint.

Constraint classes:
    - MaxWeightConstraint         : Single stock cap
    - MinWeightConstraint         : Minimum position size
    - SectorCapConstraint         : Sector-level exposure cap
    - TurnoverConstraint          : L1 turnover vs previous weights
    - ESGConstraint               : Minimum portfolio ESG score
    - TrackingErrorConstraint     : Ex-ante TE vs benchmark
    - LiquidityConstraint         : Volume-based weight cap per asset
    - LongOnlyConstraint          : No short selling (standard)
    - FullInvestmentConstraint    : Weights sum to 1

Usage:
    builder = ConstraintBuilder(n_assets, tickers)
    builder.add(MaxWeightConstraint(0.10))
    builder.add(SectorCapConstraint(sector_map, cap=0.25))
    constraints = builder.build(w, Sigma, mu)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# cvxpy imported lazily so module loads even if cvxpy unavailable
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseConstraint(ABC):
    """
    Abstract base for all portfolio constraints.

    Subclasses implement build() which receives the cvxpy weight variable
    and any auxiliary inputs, and returns a list of cvxpy constraint objects.
    """

    @abstractmethod
    def build(
        self,
        w: "cp.Variable",
        *,
        Sigma: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        extra: Optional[Dict] = None,
    ) -> List:
        """
        Return list of cvxpy constraints.

        Parameters
        ----------
        w : cp.Variable  shape (N,)
            Portfolio weight variable.
        Sigma : np.ndarray, optional  shape (N, N)
            Annualised covariance matrix (needed for TE constraint).
        mu : np.ndarray, optional  shape (N,)
            Expected returns (rarely needed in constraints).
        extra : dict, optional
            Catch-all for constraint-specific inputs passed at solve time.
        """

    def validate(self, n_assets: int) -> None:
        """Optional pre-solve validation. Raises ValueError on bad config."""


# ---------------------------------------------------------------------------
# Standard weight constraints
# ---------------------------------------------------------------------------

@dataclass
class LongOnlyConstraint(BaseConstraint):
    """w_i >= 0 for all i (no short selling)."""

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        return [w >= 0]


@dataclass
class FullInvestmentConstraint(BaseConstraint):
    """sum(w) == 1 (fully invested)."""

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        return [cp.sum(w) == 1]


@dataclass
class MaxWeightConstraint(BaseConstraint):
    """
    w_i <= max_weight for all i.

    Parameters
    ----------
    max_weight : float
        Maximum fraction allocated to any single asset (e.g. 0.10 for 10%).
    """
    max_weight: float = 0.10

    def validate(self, n_assets: int) -> None:
        min_feasible = 1.0 / n_assets
        if self.max_weight < min_feasible:
            raise ValueError(
                f"MaxWeightConstraint({self.max_weight:.2%}) is infeasible for "
                f"{n_assets} assets — minimum feasible is {min_feasible:.2%}. "
                f"Relax to at least {min_feasible:.2%}."
            )

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        return [w <= self.max_weight]


@dataclass
class MinWeightConstraint(BaseConstraint):
    """
    If asset is held, w_i >= min_weight.

    Implementation note: true min-lot is a MILP. This approximation
    enforces min_weight on all assets (i.e., every asset is in the portfolio).
    For selective inclusion, use min_weight=0 and filter post-hoc.

    Parameters
    ----------
    min_weight : float
        Minimum allocation to any held position (e.g. 0.01 for 1%).
    """
    min_weight: float = 0.01

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        return [w >= self.min_weight]


# ---------------------------------------------------------------------------
# Sector constraint
# ---------------------------------------------------------------------------

@dataclass
class SectorCapConstraint(BaseConstraint):
    """
    Sum of weights in each sector <= sector_cap.

    Parameters
    ----------
    sector_map : Dict[str, str]
        Maps ticker -> sector name. e.g. {"RELIANCE.NS": "Energy", ...}
    tickers : List[str]
        Ordered list of tickers matching weight variable indices.
    sector_cap : float
        Maximum fraction for any single sector (e.g. 0.25 for 25%).
    per_sector_caps : Dict[str, float], optional
        Override cap for specific sectors. Unspecified sectors use sector_cap.
    """
    sector_map: Dict[str, str]
    tickers: List[str]
    sector_cap: float = 0.25
    per_sector_caps: Dict[str, float] = field(default_factory=dict)

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        constraints = []
        # Build sector index groups
        sectors: Dict[str, List[int]] = {}
        for idx, ticker in enumerate(self.tickers):
            sector = self.sector_map.get(ticker, "Unknown")
            sectors.setdefault(sector, []).append(idx)

        for sector, indices in sectors.items():
            cap = self.per_sector_caps.get(sector, self.sector_cap)
            # Sum of weights in this sector <= cap
            constraints.append(cp.sum(w[indices]) <= cap)

        return constraints

    def get_sector_weights(self, weights: pd.Series) -> pd.Series:
        """Utility: compute current sector weights from a weight series."""
        result = {}
        for ticker, sector in self.sector_map.items():
            if ticker in weights.index:
                result[sector] = result.get(sector, 0.0) + weights[ticker]
        return pd.Series(result).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Turnover constraint
# ---------------------------------------------------------------------------

@dataclass
class TurnoverConstraint(BaseConstraint):
    """
    L1 turnover constraint: sum(|w - w_prev|) <= max_turnover.

    Linearised via auxiliary variable t:
        t_i >= w_i - w_prev_i
        t_i >= w_prev_i - w_i
        sum(t) <= max_turnover

    Parameters
    ----------
    prev_weights : np.ndarray  shape (N,)
        Previous portfolio weights.
    max_turnover : float
        Maximum one-way turnover allowed (e.g. 0.20 = 20%).
    """
    prev_weights: np.ndarray
    max_turnover: float = 0.20

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        w_prev = self.prev_weights
        # Auxiliary variable for absolute deviation
        t = cp.Variable(len(w_prev), nonneg=True, name="turnover_slack")
        constraints = [
            t >= w - w_prev,
            t >= w_prev - w,
            cp.sum(t) <= self.max_turnover,
        ]
        return constraints

    def estimate_turnover(self, new_weights: np.ndarray) -> float:
        """Compute actual L1 turnover given proposed weights."""
        return float(np.sum(np.abs(new_weights - self.prev_weights)))


# ---------------------------------------------------------------------------
# ESG constraint
# ---------------------------------------------------------------------------

@dataclass
class ESGConstraint(BaseConstraint):
    """
    Portfolio ESG score >= min_esg_score.

    Portfolio ESG = weighted average of individual ESG scores.
    dot(w, esg_scores) >= min_esg_score

    Parameters
    ----------
    esg_scores : np.ndarray  shape (N,)
        ESG score per asset (e.g. 0–100 scale).
    min_esg_score : float
        Minimum acceptable portfolio-level ESG score.
    """
    esg_scores: np.ndarray
    min_esg_score: float = 50.0

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        return [self.esg_scores @ w >= self.min_esg_score]

    def compute_portfolio_esg(self, weights: np.ndarray) -> float:
        """Compute current portfolio ESG score."""
        return float(self.esg_scores @ weights)


# ---------------------------------------------------------------------------
# Tracking error constraint
# ---------------------------------------------------------------------------

@dataclass
class TrackingErrorConstraint(BaseConstraint):
    """
    Ex-ante tracking error vs benchmark <= max_te.

    TE² = (w - w_bm)ᵀ Σ (w - w_bm) <= max_te²

    This is a convex (SOCP) constraint in cvxpy.

    Parameters
    ----------
    benchmark_weights : np.ndarray  shape (N,)
        Benchmark weight vector (must align with asset ordering).
    max_te : float
        Maximum annualised tracking error (e.g. 0.05 for 5%).
    """
    benchmark_weights: np.ndarray
    max_te: float = 0.05

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        if Sigma is None:
            raise ValueError("TrackingErrorConstraint requires Sigma.")
        active = w - self.benchmark_weights
        # cp.quad_form(active, Sigma) <= max_te²
        return [cp.quad_form(active, Sigma) <= self.max_te ** 2]

    def compute_te(self, weights: np.ndarray, Sigma: np.ndarray) -> float:
        """Compute realised ex-ante TE."""
        active = weights - self.benchmark_weights
        return float(np.sqrt(active @ Sigma @ active))


# ---------------------------------------------------------------------------
# Liquidity constraint
# ---------------------------------------------------------------------------

@dataclass
class LiquidityConstraint(BaseConstraint):
    """
    w_i <= liquidity_cap_i for each asset.

    Liquidity cap derived from ADV (average daily volume):
        liquidity_cap_i = participation_rate * ADV_i / portfolio_value

    Parameters
    ----------
    adv : np.ndarray  shape (N,)
        Average daily traded value per asset (in INR or USD).
    portfolio_value : float
        Total portfolio value in same currency as adv.
    participation_rate : float
        Max fraction of ADV we are willing to trade per day (e.g. 0.10).
    liquidation_days : int
        Number of days over which we assume we can liquidate.
    """
    adv: np.ndarray
    portfolio_value: float
    participation_rate: float = 0.10
    liquidation_days: int = 5

    def __post_init__(self):
        # Cap = (participation_rate * ADV * liquidation_days) / portfolio_value
        self._caps = (
            self.participation_rate * self.adv * self.liquidation_days
        ) / self.portfolio_value
        # Never let cap exceed 1.0
        self._caps = np.minimum(self._caps, 1.0)

    def build(self, w, *, Sigma=None, mu=None, extra=None):
        return [w <= self._caps]

    @property
    def caps(self) -> np.ndarray:
        return self._caps


# ---------------------------------------------------------------------------
# Constraint builder — orchestrates all constraints
# ---------------------------------------------------------------------------

class ConstraintBuilder:
    """
    Assembles a list of constraints for the optimizer.

    Usage
    -----
    builder = ConstraintBuilder(n_assets=10, tickers=tickers)
    builder.add(LongOnlyConstraint())
    builder.add(FullInvestmentConstraint())
    builder.add(MaxWeightConstraint(0.10))
    builder.add(SectorCapConstraint(sector_map, tickers, cap=0.25))
    all_constraints = builder.build(w, Sigma=Sigma_np, mu=mu_np)
    """

    def __init__(self, n_assets: int, tickers: List[str]):
        self.n_assets = n_assets
        self.tickers = tickers
        self._constraints: List[BaseConstraint] = []

    def add(self, constraint: BaseConstraint) -> "ConstraintBuilder":
        """Register a constraint. Returns self for chaining."""
        self._constraints.append(constraint)
        return self

    def validate_all(self) -> None:
        """Run pre-solve validation on all constraints."""
        for c in self._constraints:
            c.validate(self.n_assets)

    def build(
        self,
        w: "cp.Variable",
        *,
        Sigma: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        extra: Optional[Dict] = None,
    ) -> List:
        """
        Compile all constraints into a flat cvxpy list.

        Parameters
        ----------
        w : cp.Variable
        Sigma : np.ndarray, optional  — pass if any constraint needs it
        mu : np.ndarray, optional
        extra : dict, optional
        """
        all_constraints = []
        for c in self._constraints:
            all_constraints.extend(c.build(w, Sigma=Sigma, mu=mu, extra=extra))
        return all_constraints

    def summary(self) -> List[str]:
        """Return human-readable summary of registered constraints."""
        return [type(c).__name__ + ": " + repr(c) for c in self._constraints]


# ---------------------------------------------------------------------------
# Factory: build standard institutional constraint set
# ---------------------------------------------------------------------------

def build_institutional_constraints(
    n_assets: int,
    tickers: List[str],
    *,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
    sector_map: Optional[Dict[str, str]] = None,
    sector_cap: float = 0.25,
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 0.30,
    esg_scores: Optional[np.ndarray] = None,
    min_esg_score: float = 50.0,
    benchmark_weights: Optional[np.ndarray] = None,
    max_tracking_error: float = 0.05,
    adv: Optional[np.ndarray] = None,
    portfolio_value: float = 1_000_000.0,
    liquidity_participation: float = 0.10,
) -> ConstraintBuilder:
    """
    Factory that wires up the full institutional constraint set.

    Only adds constraints for which sufficient data is provided.
    Always adds LongOnly + FullInvestment + MaxWeight.

    Returns
    -------
    ConstraintBuilder
        Ready to call .build(w, Sigma=...) on.
    """
    builder = ConstraintBuilder(n_assets, tickers)

    # Always-on
    builder.add(LongOnlyConstraint())
    builder.add(FullInvestmentConstraint())
    builder.add(MaxWeightConstraint(max_weight))

    if min_weight > 0:
        builder.add(MinWeightConstraint(min_weight))

    if sector_map is not None:
        builder.add(SectorCapConstraint(sector_map, tickers, sector_cap=sector_cap))

    if prev_weights is not None:
        builder.add(TurnoverConstraint(prev_weights, max_turnover))

    if esg_scores is not None:
        builder.add(ESGConstraint(esg_scores, min_esg_score))

    if benchmark_weights is not None:
        builder.add(TrackingErrorConstraint(benchmark_weights, max_tracking_error))

    if adv is not None:
        builder.add(LiquidityConstraint(adv, portfolio_value, liquidity_participation))

    builder.validate_all()
    return builder
