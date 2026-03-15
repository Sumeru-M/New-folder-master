"""
projection_engine.py
====================
Future Projection Engine for Milestone 6 — Virtual Trade Simulation.

Responsibility
--------------
Run Geometric Brownian Motion Monte Carlo simulation (≥ 1,000 paths) for
1-year, 3-year, and 5-year horizons on both the real portfolio and the
virtual portfolio, then compare:

    • Expected portfolio value at each horizon
    • 5th percentile (severe downside) value
    • 95th percentile (upside) value
    • Downside probability: P(final value < initial value)
    • Shortfall probability: P(final value < 0.8 × initial value)  (−20%)
    • Median CAGR
    • Value-at-Risk over the full horizon

Monte Carlo model
-----------------
Each simulation path follows multivariate GBM:

    dS_i = μ_i · S_i · dt + σ_i · S_i · dW_i

where dW ~ N(0, Σ · dt) is drawn from the asset covariance structure via
Cholesky decomposition.

At each daily step:

    r_t ~ N(μ_daily · 1 − 0.5 · diag(Σ_daily), Σ_daily)   (Itô correction)
    V_t = V_{t-1} · exp(w · r_t)

The portfolio value evolves as:
    V_t = V_0 · exp(Σ_s [ w · r_s ] )

This is exact for a constant-weight portfolio rebalanced daily (a standard
assumption for analytical tractability).

Public API
----------
    ProjectionResult     dataclass — per-horizon simulation output
    MonteCarloReport     dataclass — full comparison (real vs virtual)

    ProjectionEngine
        .run(vp, real_weights, real_expected_returns, real_covariance,
             real_total_value, risk_free_rate)
          -> MonteCarloReport
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from milestone6.virtual_trade_engine import VirtualPortfolio

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Trading days per year
_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Single-portfolio Monte Carlo
# ---------------------------------------------------------------------------

def _run_mc(
    weights:          np.ndarray,
    mu_annual:        np.ndarray,
    sigma_annual:     np.ndarray,
    initial_value:    float,
    horizon_years:    int,
    n_paths:          int,
    seed:             int,
) -> np.ndarray:
    """
    Run Monte Carlo for a single portfolio.

    Parameters
    ----------
    weights        : (N,) normalised weight vector.
    mu_annual      : (N,) annualised arithmetic expected return per asset.
    sigma_annual   : (N, N) annualised covariance matrix.
    initial_value  : Starting portfolio value (INR).
    horizon_years  : Simulation horizon in years.
    n_paths        : Number of simulation paths.
    seed           : NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray shape (n_paths,)
        Simulated terminal portfolio values.
    """
    rng        = np.random.default_rng(seed)
    T          = horizon_years * _TRADING_DAYS
    N          = len(weights)
    dt         = 1.0 / _TRADING_DAYS

    # Daily parameters
    mu_daily   = mu_annual / _TRADING_DAYS
    sigma_d    = sigma_annual / _TRADING_DAYS    # daily covariance

    # Itô correction: drift − 0.5 × diag(Σ_daily)
    drift_d    = mu_daily - 0.5 * np.diag(sigma_d)

    # Cholesky decomposition for correlated returns
    try:
        # Ensure PSD
        sigma_reg = sigma_d.copy()
        eigvals   = np.linalg.eigvalsh(sigma_reg)
        if eigvals.min() < 1e-12:
            sigma_reg += (abs(eigvals.min()) + 1e-12) * np.eye(N)
        L = np.linalg.cholesky(sigma_reg)
    except np.linalg.LinAlgError:
        # Fallback: diagonal (uncorrelated) covariance
        L = np.diag(np.sqrt(np.diag(sigma_d)))

    # Simulate: cumulative log-return for each path
    # Shape: (T, N, n_paths) is memory-heavy; use (n_paths, T) via portfolio returns
    # Portfolio log-return each day = w · r_t  where r_t is the N-vector of asset returns

    # Draw all standard normals at once: shape (T, n_paths, N)
    z          = rng.standard_normal((T, n_paths, N))
    # Correlated shocks: L @ z[t, p, :].T → shape (N,)
    # Vectorised: (T, n_paths, N) @ L.T → (T, n_paths, N)
    corr_z     = z @ L.T                             # (T, n_paths, N)

    # Asset log-returns each step: drift + shock
    asset_r    = drift_d[None, None, :] + corr_z     # (T, n_paths, N)

    # Portfolio log-return each step: weighted sum over N assets
    port_r     = (asset_r * weights[None, None, :]).sum(axis=2)  # (T, n_paths)

    # Cumulative log-return → terminal value
    cum_log_r  = port_r.sum(axis=0)                  # (n_paths,)
    return initial_value * np.exp(cum_log_r)


# ---------------------------------------------------------------------------
# Projection summary
# ---------------------------------------------------------------------------

def _summarise(
    terminal_values: np.ndarray,
    initial_value:   float,
    horizon_years:   int,
    risk_free_rate:  float,
) -> "ProjectionResult":
    """Reduce a (n_paths,) terminal value array to a ProjectionResult."""
    n_paths = len(terminal_values)

    expected_value  = float(np.mean(terminal_values))
    median_value    = float(np.median(terminal_values))
    p5_value        = float(np.percentile(terminal_values, 5))
    p25_value       = float(np.percentile(terminal_values, 25))
    p75_value       = float(np.percentile(terminal_values, 75))
    p95_value       = float(np.percentile(terminal_values, 95))

    downside_prob   = float(np.mean(terminal_values < initial_value))
    shortfall_prob  = float(np.mean(terminal_values < 0.8 * initial_value))

    # Median CAGR
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            initial_value > 0,
            terminal_values / initial_value,
            1.0
        )
        cagr_per_path = np.where(
            ratio > 0,
            np.power(np.maximum(ratio, 1e-10), 1.0 / horizon_years) - 1,
            -1.0
        )
    median_cagr = float(np.median(cagr_per_path))

    # Horizon VaR (95% confidence): loss such that 5% of paths end below
    var_95 = max(0.0, initial_value - p5_value)

    # Annualised Sharpe of terminal distribution
    annual_ret = float(np.mean(cagr_per_path))
    annual_std = float(np.std(cagr_per_path))
    dist_sharpe = (annual_ret - risk_free_rate) / annual_std if annual_std > 1e-8 else 0.0

    return ProjectionResult(
        horizon_years    = horizon_years,
        n_paths          = n_paths,
        initial_value    = round(initial_value, 2),
        expected_value   = round(expected_value,  2),
        median_value     = round(median_value,    2),
        p5_value         = round(p5_value,        2),
        p25_value        = round(p25_value,        2),
        p75_value        = round(p75_value,        2),
        p95_value        = round(p95_value,        2),
        downside_prob    = round(downside_prob,    4),
        shortfall_prob   = round(shortfall_prob,   4),
        median_cagr      = round(median_cagr,      4),
        var_95_inr       = round(var_95,           2),
        dist_sharpe      = round(dist_sharpe,      4),
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    """
    Monte Carlo projection for a single portfolio at a single horizon.

    Attributes
    ----------
    horizon_years    : Projection horizon (1, 3, or 5).
    n_paths          : Number of simulated paths.
    initial_value    : Portfolio value at t=0 (INR).
    expected_value   : Mean terminal value across paths (INR).
    median_value     : Median terminal value (INR).
    p5_value         : 5th percentile terminal value — severe downside (INR).
    p25_value        : 25th percentile terminal value (INR).
    p75_value        : 75th percentile terminal value (INR).
    p95_value        : 95th percentile terminal value — strong upside (INR).
    downside_prob    : P(terminal value < initial value).
    shortfall_prob   : P(terminal value < 0.8 × initial value).
    median_cagr      : Median compound annual growth rate across paths.
    var_95_inr       : 95% VaR over full horizon (loss at 5th percentile).
    dist_sharpe      : Sharpe ratio of the terminal-value distribution.
    """
    horizon_years:  int
    n_paths:        int
    initial_value:  float
    expected_value: float
    median_value:   float
    p5_value:       float
    p25_value:      float
    p75_value:      float
    p95_value:      float
    downside_prob:  float
    shortfall_prob: float
    median_cagr:    float
    var_95_inr:     float
    dist_sharpe:    float

    def to_dict(self) -> Dict[str, Any]:
        gain_pct = (self.expected_value / self.initial_value - 1) if self.initial_value > 0 else 0.0
        return {
            "horizon_years":   self.horizon_years,
            "n_paths":         self.n_paths,
            "initial_value":   self.initial_value,
            "expected_value":  self.expected_value,
            "median_value":    self.median_value,
            "p5_value":        self.p5_value,
            "p25_value":       self.p25_value,
            "p75_value":       self.p75_value,
            "p95_value":       self.p95_value,
            "expected_gain_pct": round(gain_pct, 4),
            "downside_prob":   self.downside_prob,
            "shortfall_prob":  self.shortfall_prob,
            "median_cagr":     self.median_cagr,
            "var_95_inr":      self.var_95_inr,
            "dist_sharpe":     self.dist_sharpe,
        }


@dataclass
class HorizonComparison:
    """
    Side-by-side comparison of real vs virtual at one horizon.
    """
    horizon_years:        int
    real:                 ProjectionResult
    virtual:              ProjectionResult

    @property
    def expected_value_delta(self) -> float:
        return self.virtual.expected_value - self.real.expected_value

    @property
    def downside_prob_delta(self) -> float:
        return self.virtual.downside_prob - self.real.downside_prob

    @property
    def p5_delta(self) -> float:
        return self.virtual.p5_value - self.real.p5_value

    @property
    def median_cagr_delta(self) -> float:
        return self.virtual.median_cagr - self.real.median_cagr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_years":          self.horizon_years,
            "real":                   self.real.to_dict(),
            "virtual":                self.virtual.to_dict(),
            "delta_expected_value":   round(self.expected_value_delta, 2),
            "delta_downside_prob":    round(self.downside_prob_delta, 4),
            "delta_p5":               round(self.p5_delta, 2),
            "delta_median_cagr":      round(self.median_cagr_delta, 4),
        }


@dataclass
class MonteCarloReport:
    """
    Full Monte Carlo comparison across all horizons.

    Attributes
    ----------
    horizons         : List of HorizonComparison (1Y, 3Y, 5Y).
    best_horizon     : Year label where virtual outperforms real most.
    overall_verdict  : Plain-English overall assessment.
    trade_ticker     : Traded ticker.
    n_paths          : Paths used.
    """
    horizons:         List[HorizonComparison]
    best_horizon:     int
    overall_verdict:  str
    trade_ticker:     str
    n_paths:          int

    def to_dict(self) -> Dict[str, Any]:
        horizons_dict = {
            f"{h.horizon_years}Y": h.to_dict()
            for h in self.horizons
        }
        return {
            "n_paths":         self.n_paths,
            "trade_ticker":    self.trade_ticker,
            "horizons":        horizons_dict,
            "best_horizon":    f"{self.best_horizon}Y",
            "overall_verdict": self.overall_verdict,
        }


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class ProjectionEngine:
    """
    Runs Monte Carlo projections for real and virtual portfolios and
    returns a structured MonteCarloReport.

    Usage
    -----
    engine = ProjectionEngine(n_paths=1_000)
    report = engine.run(vp, real_weights, real_mu, real_sigma,
                        real_total_value, risk_free_rate=0.07)
    """

    HORIZONS = [1, 3, 5]   # years

    def __init__(
        self,
        n_paths:   int = 1_000,
        base_seed: int = 2024,
    ):
        """
        Parameters
        ----------
        n_paths    : Number of Monte Carlo paths per run (minimum 1,000).
        base_seed  : Base random seed; each run/horizon gets an offset.
        """
        self.n_paths   = max(1_000, n_paths)
        self.base_seed = base_seed

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        vp:                    VirtualPortfolio,
        real_weights:          pd.Series,
        real_expected_returns: pd.Series,
        real_covariance:       pd.DataFrame,
        real_total_value:      float,
        risk_free_rate:        float = 0.07,
    ) -> MonteCarloReport:
        """
        Run full Monte Carlo comparison.

        Parameters
        ----------
        vp                    : VirtualPortfolio from virtual_trade_engine.
        real_weights          : pd.Series — real portfolio weights.
        real_expected_returns : pd.Series — annualised expected returns (real universe).
        real_covariance       : pd.DataFrame — annualised covariance (real universe).
        real_total_value      : float — real portfolio value (INR).
        risk_free_rate        : float — annual risk-free rate.

        Returns
        -------
        MonteCarloReport
        """
        rf = risk_free_rate

        # ── Build aligned arrays for real portfolio ───────────────────────
        real_w, real_mu, real_sigma = self._align_real(
            real_weights, real_expected_returns, real_covariance
        )

        # ── Build aligned arrays for virtual portfolio ────────────────────
        virt_w, virt_mu, virt_sigma = self._align_virtual(vp)

        # ── Run simulations for each horizon ──────────────────────────────
        horizon_comparisons = []
        for h_idx, years in enumerate(self.HORIZONS):
            seed_r = self.base_seed + h_idx * 1000
            seed_v = self.base_seed + h_idx * 1000 + 500

            tv_real = _run_mc(
                weights       = real_w,
                mu_annual     = real_mu,
                sigma_annual  = real_sigma,
                initial_value = real_total_value,
                horizon_years = years,
                n_paths       = self.n_paths,
                seed          = seed_r,
            )

            tv_virt = _run_mc(
                weights       = virt_w,
                mu_annual     = virt_mu,
                sigma_annual  = virt_sigma,
                initial_value = vp.total_value,
                horizon_years = years,
                n_paths       = self.n_paths,
                seed          = seed_v,
            )

            real_proj = _summarise(tv_real, real_total_value,  years, rf)
            virt_proj = _summarise(tv_virt, vp.total_value,    years, rf)

            horizon_comparisons.append(
                HorizonComparison(horizon_years=years,
                                  real=real_proj, virtual=virt_proj)
            )

        # ── Best horizon (where virtual expected value gain is largest) ───
        gains = [
            (h.expected_value_delta, h.horizon_years)
            for h in horizon_comparisons
        ]
        best_horizon = max(gains, key=lambda x: x[0])[1]

        # ── Overall verdict ───────────────────────────────────────────────
        verdict = self._verdict(horizon_comparisons, rf)

        return MonteCarloReport(
            horizons        = horizon_comparisons,
            best_horizon    = best_horizon,
            overall_verdict = verdict,
            trade_ticker    = vp.trade.ticker,
            n_paths         = self.n_paths,
        )

    # ------------------------------------------------------------------
    # Internal alignment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_real(
        weights:  pd.Series,
        mu:       pd.Series,
        sigma:    pd.DataFrame,
    ):
        """Return (w, mu, sigma) as numpy arrays aligned to weights.index."""
        tickers = weights.index.tolist()
        N       = len(tickers)
        w       = weights.values.astype(float)
        w       = np.clip(w, 0, 1)
        if w.sum() > 0:
            w /= w.sum()

        mu_arr = np.array([float(mu.get(t, 0.0)) for t in tickers])

        sig    = np.zeros((N, N))
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if ti in sigma.index and tj in sigma.columns:
                    sig[i, j] = float(sigma.loc[ti, tj])
        sig = (sig + sig.T) / 2
        eigv = np.linalg.eigvalsh(sig)
        if eigv.min() < 1e-8:
            sig += (abs(eigv.min()) + 1e-8) * np.eye(N)

        return w, mu_arr, sig

    @staticmethod
    def _align_virtual(vp: VirtualPortfolio):
        """Extract (w, mu, sigma) from VirtualPortfolio as numpy arrays."""
        tickers = vp.tickers
        N       = len(tickers)
        w       = vp.weights.values.astype(float)
        w       = np.clip(w, 0, 1)
        if w.sum() > 0:
            w /= w.sum()

        mu_arr = vp.expected_returns.values.astype(float)

        sig    = vp.covariance.values.astype(float)
        sig    = (sig + sig.T) / 2
        eigv   = np.linalg.eigvalsh(sig)
        if eigv.min() < 1e-8:
            sig += (abs(eigv.min()) + 1e-8) * np.eye(N)

        return w, mu_arr, sig

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict(
        comparisons: List[HorizonComparison],
        rf:          float,
    ) -> str:
        """
        Generate a plain-English overall verdict comparing real vs virtual
        projections across all horizons.
        """
        outperform_count = sum(
            1 for h in comparisons
            if h.expected_value_delta > 0
        )
        lower_downside_count = sum(
            1 for h in comparisons
            if h.downside_prob_delta < -0.005
        )
        higher_downside_count = sum(
            1 for h in comparisons
            if h.downside_prob_delta > 0.01
        )

        # Collect key numbers from 5Y horizon
        h5 = next((h for h in comparisons if h.horizon_years == 5), comparisons[-1])
        gain_pct = h5.expected_value_delta / h5.real.initial_value * 100 if h5.real.initial_value > 0 else 0.0
        cagr_d   = h5.median_cagr_delta * 100  # in percentage points

        if outperform_count == 3 and lower_downside_count >= 1:
            verdict = (
                f"POSITIVE across all horizons: the virtual trade is projected to add "
                f"~₹{abs(h5.expected_value_delta):,.0f} ({abs(gain_pct):.1f}%) to 5-year expected value "
                f"while reducing downside probability. "
                f"Median CAGR delta: {cagr_d:+.2f}pp. Recommend executing."
            )
        elif outperform_count >= 2 and higher_downside_count == 0:
            verdict = (
                f"MOSTLY POSITIVE: the virtual trade improves projected value in "
                f"{outperform_count}/3 horizons with no material increase in downside risk. "
                f"5-year expected delta: ₹{h5.expected_value_delta:+,.0f}. "
                f"Consider executing with standard position sizing."
            )
        elif outperform_count >= 2 and higher_downside_count >= 1:
            verdict = (
                f"MIXED: the trade improves expected value ({outperform_count}/3 horizons) "
                f"but increases downside probability in {higher_downside_count} horizon(s). "
                f"5-year expected delta: ₹{h5.expected_value_delta:+,.0f}. "
                f"Review position size — consider reducing quantity."
            )
        elif outperform_count == 0:
            verdict = (
                f"NEGATIVE: the virtual trade reduces projected value across all horizons. "
                f"5-year expected delta: ₹{h5.expected_value_delta:+,.0f} ({gain_pct:+.1f}%). "
                f"Downside probability change: {h5.downside_prob_delta:+.1%}. "
                f"Do not recommend executing as-is."
            )
        else:
            verdict = (
                f"NEUTRAL: marginal projected impact across horizons. "
                f"5-year expected delta: ₹{h5.expected_value_delta:+,.0f}. "
                f"Monitor after execution."
            )

        return verdict
