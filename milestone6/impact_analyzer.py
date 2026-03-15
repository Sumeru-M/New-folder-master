"""
impact_analyzer.py
==================
Impact Simulation Engine for Milestone 6 — Virtual Trade Simulation.

Computes every analytically-meaningful delta between the real portfolio
and the virtual (post-trade) portfolio:

    Metric                   Real portfolio    Virtual portfolio
    Expected return (ann.)   r_real            r_virt
    Volatility (ann.)        sigma_real        sigma_virt
    Sharpe ratio             S_real            S_virt
    CVaR 95% (daily)         CVaR_real         CVaR_virt
    Diversification ratio    DR_real           DR_virt
    HHI / Effective-N        HHI_real          HHI_virt
    Factor exposures         beta_real         beta_virt

All analytics are computed entirely from numpy/scipy.
No M3-M5 imports — module remains fully isolated.

Public API
----------
    ImpactAnalyzer
        .analyze(virtual_portfolio, real_weights, real_daily_returns,
                 real_expected_returns, real_covariance,
                 real_total_value, risk_free_rate)
          -> ImpactReport
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from milestone6.virtual_trade_engine import VirtualPortfolio

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Portfolio metric helpers
# ---------------------------------------------------------------------------

def _portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def _portfolio_vol(weights: np.ndarray, sigma: np.ndarray) -> float:
    v = float(weights @ sigma @ weights)
    return float(np.sqrt(max(v, 0.0)))


def _sharpe(ret: float, vol: float, rf: float) -> float:
    return (ret - rf) / vol if vol > 1e-12 else 0.0


def _cvar_historical(
    weights: np.ndarray,
    daily_returns: pd.DataFrame,
    confidence: float = 0.95,
    weight_index: list = None,
) -> float:
    """Historical CVaR (Expected Shortfall). Returns positive loss magnitude."""
    ret_tickers = daily_returns.columns.tolist()

    # Build a named series from weights so we can align by ticker name
    if weight_index is not None and len(weight_index) == len(weights):
        w_named = pd.Series(weights, index=weight_index)
    else:
        # Fallback: assume weights already match daily_returns columns 1-to-1
        w_named = pd.Series(weights, index=ret_tickers[:len(weights)])

    # Align to the returns universe (zero-fill any ticker not in weights)
    aligned_w = np.array([float(w_named.get(t, 0.0)) for t in ret_tickers])

    port_ret  = daily_returns.values @ aligned_w
    threshold = np.percentile(port_ret, (1 - confidence) * 100)
    tail      = port_ret[port_ret <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-np.mean(tail))


def _diversification_ratio(weights: np.ndarray, sigma: np.ndarray) -> float:
    """DR = (w . sigma_individual) / sigma_portfolio."""
    individual_vols = np.sqrt(np.diag(sigma))
    weighted_vols   = float(weights @ individual_vols)
    port_vol        = _portfolio_vol(weights, sigma)
    if port_vol < 1e-12 or weighted_vols < 1e-12:
        return 1.0
    return weighted_vols / port_vol


def _hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index = sum(w_i^2)."""
    return float(np.sum(weights ** 2))


def _compute_factor_exposures(
    weights: np.ndarray,
    daily_returns: pd.DataFrame,
    tickers: List[str],
) -> Dict[str, float]:
    """
    Single-factor CAPM decomposition aggregated to portfolio level.
    Factor proxy = equal-weight market portfolio of all assets.
    """
    n = len(tickers)
    if n < 2 or daily_returns.empty:
        return {
            "market_beta": 1.0, "systematic_variance": 0.0,
            "idiosyncratic_variance": 0.0, "r_squared": 0.0,
            "tracking_error_vs_ew": 0.0,
        }

    aligned = pd.DataFrame(
        index=daily_returns.index,
        data={t: daily_returns[t].values
              if t in daily_returns.columns
              else np.zeros(len(daily_returns))
              for t in tickers}
    )

    mkt_returns = aligned.mean(axis=1).values

    betas, r2s = [], []
    for t in tickers:
        asset_r = aligned[t].values
        cov_am  = np.cov(asset_r, mkt_returns)
        var_mkt = cov_am[1, 1]
        if var_mkt < 1e-12:
            beta, r2 = 1.0, 0.0
        else:
            beta = cov_am[0, 1] / var_mkt
            r2   = (cov_am[0, 1] ** 2) / (var_mkt * max(cov_am[0, 0], 1e-12))
        betas.append(beta)
        r2s.append(r2)

    betas = np.array(betas)
    r2s   = np.array(r2s)

    port_beta   = float(weights @ betas)
    port_r2     = float(weights @ r2s)
    mkt_var_ann = float(np.var(mkt_returns) * 252)
    syst_var    = (port_beta ** 2) * mkt_var_ann
    total_var   = float(_portfolio_vol(weights, aligned.cov().values * 252) ** 2)
    idio_var    = max(total_var - syst_var, 0.0)

    ew_weights   = np.ones(n) / n
    diff_weights = weights - ew_weights
    te_var       = float(diff_weights @ (aligned.cov().values * 252) @ diff_weights)
    te           = float(np.sqrt(max(te_var, 0.0)))

    return {
        "market_beta":              round(port_beta, 4),
        "systematic_variance":      round(syst_var,  6),
        "idiosyncratic_variance":   round(idio_var,  6),
        "r_squared":                round(port_r2,   4),
        "tracking_error_vs_ew":     round(te,        4),
    }


def _concentration_metrics(weights: np.ndarray, n: int) -> Dict[str, float]:
    """HHI, effective-N, Shannon entropy, Gini, normalised entropy.

    Gini formula (standard, sort-based):
        G = (2 * sum(i * w_i) / (n * sum(w))) - (n+1)/n
    where w is sorted in ascending order and i is 1-indexed rank.
    This produces G=0 for equal weights and G→1 for full concentration.
    """
    w      = np.clip(weights, 1e-10, 1.0)
    hhi    = float(np.sum(w ** 2))
    eff_n  = 1.0 / hhi if hhi > 0 else float(n)
    entr   = float(-np.sum(w * np.log(w)))
    norm_e = entr / np.log(n) if n > 1 else 0.0

    # Correct Gini: ascending sort, 1-indexed ranks
    w_sorted = np.sort(w)          # ascending
    i_ranks  = np.arange(1, n + 1)
    total    = np.sum(w_sorted)
    gini     = (2.0 * float(np.sum(i_ranks * w_sorted)) / (n * total)) - (n + 1) / n if total > 0 else 0.0

    return {
        "herfindahl_index":   round(hhi,             4),
        "effective_n":        round(eff_n,            2),
        "entropy":            round(entr,             4),
        "normalised_entropy": round(norm_e,           4),
        "gini_coefficient":   round(max(gini, 0.0),  4),
    }


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMetrics:
    """Full set of analytics for one portfolio state (real or virtual)."""
    expected_return:       float
    volatility:            float
    sharpe_ratio:          float
    cvar_95:               float
    diversification_ratio: float
    hhi:                   float
    effective_n:           float
    concentration:         Dict[str, float]
    factor_exposures:      Dict[str, float]
    weights:               Dict[str, float]
    tickers:               List[str]
    label:                 str = "portfolio"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":                 self.label,
            "expected_return":       round(self.expected_return, 6),
            "volatility":            round(self.volatility, 6),
            "sharpe_ratio":          round(self.sharpe_ratio, 4),
            "cvar_95":               round(self.cvar_95, 6),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "hhi":                   round(self.hhi, 4),
            "effective_n":           round(self.effective_n, 2),
            "concentration":         self.concentration,
            "factor_exposures":      self.factor_exposures,
            "weights":               {t: round(float(w), 6) for t, w in self.weights.items()},
        }


@dataclass
class ImpactReport:
    """Complete impact analysis: before (real) vs after (virtual) metrics + deltas."""
    real_metrics:     PortfolioMetrics
    virtual_metrics:  PortfolioMetrics
    portfolio_impact: Dict[str, float]
    factor_shift:     Dict[str, float]
    risk_summary:     str
    trade_ticker:     str
    trade_value_inr:  float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_ticker":      self.trade_ticker,
            "trade_value_inr":   round(self.trade_value_inr, 2),
            "real_portfolio":    self.real_metrics.to_dict(),
            "virtual_portfolio": self.virtual_metrics.to_dict(),
            "portfolio_impact":  {k: round(v, 6) for k, v in self.portfolio_impact.items()},
            "factor_shift":      {k: round(v, 6) for k, v in self.factor_shift.items()},
            "risk_summary":      self.risk_summary,
        }


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class ImpactAnalyzer:
    """
    Computes the full analytical impact of a virtual trade.

    Usage
    -----
    analyzer = ImpactAnalyzer()
    report   = analyzer.analyze(virtual_portfolio, real_*)
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def analyze(
        self,
        vp:                    VirtualPortfolio,
        real_weights:          pd.Series,
        real_daily_returns:    pd.DataFrame,
        real_expected_returns: pd.Series,
        real_covariance:       pd.DataFrame,
        real_total_value:      float,
        risk_free_rate:        float = 0.07,
    ) -> ImpactReport:
        """Run full impact analysis."""
        rf = risk_free_rate

        real_m = self._compute_metrics(
            weights          = real_weights,
            expected_returns = real_expected_returns,
            covariance       = real_covariance,
            daily_returns    = real_daily_returns,
            rf               = rf,
            label            = "Real Portfolio",
        )

        virt_m = self._compute_metrics(
            weights          = vp.weights,
            expected_returns = vp.expected_returns,
            covariance       = vp.covariance,
            daily_returns    = vp.daily_returns,
            rf               = rf,
            label            = "Virtual Portfolio",
        )

        portfolio_impact = {
            "expected_return_change": virt_m.expected_return  - real_m.expected_return,
            "volatility_change":      virt_m.volatility       - real_m.volatility,
            "sharpe_change":          virt_m.sharpe_ratio     - real_m.sharpe_ratio,
            "cvar_change":            virt_m.cvar_95          - real_m.cvar_95,
            "diversification_change": virt_m.diversification_ratio - real_m.diversification_ratio,
            "hhi_change":             virt_m.hhi              - real_m.hhi,
            "effective_n_change":     virt_m.effective_n      - real_m.effective_n,
        }

        factor_shift = {
            k: round(float(virt_m.factor_exposures.get(k, 0))
                     - float(real_m.factor_exposures.get(k, 0)), 4)
            for k in set(list(virt_m.factor_exposures.keys()) +
                         list(real_m.factor_exposures.keys()))
        }

        risk_summary = self._generate_risk_summary(
            trade_ticker = vp.trade.ticker,
            trade_value  = vp.trade.trade_value,
            delta        = portfolio_impact,
            virt_metrics = virt_m,
            new_ticker   = vp.new_ticker,
        )

        return ImpactReport(
            real_metrics     = real_m,
            virtual_metrics  = virt_m,
            portfolio_impact = portfolio_impact,
            factor_shift     = factor_shift,
            risk_summary     = risk_summary,
            trade_ticker     = vp.trade.ticker,
            trade_value_inr  = vp.trade.trade_value,
        )

    def _compute_metrics(
        self,
        weights:          pd.Series,
        expected_returns: pd.Series,
        covariance:       pd.DataFrame,
        daily_returns:    pd.DataFrame,
        rf:               float,
        label:            str,
    ) -> PortfolioMetrics:
        tickers = weights.index.tolist()
        w       = weights.values.astype(float)
        w       = np.clip(w, 0, 1)
        if w.sum() > 0:
            w /= w.sum()

        mu    = np.array([float(expected_returns.get(t, 0.0)) for t in tickers])
        sigma = np.zeros((len(tickers), len(tickers)))
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                v = 0.0
                if ti in covariance.index and tj in covariance.columns:
                    v = float(covariance.loc[ti, tj])
                sigma[i, j] = v
        sigma    = (sigma + sigma.T) / 2
        eigvals  = np.linalg.eigvalsh(sigma)
        if eigvals.min() < 1e-8:
            sigma += (abs(eigvals.min()) + 1e-8) * np.eye(len(tickers))

        ret  = _portfolio_return(w, mu)
        vol  = _portfolio_vol(w, sigma)
        shr  = _sharpe(ret, vol, rf)
        cvar = _cvar_historical(w, daily_returns, self.confidence, weight_index=tickers)
        dr   = _diversification_ratio(w, sigma)
        hhi  = _hhi(w)
        conc = _concentration_metrics(w, len(tickers))
        fe   = _compute_factor_exposures(w, daily_returns, tickers)

        return PortfolioMetrics(
            expected_return       = ret,
            volatility            = vol,
            sharpe_ratio          = shr,
            cvar_95               = cvar,
            diversification_ratio = dr,
            hhi                   = hhi,
            effective_n           = conc["effective_n"],
            concentration         = conc,
            factor_exposures      = fe,
            weights               = dict(zip(tickers, w.tolist())),
            tickers               = tickers,
            label                 = label,
        )

    @staticmethod
    def _generate_risk_summary(
        trade_ticker: str,
        trade_value:  float,
        delta:        Dict[str, float],
        virt_metrics: PortfolioMetrics,
        new_ticker:   bool,
    ) -> str:
        lines = [
            f"Virtual trade: {trade_ticker}  |  Rs.{trade_value:,.0f}",
            ("New position -- adds a ticker not currently in portfolio."
             if new_ticker else "Increases existing position."),
            "",
        ]

        dr = delta["expected_return_change"]
        if abs(dr) < 1e-4:
            lines.append("--  Expected return: negligible change.")
        elif dr > 0:
            lines.append(f"UP  Expected return: +{dr:.2%} p.a. (positive).")
        else:
            lines.append(f"DN  Expected return: {dr:.2%} p.a. (negative).")

        dv = delta["volatility_change"]
        if abs(dv) < 1e-4:
            lines.append("--  Volatility: negligible change.")
        elif dv > 0:
            lines.append(f"UP  Volatility: +{dv:.2%} p.a. -- portfolio risk increases.")
        else:
            lines.append(f"DN  Volatility: {dv:.2%} p.a. -- portfolio risk decreases.")

        ds = delta["sharpe_change"]
        if abs(ds) < 0.01:
            lines.append("--  Sharpe ratio: negligible change.")
        elif ds > 0:
            lines.append(f"UP  Sharpe: +{ds:.2f} -- trade improves risk-adjusted returns.")
        else:
            lines.append(f"DN  Sharpe: {ds:.2f} -- trade reduces risk-adjusted efficiency.")

        dc = delta["cvar_change"]
        if abs(dc) < 1e-4:
            lines.append("--  Tail risk (CVaR 95%): negligible change.")
        elif dc > 0:
            lines.append(f"UP  CVaR 95%: +{dc:.2%} daily -- expected tail loss increases.")
        else:
            lines.append(f"DN  CVaR 95%: {dc:.2%} daily -- expected tail loss decreases.")

        dd = delta["diversification_change"]
        if abs(dd) < 0.01:
            lines.append("--  Diversification ratio: negligible change.")
        elif dd > 0:
            lines.append(f"UP  Diversification ratio: +{dd:.3f} -- portfolio becomes more diversified.")
        else:
            lines.append(f"DN  Diversification ratio: {dd:.3f} -- concentration increases.")

        pos = sum(1 for k in ("expected_return_change", "sharpe_change", "diversification_change")
                  if delta[k] > 0.001)
        neg = sum(1 for k in ("volatility_change", "cvar_change")
                  if delta[k] > 0.001)

        lines.append("")
        if pos >= 2 and neg == 0:
            verdict = "BENEFICIAL -- improves return, Sharpe, and/or diversification without increasing risk."
        elif pos >= 1 and neg <= 1:
            verdict = "MIXED -- some improvement in returns/diversification with modest risk increase. Consider position sizing."
        elif neg >= 2 and pos == 0:
            verdict = "CAUTION -- increases volatility and tail risk with no offsetting return improvement."
        else:
            verdict = "NEUTRAL -- no significant change to portfolio risk-return profile."

        lines.append(f"Verdict: {verdict}")
        lines.append(f"Post-trade Sharpe: {virt_metrics.sharpe_ratio:.2f}  |  "
                     f"Volatility: {virt_metrics.volatility:.2%}  |  "
                     f"Effective-N: {virt_metrics.effective_n:.1f}")

        return "\n".join(lines)