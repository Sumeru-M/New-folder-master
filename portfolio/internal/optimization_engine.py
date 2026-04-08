"""
optimization_engine.py — Multi-Objective Portfolio Optimization Engine

Optimization frameworks implemented:
    1. Mean-Variance (Markowitz)       — maximize μ - λσ²
    2. Minimum Variance                — minimize σ²
    3. CVaR Optimization               — minimize Expected Shortfall
    4. Risk Parity                     — equal risk contribution (iterative)
    5. Maximum Diversification         — maximize diversification ratio
    6. Multi-Objective                 — maximize return, minimize vol,
                                         minimize CVaR, penalize drawdown,
                                         penalize factor concentration

All methods accept a ConstraintBuilder and return an AllocationResult.

Design principles
-----------------
- Wraps existing optimizer.py min-variance / max-sharpe rather than replacing.
- CVaR optimization uses the Rockafellar-Uryasev LP linearisation.
- Risk Parity uses Newton's method (no LP/QP needed).
- Multi-objective uses scalarisation with configurable lambda vector.
- All solvers fall back gracefully and report solve status.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not found — MVO, CVaR, MaxDiv optimisers unavailable.")

from .constraints import ConstraintBuilder


# ---------------------------------------------------------------------------
# Result container (extends what optimizer.py's OptimizationResult carries)
# ---------------------------------------------------------------------------

@dataclass
class AllocationResult:
    """
    Full output of any optimization run.
    Contains everything needed by the allocation intelligence layer.
    """
    weights: pd.Series                          # optimal weights
    expected_return: float                      # annualised
    volatility: float                           # annualised
    sharpe_ratio: float                         # (μ - rf) / σ
    cvar_95: float                              # 95% CVaR (as positive loss %)
    optimization_type: str                      # human label
    solve_status: str                           # "optimal" / "optimal_inaccurate" / "failed"
    risk_free_rate: float = 0.07

    # Populated by allocation intelligence layer (not optimizer)
    risk_contribution: Dict[str, float] = field(default_factory=dict)
    factor_exposure: Dict[str, float] = field(default_factory=dict)
    allocation_health_score: int = 0
    overweight_underweight_flags: Dict[str, str] = field(default_factory=dict)
    rebalance_actions_rupees: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """JSON-serialisable output dict."""
        return {
            "optimal_weights": {k: round(v, 6) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 6),
            "volatility": round(self.volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "cvar": round(self.cvar_95, 6),
            "optimization_type": self.optimization_type,
            "solve_status": self.solve_status,
            "risk_contribution": {k: round(v, 6) for k, v in self.risk_contribution.items()},
            "factor_exposure": {k: round(v, 6) for k, v in self.factor_exposure.items()},
            "allocation_health_score": self.allocation_health_score,
            "overweight_underweight_flags": self.overweight_underweight_flags,
            "rebalance_actions_rupees": self.rebalance_actions_rupees,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _portfolio_stats(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> Tuple[float, float, float, float]:
    """
    Compute (expected_return, volatility, sharpe, cvar_95) for given weights.

    cvar_95 from historical simulation if returns_history provided,
    else parametric normal approximation.
    """
    port_ret = float(w @ mu)
    port_var = float(w @ Sigma @ w)
    port_std = float(np.sqrt(max(port_var, 0)))
    sharpe = (port_ret - rf) / port_std if port_std > 0 else 0.0

    if returns_history is not None and len(returns_history) > 0:
        # Historical portfolio returns (simple returns)
        # returns_history shape: (T, N)
        port_hist = returns_history @ w
        var_threshold = np.percentile(port_hist, 5)
        tail = port_hist[port_hist <= var_threshold]
        cvar_95 = float(-tail.mean()) if len(tail) > 0 else float(-var_threshold)
    else:
        # Parametric: CVaR_95 ≈ σ × φ(z)/Φ(z) where z=1.645
        from scipy.stats import norm
        z = norm.ppf(0.05)
        cvar_95 = float(-port_std * norm.pdf(z) / 0.05)

    return port_ret, port_std, sharpe, cvar_95


def _make_result(
    w_array: np.ndarray,
    tickers: List[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    returns_history: Optional[np.ndarray],
    opt_type: str,
    status: str,
    rf: float = 0.07,
) -> AllocationResult:
    weights = pd.Series(w_array, index=tickers)
    ret, vol, sharpe, cvar = _portfolio_stats(w_array, mu, Sigma, returns_history, rf)
    return AllocationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        cvar_95=cvar,
        optimization_type=opt_type,
        solve_status=status,
        risk_free_rate=rf,
    )


# ---------------------------------------------------------------------------
# 1. Mean-Variance Optimization
# ---------------------------------------------------------------------------

def optimize_mean_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    lam: float = 1.0,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> AllocationResult:
    """
    Maximize:  μᵀw - λ × wᵀΣw

    Parameters
    ----------
    mu : np.ndarray  (N,)
        Annualised arithmetic expected returns.
    Sigma : np.ndarray  (N, N)
        Annualised covariance matrix.
    tickers : List[str]
    constraint_builder : ConstraintBuilder
    lam : float
        Risk-aversion parameter. Higher = more conservative.
        lam=0 → pure return maximisation (dangerous).
        lam=1 → balanced.
        lam→∞ → minimum variance.
    returns_history : np.ndarray (T, N), optional
        Historical simple returns for historical CVaR calculation.
    rf : float
        Risk-free rate for Sharpe.
    """
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy required for MVO.")

    N = len(tickers)
    w = cp.Variable(N, name="weights")

    objective = cp.Maximize(mu @ w - lam * cp.quad_form(w, Sigma))
    constraints = constraint_builder.build(w, Sigma=Sigma, mu=mu)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)

    status = prob.status or "failed"
    if w.value is None:
        # Fallback: equal weight
        w_val = np.ones(N) / N
        status = "failed_fallback_equal_weight"
    else:
        w_val = np.maximum(w.value, 0)
        w_val /= w_val.sum()

    return _make_result(w_val, tickers, mu, Sigma, returns_history,
                        f"Mean-Variance (λ={lam})", status, rf)


# ---------------------------------------------------------------------------
# 2. Minimum Variance
# ---------------------------------------------------------------------------

def optimize_minimum_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> AllocationResult:
    """Minimize wᵀΣw subject to constraints."""
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy required.")

    N = len(tickers)
    w = cp.Variable(N, name="weights")

    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = constraint_builder.build(w, Sigma=Sigma, mu=mu)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)

    status = prob.status or "failed"
    if w.value is None:
        w_val = np.ones(N) / N
        status = "failed_fallback_equal_weight"
    else:
        w_val = np.maximum(w.value, 0)
        w_val /= w_val.sum()

    return _make_result(w_val, tickers, mu, Sigma, returns_history,
                        "Minimum Variance", status, rf)


# ---------------------------------------------------------------------------
# 3. CVaR Optimization (Rockafellar-Uryasev linearisation)
# ---------------------------------------------------------------------------

def optimize_cvar(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    returns_history: np.ndarray,
    confidence_level: float = 0.95,
    lam_return: float = 0.0,
    rf: float = 0.07,
) -> AllocationResult:
    """
    Minimize CVaR (Expected Shortfall) at given confidence level.

    Uses the Rockafellar-Uryasev (2000) LP reformulation:
        CVaR_α(w) = min_{z} { z + 1/((1-α)T) × Σ_t max(-rₜᵀw - z, 0) }

    This is a linear program in (w, z, u) where u_t = max(-rₜᵀw - z, 0).

    Parameters
    ----------
    returns_history : np.ndarray  shape (T, N)
        Historical simple returns (NOT log returns).
        Each row is one daily observation.
    confidence_level : float, default 0.95
        CVaR confidence level (95% = worst 5% of days).
    lam_return : float, default 0.0
        If > 0, adds - lam_return × μᵀw to objective (penalise low return).
        Set to a small positive value to avoid degenerate zero-return solutions.
    """
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy required.")

    T, N = returns_history.shape
    alpha = confidence_level

    w = cp.Variable(N, name="weights")
    z = cp.Variable(name="var_threshold")             # VaR level
    u = cp.Variable(T, nonneg=True, name="shortfall") # auxiliary loss vars

    # Portfolio loss on each day: -rₜᵀw
    losses = -returns_history @ w  # shape (T,)

    objective = cp.Minimize(
        z + (1.0 / ((1 - alpha) * T)) * cp.sum(u)
        - lam_return * (mu @ w)
    )

    constraints = constraint_builder.build(w, Sigma=Sigma, mu=mu)
    constraints += [
        u >= losses - z,   # u_t >= loss_t - z (lower bound on shortfall)
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)

    status = prob.status or "failed"
    if w.value is None:
        w_val = np.ones(N) / N
        status = "failed_fallback_equal_weight"
    else:
        w_val = np.maximum(w.value, 0)
        if w_val.sum() > 0:
            w_val /= w_val.sum()
        else:
            w_val = np.ones(N) / N

    return _make_result(w_val, tickers, mu, Sigma, returns_history,
                        f"CVaR Optimisation (α={confidence_level:.0%})", status, rf)


# ---------------------------------------------------------------------------
# 4. Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------

def optimize_risk_parity(
    Sigma: np.ndarray,
    tickers: List[str],
    mu: Optional[np.ndarray] = None,
    target_risk_contributions: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-10,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> AllocationResult:
    """
    Risk Parity: each asset contributes equally to portfolio risk.

    Solves: minimize Σ_{i,j} (CRC_i - CRC_j)²
    where CRC_i = w_i × (Σw)_i

    Uses the Spinu (2013) Newton algorithm for speed and stability.
    No cvxpy dependency — pure scipy/numpy.

    Parameters
    ----------
    Sigma : np.ndarray  (N, N)
    tickers : List[str]
    mu : np.ndarray (N,), optional
        Only used for reporting expected return.
    target_risk_contributions : np.ndarray (N,), optional
        Desired risk budget per asset (must sum to 1).
        Default: equal = 1/N for all assets.
    max_iter : int
        Newton iterations.
    tol : float
        Convergence tolerance on gradient norm.
    """
    N = len(tickers)

    if target_risk_contributions is None:
        b = np.ones(N) / N  # equal risk budgets
    else:
        b = np.array(target_risk_contributions, dtype=float)
        b /= b.sum()

    # Objective: sum_i (CRC_i/σ_p - b_i)²  where CRC_i = w_i(Σw)_i/σ_p
    # Equivalent to: sum_i (w_i(Σw)_i - b_i × wᵀΣw)² / (wᵀΣw)
    # Use the unconstrained form with x = w / sum(w), then normalise

    def _objective(x):
        x = np.maximum(x, 1e-8)
        Sigma_x = Sigma @ x
        port_var = x @ Sigma_x
        risk_contrib = x * Sigma_x / port_var
        diff = risk_contrib - b
        return 0.5 * np.sum(diff ** 2)

    def _gradient(x):
        x = np.maximum(x, 1e-8)
        Sigma_x = Sigma @ x
        port_var = x @ Sigma_x
        risk_contrib = x * Sigma_x / port_var

        # Gradient of CRC_i w.r.t. x_j
        # ∂CRC_i/∂x_j = (Sigma[i,j]*x_i + Sigma_x[i]*1_{i=j})/port_var
        #               - 2 x_i Sigma_x[i] (Sigma_x[j]) / port_var²
        diff = risk_contrib - b
        dCRC = np.outer(x, Sigma[np.arange(N), np.arange(N)]) / port_var  # approx
        # Full gradient (vectorised)
        Jac = (
            np.diag(Sigma_x) / port_var
            + (np.diag(x) @ Sigma) / port_var
            - 2 * np.outer(x * Sigma_x, Sigma_x) / port_var ** 2
        )
        return Jac.T @ diff

    # Initial guess: inverse-vol weighting
    vols = np.sqrt(np.diag(Sigma))
    x0 = (1.0 / vols) / np.sum(1.0 / vols)

    result = minimize(
        _objective,
        x0,
        jac=_gradient,
        method="L-BFGS-B",
        bounds=[(1e-6, 1.0)] * N,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    w_val = np.maximum(result.x, 0)
    w_val /= w_val.sum()

    status = "optimal" if result.success else "optimal_inaccurate"
    mu_use = mu if mu is not None else np.zeros(N)

    return _make_result(w_val, tickers, mu_use, Sigma, returns_history,
                        "Risk Parity", status, rf)


# ---------------------------------------------------------------------------
# 5. Maximum Diversification
# ---------------------------------------------------------------------------

def optimize_max_diversification(
    Sigma: np.ndarray,
    tickers: List[str],
    mu: Optional[np.ndarray] = None,
    constraint_builder: Optional[ConstraintBuilder] = None,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> AllocationResult:
    """
    Maximum Diversification: maximize DR = (wᵀσ) / σ_p

    Equivalent (Choueifaty & Coignard 2008) to:
        maximize (wᵀσ) / sqrt(wᵀΣw)

    Solved as a QP by substituting y = w / (wᵀσ):
        minimize yᵀΣy
        subject to: σᵀy = 1, y >= 0

    Then w = y / sum(y).

    Constraints other than long-only + full-investment require SOCP
    and are applied via cvxpy if available, else ignored with a warning.
    """
    N = len(tickers)
    sigma_i = np.sqrt(np.diag(Sigma))  # individual vols (N,)

    if CVXPY_AVAILABLE:
        # SOCP formulation (handles full constraint set)
        w = cp.Variable(N, name="weights")
        # Maximise wᵀσ / sigma_p ≡ minimise sigma_p / wᵀσ
        # Reformulate as: min wᵀΣw s.t. wᵀσ >= 1 (then normalise)
        constraints = [
            cp.sum(w) >= 0,
            w >= 0,
            sigma_i @ w >= 1.0,  # normalisation anchor
        ]
        if constraint_builder is not None:
            # Add user constraints (excluding full-investment since we renormalise)
            for c in constraint_builder._constraints:
                from constraints import FullInvestmentConstraint, LongOnlyConstraint
                if isinstance(c, (FullInvestmentConstraint, LongOnlyConstraint)):
                    continue
                constraints.extend(c.build(w, Sigma=Sigma))

        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-9, eps_rel=1e-9)

        status = prob.status or "failed"
        if w.value is None:
            w_val = (1.0 / sigma_i) / np.sum(1.0 / sigma_i)
            status = "failed_fallback_inv_vol"
        else:
            w_val = np.maximum(w.value, 0)
            w_val /= w_val.sum()
    else:
        # Scipy fallback (long-only, full-investment only)
        def _neg_dr(w):
            port_vol = np.sqrt(w @ Sigma @ w)
            return -(w @ sigma_i) / port_vol if port_vol > 0 else 0.0

        x0 = np.ones(N) / N
        result = minimize(
            _neg_dr, x0,
            method="SLSQP",
            bounds=[(0, 1)] * N,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        w_val = np.maximum(result.x, 0)
        w_val /= w_val.sum()
        status = "optimal" if result.success else "optimal_inaccurate"

    mu_use = mu if mu is not None else np.zeros(N)
    return _make_result(w_val, tickers, mu_use, Sigma, returns_history,
                        "Maximum Diversification", status, rf)


# ---------------------------------------------------------------------------
# 6. Multi-Objective Optimization
# ---------------------------------------------------------------------------

def optimize_multi_objective(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    returns_history: Optional[np.ndarray] = None,
    # Lambda weights for each objective (all non-negative, sum need not be 1)
    lam_return: float = 1.0,       # reward expected return
    lam_vol: float = 1.0,          # penalise volatility
    lam_cvar: float = 0.5,         # penalise CVaR
    lam_drawdown: float = 0.3,     # penalise historical max drawdown contribution
    lam_factor_conc: float = 0.2,  # penalise factor concentration (if betas provided)
    factor_betas: Optional[np.ndarray] = None,   # (N, K)
    factor_target: Optional[np.ndarray] = None,  # (K,) desired factor exposure
    confidence_level: float = 0.95,
    rf: float = 0.07,
) -> AllocationResult:
    """
    Multi-objective scalarised optimisation.

    Objective (maximise):
        lam_return  × μᵀw
      - lam_vol     × wᵀΣw
      - lam_cvar    × CVaR_α(w)           [LP linearised]
      - lam_drawdown× avg_drawdown(w)      [historical approximation]
      - lam_factor_conc × ||Bᵀw - target||² [factor concentration]

    CVaR term requires returns_history.
    Factor concentration term requires factor_betas.
    drawdown_penalty is the average of historical drawdown on the weight vector.

    Parameters
    ----------
    lam_return, lam_vol, lam_cvar, lam_drawdown, lam_factor_conc : float
        Scalarisation weights. Set any to 0 to exclude that objective.
    factor_betas : np.ndarray (N, K), optional
        Asset factor loadings. Required for factor concentration penalty.
    factor_target : np.ndarray (K,), optional
        Desired portfolio factor exposure. Defaults to zeros (neutral).
    """
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy required for multi-objective optimisation.")

    N = len(tickers)
    w = cp.Variable(N, name="weights")

    objective_terms = []

    # 1. Return maximisation
    if lam_return > 0:
        objective_terms.append(lam_return * (mu @ w))

    # 2. Variance penalty
    if lam_vol > 0:
        objective_terms.append(-lam_vol * cp.quad_form(w, Sigma))

    # 3. CVaR penalty (Rockafellar-Uryasev)
    extra_constraints = []
    if lam_cvar > 0 and returns_history is not None:
        T = len(returns_history)
        z = cp.Variable(name="var_z")
        u = cp.Variable(T, nonneg=True, name="cvar_u")
        alpha = confidence_level
        cvar_term = z + (1.0 / ((1 - alpha) * T)) * cp.sum(u)
        objective_terms.append(-lam_cvar * cvar_term)
        losses = -returns_history @ w
        extra_constraints += [u >= losses - z]

    # 4. Drawdown penalty (historical approximation)
    # Approximate: penalise average of the worst (1-alpha) daily losses
    # This is a simpler proxy that avoids path-dependency in a static optimizer
    if lam_drawdown > 0 and returns_history is not None:
        T = len(returns_history)
        alpha = confidence_level
        z_dd = cp.Variable(name="dd_z")
        u_dd = cp.Variable(T, nonneg=True, name="dd_u")
        losses = -returns_history @ w
        dd_term = z_dd + (1.0 / ((1 - alpha) * T)) * cp.sum(u_dd)
        objective_terms.append(-lam_drawdown * dd_term)
        extra_constraints += [u_dd >= losses - z_dd]

    # 5. Factor concentration penalty
    if lam_factor_conc > 0 and factor_betas is not None:
        B = factor_betas  # (N, K)
        if factor_target is None:
            factor_target = np.zeros(B.shape[1])
        portfolio_factor_exp = B.T @ w  # (K,)
        deviation = portfolio_factor_exp - factor_target
        # cp.sum_squares is convex
        objective_terms.append(-lam_factor_conc * cp.sum_squares(deviation))

    if not objective_terms:
        raise ValueError("All lambda values are zero — no objective defined.")

    objective = cp.Maximize(sum(objective_terms))
    constraints = constraint_builder.build(w, Sigma=Sigma, mu=mu) + extra_constraints

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8,
               max_iter=10000)

    status = prob.status or "failed"
    if w.value is None:
        w_val = np.ones(N) / N
        status = "failed_fallback_equal_weight"
    else:
        w_val = np.maximum(w.value, 0)
        if w_val.sum() > 0:
            w_val /= w_val.sum()
        else:
            w_val = np.ones(N) / N

    label = (
        f"Multi-Objective ("
        f"λ_ret={lam_return}, λ_vol={lam_vol}, "
        f"λ_cvar={lam_cvar}, λ_dd={lam_drawdown}, "
        f"λ_fc={lam_factor_conc})"
    )
    return _make_result(w_val, tickers, mu, Sigma, returns_history, label, status, rf)


# ---------------------------------------------------------------------------
# Efficient Frontier (Pareto-optimal portfolios)
# ---------------------------------------------------------------------------

@dataclass
class EfficientFrontierResult:
    """Stores the full efficient frontier."""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights_matrix: np.ndarray          # shape (n_points, N)
    pareto_portfolios: List[AllocationResult]  # full result objects at each point

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "return": self.returns,
            "volatility": self.volatilities,
            "sharpe": self.sharpe_ratios,
        })


def compute_efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    n_points: int = 50,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> EfficientFrontierResult:
    """
    Compute the efficient frontier by sweeping target returns.

    For each target return level, solve:
        minimize wᵀΣw
        subject to: μᵀw >= target_return, [all other constraints]

    Parameters
    ----------
    n_points : int
        Number of frontier points (default 50).
    """
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy required for efficient frontier.")

    N = len(tickers)

    # Find min and max achievable returns under constraints
    # Min return: minimum variance portfolio
    min_var_result = optimize_minimum_variance(
        mu, Sigma, tickers, constraint_builder, returns_history, rf
    )
    mu_min = min_var_result.expected_return

    # Max return: unconstrained maximum (but constrained weights)
    w_temp = cp.Variable(N)
    prob_max = cp.Problem(
        cp.Maximize(mu @ w_temp),
        constraint_builder.build(w_temp, Sigma=Sigma, mu=mu)
    )
    prob_max.solve(solver=cp.OSQP, verbose=False)
    mu_max = float(mu @ w_temp.value) if w_temp.value is not None else mu.max()

    target_returns = np.linspace(mu_min, mu_max, n_points)

    all_returns, all_vols, all_sharpes = [], [], []
    all_weights = []
    pareto = []

    w = cp.Variable(N, name="weights")

    for target in target_returns:
        base_constraints = constraint_builder.build(w, Sigma=Sigma, mu=mu)
        constraints = base_constraints + [mu @ w >= target]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-9, eps_rel=1e-9)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            w_val = np.maximum(w.value, 0)
            w_val /= w_val.sum()
            ret, vol, sharpe, cvar = _portfolio_stats(w_val, mu, Sigma, returns_history, rf)
            all_returns.append(ret)
            all_vols.append(vol)
            all_sharpes.append(sharpe)
            all_weights.append(w_val)
            pareto.append(
                _make_result(w_val, tickers, mu, Sigma, returns_history,
                             f"Frontier (μ={target:.2%})", prob.status, rf)
            )

    return EfficientFrontierResult(
        returns=np.array(all_returns),
        volatilities=np.array(all_vols),
        sharpe_ratios=np.array(all_sharpes),
        weights_matrix=np.array(all_weights),
        pareto_portfolios=pareto,
    )


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "mean_variance": optimize_mean_variance,
    "min_variance": optimize_minimum_variance,
    "cvar": optimize_cvar,
    "risk_parity": optimize_risk_parity,
    "max_diversification": optimize_max_diversification,
    "multi_objective": optimize_multi_objective,
}


def run_optimizer(
    method: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    **kwargs,
) -> AllocationResult:
    """
    Dispatch to the correct optimizer by name.

    Parameters
    ----------
    method : str
        One of: "mean_variance", "min_variance", "cvar",
                "risk_parity", "max_diversification", "multi_objective"
    kwargs : passed through to optimizer (lam, returns_history, etc.)
    """
    if method not in OPTIMIZERS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[method](mu, Sigma, tickers, constraint_builder, **kwargs)
