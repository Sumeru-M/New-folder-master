"""
robust_optimizer.py — Robust & Stress-Aware Portfolio Optimization

Three robustness strategies:

1. Shrinkage-based MVO
   Use Ledoit-Wolf shrinkage covariance instead of sample covariance.
   Wraps optimizer.py's compute_ledoit_wolf_shrinkage (with the ρ-hat fix applied).

2. Worst-Case Covariance Band
   Optimize over the worst covariance matrix within an uncertainty ellipsoid:
       max_{Σ ∈ U} wᵀΣw  (worst-case variance)
   Approximated via eigenvalue perturbation: inflate each eigenvalue by δ × λ_max.

3. Scenario-Weighted Loss Optimization
   Given K stress scenarios with probabilities p_k:
       minimize Σ_k p_k × CVaR_k(w)
   where CVaR_k is computed under the k-th stressed return distribution.
   Wraps scenario_engine.ScenarioEngine.

All functions return AllocationResult objects for compatibility with the
allocation intelligence layer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .optimization_engine import (
    AllocationResult, _make_result, _portfolio_stats, optimize_minimum_variance
)
from .constraints import ConstraintBuilder


# ---------------------------------------------------------------------------
# 1. Shrinkage-based robust MVO
# ---------------------------------------------------------------------------

def compute_ledoit_wolf_shrinkage_fixed(
    returns: pd.DataFrame,
    annualized: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf shrinkage with the corrected ρ-hat term (per our quant review).

    Σ_shrink = (1 - δ) × S + δ × F
    where F = (tr(S)/N) × I  (scaled identity target)

    δ* = (π - ρ) / γ
        π = asymptotic variance of sample cov entries
        ρ = correction term for scaled-identity target  ← WAS MISSING
        γ = ||S - F||²_F

    Parameters
    ----------
    returns : pd.DataFrame  shape (T, N)
    annualized : bool

    Returns
    -------
    (shrunk_cov DataFrame, delta float)
    """
    X = returns.values  # (T, N)
    T, N = X.shape

    if T < 2:
        raise ValueError("Need at least 2 observations.")

    X_c = X - X.mean(axis=0, keepdims=True)  # centred
    S = (X_c.T @ X_c) / (T - 1)              # sample cov (N, N)

    # Shrinkage target: F = mu_hat × I
    mu_hat = np.trace(S) / N
    F = mu_hat * np.eye(N)

    # π-hat: sum of asymptotic variances of S_{ij}
    pi_hat = 0.0
    for i in range(N):
        for j in range(N):
            cross = X_c[:, i] * X_c[:, j]
            pi_hat += np.var(cross, ddof=1)
    pi_hat /= T

    # ρ-hat: correction term for scaled-identity target (LW 2004, Appendix B)
    rho_hat = 0.0
    for i in range(N):
        cross_ii = X_c[:, i] ** 2
        asym_var_ii = np.var(cross_ii, ddof=1) / T
        if mu_hat > 0:
            rho_hat += asym_var_ii * (S[i, i] - mu_hat) / mu_hat

    # γ-hat: ||S - F||²_F
    gamma_hat = float(np.sum((S - F) ** 2))

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 1.0
    delta = float(np.clip(kappa, 0.0, 1.0))

    Sigma_shrink = (1 - delta) * S + delta * F
    Sigma_shrink = (Sigma_shrink + Sigma_shrink.T) / 2  # enforce symmetry

    cov_df = pd.DataFrame(Sigma_shrink, index=returns.columns, columns=returns.columns)
    if annualized:
        cov_df *= 252

    return cov_df, delta


def optimize_with_shrinkage(
    mu: np.ndarray,
    returns_df: pd.DataFrame,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    optimization_method: str = "mean_variance",
    lam: float = 1.0,
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
) -> Tuple[AllocationResult, float]:
    """
    Run any optimizer using Ledoit-Wolf shrinkage covariance.

    Parameters
    ----------
    mu : np.ndarray (N,)
        Expected returns (arithmetic, annualised).
    returns_df : pd.DataFrame  shape (T, N)
        Historical log returns for shrinkage estimation.
    optimization_method : str
        "mean_variance" | "min_variance" | "cvar" | "risk_parity"
        | "max_diversification" | "multi_objective"
    lam : float
        Risk-aversion parameter (for mean_variance).

    Returns
    -------
    (AllocationResult, shrinkage_delta)
    """
    from optimization_engine import run_optimizer

    cov_shrink, delta = compute_ledoit_wolf_shrinkage_fixed(returns_df)
    Sigma = cov_shrink.values

    result = run_optimizer(
        optimization_method,
        mu, Sigma, tickers, constraint_builder,
        lam=lam, returns_history=returns_history, rf=rf,
    )
    result.optimization_type = f"{result.optimization_type} [LW δ={delta:.3f}]"
    return result, delta


# ---------------------------------------------------------------------------
# 2. Worst-case covariance band
# ---------------------------------------------------------------------------

def compute_worst_case_covariance(
    Sigma: np.ndarray,
    uncertainty_level: float = 0.10,
    method: str = "eigenvalue_inflation",
) -> np.ndarray:
    """
    Construct worst-case covariance within an uncertainty band.

    Two methods:
    A) eigenvalue_inflation
       Inflate each eigenvalue by (1 + uncertainty_level × relative_rank).
       Largest eigenvalue gets the most inflation, smallest the least.
       Preserves eigenvectors (directions of risk unchanged).

    B) frobenius_ball
       Sigma_wc = Sigma + uncertainty_level × ||Sigma||_F × I
       Adds uniform uncertainty to all directions.
       Simpler but less discriminating.

    Parameters
    ----------
    Sigma : np.ndarray (N, N)
        Base covariance matrix (annualised).
    uncertainty_level : float
        Fractional perturbation (e.g. 0.10 = 10%).
    method : str
        "eigenvalue_inflation" or "frobenius_ball".

    Returns
    -------
    np.ndarray (N, N)
        Worst-case covariance matrix (PSD guaranteed).
    """
    if method == "eigenvalue_inflation":
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        # Rank-weighted inflation: most important factor gets most inflation
        N = len(eigvals)
        ranks = np.arange(N) / max(N - 1, 1)          # 0 = smallest, 1 = largest
        inflation = 1.0 + uncertainty_level * ranks
        inflated_eigvals = eigvals * inflation
        inflated_eigvals = np.maximum(inflated_eigvals, 0)  # PSD safety
        Sigma_wc = eigvecs @ np.diag(inflated_eigvals) @ eigvecs.T

    elif method == "frobenius_ball":
        delta = uncertainty_level * np.linalg.norm(Sigma, "fro")
        Sigma_wc = Sigma + delta * np.eye(len(Sigma))

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return (Sigma_wc + Sigma_wc.T) / 2  # symmetry


def optimize_worst_case(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    uncertainty_level: float = 0.10,
    uncertainty_method: str = "eigenvalue_inflation",
    optimization_method: str = "min_variance",
    returns_history: Optional[np.ndarray] = None,
    rf: float = 0.07,
    **kwargs,
) -> AllocationResult:
    """
    Optimize under worst-case covariance within the uncertainty band.

    This is a conservative strategy: the portfolio that performs best
    under the most adversarial (but plausible) covariance matrix.

    Parameters
    ----------
    uncertainty_level : float
        How large the covariance uncertainty band is (0.10 = 10%).
    uncertainty_method : str
        "eigenvalue_inflation" | "frobenius_ball"
    optimization_method : str
        Optimizer to apply on worst-case covariance.
    """
    from optimization_engine import run_optimizer

    Sigma_wc = compute_worst_case_covariance(Sigma, uncertainty_level, uncertainty_method)

    result = run_optimizer(
        optimization_method,
        mu, Sigma_wc, tickers, constraint_builder,
        returns_history=returns_history, rf=rf, **kwargs,
    )
    result.optimization_type = (
        f"{result.optimization_type} [Worst-Case Σ, ε={uncertainty_level:.0%}]"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Scenario-weighted loss optimization
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """
    A single stress scenario for the robust optimizer.

    probability : float
        Weight of this scenario in the composite objective (must sum to 1 across scenarios).
    shocked_mu : np.ndarray (N,)
        Expected returns under this scenario.
    shocked_Sigma : np.ndarray (N, N)
        Covariance under this scenario.
    scenario_returns : np.ndarray (T, N), optional
        Simulated or historical returns under this scenario.
        If None, parametric CVaR is used.
    label : str
    """
    probability: float
    shocked_mu: np.ndarray
    shocked_Sigma: np.ndarray
    scenario_returns: Optional[np.ndarray] = None
    label: str = "Unnamed Scenario"


def build_stress_scenarios_from_engine(
    base_mu: pd.Series,
    base_Sigma: pd.DataFrame,
    base_returns: pd.DataFrame,
    scenario_definitions: Optional[List[Dict]] = None,
    n_simulated_paths: int = 2000,
) -> List[StressScenario]:
    """
    Build StressScenario list using ScenarioEngine (scenario_engine.py).

    Simulates return paths under each stressed (mu, Sigma) pair using
    multivariate normal draws. In production, replace with historical
    factor scenario returns.

    Parameters
    ----------
    base_mu : pd.Series
    base_Sigma : pd.DataFrame
    base_returns : pd.DataFrame  shape (T, N) — historical daily simple returns
    scenario_definitions : List[Dict], optional
        Each dict: {"label": str, "probability": float,
                    "return_shock": float, "volatility_shock": float,
                    "correlation_shock": float}
        If None, uses 4 canonical scenarios.
    n_simulated_paths : int
        Number of simulated daily observations per scenario.
    """
    try:
        from scenario_engine import ScenarioEngine, MarketShock
    except ImportError:
        raise ImportError("scenario_engine.py must be on the Python path.")

    if scenario_definitions is None:
        scenario_definitions = [
            {
                "label": "Base (Normal)",
                "probability": 0.50,
                "return_shock": 0.0,
                "volatility_shock": 1.0,
                "correlation_shock": 0.0,
            },
            {
                "label": "Moderate Stress",
                "probability": 0.25,
                "return_shock": -0.15,
                "volatility_shock": 1.5,
                "correlation_shock": 0.15,
            },
            {
                "label": "Severe Crash",
                "probability": 0.15,
                "return_shock": -0.40,
                "volatility_shock": 2.5,
                "correlation_shock": 0.40,
            },
            {
                "label": "Recovery / Boom",
                "probability": 0.10,
                "return_shock": 0.15,
                "volatility_shock": 0.8,
                "correlation_shock": -0.10,
            },
        ]

    engine = ScenarioEngine(
        expected_returns=base_mu,
        covariance_matrix=base_Sigma,
    )

    tickers = base_mu.index.tolist()
    scenarios: List[StressScenario] = []

    for defn in scenario_definitions:
        shock = MarketShock(
            name=defn["label"],
            return_shock=defn["return_shock"],
            volatility_shock=defn["volatility_shock"],
            correlation_shock=defn["correlation_shock"],
        )
        shocked_mu, shocked_Sigma = engine.apply_scenario(shock)

        # Simulate returns under this scenario (daily, simple returns)
        # Convert annualised parameters to daily
        mu_daily = shocked_mu.values / 252
        Sigma_daily = shocked_Sigma.values / 252

        # Simulate via multivariate normal (GBM approximation)
        rng = np.random.default_rng(seed=42)
        sim_log_returns = rng.multivariate_normal(
            mean=mu_daily, cov=Sigma_daily, size=n_simulated_paths
        )
        sim_simple_returns = np.expm1(sim_log_returns)  # exp(r) - 1

        scenarios.append(
            StressScenario(
                probability=defn["probability"],
                shocked_mu=shocked_mu.values,
                shocked_Sigma=shocked_Sigma.values,
                scenario_returns=sim_simple_returns,
                label=defn["label"],
            )
        )

    # Normalise probabilities just in case
    total_prob = sum(s.probability for s in scenarios)
    for s in scenarios:
        s.probability /= total_prob

    return scenarios


def optimize_scenario_weighted(
    scenarios: List[StressScenario],
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    confidence_level: float = 0.95,
    rf: float = 0.07,
    n_sim_paths_parametric: int = 5000,
) -> AllocationResult:
    """
    Scenario-weighted CVaR optimisation.

    Objective: minimize Σ_k p_k × CVaR_α(w; scenario_k)

    Using Rockafellar-Uryasev for each scenario:
        CVaR_α(w; k) = z_k + 1/((1-α)T_k) × Σ_t u_{k,t}

    The combined objective is:
        Σ_k p_k × [z_k + 1/((1-α)T_k) × Σ_t u_{k,t}]

    This is a single large LP (one (z_k, u_k) pair per scenario).

    Parameters
    ----------
    scenarios : List[StressScenario]
        Scenarios with probabilities summing to 1.
    confidence_level : float
        CVaR confidence level (e.g. 0.95).
    """
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy required for scenario-weighted optimisation.")

    N = len(tickers)
    alpha = confidence_level

    w = cp.Variable(N, name="weights")
    all_constraints = constraint_builder.build(w)

    objective_terms = []

    for scenario in scenarios:
        if scenario.scenario_returns is not None:
            T_k = len(scenario.scenario_returns)
            ret_matrix = scenario.scenario_returns  # (T_k, N)
        else:
            # Generate parametric draws
            mu_d = scenario.shocked_mu / 252
            Sig_d = scenario.shocked_Sigma / 252
            rng = np.random.default_rng(42)
            draws = rng.multivariate_normal(mu_d, Sig_d, size=n_sim_paths_parametric)
            ret_matrix = np.expm1(draws)
            T_k = n_sim_paths_parametric

        losses = -ret_matrix @ w  # (T_k,)  portfolio loss per period
        z_k = cp.Variable(name=f"z_{scenario.label[:8]}")
        u_k = cp.Variable(T_k, nonneg=True, name=f"u_{scenario.label[:8]}")

        cvar_k = z_k + (1.0 / ((1 - alpha) * T_k)) * cp.sum(u_k)
        objective_terms.append(scenario.probability * cvar_k)
        all_constraints += [u_k >= losses - z_k]

    objective = cp.Minimize(sum(objective_terms))
    prob = cp.Problem(objective, all_constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8,
               max_iter=20000)

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

    # Use blended mu and Sigma for reporting
    blended_mu = sum(s.probability * s.shocked_mu for s in scenarios)
    blended_Sigma = sum(s.probability * s.shocked_Sigma for s in scenarios)
    blended_returns = None
    if all(s.scenario_returns is not None for s in scenarios):
        # Weighted sample of returns for CVaR reporting
        blended_returns = np.vstack([s.scenario_returns for s in scenarios])

    scenario_labels = ", ".join(s.label for s in scenarios)
    return _make_result(
        w_val, tickers, blended_mu, blended_Sigma, blended_returns,
        f"Scenario-Weighted CVaR [{scenario_labels}]", status, rf
    )


# ---------------------------------------------------------------------------
# Convenience: run all three robust strategies and return best by Sharpe
# ---------------------------------------------------------------------------

def optimize_robust_ensemble(
    mu: np.ndarray,
    returns_df: pd.DataFrame,
    tickers: List[str],
    constraint_builder: ConstraintBuilder,
    scenarios: Optional[List[StressScenario]] = None,
    uncertainty_level: float = 0.10,
    rf: float = 0.07,
) -> Dict[str, AllocationResult]:
    """
    Run all three robust strategies. Returns dict of results.

    Keys: "shrinkage", "worst_case", "scenario_weighted"
    """
    Sigma_sample = returns_df.cov().values * 252
    results: Dict[str, AllocationResult] = {}

    # 1. Shrinkage
    shrink_result, _ = optimize_with_shrinkage(
        mu, returns_df, tickers, constraint_builder,
        optimization_method="min_variance", rf=rf,
    )
    results["shrinkage"] = shrink_result

    # 2. Worst-case
    wc_result = optimize_worst_case(
        mu, Sigma_sample, tickers, constraint_builder,
        uncertainty_level=uncertainty_level,
        optimization_method="min_variance", rf=rf,
    )
    results["worst_case"] = wc_result

    # 3. Scenario-weighted (only if scenarios provided)
    if scenarios is not None:
        sw_result = optimize_scenario_weighted(
            scenarios, tickers, constraint_builder, rf=rf
        )
        results["scenario_weighted"] = sw_result

    return results
