"""
Milestone 5: Institutional Portfolio Optimization & Allocation Intelligence

This script runs the full Milestone 5 optimization pipeline on real NSE data.
It demonstrates all six optimization frameworks, robust strategies, and the
allocation intelligence layer — producing a complete JSON-ready output alongside
CSV artifacts.

Usage:
    # Interactive mode (recommended for first run):
    python run_milestone5.py

    # Command line mode:
    python run_milestone5.py --tickers "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"

    # With custom settings:
    python run_milestone5.py \\
        --tickers "TCS.NS,INFY.NS,WIPRO.NS,HDFCBANK.NS,ICICIBANK.NS,AXISBANK.NS" \\
        --portfolio-value 5000000 \\
        --method mean_variance \\
        --risk-aversion 2.0 \\
        --max-weight 0.20 \\
        --sector-cap 0.40 \\
        --risk-free-rate 0.07

Available optimization methods:
    mean_variance       Maximize return - λ×risk  (configurable risk-aversion)
    min_variance        Pure risk minimisation
    cvar                Minimise Expected Shortfall (CVaR)
    risk_parity         Equal risk contribution across all assets
    max_diversification Maximise diversification ratio
    multi_objective     Balance return, vol, CVaR, drawdown simultaneously
    all                 Run all six and compare (default)
"""

import sys
import os
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — same pattern as M3/M4
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Existing modules (M1–M4)
from portfolio.portfolio_complete import (
    load_price_data,
    compute_daily_returns,
)
# (scenario_engine and risk_metrics are used internally by M5 modules)

# Milestone 5 modules
from portfolio.constraints import build_institutional_constraints
from portfolio.optimization_engine import (
    optimize_mean_variance,
    optimize_minimum_variance,
    optimize_cvar,
    optimize_risk_parity,
    optimize_max_diversification,
    optimize_multi_objective,
    compute_efficient_frontier,
    AllocationResult,
)
from portfolio.robust_optimizer import (
    compute_ledoit_wolf_shrinkage_fixed,
    optimize_worst_case,
    build_stress_scenarios_from_engine,
    optimize_scenario_weighted,
)
from portfolio.risk_contribution import build_risk_attribution_report
from portfolio.allocation_scorer import enrich_allocation_result


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Milestone 5: Institutional Portfolio Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode:
  python run_milestone5.py

  # Analyze specific stocks:
  python run_milestone5.py --tickers "TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,RELIANCE.NS"

  # Risk parity with sector cap:
  python run_milestone5.py --tickers "TCS.NS,INFY.NS,HDFCBANK.NS" --method risk_parity

  # CVaR minimisation with 10% max weight:
  python run_milestone5.py --tickers "TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS" \\
      --method cvar --max-weight 0.10

  # Run all methods and compare:
  python run_milestone5.py --tickers "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ITC.NS" --method all
        """
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="NSE tickers, comma-separated. Example: RELIANCE.NS,TCS.NS,INFY.NS"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Current allocation as percentages (comma-separated). Example: 30,25,25,20. Leave blank for equal weights."
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=10_00_000.0,
        help="Portfolio value in INR (default: ₹10,00,000)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["mean_variance", "min_variance", "cvar", "risk_parity",
                 "max_diversification", "multi_objective", "all"],
        help="Optimization method (default: all)"
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=2.0,
        help="Risk-aversion λ for mean_variance (default: 2.0). Higher = more conservative."
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.40,
        help="Max single-stock weight, e.g. 0.20 for 20%% (default: 0.40)"
    )
    parser.add_argument(
        "--sector-cap",
        type=float,
        default=0.50,
        help="Max sector weight, e.g. 0.40 for 40%% (default: 0.50)"
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=1.0,
        help="Max one-way turnover vs current weights (default: 1.0 = unconstrained)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.07,
        help="Risk-free rate (default: 0.07 = 7%%)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Historical data period: 6mo, 1y, 2y, 5y (default: 2y)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="CVaR/VaR confidence level (default: 0.95)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sector map for NSE stocks (covers most common tickers)
# ---------------------------------------------------------------------------

# NSE_SECTOR_MAP and get_sector are now in portfolio.portfolio_complete
from portfolio.portfolio_complete import NSE_SECTOR_MAP, get_sector


# ---------------------------------------------------------------------------
# Helper: print section header (M3/M4 style)
# ---------------------------------------------------------------------------

def section(title: str, width: int = 80):
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def subsection(title: str, width: int = 80):
    print()
    print("-" * width)
    print(title)
    print("-" * width)


# ---------------------------------------------------------------------------
# Helper: print weights table
# ---------------------------------------------------------------------------

def print_weights(weights: pd.Series, label: str = "Weights"):
    print(f"\n  {label}:")
    for ticker, w in weights.sort_values(ascending=False).items():
        bar = "█" * int(w * 40)
        print(f"    {ticker:22s}  {w:>6.1%}  {bar}")


# ---------------------------------------------------------------------------
# Helper: print allocation result summary
# ---------------------------------------------------------------------------

def print_result(result: AllocationResult, portfolio_value: float):
    print_weights(result.weights, "Optimal Weights")
    print()
    print(f"  Expected Return : {result.expected_return:>7.2%} p.a.")
    print(f"  Volatility      : {result.volatility:>7.2%} p.a.")
    print(f"  Sharpe Ratio    : {result.sharpe_ratio:>7.2f}")
    print(f"  CVaR (95%)      : {result.cvar_95:>7.2%}")
    print(f"  Solve Status    : {result.solve_status}")

    if result.allocation_health_score > 0:
        score = result.allocation_health_score
        if score >= 80:
            grade = "🟢 Excellent"
        elif score >= 60:
            grade = "🟡 Good"
        elif score >= 40:
            grade = "🟠 Acceptable"
        else:
            grade = "🔴 Poor"
        print(f"  Health Score    : {score}/100  {grade}")

    if result.risk_contribution:
        print(f"\n  Risk Contribution (% of portfolio risk):")
        for ticker, rc in sorted(result.risk_contribution.items(),
                                  key=lambda x: -x[1]):
            bar = "█" * int(rc * 30)
            print(f"    {ticker:22s}  {rc:>6.1%}  {bar}")

    if result.rebalance_actions_rupees:
        print(f"\n  Rebalance Actions (₹):")
        for ticker, amt in sorted(result.rebalance_actions_rupees.items(),
                                   key=lambda x: -abs(x[1])):
            action = "BUY " if amt > 0 else "SELL"
            print(f"    {action} {ticker:20s}  ₹{abs(amt):>12,.0f}")


# ---------------------------------------------------------------------------
# Interactive input
# ---------------------------------------------------------------------------

def interactive_input(args):
    section("MILESTONE 5: INSTITUTIONAL PORTFOLIO OPTIMIZATION")
    print()
    print("This tool finds the mathematically optimal allocation for your portfolio")
    print("using institutional-grade techniques: MVO, CVaR, Risk Parity, and more.")
    print()

    # Tickers
    print("📈 STEP 1: Which stocks do you want to optimize?")
    print("   Enter NSE stock symbols (with .NS suffix)")
    print("   Example: RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ITC.NS")
    print()
    ti = input("   Your stocks (comma-separated): ").strip()
    args.tickers = ti if ti else "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"

    # Current weights
    print()
    print("💼 STEP 2: Your current allocation (optional)")
    print("   Enter percentages matching your stocks above.")
    print("   Example: 30,25,25,20  means 30% in first stock, etc.")
    print("   Press Enter for equal weights.")
    wts = input("   Current allocation (%): ").strip()
    args.weights = wts if wts else None

    # Portfolio value
    print()
    pv = input(f"💰 Portfolio value in ₹ [{args.portfolio_value:,.0f}]: ").strip()
    if pv:
        try:
            args.portfolio_value = float(pv.replace(",", "").replace("₹", ""))
        except Exception:
            pass

    # Method
    print()
    print("🔬 STEP 3: Choose optimization method")
    print("   mean_variance       Best balance of return and risk (recommended)")
    print("   min_variance        Lowest possible risk")
    print("   cvar                Minimise worst-case losses")
    print("   risk_parity         Equal risk from all stocks")
    print("   max_diversification Maximum diversification benefit")
    print("   multi_objective     Balance all objectives simultaneously")
    print("   all                 Run all methods and compare (default)")
    method = input("   Method [all]: ").strip().lower()
    args.method = method if method in [
        "mean_variance", "min_variance", "cvar", "risk_parity",
        "max_diversification", "multi_objective", "all"
    ] else "all"

    # Risk-free rate
    print()
    rfr = input(f"🏦 Risk-free rate % (FD/bond rate) [{args.risk_free_rate*100:.0f}]: ").strip()
    if rfr:
        try:
            r = float(rfr)
            args.risk_free_rate = r / 100 if r > 1 else r
        except Exception:
            pass

    # Max weight
    print()
    mw = input(f"📏 Max weight per stock % [{args.max_weight*100:.0f}]: ").strip()
    if mw:
        try:
            m = float(mw)
            args.max_weight = m / 100 if m > 1 else m
        except Exception:
            pass

    return args


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    results_dict: dict,
    ef_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    artifacts_dir: str,
):
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Full JSON output
    json_path = os.path.join(artifacts_dir, "optimization_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    # 2. Efficient frontier CSV
    ef_path = os.path.join(artifacts_dir, "efficient_frontier.csv")
    ef_df.to_csv(ef_path, index=False)

    # 3. Method comparison CSV
    comp_path = os.path.join(artifacts_dir, "method_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)

    return json_path, ef_path, comp_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Interactive fallback (M3/M4 pattern)
    if args.tickers is None:
        args = interactive_input(args)

    # Normalise tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers]
    N = len(tickers)

    if N < 2:
        print("❌ Need at least 2 stocks to optimize. Please add more tickers.")
        return

    rf = args.risk_free_rate
    pv = args.portfolio_value
    method = args.method

    section("PORTFOLIO OPTIMIZATION ENGINE — MILESTONE 5")
    print(f"  Stocks          : {', '.join(tickers)}")
    print(f"  Portfolio Value : ₹{pv:,.0f}")
    print(f"  Method          : {method}")
    print(f"  Risk-Free Rate  : {rf:.1%}")
    print(f"  Max Weight      : {args.max_weight:.0%} per stock")
    print(f"  Sector Cap      : {args.sector_cap:.0%} per sector")
    print(f"  Data Period     : {args.period}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Load Data
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 1: Loading Market Data")
    try:
        prices = load_price_data(tickers, period=args.period)
        print(f"  ✓ Loaded {len(prices)} trading days for {N} stocks")
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        return

    # Remove any tickers with too many NaNs
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    tickers = list(prices.columns)
    N = len(tickers)
    print(f"  ✓ {N} stocks after data quality check")
    print(f"  Period: {prices.index[0].date()} → {prices.index[-1].date()}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Compute Returns & Risk Parameters
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 2: Computing Risk Parameters")

    log_returns = compute_daily_returns(prices)      # log returns DataFrame

    # Arithmetic annualised mu (lognormal correction — correct for MVO)
    mu_log_daily = log_returns.mean()
    var_log_daily = log_returns.var()
    mu_arithmetic = np.exp(mu_log_daily * 252 + 0.5 * var_log_daily * 252) - 1
    mu_series = mu_arithmetic                        # pd.Series, indexed by ticker

    # Annualised covariance (sample)
    Sigma_sample_df = log_returns.cov() * 252
    Sigma_sample = Sigma_sample_df.values

    # Ledoit-Wolf shrinkage covariance
    Sigma_shrink_df, lw_delta = compute_ledoit_wolf_shrinkage_fixed(log_returns)
    Sigma_shrink = Sigma_shrink_df.values

    # Historical simple returns for CVaR optimisation
    simple_returns = np.expm1(log_returns.values)    # shape (T, N)

    print(f"  ✓ Expected returns (arithmetic, annualised):")
    for t, r in mu_series.sort_values(ascending=False).items():
        print(f"      {t:22s}  {r:>7.2%} p.a.")

    print(f"\n  ✓ Ledoit-Wolf shrinkage intensity: δ = {lw_delta:.4f}")
    print(f"    (0 = sample covariance, 1 = identity target)")
    print(f"    δ = {lw_delta:.4f} → {'heavy shrinkage - useful with limited history' if lw_delta > 0.3 else 'mild shrinkage - sample cov is well-conditioned'}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Parse Current Weights
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 3: Current Portfolio")

    if args.weights:
        try:
            raw_w = [float(x.strip()) for x in args.weights.split(",")]
            if len(raw_w) != N:
                print(f"  ⚠ Weight count ({len(raw_w)}) ≠ ticker count ({N}). Using equal weights.")
                raw_w = [1.0 / N] * N
            else:
                total = sum(raw_w)
                raw_w = [w / total for w in raw_w]
        except Exception:
            raw_w = [1.0 / N] * N
    else:
        raw_w = [1.0 / N] * N

    current_weights = pd.Series(raw_w, index=tickers)

    print(f"  Current allocation:")
    for t, w in current_weights.items():
        print(f"    {t:22s}  {w:>6.1%}")

    # Current portfolio stats
    w_arr = current_weights.values
    curr_ret = float(w_arr @ mu_series.values)
    curr_vol = float(np.sqrt(w_arr @ Sigma_shrink @ w_arr))
    curr_sharpe = (curr_ret - rf) / curr_vol if curr_vol > 0 else 0.0
    print(f"\n  Current: Return {curr_ret:.2%}  |  Vol {curr_vol:.2%}  |  Sharpe {curr_sharpe:.2f}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Build Constraints
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 4: Applying Institutional Constraints")

    sector_map = {t: get_sector(t) for t in tickers}
    print(f"  Sector assignments:")
    for t, s in sector_map.items():
        print(f"    {t:22s}  {s}")

    # Feasibility: max_weight must be >= 1/N
    min_feasible = 1.0 / N
    max_weight = max(args.max_weight, min_feasible + 0.01)
    if max_weight != args.max_weight:
        print(f"\n  ⚠ max_weight adjusted to {max_weight:.1%} (minimum feasible for {N} stocks)")

    # Build with previous weights for turnover constraint (if not too tight)
    effective_turnover = min(args.max_turnover, 2.0)  # cap at 200% to avoid infeasibility

    constraint_builder = build_institutional_constraints(
        n_assets=N,
        tickers=tickers,
        max_weight=max_weight,
        sector_map=sector_map,
        sector_cap=args.sector_cap,
        prev_weights=current_weights.values,
        max_turnover=effective_turnover,
    )

    print(f"\n  Active constraints:")
    for c in constraint_builder.summary():
        print(f"    • {c}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Run Optimizations
    # ──────────────────────────────────────────────────────────────────────
    section("Step 5: Running Optimization")

    mu_np = mu_series.values
    opt_results = {}   # method_name → AllocationResult

    METHODS_TO_RUN = (
        ["mean_variance", "min_variance", "cvar", "risk_parity",
         "max_diversification", "multi_objective"]
        if method == "all"
        else [method]
    )

    for m in METHODS_TO_RUN:
        print(f"\n  [{m.upper().replace('_', ' ')}]")
        try:
            if m == "mean_variance":
                result = optimize_mean_variance(
                    mu=mu_np, Sigma=Sigma_shrink, tickers=tickers,
                    constraint_builder=constraint_builder,
                    lam=args.risk_aversion,
                    returns_history=simple_returns, rf=rf,
                )

            elif m == "min_variance":
                result = optimize_minimum_variance(
                    mu=mu_np, Sigma=Sigma_shrink, tickers=tickers,
                    constraint_builder=constraint_builder,
                    returns_history=simple_returns, rf=rf,
                )

            elif m == "cvar":
                result = optimize_cvar(
                    mu=mu_np, Sigma=Sigma_shrink, tickers=tickers,
                    constraint_builder=constraint_builder,
                    returns_history=simple_returns,
                    confidence_level=args.confidence_level,
                    lam_return=0.3, rf=rf,
                )

            elif m == "risk_parity":
                result = optimize_risk_parity(
                    Sigma=Sigma_shrink, tickers=tickers, mu=mu_np,
                    returns_history=simple_returns, rf=rf,
                )

            elif m == "max_diversification":
                result = optimize_max_diversification(
                    Sigma=Sigma_shrink, tickers=tickers, mu=mu_np,
                    constraint_builder=constraint_builder,
                    returns_history=simple_returns, rf=rf,
                )

            elif m == "multi_objective":
                result = optimize_multi_objective(
                    mu=mu_np, Sigma=Sigma_shrink, tickers=tickers,
                    constraint_builder=constraint_builder,
                    returns_history=simple_returns,
                    lam_return=1.0, lam_vol=1.0, lam_cvar=0.5,
                    lam_drawdown=0.3, lam_factor_conc=0.0,
                    confidence_level=args.confidence_level, rf=rf,
                )

            else:
                continue

            # Enrich with intelligence layer
            result = enrich_allocation_result(
                result=result,
                covariance_matrix=Sigma_shrink_df,
                mu=mu_series,
                current_weights=current_weights,
                portfolio_value=pv,
                rf=rf,
            )

            opt_results[m] = result

            # Print summary
            print(f"    Status  : {result.solve_status}")
            print(f"    Return  : {result.expected_return:.2%}  |  Vol: {result.volatility:.2%}  |  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"    CVaR 95%: {result.cvar_95:.2%}  |  Health: {result.allocation_health_score}/100")

        except Exception as e:
            print(f"    ✗ Failed: {e}")

    if not opt_results:
        print("\n❌ All optimizations failed. Check your data and constraints.")
        return

    # ──────────────────────────────────────────────────────────────────────
    # Step 6: Efficient Frontier
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 6: Efficient Frontier")

    ef_df = pd.DataFrame()
    try:
        print("  Computing efficient frontier (20 Pareto-optimal portfolios)...")
        ef = compute_efficient_frontier(
            mu=mu_np, Sigma=Sigma_shrink, tickers=tickers,
            constraint_builder=constraint_builder,
            n_points=20,
            returns_history=simple_returns, rf=rf,
        )
        ef_df = ef.to_dataframe()

        print(f"  ✓ Frontier computed: {len(ef_df)} points")
        print(f"\n  Return range : {ef_df['return'].min():.2%} → {ef_df['return'].max():.2%}")
        print(f"  Vol range    : {ef_df['volatility'].min():.2%} → {ef_df['volatility'].max():.2%}")
        print(f"  Best Sharpe  : {ef_df['sharpe'].max():.2f} (Return {ef_df.loc[ef_df['sharpe'].idxmax(), 'return']:.2%}, Vol {ef_df.loc[ef_df['sharpe'].idxmax(), 'volatility']:.2%})")

    except Exception as e:
        print(f"  ⚠ Efficient frontier skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 7: Robust Optimization (stress-aware)
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 7: Robust Optimization (Stress-Aware)")

    robust_results = {}

    # 7a. Worst-case covariance
    print("  [WORST-CASE COVARIANCE]  (10% eigenvalue uncertainty band)")
    try:
        wc_result = optimize_worst_case(
            mu=mu_np, Sigma=Sigma_shrink, tickers=tickers,
            constraint_builder=constraint_builder,
            uncertainty_level=0.10,
            optimization_method="min_variance",
            returns_history=simple_returns, rf=rf,
        )
        wc_result = enrich_allocation_result(
            wc_result, Sigma_shrink_df, mu_series, current_weights, pv, rf=rf
        )
        robust_results["worst_case"] = wc_result
        print(f"    Return: {wc_result.expected_return:.2%}  |  Vol: {wc_result.volatility:.2%}  |  Sharpe: {wc_result.sharpe_ratio:.2f}")
    except Exception as e:
        print(f"    ⚠ Skipped: {e}")

    # 7b. Scenario-weighted CVaR
    print("  [SCENARIO-WEIGHTED CVaR]  (4 macro scenarios: base + moderate + crash + boom)")
    try:
        scenarios = build_stress_scenarios_from_engine(
            base_mu=mu_series,
            base_Sigma=Sigma_shrink_df,
            base_returns=pd.DataFrame(simple_returns, columns=tickers),
            scenario_definitions=[
                {"label": "Base (Normal)",    "probability": 0.50, "return_shock": 0.00, "volatility_shock": 1.0, "correlation_shock": 0.00},
                {"label": "Moderate Stress",  "probability": 0.25, "return_shock": -0.15, "volatility_shock": 1.5, "correlation_shock": 0.15},
                {"label": "Severe Crash",     "probability": 0.15, "return_shock": -0.40, "volatility_shock": 2.5, "correlation_shock": 0.40},
                {"label": "Recovery / Boom",  "probability": 0.10, "return_shock": 0.15, "volatility_shock": 0.8, "correlation_shock": -0.10},
            ],
        )
        sw_result = optimize_scenario_weighted(
            scenarios=scenarios, tickers=tickers,
            constraint_builder=constraint_builder,
            confidence_level=args.confidence_level, rf=rf,
        )
        sw_result = enrich_allocation_result(
            sw_result, Sigma_shrink_df, mu_series, current_weights, pv, rf=rf
        )
        robust_results["scenario_weighted"] = sw_result
        print(f"    Return: {sw_result.expected_return:.2%}  |  Vol: {sw_result.volatility:.2%}  |  Sharpe: {sw_result.sharpe_ratio:.2f}")
    except Exception as e:
        print(f"    ⚠ Skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 8: Risk Attribution (best result)
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 8: Risk Attribution")

    # Pick best result by Sharpe for detailed attribution
    best_method = max(opt_results, key=lambda m: opt_results[m].sharpe_ratio)
    best = opt_results[best_method]

    print(f"  Best method by Sharpe: {best_method.upper().replace('_', ' ')} ({best.sharpe_ratio:.2f})")

    print_weights(best.weights, "Optimal Weights")
    print()
    print(f"  Expected Return : {best.expected_return:.2%} p.a.")
    print(f"  Volatility      : {best.volatility:.2%} p.a.")
    print(f"  Sharpe Ratio    : {best.sharpe_ratio:.2f}")
    print(f"  CVaR (95%)      : {best.cvar_95:.2%}")

    if best.risk_contribution:
        print(f"\n  Risk Contribution (% of portfolio risk):")
        for t, rc in sorted(best.risk_contribution.items(), key=lambda x: -x[1]):
            bar = "█" * int(rc * 30)
            print(f"    {t:22s}  {rc:>6.1%}  {bar}")

    if best.diagnostics.get("concentration"):
        conc = best.diagnostics["concentration"]
        print(f"\n  Concentration Metrics:")
        print(f"    HHI (Herfindahl Index) : {conc.get('herfindahl_index', 0):.4f}  (0=diversified, 1=concentrated)")
        print(f"    Effective N            : {conc.get('effective_n', 0):.1f}  effective positions (out of {N})")
        print(f"    Diversification Ratio  : {best.diagnostics.get('diversification_ratio', 0):.3f}")

    if best.diagnostics.get("factor_imbalance_warning"):
        print(f"\n  ⚠️  {best.diagnostics['factor_imbalance_warning']}")

    # Reporting-only risk attribution (does not affect optimization outputs)
    risk_attribution_report = build_risk_attribution_report(
        weights=best.weights,
        covariance_matrix=Sigma_shrink_df,
    )
    print(f"\n  Diversification Ratio (best): {risk_attribution_report['diversification_ratio']:.3f}")
    conc_eff_n = risk_attribution_report.get("concentration", {}).get("effective_n", 0.0)
    if conc_eff_n:
        print(f"  Effective Number of Positions: {conc_eff_n:.2f}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 9: Rebalance Actions
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 9: Rebalance Actions")

    print(f"  To move from your current portfolio to the {best_method.replace('_', ' ')} optimal:")
    print()
    if best.rebalance_actions_rupees:
        buys = {t: a for t, a in best.rebalance_actions_rupees.items() if a > 0}
        sells = {t: a for t, a in best.rebalance_actions_rupees.items() if a < 0}

        if buys:
            print(f"  BUY:")
            for t, amt in sorted(buys.items(), key=lambda x: -x[1]):
                print(f"    {t:22s}  ₹{amt:>12,.0f}")

        if sells:
            print(f"\n  SELL:")
            for t, amt in sorted(sells.items(), key=lambda x: x[1]):
                print(f"    {t:22s}  ₹{abs(amt):>12,.0f}")

        turnover = sum(abs(a) for a in best.rebalance_actions_rupees.values()) / pv / 2
        print(f"\n  One-way turnover: {turnover:.1%}")
    else:
        print("  No rebalance needed (current allocation matches optimal).")

    if best.overweight_underweight_flags:
        print(f"\n  Overweight / Underweight vs equal-weight benchmark:")
        for t, flag in best.overweight_underweight_flags.items():
            if flag != "Neutral":
                icon = "🔺" if "OW" in flag else "🔻"
                print(f"    {icon} {t:22s}  {flag}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 10: Method Comparison Table
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 10: Method Comparison")

    all_results = {**opt_results, **robust_results}
    comparison_rows = []

    print(f"\n  {'Method':<28}  {'Return':>8}  {'Vol':>7}  {'Sharpe':>7}  {'CVaR':>7}  {'Health':>7}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for mname, res in all_results.items():
        label = mname.replace("_", " ").title()
        print(f"  {label:<28}  {res.expected_return:>8.2%}  {res.volatility:>7.2%}  "
              f"{res.sharpe_ratio:>7.2f}  {res.cvar_95:>7.2%}  {res.allocation_health_score:>6}/100")
        comparison_rows.append({
            "Method": label,
            "Expected_Return": f"{res.expected_return:.4f}",
            "Volatility": f"{res.volatility:.4f}",
            "Sharpe_Ratio": f"{res.sharpe_ratio:.4f}",
            "CVaR_95": f"{res.cvar_95:.4f}",
            "Health_Score": res.allocation_health_score,
            "Solve_Status": res.solve_status,
        })

    # ──────────────────────────────────────────────────────────────────────
    # Step 11: Save Artifacts
    # ──────────────────────────────────────────────────────────────────────
    subsection("Step 11: Saving Results")

    artifacts_dir = "artifacts/milestone5"

    # Build full JSON output
    results_dict = {
        "portfolio": {
            "tickers": tickers,
            "portfolio_value_inr": pv,
            "risk_free_rate": rf,
            "data_period": args.period,
            "current_weights": current_weights.to_dict(),
            "current_stats": {
                "expected_return": round(curr_ret, 6),
                "volatility": round(curr_vol, 6),
                "sharpe_ratio": round(curr_sharpe, 4),
            },
            "shrinkage_delta": round(lw_delta, 4),
        },
        "optimization_results": {
            m: res.to_dict() for m, res in all_results.items()
        },
        "recommendation": {
            "best_method": best_method,
            "reason": f"Highest Sharpe ratio ({best.sharpe_ratio:.2f}) among all methods",
            "summary": {
                "expected_return": round(best.expected_return, 4),
                "volatility": round(best.volatility, 4),
                "sharpe_ratio": round(best.sharpe_ratio, 4),
                "health_score": best.allocation_health_score,
            },
            "risk_attribution": risk_attribution_report,
        }
    }

    comparison_df = pd.DataFrame(comparison_rows)
    json_path, ef_path, comp_path = save_artifacts(
        results_dict, ef_df, comparison_df, artifacts_dir
    )

    # Individual weight CSVs per method
    weight_rows = []
    for mname, res in all_results.items():
        for ticker, w in res.weights.items():
            weight_rows.append({
                "Method": mname.replace("_", " ").title(),
                "Ticker": ticker,
                "Weight": round(w, 6),
                "Weight_Pct": f"{w:.2%}",
                "Rupees": round(w * pv, 0),
            })
    weights_df = pd.DataFrame(weight_rows)
    weights_path = os.path.join(artifacts_dir, "all_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    risk_attr_path = os.path.join(artifacts_dir, "risk_attribution_report.json")
    with open(risk_attr_path, "w") as f:
        json.dump(risk_attribution_report, f, indent=2, default=str)

    print(f"  ✓ {json_path}")
    print(f"  ✓ {comp_path}")
    print(f"  ✓ {weights_path}")
    print(f"  ✓ {risk_attr_path}")
    if not ef_df.empty:
        print(f"  ✓ {ef_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────────────
    section("FINAL SUMMARY — MILESTONE 5 RESULTS")
    print()
    print(f"  Your Current Portfolio:")
    print(f"    Return: {curr_ret:.2%}  |  Vol: {curr_vol:.2%}  |  Sharpe: {curr_sharpe:.2f}")
    print()
    print(f"  ✅ Recommended Allocation  [{best_method.replace('_', ' ').upper()}]:")
    print(f"    Return: {best.expected_return:.2%}  |  Vol: {best.volatility:.2%}  |  Sharpe: {best.sharpe_ratio:.2f}")
    print(f"    CVaR (95%): {best.cvar_95:.2%}  |  Health Score: {best.allocation_health_score}/100")

    delta_return = best.expected_return - curr_ret
    delta_vol = best.volatility - curr_vol
    delta_sharpe = best.sharpe_ratio - curr_sharpe
    print()
    print(f"  Improvement over current:")
    print(f"    Return  : {delta_return:>+7.2%}")
    print(f"    Vol     : {delta_vol:>+7.2%}  {'(lower is better)' if delta_vol < 0 else '(higher risk accepted for return)'}")
    print(f"    Sharpe  : {delta_sharpe:>+7.2f}")

    print()
    print(f"  📁 Results saved to: {artifacts_dir}/")
    print(f"     optimization_results.json  — complete JSON output")
    print(f"     method_comparison.csv      — all methods side by side")
    print(f"     all_weights.csv            — weights per method in rupees")
    print(f"     risk_attribution_report.json — diversification and concentration breakdown")
    if not ef_df.empty:
        print(f"     efficient_frontier.csv     — {len(ef_df)} Pareto-optimal portfolios")
    print()
    print("=" * 80)
    print("✅ Milestone 5 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
