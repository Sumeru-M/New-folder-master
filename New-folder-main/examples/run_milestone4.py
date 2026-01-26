"""
Milestone 4: Scenario Analysis & Stress Testing

This script demonstrates the risk management capabilities:
1. Risk Metrics (VaR, CVaR, Component VaR)
2. Scenario Analysis (Hypothetical Shocks)
3. Historical Stress Testing (2008, 2020, etc.)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portfolio.data_loader import load_price_data
from portfolio.optimizer import (
    PortfolioOptimizer,
    compute_daily_returns,
    compute_expected_returns,
    compute_covariance_matrix
)
from portfolio.scenario_engine import ScenarioEngine, MarketShock
from portfolio.stress_testing import StressTester
from portfolio.risk_metrics import (
    compute_parametric_var,
    compute_historical_var,
    compute_cvar,
    compute_component_var
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Milestone 4: Risk & Stress Testing")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers (if omitted, interactive mode is used)"
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=1_000_000.0,
        help="Portfolio value for VaR calculations (default: $1,000,000)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for VaR/CVaR (default: 0.95)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Risk-free rate (default: 0.05 for 5%%)"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Interactive mode if no tickers provided
    if args.tickers is None:
        print("\n=== Portfolio Risk Analysis Interactive Mode ===")
        print("Please enter the following details (or press Enter for defaults):\n")
        
        # Get tickers
        tickers_input = input("Enter tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS) [default: RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS]: ").strip()
        
        if tickers_input:
            # Validate ticker input
            if '"' in tickers_input or '\\' in tickers_input or '/' in tickers_input:
                print("\n[ERROR] Invalid ticker input detected. Using default tickers.")
                print("Tickers should be comma-separated symbols like: RELIANCE.NS,TCS.NS\n")
                args.tickers = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
            else:
                args.tickers = tickers_input
        else:
            args.tickers = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
        
        # Get portfolio value
        pv_input = input(f"Enter portfolio value [default: ${args.portfolio_value:,.0f}]: ").strip()
        if pv_input:
            try:
                args.portfolio_value = float(pv_input.replace(',', '').replace('$', ''))
            except ValueError:
                print(f"Invalid input, using default: ${args.portfolio_value:,.0f}")
        
        # Get confidence level
        conf_input = input(f"Enter confidence level (e.g., 0.95 for 95%) [default: {args.confidence_level}]: ").strip()
        if conf_input:
            try:
                conf_val = float(conf_input)
                if 0 < conf_val < 1:
                    args.confidence_level = conf_val
                else:
                    print(f"Confidence level must be between 0 and 1, using default: {args.confidence_level}")
            except ValueError:
                print(f"Invalid input, using default: {args.confidence_level}")
        
        # Get risk-free rate
        rf_input = input(f"Enter risk-free rate (e.g., 0.05 for 5%) [default: {args.risk_free_rate}]: ").strip()
        if rf_input:
            try:
                args.risk_free_rate = float(rf_input)
            except ValueError:
                print(f"Invalid input, using default: {args.risk_free_rate}")
        print()
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    
    # Validate tickers
    if not tickers:
        print("[ERROR] No valid tickers provided. Exiting.")
        return
    
    print("=" * 80)
    print("Milestone 4: Scenario Analysis & Stress Testing")
    print("=" * 80)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Portfolio Value: ${args.portfolio_value:,.2f}")
    print(f"Confidence Level: {args.confidence_level:.1%}")
    print(f"Risk-Free Rate: {args.risk_free_rate:.2%}")
    print()

    # ---------------------------------------------------------
    # 1. Base Portfolio Construction
    # ---------------------------------------------------------
    print("Step 1: Constructing Base Portfolio (Max Sharpe)")
    print("-" * 80)
    
    try:
        # Load recent data for optimization (2 years)
        prices_opt = load_price_data(tickers, period="2y")
        print(f"[OK] Loaded {len(prices_opt)} days of price data")
    except Exception as e:
        print(f"[ERROR] Failed to load price data: {e}")
        return
    
    daily_returns = compute_daily_returns(prices_opt)
    mu = compute_expected_returns(daily_returns)
    sigma = compute_covariance_matrix(daily_returns)
    
    print("\nOptimizing portfolio for Maximum Sharpe Ratio...")
    optimizer = PortfolioOptimizer(mu, sigma, risk_free_rate=args.risk_free_rate)
    result = optimizer.optimize_max_sharpe()
    weights = result.weights
    
    print("\n[OK] Optimal Weights (Max Sharpe):")
    for ticker, weight in weights.items():
        print(f"  {ticker:20s}: {weight:8.2%}")
    print(f"\nExpected Return: {result.expected_return:8.2%}")
    print(f"Volatility:      {result.volatility:8.2%}")
    print(f"Sharpe Ratio:    {result.sharpe_ratio:8.3f}")
    print("-" * 80)
    print()

    # ---------------------------------------------------------
    # 2. Risk Metrics (VaR / CVaR / Component VaR)
    # ---------------------------------------------------------
    print("Step 2: Computing Risk Metrics")
    print("-" * 80)
    
    # Calculate portfolio daily returns
    portfolio_daily_ret = daily_returns.dot(weights)
    
    # Parametric VaR
    p_var = compute_parametric_var(
        weights, mu, sigma,
        confidence_level=args.confidence_level,
        portfolio_value=args.portfolio_value
    )
    print(f"Parametric VaR ({args.confidence_level:.0%}, 1-day):")
    print(f"  As percentage: {p_var['var_percent']:8.2%}")
    print(f"  Dollar amount: ${p_var['var_amount']:,.2f}")
    print()
    
    # Historical VaR
    h_var = compute_historical_var(
        portfolio_daily_ret,
        confidence_level=args.confidence_level,
        portfolio_value=args.portfolio_value
    )
    print(f"Historical VaR ({args.confidence_level:.0%}, 1-day):")
    print(f"  As percentage: {h_var['var_percent']:8.2%}")
    print(f"  Dollar amount: ${h_var['var_amount']:,.2f}")
    print()
    
    # CVaR (Expected Shortfall)
    cvar = compute_cvar(
        portfolio_daily_ret,
        confidence_level=args.confidence_level,
        portfolio_value=args.portfolio_value
    )
    print(f"CVaR / Expected Shortfall ({args.confidence_level:.0%}):")
    print(f"  As percentage: {cvar['cvar_percent']:8.2%}")
    print(f"  Dollar amount: ${cvar['cvar_amount']:,.2f}")
    print()
    
    # Component VaR (Risk Contribution by Asset)
    print("Risk Contribution by Asset:")
    comp_var = compute_component_var(
        weights, sigma,
        confidence_level=args.confidence_level
    )
    print(comp_var[['Weight', 'Marginal VaR', 'Component VaR', '% Contribution']].round(4))
    print("\nInterpretation: % Contribution shows how much each asset contributes to total portfolio VaR")
    print("-" * 80)
    print()

    # ---------------------------------------------------------
    # 3. Scenario Analysis (Hypothetical Shocks)
    # ---------------------------------------------------------
    print("Step 3: Scenario Analysis (Hypothetical Market Shocks)")
    print("-" * 80)
    
    engine = ScenarioEngine(mu, sigma)
    scenarios = ScenarioEngine.create_standard_scenarios()
    
    scenario_results = []
    
    print(f"{'Scenario':<30} | {'Return':>10} | {'Vol':>10} | {'Sharpe':>8} | {'Change':>10}")
    print("-" * 80)
    
    # Base Case
    base_sharpe = result.sharpe_ratio
    print(f"{'Base Case':<30} | {result.expected_return:>10.2%} | {result.volatility:>10.2%} | {base_sharpe:>8.3f} | {'-':>10}")
    
    for shock in scenarios:
        # Apply shock to get new parameters
        new_mu, new_sigma = engine.apply_scenario(shock)
        
        # Evaluate portfolio with FIXED weights under new conditions
        w = weights.values
        port_ret = np.dot(w, new_mu.values)
        port_vol = np.sqrt(np.dot(w, np.dot(new_sigma.values, w)))
        sharpe = (port_ret - args.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        
        sharpe_change = sharpe - base_sharpe
        change_str = f"{sharpe_change:+.3f}"
        
        print(f"{shock.name:<30} | {port_ret:>10.2%} | {port_vol:>10.2%} | {sharpe:>8.3f} | {change_str:>10}")
        
        scenario_results.append({
            "Scenario": shock.name,
            "Return": port_ret,
            "Volatility": port_vol,
            "Sharpe": sharpe,
            "Sharpe_Change": sharpe_change
        })
    
    print("-" * 80)
    print("\nInterpretation: Shows how portfolio performs under various market conditions")
    print("with current weights (no rebalancing)")
    print()

    # ---------------------------------------------------------
    # 4. Historical Stress Testing
    # ---------------------------------------------------------
    print("Step 4: Historical Stress Testing")
    print("-" * 80)
    
    print("Loading long-term historical data for stress tests...")
    
    try:
        # Load maximum available history
        prices_long = load_price_data(tickers, period="max")
        print(f"[OK] Loaded {len(prices_long)} days of historical data")
        print(f"    Date range: {prices_long.index[0].date()} to {prices_long.index[-1].date()}")
        print()
        
        tester = StressTester(prices_long)
        
        # Get summary table
        stress_summary = tester.compute_stress_summary(weights)
        
        if not stress_summary.empty:
            print("Historical Stress Test Summary:")
            print(stress_summary.to_string(index=False))
            print()
            
            # Detailed results
            stress_results = []
            historical_scenarios = StressTester.get_historical_scenarios()
            
            print("\nDetailed Results:")
            print("-" * 80)
            
            for name, (start, end) in historical_scenarios.items():
                res = tester.replay_period(weights, start, end)
                
                if "error" in res:
                    print(f"[SKIP] {name}: {res['error']}")
                    continue
                
                # Add warning for very short periods
                short_period_warning = " (⚠️ Short period)" if res['n_days'] < 60 else ""
                adjusted_warning = " (📅 Dates adjusted)" if res.get('dates_adjusted', False) else ""
                
                print(f"\n{name}{short_period_warning}{adjusted_warning}:")
                print(f"  Period:            {res['period_label']}")
                if res.get('dates_adjusted', False):
                    print(f"    (Requested: {res['original_period']})")
                print(f"  Total Return:      {res['total_return']:8.2%}")
                print(f"  Annualized Return: {res['annualized_return']:8.2%}")
                if res['n_days'] < 60:
                    print(f"    Note: Annualized return over {res['n_days']} days - use with caution")
                print(f"  Max Drawdown:      {res['max_drawdown']:8.2%}")
                print(f"  Volatility:        {res['volatility']:8.2%}")
                print(f"  Sharpe Ratio:      {res['sharpe_ratio']:8.3f}")
                print(f"  Days:              {res['n_days']:8d}")
                
                stress_results.append({
                    "Scenario": name,
                    "Start": start,
                    "End": end,
                    "Total_Return": res['total_return'],
                    "Annualized_Return": res['annualized_return'],
                    "Max_Drawdown": res['max_drawdown'],
                    "Volatility": res['volatility'],
                    "Sharpe_Ratio": res['sharpe_ratio'],
                    "Days": res['n_days']
                })
        else:
            print("[WARNING] No historical scenarios had sufficient data")
            stress_results = []
            
    except Exception as e:
        print(f"[ERROR] Historical stress testing failed: {e}")
        import traceback
        traceback.print_exc()
        stress_results = []
    
    print("-" * 80)
    print()
    
    # ---------------------------------------------------------
    # 5. Saving Results
    # ---------------------------------------------------------
    print("Step 5: Saving Results")
    print("-" * 80)
    
    output_dir = "artifacts/milestone4"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save portfolio weights
    weights_df = pd.DataFrame({
        'Ticker': weights.index,
        'Weight': weights.values
    })
    weights_df.to_csv(os.path.join(output_dir, "portfolio_weights.csv"), index=False)
    print(f"[OK] Portfolio weights saved")
    
    # Save risk metrics
    risk_metrics_df = pd.DataFrame({
        'Metric': ['Parametric VaR', 'Historical VaR', 'CVaR'],
        'Percent': [p_var['var_percent'], h_var['var_percent'], cvar['cvar_percent']],
        'Amount': [p_var['var_amount'], h_var['var_amount'], cvar['cvar_amount']]
    })
    risk_metrics_df.to_csv(os.path.join(output_dir, "risk_metrics.csv"), index=False)
    print(f"[OK] Risk metrics saved")
    
    # Save Component VaR
    comp_var.to_csv(os.path.join(output_dir, "component_var.csv"))
    print(f"[OK] Component VaR saved")
    
    # Save Scenario Analysis Results
    if scenario_results:
        pd.DataFrame(scenario_results).to_csv(
            os.path.join(output_dir, "scenario_analysis.csv"),
            index=False
        )
        print(f"[OK] Scenario analysis saved")
    
    # Save Stress Test Results
    if 'stress_results' in locals() and stress_results:
        pd.DataFrame(stress_results).to_csv(
            os.path.join(output_dir, "historical_stress_test.csv"),
            index=False
        )
        print(f"[OK] Historical stress test results saved")
    
    print()
    print("=" * 80)
    print(f"All results saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()