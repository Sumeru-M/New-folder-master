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
from portfolio.investor_guide import (
    format_risk_summary,
    format_scenario_result,
    format_stress_test_result,
    get_scenario_menu,
    interpret_var
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
        default=10_00_000.0,  # 10 lakhs (₹10,00,000)
        help="Portfolio value for VaR calculations (default: ₹10,00,000)"
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
        pv_input = input(f"Enter portfolio value [default: ₹{args.portfolio_value:,.0f}]: ").strip()
        if pv_input:
            try:
                args.portfolio_value = float(pv_input.replace(',', '').replace('₹', '').replace('$', ''))
            except ValueError:
                print(f"Invalid input, using default: ₹{args.portfolio_value:,.0f}")
        
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
    print(f"Portfolio Value: ₹{args.portfolio_value:,.2f}")
    print(f"Confidence Level: {args.confidence_level:.1%}")
    print(f"Risk-Free Rate: {args.risk_free_rate:.2%}")
    print()

    # ---------------------------------------------------------
    # 1. Base Portfolio Construction
    # ---------------------------------------------------------
    print("Step 1: Building Your Optimal Portfolio")
    print("-" * 80)
    print()
    
    try:
        # Load recent data for optimization (2 years)
        prices_opt = load_price_data(tickers, period="2y")
        print(f"[OK] Loaded {len(prices_opt)} days of price data")
        print(f"    Analyzing: {', '.join(tickers)}")
    except Exception as e:
        print(f"[ERROR] Failed to load price data: {e}")
        return
    
    daily_returns = compute_daily_returns(prices_opt)
    mu = compute_expected_returns(daily_returns)
    sigma = compute_covariance_matrix(daily_returns)
    
    print("\nOptimizing portfolio for best risk-adjusted returns...")
    optimizer = PortfolioOptimizer(mu, sigma, risk_free_rate=args.risk_free_rate)
    result = optimizer.optimize_max_sharpe()
    weights = result.weights
    
    # Get investor-friendly summary
    from portfolio.optimizer import compute_weight_dispersion
    from portfolio.investor_guide import format_portfolio_summary
    
    dispersion = compute_weight_dispersion(weights)
    
    print("\n[OK] Portfolio Optimized!")
    print(format_portfolio_summary(
        result,
        dispersion,
        "YOUR OPTIMIZED PORTFOLIO"
    ))
    
    print("\n📋 RECOMMENDED ALLOCATION:")
    print("   (How much to invest in each stock)")
    print()
    for ticker, weight in weights.items():
        print(f"   {ticker:20s}: {weight:>6.1%} ({weight*100:>5.1f}% of your portfolio)")
    print()
    print("-" * 80)
    print()

    # ---------------------------------------------------------
    # 2. Risk Metrics (VaR / CVaR / Component VaR)
    # ---------------------------------------------------------
    print("Step 2: Understanding Your Portfolio Risk")
    print("-" * 80)
    print()
    
    # Calculate portfolio daily returns
    portfolio_daily_ret = daily_returns.dot(weights)
    
    # Parametric VaR
    p_var = compute_parametric_var(
        weights, mu, sigma,
        confidence_level=args.confidence_level,
        portfolio_value=args.portfolio_value
    )
    
    # Historical VaR
    h_var = compute_historical_var(
        portfolio_daily_ret,
        confidence_level=args.confidence_level,
        portfolio_value=args.portfolio_value
    )
    
    # CVaR (Expected Shortfall)
    cvar = compute_cvar(
        portfolio_daily_ret,
        confidence_level=args.confidence_level,
        portfolio_value=args.portfolio_value
    )
    
    # Use the more conservative (higher) VaR for display
    display_var = p_var if abs(p_var['var_amount']) > abs(h_var['var_amount']) else h_var
    
    # Print investor-friendly risk summary
    print(format_risk_summary(display_var, cvar, args.portfolio_value, args.confidence_level))
    print()
    
    # Component VaR (Risk Contribution by Asset) - simplified
    comp_var = compute_component_var(
        weights, sigma,
        confidence_level=args.confidence_level
    )
    
    print("📊 RISK CONTRIBUTION BY STOCK:")
    print("   (Which stocks contribute most to your portfolio risk)")
    print()
    for ticker, row in comp_var.iterrows():
        print(f"   {ticker:20s}: {row['% Contribution']:>6.1%} of total risk")
    print()
    print("-" * 80)
    print()

    # ---------------------------------------------------------
    # 3. Scenario Selection and Analysis
    # ---------------------------------------------------------
    print("Step 3: Choose a Scenario to Test")
    print("-" * 80)
    print()
    
    engine = ScenarioEngine(mu, sigma)
    base_sharpe = result.sharpe_ratio
    
    # Get scenario menu
    scenario_menu = get_scenario_menu()
    
    # Display menu
    print("📋 AVAILABLE SCENARIOS:")
    print()
    print("HYPOTHETICAL SCENARIOS (What-if situations):")
    hypothetical_count = 0
    for key, info in scenario_menu.items():
        if info['type'] == 'hypothetical':
            hypothetical_count += 1
            print(f"   {key}: {info['name']}")
    
    print()
    print("HISTORICAL SCENARIOS (Real past events):")
    historical_count = 0
    for key, info in scenario_menu.items():
        if info['type'] == 'historical':
            historical_count += 1
            start, end = info['dates']
            print(f"   {key}: {info['name']} ({start} to {end})")
    
    print()
    print("=" * 80)
    
    # Get user selection
    selected_scenario = None
    while True:
        choice = input(f"\nEnter scenario code (H1-H{hypothetical_count} or S1-S{historical_count}) to test: ").strip().upper()
        if choice in scenario_menu:
            selected_scenario = scenario_menu[choice]
            break
        else:
            print(f"Invalid choice. Please enter H1-H{hypothetical_count} or S1-S{historical_count}")
    
    if not selected_scenario:
        print("[ERROR] No scenario selected. Exiting.")
        return
    
    print()
    print("=" * 80)
    print(f"ANALYZING SCENARIO: {selected_scenario['name']}")
    print("=" * 80)
    print()
    
    scenario_results = []
    
    # Process selected scenario
    if selected_scenario['type'] == 'hypothetical':
        # Hypothetical scenario
        shock = selected_scenario['shock']
        new_mu, new_sigma = engine.apply_scenario(shock)
        
        # Evaluate portfolio with FIXED weights under new conditions
        w = weights.values
        port_ret = np.dot(w, new_mu.values)
        port_vol = np.sqrt(np.dot(w, np.dot(new_sigma.values, w)))
        sharpe = (port_ret - args.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        
        # Display investor-friendly result
        print(format_scenario_result(
            selected_scenario['name'],
            port_ret,
            port_vol,
            sharpe,
            base_sharpe,
            args.portfolio_value
        ))
        
        scenario_results.append({
            "Scenario": shock.name,
            "Return": port_ret,
            "Volatility": port_vol,
            "Sharpe": sharpe,
            "Sharpe_Change": sharpe - base_sharpe
        })
    else:
        # Historical scenario - will be processed in Step 4
        pass

    # ---------------------------------------------------------
    # 4. Historical Stress Testing (if historical scenario selected)
    # ---------------------------------------------------------
   # Step 4: Historical Stress Testing (if historical scenario selected)
    stress_results = []

    if selected_scenario and selected_scenario['type'] == 'historical':
            print("Step 4: Loading Historical Data for Stress Test")
    
    # Check if 'dates' key exists in the selected_scenario dictionary
    if 'dates' in selected_scenario:
        start_date, end_date = selected_scenario['dates']
    else:
        print("[ERROR] The selected scenario does not contain 'dates'. Please check the input.")
        return  # Exit or handle the error appropriately
    
    try:
        # Load maximum available history
        prices_long = load_price_data(tickers, period="max")
        print(f"[OK] Loaded {len(prices_long)} days of historical data")
        print(f"    Date range: {prices_long.index[0].date()} to {prices_long.index[-1].date()}")
        print()
        
        tester = StressTester(prices_long)
        
            # Get the selected historical scenario
        scenario_name = selected_scenario['name']
        start_date, end_date = selected_scenario['dates']
            
        print("=" * 80)
        print(f"RUNNING STRESS TEST: {scenario_name}")
        print("=" * 80)
        print()
            
            # Run stress test for selected scenario only
        res = tester.replay_period(weights, start_date, end_date, initial_investment=args.portfolio_value)
                
        if "error" in res:
                print(f"[ERROR] {scenario_name}: {res['error']}")
                print("This scenario may not have sufficient data for your selected stocks.")
        else:
                # Display investor-friendly result
                print(format_stress_test_result(
                    scenario_name,
                    res,
                    args.portfolio_value
                ))
                
                stress_results.append({
                    "Scenario": scenario_name,
                    "Start": start_date,
                    "End": end_date,
                    "Total_Return": res['total_return'],
                    "Annualized_Return": res['annualized_return'],
                    "Max_Drawdown": res['max_drawdown'],
                    "Volatility": res['volatility'],
                    "Sharpe_Ratio": res['sharpe_ratio'],
                    "Days": res['n_days']
                })
            
    except Exception as e:
        print(f"[ERROR] Historical stress testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # ---------------------------------------------------------
    # 5. Saving Results
    # ---------------------------------------------------------
    print("=" * 80)
    print("Saving Results")
    print("=" * 80)
    print()
    
    output_dir = "artifacts/milestone4"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save portfolio weights (investor-friendly)
    weights_df = pd.DataFrame({
        'Stock': weights.index,
        'Allocation (%)': (weights.values * 100).round(2),
        'What This Means': [
            f"Invest {weight*100:.1f}% of your portfolio in {ticker}" 
            for ticker, weight in weights.items()
        ]
    })
    weights_df.to_csv(os.path.join(output_dir, "portfolio_allocation.csv"), index=False)
    print(f"[OK] Portfolio allocation saved")
    
    # Save risk metrics (investor-friendly)
    risk_metrics_df = pd.DataFrame({
        'Risk Metric': [
            'Maximum Daily Loss (Worst Case)',
            'Average Loss (If Bad Day Occurs)'
        ],
        'Percentage': [
            f"{abs(display_var['var_percent']):.2%}",
            f"{abs(cvar['cvar_percent']):.2%}"
        ],
        'Amount (INR)': [
            f"₹{abs(display_var['var_amount']):,.2f}",
            f"₹{abs(cvar['cvar_amount']):,.2f}"
        ],
        'What This Means': [
            f"On {args.confidence_level:.0%} of days, you might lose at most this amount",
            f"If a bad day occurs, expect losses around this amount on average"
        ]
    })
    risk_metrics_df.to_csv(os.path.join(output_dir, "risk_assessment.csv"), index=False)
    print(f"[OK] Risk assessment saved")
    
    # Save Component VaR (simplified)
    comp_var_simple = pd.DataFrame({
        'Stock': comp_var.index,
        'Risk Contribution (%)': (comp_var['% Contribution'] * 100).round(2),
        'What This Means': [
            f"This stock contributes {row['% Contribution']*100:.1f}% to your total portfolio risk"
            for _, row in comp_var.iterrows()
        ]
    })
    comp_var_simple.to_csv(os.path.join(output_dir, "risk_by_stock.csv"), index=False)
    print(f"[OK] Risk by stock saved")
    
    # Save Scenario Analysis Results (if hypothetical)
    if scenario_results:
        scenario_df = pd.DataFrame({
            'Scenario': [scenario_results[0]['Scenario']],
            'Expected Return': [f"{scenario_results[0]['Return']:.2%}"],
            'Volatility': [f"{scenario_results[0]['Volatility']:.2%}"],
            'Performance Score': [f"{scenario_results[0]['Sharpe']:.2f}"],
            'What This Means': [
                f"Your portfolio would have {scenario_results[0]['Return']:.2%} annual return "
                f"if this scenario occurs"
            ]
        })
        scenario_df.to_csv(
            os.path.join(output_dir, "scenario_analysis.csv"),
            index=False
        )
        print(f"[OK] Scenario analysis saved")
    
    # Save Stress Test Results (if historical)
    if stress_results:
        stress_df = pd.DataFrame({
            'Scenario': [stress_results[0]['Scenario']],
            'Period': [f"{stress_results[0]['Start']} to {stress_results[0]['End']}"],
            'Total Return': [f"{stress_results[0]['Total_Return']:.2%}"],
            'Annual Return': [f"{stress_results[0]['Annualized_Return']:.2%}"],
            'Maximum Loss': [f"{stress_results[0]['Max_Drawdown']:.2%}"],
            'What This Means': [
                f"Your portfolio would have {stress_results[0]['Total_Return']:.2%} total return "
                f"during this historical period"
            ]
        })
        stress_df.to_csv(
            os.path.join(output_dir, "stress_test_result.csv"),
            index=False
        )
        print(f"[OK] Stress test result saved")
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 All results saved to: {output_dir}/")
    print()
    print("Files created:")
    print("  • portfolio_allocation.csv - How your portfolio is allocated")
    print("  • risk_assessment.csv - Your portfolio's risk metrics")
    print("  • risk_by_stock.csv - Which stocks contribute most to risk")
    if scenario_results:
        print("  • scenario_analysis.csv - Results for selected scenario")
    if stress_results:
        print("  • stress_test_result.csv - Historical stress test results")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()