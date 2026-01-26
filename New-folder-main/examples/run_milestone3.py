"""
Milestone 3: Portfolio Optimization Example

This script demonstrates how to use the portfolio optimization engine
to compute optimal portfolios using Markowitz mean-variance optimization.

Usage:
    python examples/run_milestone3.py
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
    compute_covariance_matrix,
    compare_covariance_methods
)
from portfolio.plotting import plot_efficient_frontier
from portfolio.investor_guide import (
    format_portfolio_summary,
    format_comparison_summary,
    create_investor_friendly_csv,
    create_simple_weights_csv
)


def parse_args():
    """Parse command line arguments for dynamic user portfolio input."""
    parser = argparse.ArgumentParser(description="Milestone 3 Portfolio Optimization")
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        default=None,
        help="Comma-separated list of tickers (e.g., RELIANCE.NS,TCS.NS,INFY.NS). If omitted, interactive mode is used."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated weights matching tickers (e.g., 0.3,0.4,0.3). If omitted, equal weights are used."
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Data period (e.g., 6mo,1y,2y,5y) if start/end not provided."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides period if provided with end_date."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Overrides period if provided with start_date."
    )
    parser.add_argument(
        "--risk_free_rate",
        type=float,
        default=0.05,
        help="Annual risk-free rate (e.g., 0.05 for 5%%)."
    )
    return parser.parse_args()


def compute_portfolio_point(weights: np.ndarray, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_free_rate: float):
    """Compute portfolio return, volatility, Sharpe for given weights."""
    mu = expected_returns.values
    Sigma = covariance_matrix.values
    w = weights
    port_return = float(np.dot(w, mu))
    port_vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0
    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe
    }


def main():
    """Main function to run portfolio optimization example."""
    args = parse_args()

    # Interactive Input Handling
    if args.tickers:
        tickers_str = args.tickers
        weights_str = args.weights
        period = args.period
        start_date = args.start_date
        end_date = args.end_date
        risk_free_rate = args.risk_free_rate
    else:
        print("\n=== Portfolio Optimization Interactive Mode ===")
        print("Please enter the following details:\n")
        
        tickers_str = input("Enter tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS): ").strip()
        while not tickers_str:
             print("Tickers are required.")
             tickers_str = input("Enter tickers (comma-separated): ").strip()

        weights_str = input("Enter weights (optional, comma-separated, press Enter for equal weights): ").strip() or None
        
        # Date selection
        period_input = input("Enter data period (e.g., 2y, 1y, 6mo) [default: 2y]: ").strip()
        period = period_input if period_input else "2y"
        
        # Validate period format
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        # Try to convert common mistakes (e.g., "1" -> "1y", "2" -> "2y")
        if period.isdigit():
            period_num = int(period)
            if period_num <= 10:
                period = f"{period_num}y"
                print(f"Interpreting '{period_input}' as '{period}'")
            else:
                print(f"Warning: '{period_input}' is not a valid period. Using default '2y'")
                period = "2y"
        elif period not in valid_periods:
            print(f"Warning: '{period}' is not a valid period. Valid periods: {', '.join(valid_periods)}")
            print(f"Using default '2y'")
            period = "2y"
        
        start_date = None
        end_date = None
        # Simple heuristic: if user typed a date format like YYYY-MM-DD instead of a period code, handle that?
        # For now, let's keep it simple. If they want custom dates, they might need to use CLI or I can add a prompt.
        # Let's add a quick check if they want custom dates.
        if period.lower() == 'custom':
             start_date = input("Enter start date (YYYY-MM-DD): ").strip()
             end_date = input("Enter end date (YYYY-MM-DD): ").strip()
             period = None

        rf_input = input("Enter risk-free rate (decimal, e.g., 0.05 for 5%) [default: 0.05]: ").strip()
        try:
            risk_free_rate = float(rf_input) if rf_input else 0.05
        except ValueError:
            print("Invalid number for risk-free rate. Using default 0.05.")
            risk_free_rate = 0.05

    # Parse tickers and optional weights
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        print("[ERROR] No tickers provided.")
        return

    weights = None
    if weights_str:
        parts = [p.strip() for p in weights_str.split(",") if p.strip()]
        if len(parts) != len(tickers):
            print(f"[ERROR] Number of weights ({len(parts)}) must match number of tickers ({len(tickers)}).")
            return
        try:
            weights = np.array([float(p) for p in parts], dtype=float)
            if weights.sum() <= 0:
                raise ValueError("Weights must sum to a positive number.")
            weights = weights / weights.sum()
        except Exception as e:
            print(f"[ERROR] Invalid weights: {e}")
            return
    else:
        weights = np.ones(len(tickers)) / len(tickers)

    print("=" * 80)
    print("Milestone 3: Portfolio Optimization Engine")
    print("=" * 80)
    print()
    print(f"Portfolio Tickers: {', '.join(tickers)}")
    print(f"Weights: {weights}")
    print(f"Risk-free Rate: {risk_free_rate:.2%}")
    if start_date and end_date:
        print(f"Date Range: {start_date} to {end_date}")
    else:
        print(f"Data Period: {period}")
    print()
    
    # Step 1: Load price data
    print("-" * 80)
    print("Step 1: Loading Price Data")
    print("-" * 80)
    print()
    
    try:
        prices = load_price_data(
            tickers,
            period=period if not (start_date and end_date) else None,
            start_date=start_date,
            end_date=end_date
        )
        if prices.empty:
            print("[ERROR] No data loaded.")
            print("Possible issues:")
            print("  - Ticker symbols may be incorrect (ensure .NS suffix for NSE stocks)")
            print("  - Tickers may be delisted or not available")
            print("  - Period may be invalid or too short")
            print("  - Network connection issues")
            print(f"\nTried to fetch: {', '.join(tickers)}")
            if start_date and end_date:
                print(f"Date range: {start_date} to {end_date}")
            else:
                print(f"Period: {period}")
            return

        print(f"[OK] Data loaded successfully")
        print(f"  Shape: {prices.shape}")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"  Tickers with data: {', '.join(prices.columns.tolist())}")
        print()
    except ValueError as e:
        print(f"[ERROR] Failed to load data: {e}")
        print("\nTroubleshooting tips:")
        print("  - Verify ticker symbols are correct (e.g., RELIANCE.NS, TCS.NS, INFY.NS)")
        print("  - Try a longer period (e.g., '2y' or '5y')")
        print("  - Check your internet connection")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Compute daily returns
    print("-" * 80)
    print("Step 2: Computing Daily Returns")
    print("-" * 80)
    print()
    
    daily_returns = compute_daily_returns(prices)
    print(f"[OK] Daily returns computed")
    print(f"  Shape: {daily_returns.shape}")
    print(f"  Sample returns:")
    print(daily_returns.head().round(4))
    print()
    
    # Step 3: Compute expected returns and covariance matrix
    print("-" * 80)
    print("Step 3: Computing Expected Returns and Covariance Matrix")
    print("-" * 80)
    print()
    
    expected_returns = compute_expected_returns(daily_returns, annualized=True)
    covariance_matrix = compute_covariance_matrix(daily_returns, annualized=True)
    
    print("[OK] Expected annual returns:")
    for ticker, ret in expected_returns.items():
        print(f"  {ticker:20s}: {ret:8.2%}")
    print()
    
    print("[OK] Covariance matrix (annualized):")
    print(covariance_matrix.round(6))
    print()
    
    # Step 4: Initialize optimizer
    print("-" * 80)
    print("Step 4: Portfolio Optimization")
    print("-" * 80)
    print()
    
    optimizer = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate
    )
    
    # Step 4a: Minimum Variance Portfolio
    print("4a. Computing Lowest Risk Portfolio...")
    try:
        min_var_result = optimizer.optimize_min_variance()
        print(f"[OK] Lowest Risk Portfolio Found!")
        print()
        
        # Get dispersion metrics for investor-friendly output
        from portfolio.optimizer import compute_weight_dispersion
        minvar_dispersion = compute_weight_dispersion(min_var_result.weights)
        
        # Print investor-friendly summary
        print(format_portfolio_summary(
            min_var_result,
            minvar_dispersion,
            "LOWEST RISK PORTFOLIO"
        ))
        
        print("\n📋 RECOMMENDED ALLOCATION:")
        print("   (How much to invest in each stock)")
        print()
        for ticker, weight in min_var_result.weights.items():
            print(f"   {ticker:20s}: {weight:>6.1%} ({weight*100:>5.1f}% of your portfolio)")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to compute minimum variance portfolio: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4b: Maximum Sharpe Ratio Portfolio
    print("4b. Computing Best Risk-Adjusted Returns Portfolio...")
    try:
        max_sharpe_result = optimizer.optimize_max_sharpe()
        print(f"[OK] Best Risk-Adjusted Returns Portfolio Found!")
        print()
        
        # Get dispersion metrics for investor-friendly output
        from portfolio.optimizer import compute_weight_dispersion
        maxsharpe_dispersion = compute_weight_dispersion(max_sharpe_result.weights)
        
        # Print investor-friendly summary
        print(format_portfolio_summary(
            max_sharpe_result,
            maxsharpe_dispersion,
            "BEST RISK-ADJUSTED RETURNS PORTFOLIO"
        ))
        
        print("\n📋 RECOMMENDED ALLOCATION:")
        print("   (How much to invest in each stock)")
        print()
        for ticker, weight in max_sharpe_result.weights.items():
            print(f"   {ticker:20s}: {weight:>6.1%} ({weight*100:>5.1f}% of your portfolio)")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to compute maximum Sharpe portfolio: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4c: Robust Portfolio Optimization - Sample vs Shrinkage Covariance
    print("-" * 80)
    print("Step 4c: Robust Portfolio Optimization (Sample vs Shrinkage Covariance)")
    print("-" * 80)
    print()
    
    comparison_results = {}
    
    # Compare Minimum Variance portfolios
    print("Comparing Lowest Risk Portfolios (Standard vs Robust Method)...")
    try:
        comparison_minvar = compare_covariance_methods(
            returns=daily_returns,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate,
            optimization_type="min_variance"
        )
        comparison_results['min_variance'] = comparison_minvar
        
        print(format_comparison_summary(comparison_minvar, "min_variance"))
        
        print("\n📋 RECOMMENDED ALLOCATION (Robust Method - More Stable):")
        print()
        for ticker, weight in comparison_minvar.shrinkage_result.weights.items():
            print(f"   {ticker:20s}: {weight:>6.1%} ({weight*100:>5.1f}% of your portfolio)")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to compare minimum variance portfolios: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare Maximum Sharpe portfolios
    print("Comparing Best Returns Portfolios (Standard vs Robust Method)...")
    try:
        comparison_maxsharpe = compare_covariance_methods(
            returns=daily_returns,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate,
            optimization_type="max_sharpe"
        )
        comparison_results['max_sharpe'] = comparison_maxsharpe
        
        print(format_comparison_summary(comparison_maxsharpe, "max_sharpe"))
        
        print("\n📋 RECOMMENDED ALLOCATION (Robust Method - More Stable):")
        print()
        for ticker, weight in comparison_maxsharpe.shrinkage_result.weights.items():
            print(f"   {ticker:20s}: {weight:>6.1%} ({weight*100:>5.1f}% of your portfolio)")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to compare maximum Sharpe portfolios: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Compute Efficient Frontier
    print("-" * 80)
    print("Step 5: Computing Efficient Frontier")
    print("-" * 80)
    print()
    
    try:
        print("Computing efficient frontier points...")
        ef_returns, ef_volatilities, ef_sharpe_ratios = optimizer.compute_efficient_frontier(n_points=50)
        print(f"[OK] Efficient frontier computed with {len(ef_returns)} points")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to compute efficient frontier: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Prepare individual asset data for plotting
    individual_assets = pd.DataFrame({
        'return': expected_returns,
        'volatility': np.sqrt(np.diag(covariance_matrix))
    })

    # Compute user portfolio point
    user_portfolio_point = compute_portfolio_point(weights, expected_returns, covariance_matrix, risk_free_rate)
    user_portfolio_point['label'] = "User Portfolio"
    
    # Step 7: Plot Efficient Frontier
    print("-" * 80)
    print("Step 6: Plotting Efficient Frontier")
    print("-" * 80)
    print()
    
    # Create artifacts directory if it doesn't exist
    artifacts_dir = "artifacts/milestone3"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    try:
        fig = plot_efficient_frontier(
            returns=ef_returns,
            volatilities=ef_volatilities,
            sharpe_ratios=ef_sharpe_ratios,
            min_var_result=min_var_result,
            max_sharpe_result=max_sharpe_result,
            individual_assets=individual_assets,
            user_portfolio={
                "expected_return": user_portfolio_point["expected_return"],
                "volatility": user_portfolio_point["volatility"],
                "label": user_portfolio_point["label"]
            },
            risk_free_rate=risk_free_rate,
            save_path=os.path.join(artifacts_dir, "efficient_frontier.png")
        )
        print(f"[OK] Efficient frontier plot saved to: {artifacts_dir}/efficient_frontier.png")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to plot efficient frontier: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 8: Save results to CSV
    print("-" * 80)
    print("Step 7: Saving Results")
    print("-" * 80)
    print()
    
    try:
        # Save minimum variance portfolio (investor-friendly format)
        min_var_df = create_simple_weights_csv(min_var_result, "Lowest Risk Portfolio")
        min_var_df.to_csv(
            os.path.join(artifacts_dir, "lowest_risk_portfolio_allocation.csv"),
            index=False
        )
        print(f"[OK] Lowest risk portfolio allocation saved")
        
        # Save maximum Sharpe portfolio (investor-friendly format)
        max_sharpe_df = create_simple_weights_csv(max_sharpe_result, "Best Risk-Adjusted Returns Portfolio")
        max_sharpe_df.to_csv(
            os.path.join(artifacts_dir, "best_returns_portfolio_allocation.csv"),
            index=False
        )
        print(f"[OK] Best returns portfolio allocation saved")
        
        # Save efficient frontier data
        ef_df = pd.DataFrame({
            'expected_return': ef_returns,
            'volatility': ef_volatilities,
            'sharpe_ratio': ef_sharpe_ratios
        })
        ef_df.to_csv(
            os.path.join(artifacts_dir, "efficient_frontier.csv"),
            index=False
        )
        print(f"[OK] Efficient frontier data saved")
        
        # Save comparison results if available
        if comparison_results:
            print()
            print("Saving comparison results...")
            
            # Save minimum variance comparison (investor-friendly format)
            if 'min_variance' in comparison_results:
                comp = comparison_results['min_variance']
                
                # Create investor-friendly comparison
                comparison_df = create_investor_friendly_csv(comp, "min_variance")
                comparison_df.to_csv(
                    os.path.join(artifacts_dir, "lowest_risk_comparison.csv"),
                    index=False
                )
                
                # Save recommended weights (robust method)
                recommended_weights = create_simple_weights_csv(
                    comp.shrinkage_result,
                    "Robust Method - Lowest Risk"
                )
                recommended_weights.to_csv(
                    os.path.join(artifacts_dir, "lowest_risk_recommended_allocation.csv"),
                    index=False
                )
                
                print(f"[OK] Lowest risk comparison saved")
            
            # Save maximum Sharpe comparison (investor-friendly format)
            if 'max_sharpe' in comparison_results:
                comp = comparison_results['max_sharpe']
                
                # Create investor-friendly comparison
                comparison_df = create_investor_friendly_csv(comp, "max_sharpe")
                comparison_df.to_csv(
                    os.path.join(artifacts_dir, "best_returns_comparison.csv"),
                    index=False
                )
                
                # Save recommended weights (robust method)
                recommended_weights = create_simple_weights_csv(
                    comp.shrinkage_result,
                    "Robust Method - Best Returns"
                )
                recommended_weights.to_csv(
                    os.path.join(artifacts_dir, "best_returns_recommended_allocation.csv"),
                    index=False
                )
                
                print(f"[OK] Best returns comparison saved")
        
        print()
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("=" * 80)
    print("📊 FINAL SUMMARY - YOUR PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    
    print("🎯 QUICK COMPARISON:")
    print()
    print(f"Your Current Portfolio:")
    print(f"  • Expected Return: {user_portfolio_point['expected_return']:.2%} per year")
    print(f"  • Risk Level: {user_portfolio_point['volatility']:.2%} volatility")
    print(f"  • Performance Score: {user_portfolio_point['sharpe']:.2f}")
    print()
    
    print(f"💡 RECOMMENDED OPTIONS:")
    print()
    print(f"Option 1: Lowest Risk Portfolio")
    print(f"  • Expected Return: {min_var_result.expected_return:.2%} per year")
    print(f"  • Risk Level: {min_var_result.volatility:.2%} volatility")
    print(f"  • Performance Score: {min_var_result.sharpe_ratio:.2f}")
    print(f"  • Best for: Conservative investors who want to minimize risk")
    print()
    
    print(f"Option 2: Best Risk-Adjusted Returns Portfolio")
    print(f"  • Expected Return: {max_sharpe_result.expected_return:.2%} per year")
    print(f"  • Risk Level: {max_sharpe_result.volatility:.2%} volatility")
    print(f"  • Performance Score: {max_sharpe_result.sharpe_ratio:.2f}")
    print(f"  • Best for: Investors seeking the best balance of risk and return")
    print()
    
    if comparison_results:
        print("💎 RECOMMENDATION:")
        print("   We've compared two methods (Standard vs Robust) and recommend")
        print("   using the Robust Method allocations for more stable and reliable results.")
        print()
    
    print("📁 ALL RESULTS SAVED:")
    print(f"   Location: {artifacts_dir}/")
    print()
    print("   Files created:")
    print("   • lowest_risk_portfolio_allocation.csv - How to allocate for lowest risk")
    print("   • best_returns_portfolio_allocation.csv - How to allocate for best returns")
    if comparison_results:
        print("   • lowest_risk_comparison.csv - Detailed comparison (lowest risk)")
        print("   • best_returns_comparison.csv - Detailed comparison (best returns)")
        print("   • *_recommended_allocation.csv - Recommended allocations (robust method)")
    print()
    print("=" * 80)
    print("✅ Analysis complete! Check the CSV files for detailed recommendations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
