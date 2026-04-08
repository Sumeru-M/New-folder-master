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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portfolio.portfolio_complete import (
    load_price_data,
    PortfolioOptimizer,
    compute_daily_returns,
    compute_expected_returns,
    compute_covariance_matrix,
    compare_covariance_methods,
    plot_efficient_frontier,
    format_portfolio_summary,
    format_comparison_summary,
    create_investor_friendly_csv,
    create_simple_weights_csv,
)


def parse_args():
    """Parse command line arguments for dynamic user portfolio input."""
    parser = argparse.ArgumentParser(
        description="Portfolio Optimization - Find the best way to allocate your investments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (interactive mode):
  python run_milestone3.py
  
  # Analyze specific stocks with equal weights:
  python run_milestone3.py --tickers "RELIANCE.NS,TCS.NS,INFY.NS"
  
  # Analyze with your current allocation:
  python run_milestone3.py --tickers "RELIANCE.NS,TCS.NS" --weights "60,40"
  
  # Use 5 years of data with 8% safe investment rate:
  python run_milestone3.py --tickers "ITC.NS,HDFCBANK.NS" --period "5y" --risk_free_rate 8
        """
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        default=None,
        help="Stock symbols to analyze (comma-separated). Example: RELIANCE.NS,TCS.NS,INFY.NS"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Your current allocation as percentages (comma-separated). Example: 30,40,30 means 30%% in first stock, 40%% in second, 30%% in third. Leave blank for equal weights."
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="How much historical data to analyze. Options: 6mo, 1y, 2y, 5y, 10y. Default: 2y (recommended)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Custom start date (YYYY-MM-DD). Use with --end_date to override --period."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Custom end date (YYYY-MM-DD). Use with --start_date to override --period."
    )
    parser.add_argument(
        "--risk_free_rate",
        type=float,
        default=0.07,
        help="Safe investment return rate (as percentage, e.g., 7 for 7%%). This is your benchmark (FD/bond rate). Default: 7%%"
    )
    return parser.parse_args()


def validate_tickers(tickers_list):
    """
    Validate ticker symbols and suggest corrections for common mistakes.
    
    Returns
    -------
    tuple: (valid_tickers, warnings)
    """
    # Common ticker mistakes and corrections
    corrections = {
        'ICICI.NS': 'ICICIBANK.NS',
        'HDFC.NS': 'HDFCBANK.NS',
        'SBI.NS': 'SBIN.NS',
        'AXIS.NS': 'AXISBANK.NS',
        'KOTAK.NS': 'KOTAKBANK.NS',
    }
    
    valid_tickers = []
    warnings = []
    
    for ticker in tickers_list:
        ticker = ticker.strip().upper()
        
        # Check for .NS suffix
        if not ticker.endswith('.NS'):
            warnings.append(f"⚠️  {ticker} doesn't have .NS suffix. For NSE stocks, use {ticker}.NS")
            continue
        
        # Check for common mistakes
        if ticker in corrections:
            correct = corrections[ticker]
            warnings.append(f"❌ {ticker} is invalid. Did you mean {correct}?")
            print(f"   Auto-correcting: {ticker} → {correct}")
            valid_tickers.append(correct)
        else:
            valid_tickers.append(ticker)
    
    return valid_tickers, warnings


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
        
        # Handle risk_free_rate - accept as percentage or decimal
        risk_free_rate = args.risk_free_rate
        if risk_free_rate > 1:  # User entered as percentage (e.g., 7 for 7%)
            risk_free_rate = risk_free_rate / 100
            print(f"✓ Using {args.risk_free_rate}% as safe investment rate")
        else:  # User entered as decimal (e.g., 0.07)
            print(f"✓ Using {risk_free_rate*100:.1f}% as safe investment rate")
    else:
        print("\n=== Portfolio Optimization - Let's Build Your Best Portfolio ===")
        print("\n📈 STEP 1: Which stocks do you want to analyze?")
        print("Enter Indian stock symbols (add .NS for NSE stocks)")
        print("Examples: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ITC.NS")
        print()
        
        tickers_str = input("Enter your stocks (comma-separated): ").strip()
        while not tickers_str:
             print("❌ You need to enter at least one stock symbol.")
             tickers_str = input("Enter your stocks (comma-separated): ").strip()

        print("\n💰 CURRENT PORTFOLIO ALLOCATION (Optional)")
        print("If you already own these stocks, tell us how much of each you have.")
        print("Examples:")
        print("  • '30,40,30' means 30% in first stock, 40% in second, 30% in third")
        print("  • '50,50' means equal 50-50 split between two stocks")
        print("  • Just press Enter to compare equal allocation across all stocks")
        weights_str = input("\nYour current allocation (comma-separated percentages, or press Enter): ").strip() or None
                
        # Date selection
        print("\n📅 STEP 2: How much historical data should we analyze?")
        print("More data = more reliable, but older data may be less relevant")
        print("Recommended: 2 years (good balance)")
        print("Options: 6mo, 1y, 2y, 5y, 10y")
        period_input = input("\nEnter time period [default: 2y]: ").strip()
        period = period_input if period_input else "2y"
        
        # Validate period format
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        # Try to convert common mistakes (e.g., "1" -> "1y", "2" -> "2y")
        if period.isdigit():
            period_num = int(period)
            if period_num <= 10:
                period = f"{period_num}y"
                print(f"✓ Got it! Using {period_num} years of data")
            else:
                print(f"⚠️  '{period_input}' is too long. Using default 2 years")
                period = "2y"
        elif period not in valid_periods:
            print(f"⚠️  '{period}' is not valid. Valid options: {', '.join(valid_periods)}")
            print(f"   Using default: 2 years")
            period = "2y"
        else:
            print(f"✓ Using {period} of historical data")
        
        start_date = None
        end_date = None
        # Custom date range option
        if period.lower() == 'custom':
             print("\n📅 Custom Date Range")
             start_date = input("Enter start date (YYYY-MM-DD): ").strip()
             end_date = input("Enter end date (YYYY-MM-DD): ").strip()
             period = None

        print("\n📊 SAFE INVESTMENT RATE (Benchmark)")
        print("This is the return you'd get from a 'safe' investment like:")
        print("  • Fixed Deposits (FDs): ~6-7% per year")
        print("  • Government Bonds: ~7-8% per year")
        print("  • Savings Account: ~3-4% per year")
        print("\nWe use this to measure if your stock portfolio is worth the extra risk.")
        print("Default is 7% (typical for Indian FDs/bonds)")
        
        rf_input = input("\nEnter safe investment return rate (just the number, e.g., 7 for 7%) [default: 7]: ").strip()
        try:
            if rf_input:
                # User entered a number
                rf_value = float(rf_input)
                # If user entered as percentage (e.g., 7), convert to decimal
                if rf_value > 1:
                    risk_free_rate = rf_value / 100
                    print(f"Using {rf_value}% ({risk_free_rate:.4f}) as safe investment rate")
                else:
                    # User already entered as decimal (e.g., 0.07)
                    risk_free_rate = rf_value
                    print(f"Using {rf_value*100}% as safe investment rate")
            else:
                risk_free_rate = 0.07  # Changed default to 7% for India
                print("Using default: 7% safe investment rate")
        except ValueError:
            print("Invalid number. Using default 7% safe investment rate.")
            risk_free_rate = 0.07

    # Parse tickers and optional weights
    tickers_raw = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if not tickers_raw:
        print("[ERROR] No tickers provided.")
        return
    
    # Validate and auto-correct tickers
    print("\nValidating ticker symbols...")
    tickers, warnings = validate_tickers(tickers_raw)
    
    if warnings:
        for warning in warnings:
            print(warning)
        print()
    
    if not tickers:
        print("❌ ERROR: No valid tickers after validation.")
        return
    
    if len(tickers) < 2:
        print(f"❌ ERROR: Portfolio optimization requires at least 2 stocks.")
        print(f"   You provided: {', '.join(tickers)}")
        return
    
    print(f"✓ Using tickers: {', '.join(tickers)}\n")

    weights = None
    if weights_str:
        parts = [p.strip() for p in weights_str.split(",") if p.strip()]
        if len(parts) != len(tickers):
            print(f"\n❌ ERROR: You entered {len(parts)} weights but have {len(tickers)} stocks.")
            print(f"   Please enter {len(tickers)} weights (one for each stock).")
            return
        try:
            weights = np.array([float(p) for p in parts], dtype=float)
            
            # If user entered percentages (e.g., 30, 40, 30), convert to decimals
            if weights.sum() > 1.5:  # Likely percentages
                print(f"\n✓ Interpreting as percentages: {', '.join([f'{w:.1f}%' for w in weights])}")
                weights = weights / 100
            
            if weights.sum() <= 0:
                raise ValueError("Weights must be positive numbers.")
            
            # Normalize weights to sum to 1
            original_sum = weights.sum()
            weights = weights / weights.sum()
            
            if abs(original_sum - 1.0) > 0.01:  # If not already close to 100%
                print(f"✓ Normalized weights to 100%: {', '.join([f'{w:.1%}' for w in weights])}")
            else:
                print(f"✓ Using weights: {', '.join([f'{w:.1%}' for w in weights])}")
                
        except Exception as e:
            print(f"\n❌ ERROR: Invalid weights - {e}")
            print("   Weights should be numbers like: 30,40,30 or 0.3,0.4,0.3")
            return
    else:
        weights = np.ones(len(tickers)) / len(tickers)
        print(f"\n✓ Using equal weights: {', '.join([f'{w:.1%}' for w in weights])}")


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
        
        # VALIDATION: Check for columns with excessive NaN values
        print("Validating data quality...")
        nan_counts = prices.isna().sum()
        total_rows = len(prices)
        problematic_tickers = []
        
        for ticker in prices.columns:
            nan_pct = (nan_counts[ticker] / total_rows) * 100
            if nan_pct > 50:  # More than 50% missing data
                problematic_tickers.append((ticker, nan_pct))
        
        if problematic_tickers:
            print("\n⚠️  WARNING: Some tickers have insufficient data:")
            for ticker, pct in problematic_tickers:
                print(f"   • {ticker}: {pct:.1f}% missing data (likely invalid or delisted)")
            
            # Remove problematic tickers
            good_tickers = [t for t in prices.columns if t not in [x[0] for x in problematic_tickers]]
            
            if len(good_tickers) == 0:
                print("\n❌ ERROR: No valid tickers with sufficient data.")
                print("\nPlease check your ticker symbols. Common NSE tickers:")
                print("  Banks: ICICIBANK.NS, HDFCBANK.NS, SBIN.NS, AXISBANK.NS")
                print("  IT: TCS.NS, INFY.NS, WIPRO.NS, HCLTECH.NS")
                print("  Others: RELIANCE.NS, ITC.NS, HINDUNILVR.NS, BHARTIARTL.NS")
                return
            
            if len(good_tickers) < 2:
                print(f"\n❌ ERROR: Only {len(good_tickers)} valid ticker(s) found.")
                print("   Portfolio optimization requires at least 2 stocks.")
                return
            
            print(f"\n✓ Proceeding with {len(good_tickers)} valid ticker(s): {', '.join(good_tickers)}")
            prices = prices[good_tickers]
            
            # Update tickers list and weights
            tickers = good_tickers
            if len(weights) != len(tickers):
                print(f"⚠️  Adjusting to equal allocation for {len(tickers)} valid stocks")
                weights = np.ones(len(tickers)) / len(tickers)
                print(f"   New weights: {', '.join([f'{w:.1%}' for w in weights])}")
        
        # Verify we have actual valid data points
        valid_data = prices.dropna()
        if valid_data.empty:
            print("\n❌ ERROR: No valid price data after removing missing values.")
            print("This usually means:")
            print("  1. Invalid ticker symbols (check NSE website)")
            print("  2. Tickers are delisted")
            print("  3. Data for this period is not available")
            return
        
        if len(valid_data) < 50:
            print(f"\n⚠️  WARNING: Only {len(valid_data)} valid data points.")
            print("   Optimization results may be unreliable with limited data.")
            print("   Recommendation: Use a longer time period or check ticker validity.")

        print(f"[OK] Data loaded successfully")
        print(f"  Shape: {prices.shape}")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"  Tickers with data: {', '.join(prices.columns.tolist())}")
        print(f"  Valid data points: {len(valid_data)} days")
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
        from portfolio.portfolio_complete import compute_weight_dispersion
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
        from portfolio.portfolio_complete import compute_weight_dispersion
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

    # Step 8: Institutional Risk Analysis (New Step)
    print("-" * 80)
    print("Step 8: Institutional Risk Analysis")
    print("-" * 80)
    print()
    
    try:
        # Import new modules here (or at top)
        # Assuming imports are added at top, but for safety in this block:
        from portfolio.portfolio_complete import (
            FactorModel,
            compute_portfolio_risk_metrics,
            compute_max_drawdown,
            compute_ulcer_index,
            detect_market_regime,
            plot_correlation_heatmap,
            plot_drawdown_chart,
        )

        print("[OK] Analyzing Maximum Sharpe Portfolio for institutional metrics...")
        
        # Use Max Sharpe weights for analysis (most common recommendation)
        # Ensure we have the returns for just the assets in the portfolio
        # daily_returns already contains asset returns
        
        # 1. Factor Analysis (vs Nifty)
        # We need Nifty 50 data. It wasn't loaded in Step 1 unless requested.
        # Let's try to fetch it now if not present, or skip if unavailable.
        try:
            index_ticker = "^NSEI"
            if index_ticker not in prices.columns:
                print(f"    Fetching {index_ticker} for factor analysis...")
                idx_data = load_price_data([index_ticker], period=period if not (start_date and end_date) else None, start_date=start_date, end_date=end_date)
                idx_returns = compute_daily_returns(idx_data)
            else:
                 # If user included ^NSEI in their tickers
                idx_returns = daily_returns[[index_ticker]]
            
            # Align dates
            common_idx = daily_returns.index.intersection(idx_returns.index)
            asset_ret_aligned = daily_returns.loc[common_idx]
            factor_ret_aligned = idx_returns.loc[common_idx]
            
            fm = FactorModel(factor_ret_aligned)
            # Use max_sharpe_weights
            max_sharpe_series = max_sharpe_result.weights
            
            decomp = fm.decompose_portfolio_risk(max_sharpe_series, asset_ret_aligned)
            
            print(f"\n    🔍 FACTOR ANALYSIS (What drives your risk?):")
            print(f"      • Market Risk (Systematic):  {decomp['systematic_volatility']:.2%} (Explains {decomp['r_squared']:.1%} of price moves)")
            print(f"        (Risk from the overall market moving up/down - cannot be diversified away)")
            print(f"      • Specific Risk (Unique):    {decomp['idiosyncratic_volatility']:.2%}")
            print(f"        (Risk unique to your selected stocks - can be reduced by diversifying)")
            
            # Regime Analysis
            regime = detect_market_regime(factor_ret_aligned.iloc[:, 0])
            print(f"\n    🌍 MARKET REGIME (Current Environment):")
            print(f"      • Status:              {regime['regime'].upper()}")
            print(f"      • Volatility Level:    {regime['current_vol_annualized']:.2%}")

        except Exception as e:
            print(f"    [WARNING] Factor analysis skipped: {e}")

        # 2. Advanced Risk Metrics
        # Calculate portfolio daily returns series
        port_daily_ret = (daily_returns * max_sharpe_result.weights).sum(axis=1)
        
        # VaR / CVaR
        risk_stats = compute_portfolio_risk_metrics(
            max_sharpe_result.weights, 
            port_daily_ret, 
            expected_returns, 
            covariance_matrix
        )
        
        print(f"\n    🛡️ DOWNSIDE RISK METRICS (What's the worst case?):")
        print(f"      • Max Likely Daily Loss (VaR 95%):   {risk_stats['parametric_var']['var_percent']:.2%}")
        print(f"        (On 95 out of 100 days, your loss won't exceed this)")
        print(f"      • Avg Loss on Bad Days (CVaR):       {risk_stats['cvar']['cvar_percent']:.2%}")
        print(f"        (If a bad day happens, this is the average loss expected)")
        
        # Drawdown & Ulcer Index
        dd_stats = compute_max_drawdown(port_daily_ret)
        ulcer_idx = compute_ulcer_index(port_daily_ret)
        
        print(f"\n    📉 HISTORICAL STRESS (Based on past performance):")
        print(f"      • Worst Drop (Max Drawdown):         {dd_stats['max_drawdown']:.2%}")
        print(f"        (The biggest drop from peak to bottom in the analyzed period)")
        print(f"      • Recovery Time:                     {dd_stats['max_drawdown_duration_days']} days")
        print(f"        (Time taken to recover from the worst drop)")
        print(f"      • Ulcer Index (Pain Score):          {ulcer_idx:.4f}")
        print(f"        (Measures the depth and duration of drawdowns; lower is better)")

        
        # 3. Visualizations
        print(f"\n    Generating additional visual reports...")
        
        # Heatmap
        plot_correlation_heatmap(
            daily_returns.corr(), 
            title="Asset Correlation Matrix",
            save_path=os.path.join(artifacts_dir, "correlation_heatmap.png")
        )
        print(f"      [OK] Heatmap saved: correlation_heatmap.png")
        
        # Drawdown Chart
        # Calculate cumulative returns and drawdowns for plotting
        cum_ret = (1 + port_daily_ret).cumprod()
        wealth_index = (1 + port_daily_ret).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        plot_drawdown_chart(
            cum_ret, 
            drawdowns, 
            title="Max Sharpe Portfolio Drawdown Analysis",
            save_path=os.path.join(artifacts_dir, "drawdown_analysis.png")
        )
        print(f"      [OK] Drawdown chart saved: drawdown_analysis.png")
        
        # Save Report CSV
        risk_report = pd.DataFrame({
            "Metric": ["Annual Return", "Annual Volatility", "Sharpe", "Max Drawdown", "VaR (95%)", "CVaR (95%)"],
            "Value": [
                f"{max_sharpe_result.expected_return:.2%}",
                f"{max_sharpe_result.volatility:.2%}",
                f"{max_sharpe_result.sharpe_ratio:.2f}",
                f"{dd_stats['max_drawdown']:.2%}",
                f"{risk_stats['parametric_var']['var_percent']:.2%}",
                f"{risk_stats['cvar']['cvar_percent']:.2%}"
            ]
        })
        risk_report.to_csv(os.path.join(artifacts_dir, "institutional_risk_report.csv"), index=False)
        print(f"      [OK] Risk report saved: institutional_risk_report.csv")

    except Exception as e:
        print(f"[ERROR] Failed to run institutional analysis: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("✅ Analysis complete! Check the CSV files for detailed recommendations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
