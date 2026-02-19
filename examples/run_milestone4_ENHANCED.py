"""
Milestone 4: Enhanced Scenario Analysis & Risk Testing

ENHANCED VERSION - Focuses on Forward-Looking Hypothetical Scenarios
====================================================================

This version removes historical stress testing (backward-looking) and provides
enhanced hypothetical scenario analysis with empirical calibration based on:

1. Historical Indian market behavior (Nifty 50, sectoral indices)
2. Empirically observed volatility patterns during stress
3. Correlation behavior during different market regimes
4. Recovery time analysis from past events

Each scenario is calibrated using actual historical data but applied as a
forward-looking "what-if" stress test rather than replaying past periods.

WHY THIS IS BETTER:
- Historical stress testing shows what DID happen (may not repeat)
- Hypothetical scenarios show what COULD happen (more predictive)
- Calibrated to Indian market empirics (not just guesswork)
- Multiple scenarios can be tested quickly
- Clear actionable insights

Changes from Original:
- ✅ Removed all historical stress testing code
- ✅ Enhanced scenario library with 13 empirically calibrated scenarios
- ✅ Added Indian market-specific scenarios (RBI policy, monsoon, geopolitical)
- ✅ Detailed calibration notes for each scenario
- ✅ Multi-scenario testing capability
- ✅ Improved impact reporting with recovery estimates
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

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
from portfolio.risk_metrics import (
    compute_parametric_var,
    compute_historical_var,
    compute_cvar,
    compute_component_var
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Milestone 4: Enhanced Scenario Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode:
  python run_milestone4.py
  
  # Command line with specific stocks:
  python run_milestone4.py --tickers "TCS.NS,ICICIBANK.NS,INFY.NS"
  
  # With custom settings:
  python run_milestone4.py --portfolio-value 2000000 --risk-free-rate 0.07
        """
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list"
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=10_00_000.0,
        help="Portfolio value in INR (default: ₹10,00,000)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="VaR confidence level (default: 0.95)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.07,
        help="Risk-free rate (default: 0.07 = 7%%)"
    )
    return parser.parse_args()


def get_enhanced_scenarios():
    """
    Enhanced scenario library with empirical calibration.
    
    Each scenario includes:
    - Empirically calibrated shock parameters
    - Historical basis for calibration
    - Likelihood estimates
    - Duration and recovery estimates
    - India-specific context
    """
    return {
        # ==============================================================
        # SEVERE CRISIS SCENARIOS
        # ==============================================================
        "CRISIS_1": {
            "code": "CRISIS_1",
            "name": "Global Financial Crisis",
            "category": "🔴 SEVERE",
            "description": "Severe global credit crisis with liquidity freeze",
            "shock": MarketShock(
                name="Global Financial Crisis",
                return_shock=-0.45,      # Nifty fell 52% in 2008
                volatility_shock=2.5,     # VIX India spiked 150%
                correlation_shock=0.40    # Correlations → 0.85+
            ),
            "calibration": "2008: Nifty -52%, Vol 18%→45%, Corr 0.45→0.85",
            "likelihood": "Very Rare (1-2% annual)",
            "duration": "12-18 months",
            "recovery": "24-36 months",
            "impact": "40-50% portfolio decline"
        },
        
        "CRISIS_2": {
            "code": "CRISIS_2",
            "name": "Black Swan Pandemic/War",
            "category": "🔴 SEVERE",
            "description": "Unexpected catastrophic event",
            "shock": MarketShock(
                name="Black Swan Event",
                return_shock=-0.40,
                volatility_shock=2.8,
                correlation_shock=0.45
            ),
            "calibration": "COVID Mar 2020: Nifty -38% in 1 month, Vol tripled",
            "likelihood": "Extremely Rare (<1% annual)",
            "duration": "1 month crash, 6 months total",
            "recovery": "6-12 months (if strong policy response)",
            "impact": "35-40% rapid drop"
        },
        
        # ==============================================================
        # MODERATE SCENARIOS
        # ==============================================================
        "MOD_1": {
            "code": "MOD_1",
            "name": "India Economic Slowdown",
            "category": "🟠 MODERATE",
            "description": "GDP slowdown, policy uncertainty",
            "shock": MarketShock(
                name="India Economic Slowdown",
                return_shock=-0.22,
                volatility_shock=1.5,
                correlation_shock=0.18
            ),
            "calibration": "2013 Taper Tantrum, 2018 NBFC crisis: -12% to -18%",
            "likelihood": "Moderate (10-15% annual)",
            "duration": "6-12 months",
            "recovery": "12-18 months",
            "impact": "15-25% decline"
        },
        
        "MOD_2": {
            "code": "MOD_2",
            "name": "US Fed Rate Shock",
            "category": "🟠 MODERATE",
            "description": "Fed tightening, FII outflows",
            "shock": MarketShock(
                name="Fed Rate Hikes",
                return_shock=-0.18,
                volatility_shock=1.4,
                correlation_shock=0.12
            ),
            "calibration": "2022 Fed hikes: Nifty -15% to -20%",
            "likelihood": "Moderate (15-20% annual)",
            "duration": "3-9 months",
            "recovery": "6-12 months",
            "impact": "12-20% decline"
        },
        
        "MOD_3": {
            "code": "MOD_3",
            "name": "Banking/NBFC Crisis",
            "category": "🟠 MODERATE",
            "description": "Major bank failure, credit crunch",
            "shock": MarketShock(
                name="Banking Crisis",
                return_shock=-0.28,
                volatility_shock=1.7,
                correlation_shock=0.22
            ),
            "calibration": "IL&FS 2018, Yes Bank 2020: -15% to -25%, BankNifty -40%",
            "likelihood": "Low-Moderate (5-8% annual)",
            "duration": "6-18 months",
            "recovery": "18-30 months",
            "impact": "20-30% decline"
        },
        
        # ==============================================================
        # MILD SCENARIOS
        # ==============================================================
        "MILD_1": {
            "code": "MILD_1",
            "name": "Healthy Market Correction",
            "category": "🟡 MILD",
            "description": "Normal profit booking",
            "shock": MarketShock(
                name="Market Correction",
                return_shock=-0.11,
                volatility_shock=1.25,
                correlation_shock=0.06
            ),
            "calibration": "Typical 10-12% corrections (occur almost yearly)",
            "likelihood": "High (40-50% annual)",
            "duration": "1-3 months",
            "recovery": "3-6 months",
            "impact": "8-12% pullback"
        },
        
        "MILD_2": {
            "code": "MILD_2",
            "name": "Profit Booking/Rotation",
            "category": "🟡 MILD",
            "description": "Tactical selling, sector rotation",
            "shock": MarketShock(
                name="Profit Booking",
                return_shock=-0.07,
                volatility_shock=1.15,
                correlation_shock=0.03
            ),
            "calibration": "Regular intra-month volatility: -5% to -8%",
            "likelihood": "Very High (60-70% annual)",
            "duration": "2-6 weeks",
            "recovery": "1-3 months",
            "impact": "5-8% dip"
        },
        
        # ==============================================================
        # INDIA-SPECIFIC
        # ==============================================================
        "INDIA_1": {
            "code": "INDIA_1",
            "name": "RBI Rate Hike Cycle",
            "category": "🔵 INDIA-SPECIFIC",
            "description": "RBI aggressive rate hikes",
            "shock": MarketShock(
                name="RBI Rate Hikes",
                return_shock=-0.15,
                volatility_shock=1.35,
                correlation_shock=0.10
            ),
            "calibration": "2022-23: RBI +250bps in 9 months, Nifty -8% to -15%",
            "likelihood": "Moderate (15-20% annual)",
            "duration": "6-15 months",
            "recovery": "9-18 months",
            "impact": "10-15% decline"
        },
        
        "INDIA_2": {
            "code": "INDIA_2",
            "name": "Monsoon Failure",
            "category": "🔵 INDIA-SPECIFIC",
            "description": "Poor monsoon, rural demand hit",
            "shock": MarketShock(
                name="Monsoon Failure",
                return_shock=-0.09,
                volatility_shock=1.20,
                correlation_shock=0.05
            ),
            "calibration": "2014-15 deficits: -5% to -10%, FMCG/Auto affected",
            "likelihood": "Moderate (10-15% annual)",
            "duration": "6-9 months",
            "recovery": "6-12 months",
            "impact": "6-10% decline"
        },
        
        "INDIA_3": {
            "code": "INDIA_3",
            "name": "Geopolitical Tension",
            "category": "🔵 INDIA-SPECIFIC",
            "description": "Border tensions (Pak/China)",
            "shock": MarketShock(
                name="Geopolitical Risk",
                return_shock=-0.10,
                volatility_shock=1.30,
                correlation_shock=0.08
            ),
            "calibration": "Pulwama 2019, Galwan 2020: -3% to -8% (short-lived)",
            "likelihood": "Moderate (20-30% annual)",
            "duration": "2-12 weeks",
            "recovery": "1-3 months",
            "impact": "5-10% decline"
        },
        
        # ==============================================================
        # SECTORAL
        # ==============================================================
        "SECTOR_1": {
            "code": "SECTOR_1",
            "name": "IT Sector Crash",
            "category": "🔷 SECTORAL",
            "description": "Global tech selloff",
            "shock": MarketShock(
                name="Tech Crash",
                return_shock=-0.32,
                volatility_shock=1.9,
                correlation_shock=0.08
            ),
            "calibration": "2022 selloff: Nifty IT -30% to -40%",
            "likelihood": "Moderate (10-15% annual)",
            "duration": "6-18 months",
            "recovery": "18-36 months",
            "impact": "Severe if IT-heavy (15-35%)"
        },
        
        "SECTOR_2": {
            "code": "SECTOR_2",
            "name": "Oil Price Shock",
            "category": "🔷 SECTORAL",
            "description": "Crude spike to $120+",
            "shock": MarketShock(
                name="Oil Shock",
                return_shock=-0.16,
                volatility_shock=1.4,
                correlation_shock=0.10
            ),
            "calibration": "2008 oil at $147, 2022 at $130: -10% to -18%",
            "likelihood": "Moderate (15-20% annual)",
            "duration": "6-18 months",
            "recovery": "12-24 months",
            "impact": "12-18% decline"
        },
        
        # ==============================================================
        # POSITIVE
        # ==============================================================
        "POS_1": {
            "code": "POS_1",
            "name": "India Growth Boom",
            "category": "🟢 POSITIVE",
            "description": "Strong growth, reforms",
            "shock": MarketShock(
                name="Growth Boom",
                return_shock=0.25,
                volatility_shock=0.85,
                correlation_shock=-0.08
            ),
            "calibration": "2014-17, 2020-21 rallies: +20% to +40%",
            "likelihood": "Moderate (15-20% annual)",
            "duration": "12-36 months",
            "recovery": "N/A (positive)",
            "impact": "20-35% gains"
        },
        
        "POS_2": {
            "code": "POS_2",
            "name": "Sector Bull Run",
            "category": "🟢 POSITIVE",
            "description": "Sectoral outperformance",
            "shock": MarketShock(
                name="Sector Rally",
                return_shock=0.18,
                volatility_shock=0.90,
                correlation_shock=-0.04
            ),
            "calibration": "2020 Pharma, 2021 Metals, 2023 PSU: +30% to +100%",
            "likelihood": "Moderate-High (25-35% annual)",
            "duration": "12-30 months",
            "recovery": "N/A (positive)",
            "impact": "15-25% if well-positioned"
        }
    }


def print_menu(scenarios):
    """Print formatted scenario menu."""
    print("\n" + "=" * 80)
    print("🎯 ENHANCED SCENARIO ANALYSIS - EMPIRICALLY CALIBRATED")
    print("=" * 80)
    print()
    
    # Group by category
    grouped = {}
    for code, info in scenarios.items():
        cat = info['category']
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append((code, info))
    
    # Print by category
    for cat in sorted(grouped.keys()):
        print(f"\n{cat}")
        print("-" * 80)
        for code, info in grouped[cat]:
            print(f"[{code:8s}] {info['name']}")
            print(f"            {info['description']}")
            print(f"            Impact: {info['impact']} | Likelihood: {info['likelihood']}")
    
    print("\n" + "=" * 80)


def analyze_impact(weights, base_mu, base_sigma, stressed_mu, stressed_sigma, rf, pv):
    """Calculate scenario impact metrics."""
    w = weights.values
    
    base_ret = np.dot(w, base_mu.values)
    base_vol = np.sqrt(w @ base_sigma.values @ w)
    base_sharpe = (base_ret - rf) / base_vol if base_vol > 0 else 0
    
    stress_ret = np.dot(w, stressed_mu.values)
    stress_vol = np.sqrt(w @ stressed_sigma.values @ w)
    stress_sharpe = (stress_ret - rf) / stress_vol if stress_vol > 0 else 0
    
    ret_chg = stress_ret - base_ret
    vol_chg = (stress_vol / base_vol - 1) if base_vol > 0 else 0
    
    return {
        "base_return": base_ret,
        "stressed_return": stress_ret,
        "return_change": ret_chg,
        "base_vol": base_vol,
        "stressed_vol": stress_vol,
        "vol_change_pct": vol_chg,
        "base_sharpe": base_sharpe,
        "stressed_sharpe": stress_sharpe,
        "sharpe_change": stress_sharpe - base_sharpe,
        "portfolio_loss": ret_chg * pv,
        "loss_pct": ret_chg
    }


def main():
    """Main execution."""
    args = parse_args()
    
    # Interactive mode
    if args.tickers is None:
        print("\n" + "=" * 80)
        print("MILESTONE 4: ENHANCED SCENARIO ANALYSIS")
        print("=" * 80)
        print()
        
        ti = input("Tickers (comma-separated) [RELIANCE.NS,TCS.NS,INFY.NS]: ").strip()
        args.tickers = ti if ti else "RELIANCE.NS,TCS.NS,INFY.NS"
        
        pv = input(f"Portfolio value [₹{args.portfolio_value:,.0f}]: ").strip()
        if pv:
            try:
                args.portfolio_value = float(pv.replace(',', '').replace('₹', ''))
            except:
                pass
        
        rf = input(f"Risk-free rate %(Typically 4-7% offered by banks and govt bonds)[{args.risk_free_rate*100:.0f}]: ").strip()
        if rf:
            try:
                r = float(rf)
                args.risk_free_rate = r/100 if r > 1 else r
            except:
                pass
    
    tickers = [t.strip() for t in args.tickers.split(",")]
    tickers = [t if t.endswith('.NS') else t+'.NS' for t in tickers]
    
    print("\n" + "=" * 80)
    print("PORTFOLIO SCENARIO ANALYSIS")
    print("=" * 80)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Portfolio: ₹{args.portfolio_value:,.0f}")
    print(f"Risk-Free Rate %(Typically 4-7% offered by banks and govt bonds): {args.risk_free_rate:.1%}")
    print()
    
    # Build portfolio
    print("Step 1: Building Optimal Portfolio...")
    try:
        prices = load_price_data(tickers, period="2y")
        print(f"✓ Loaded {len(prices)} days")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    returns = compute_daily_returns(prices)
    mu = compute_expected_returns(returns)
    sigma = compute_covariance_matrix(returns)
    
    opt = PortfolioOptimizer(mu, sigma, args.risk_free_rate)
    result = opt.optimize_max_sharpe()
    weights = result.weights
    
    print("\n✅ OPTIMAL PORTFOLIO:")
    for t, w in weights.items():
        print(f"  {t:20s} {w:>6.1%}")
    print(f"\n  Return: {result.expected_return:>6.2%} | Vol: {result.volatility:>6.2%} | Sharpe: {result.sharpe_ratio:>5.2f}")
    print()
    
    # Risk metrics
    print("Step 2: Risk Metrics...")
    port_ret = returns[tickers].dot(weights)
    
    pvar = compute_parametric_var(weights, mu, sigma, args.confidence_level, 1, args.portfolio_value)
    hvar = compute_historical_var(port_ret, args.confidence_level, args.portfolio_value)
    cvar = compute_cvar(port_ret, args.confidence_level, args.portfolio_value)
    cvar_comp = compute_component_var(weights, sigma, args.confidence_level)
    
    print(f"\n  VaR (95%, Param): ₹{pvar['var_amount']:>10,.0f}")
    print(f"  VaR (95%, Hist):  ₹{hvar['var_amount']:>10,.0f}")
    print(f"  CVaR (95%):       ₹{cvar['cvar_amount']:>10,.0f}")
    print("\n  Risk Contribution:")
    for t in cvar_comp.index:
        print(f"    {t:20s} {cvar_comp.loc[t, '% Contribution']:>5.1f}%")
    print()
    
    # Scenario analysis
    print("=" * 80)
    print("Step 3: Scenario Analysis")
    print("=" * 80)
    
    scenarios = get_enhanced_scenarios()
    engine = ScenarioEngine(mu, sigma)
    
    print_menu(scenarios)
    
    print("\nSelect scenarios:")
    print("  • Comma-separated codes (e.g., CRISIS_1,MOD_1,MILD_1)")
    print("  • 'ALL' for all scenarios")
    print("  • 'SEVERE' for severe only")
    print()
    
    choice = input("Selection: ").strip().upper()
    
    if choice == 'ALL':
        selected = list(scenarios.keys())
    elif choice == 'SEVERE':
        selected = [k for k, v in scenarios.items() if 'SEVERE' in v['category']]
    else:
        selected = [c.strip() for c in choice.split(',') if c.strip() in scenarios]
    
    if not selected:
        print("✗ No valid selection")
        return
    
    print(f"\n✓ Testing {len(selected)} scenario(s)\n")
    
    # Test scenarios
    results = []
    
    for code in selected:
        info = scenarios[code]
        shock = info['shock']
        
        print("=" * 80)
        print(f"{info['name']} [{code}]")
        print("=" * 80)
        print(f"Category: {info['category']}")
        print(f"{info['description']}")
        print(f"\nCalibration: {info['calibration']}")
        print(f"Likelihood:  {info['likelihood']}")
        print(f"Duration:    {info['duration']}")
        print(f"Recovery:    {info['recovery']}")
        print()
        
        stressed_mu, stressed_sigma = engine.apply_scenario(shock)
        impact = analyze_impact(weights, mu, sigma, stressed_mu, stressed_sigma, args.risk_free_rate, args.portfolio_value)
        
        print("IMPACT:")
        print(f"  Return:  {impact['base_return']:>6.2%} → {impact['stressed_return']:>6.2%}  ({impact['return_change']:>+6.2%})")
        print(f"  Vol:     {impact['base_vol']:>6.2%} → {impact['stressed_vol']:>6.2%}  ({impact['vol_change_pct']:>+6.1%})")
        print(f"  Sharpe:  {impact['base_sharpe']:>6.2f} → {impact['stressed_sharpe']:>6.2f}  ({impact['sharpe_change']:>+6.2f})")
        print(f"\n  Loss:    ₹{abs(impact['portfolio_loss']):>10,.0f}  ({abs(impact['loss_pct']):>5.2%})")
        print()
        
        results.append({
            "Scenario": info['name'],
            "Code": code,
            "Category": info['category'],
            "Return_Chg_%": impact['return_change'] * 100,
            "Vol_Chg_%": impact['vol_change_pct'] * 100,
            "Sharpe_Chg": impact['sharpe_change'],
            "Loss_INR": abs(impact['portfolio_loss']),
            "Loss_%": abs(impact['loss_pct']) * 100,
            "Likelihood": info['likelihood'],
            "Duration": info['duration'],
            "Recovery": info['recovery']
        })
    
    # Save
    os.makedirs("artifacts/milestone4", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("artifacts/milestone4/scenario_analysis.csv", index=False)
    
    print("=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"\n📁 artifacts/milestone4/scenario_analysis.csv\n")
    print("SUMMARY:")
    print(f"  Tested:      {len(selected)}")
    print(f"  Worst Loss:  ₹{df['Loss_INR'].max():,.0f} ({df['Loss_%'].max():.1f}%)")
    if df['Loss_%'].min() < 0:
        print(f"  Best Gain:   ₹{abs(df['Loss_INR'].min()):,.0f} ({abs(df['Loss_%'].min()):.1f}%)")
    
    print("\nRISK ASSESSMENT:")
    max_loss = df['Loss_%'].max()
    if max_loss > 35:
        print("  🔴 HIGH RISK - Consider hedging")
    elif max_loss > 20:
        print("  🟠 MODERATE RISK - Monitor closely")
    else:
        print("  🟢 ACCEPTABLE RISK - Continue monitoring")
    print()


if __name__ == "__main__":
    main()
