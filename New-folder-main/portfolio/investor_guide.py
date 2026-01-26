"""
Investor-Friendly Output Module

This module provides simplified, easy-to-understand explanations of portfolio
optimization results for day-to-day investors.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def interpret_volatility(volatility: float) -> Dict[str, Any]:
    """
    Interpret volatility in investor-friendly terms.
    
    Parameters
    ----------
    volatility : float
        Annual volatility (e.g., 0.20 for 20%)
    
    Returns
    -------
    Dict with interpretation, risk level, and explanation
    """
    vol_pct = volatility * 100
    
    if vol_pct < 10:
        risk_level = "Very Low"
        interpretation = "Conservative"
        explanation = "Your portfolio is very stable with minimal price swings. Suitable for risk-averse investors."
    elif vol_pct < 15:
        risk_level = "Low"
        interpretation = "Moderate"
        explanation = "Your portfolio has modest price movements. Good balance of stability and growth potential."
    elif vol_pct < 25:
        risk_level = "Moderate"
        interpretation = "Balanced"
        explanation = "Your portfolio has moderate price swings. Suitable for investors comfortable with some risk."
    elif vol_pct < 35:
        risk_level = "High"
        interpretation = "Aggressive"
        explanation = "Your portfolio experiences significant price movements. Higher risk, higher potential returns."
    else:
        risk_level = "Very High"
        interpretation = "Very Aggressive"
        explanation = "Your portfolio is highly volatile with large price swings. Only for risk-tolerant investors."
    
    return {
        'risk_level': risk_level,
        'interpretation': interpretation,
        'explanation': explanation,
        'daily_range': f"±{vol_pct/np.sqrt(252):.2f}%"
    }


def interpret_sharpe_ratio(sharpe: float) -> Dict[str, Any]:
    """
    Interpret Sharpe ratio in investor-friendly terms.
    
    Parameters
    ----------
    sharpe : float
        Sharpe ratio
    
    Returns
    -------
    Dict with interpretation and explanation
    """
    if sharpe < 0:
        quality = "Poor"
        explanation = "Your portfolio is not providing adequate returns for the risk taken. Consider rebalancing."
    elif sharpe < 0.5:
        quality = "Below Average"
        explanation = "Your portfolio's risk-adjusted returns are modest. There may be better allocation options."
    elif sharpe < 1.0:
        quality = "Good"
        explanation = "Your portfolio provides decent returns relative to risk. A solid investment choice."
    elif sharpe < 2.0:
        quality = "Very Good"
        explanation = "Your portfolio offers excellent risk-adjusted returns. Well-optimized allocation."
    else:
        quality = "Excellent"
        explanation = "Your portfolio has outstanding risk-adjusted returns. Exceptional performance."
    
    return {
        'quality': quality,
        'explanation': explanation
    }


def interpret_diversification(effective_n: float, total_assets: int) -> Dict[str, Any]:
    """
    Interpret diversification level.
    
    Parameters
    ----------
    effective_n : float
        Effective number of assets
    total_assets : int
        Total number of assets in portfolio
    
    Returns
    -------
    Dict with interpretation and explanation
    """
    diversification_pct = (effective_n / total_assets) * 100
    
    if effective_n < 2:
        level = "Highly Concentrated"
        explanation = f"Your portfolio is heavily focused on just {effective_n:.1f} assets. High risk if one asset underperforms."
        recommendation = "Consider diversifying across more assets to reduce risk."
    elif effective_n < total_assets * 0.5:
        level = "Concentrated"
        explanation = f"Your portfolio is somewhat concentrated with effective diversification of {effective_n:.1f} assets."
        recommendation = "Adding more diversification could help reduce risk."
    elif effective_n < total_assets * 0.8:
        level = "Well Diversified"
        explanation = f"Your portfolio is well-diversified with effective exposure to {effective_n:.1f} assets."
        recommendation = "Good diversification level. Maintain this balance."
    else:
        level = "Highly Diversified"
        explanation = f"Your portfolio is highly diversified with effective exposure to {effective_n:.1f} assets."
        recommendation = "Excellent diversification. This helps reduce risk from individual asset movements."
    
    return {
        'level': level,
        'explanation': explanation,
        'recommendation': recommendation,
        'diversification_pct': diversification_pct
    }


def format_portfolio_summary(
    result,
    dispersion: Dict[str, float],
    portfolio_name: str = "Portfolio"
) -> str:
    """
    Create an investor-friendly portfolio summary.
    
    Parameters
    ----------
    result : OptimizationResult
        Portfolio optimization result
    dispersion : Dict
        Weight dispersion metrics
    portfolio_name : str
        Name of the portfolio
    
    Returns
    -------
    str
        Formatted summary text
    """
    vol_info = interpret_volatility(result.volatility)
    sharpe_info = interpret_sharpe_ratio(result.sharpe_ratio)
    div_info = interpret_diversification(
        dispersion['effective_n'],
        len(result.weights)
    )
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  {portfolio_name:^58}  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📈 EXPECTED RETURNS                                           ║
║     {result.expected_return:>6.2%} per year (annualized)                    ║
║                                                                ║
║  📊 RISK LEVEL                                                 ║
║     {vol_info['risk_level']:^20} ({result.volatility:.2%} volatility)        ║
║     {vol_info['explanation']:<58} ║
║     Daily price swings: typically {vol_info['daily_range']} per day        ║
║                                                                ║
║  ⭐ RISK-ADJUSTED PERFORMANCE                                   ║
║     Sharpe Ratio: {result.sharpe_ratio:.2f} ({sharpe_info['quality']:^20})  ║
║     {sharpe_info['explanation']:<58} ║
║                                                                ║
║  🎯 DIVERSIFICATION                                            ║
║     {div_info['level']:^20} ({dispersion['effective_n']:.1f} effective assets)  ║
║     {div_info['explanation']:<58} ║
║     💡 {div_info['recommendation']:<54} ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def format_comparison_summary(comparison, optimization_type: str) -> str:
    """
    Create investor-friendly comparison summary.
    
    Parameters
    ----------
    comparison : ComparisonResult
        Comparison result
    optimization_type : str
        Type of optimization ("min_variance" or "max_sharpe")
    
    Returns
    -------
    str
        Formatted comparison text
    """
    opt_name = "Lowest Risk" if optimization_type == "min_variance" else "Best Risk-Adjusted Returns"
    
    sample_vol = interpret_volatility(comparison.sample_result.volatility)
    shrink_vol = interpret_volatility(comparison.shrinkage_result.volatility)
    
    sample_sharpe = interpret_sharpe_ratio(comparison.sample_result.sharpe_ratio)
    shrink_sharpe = interpret_sharpe_ratio(comparison.shrinkage_result.sharpe_ratio)
    
    sample_div = interpret_diversification(
        comparison.sample_dispersion['effective_n'],
        len(comparison.sample_result.weights)
    )
    shrink_div = interpret_diversification(
        comparison.shrinkage_dispersion['effective_n'],
        len(comparison.shrinkage_result.weights)
    )
    
    # Determine which is better
    better_volatility = "Robust Method" if comparison.volatility_difference < 0 else "Standard Method"
    better_sharpe = "Robust Method" if comparison.sharpe_difference > 0 else "Standard Method"
    better_div = "Robust Method" if comparison.shrinkage_dispersion['effective_n'] > comparison.sample_dispersion['effective_n'] else "Standard Method"
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PORTFOLIO COMPARISON: {opt_name:^45}  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 TWO METHODS COMPARED                                                    ║
║     • Standard Method: Uses historical data directly                        ║
║     • Robust Method: Uses advanced technique to reduce estimation errors   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STANDARD METHOD                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Expected Return:     {comparison.sample_result.expected_return:>6.2%} per year              ║
║  Risk Level:          {sample_vol['risk_level']:^20} ({comparison.sample_result.volatility:.2%})  ║
║  Performance Quality: {sample_sharpe['quality']:^20} (Sharpe: {comparison.sample_result.sharpe_ratio:.2f}) ║
║  Diversification:     {sample_div['level']:^20} ({comparison.sample_dispersion['effective_n']:.1f} assets) ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ROBUST METHOD (Recommended)                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Expected Return:     {comparison.shrinkage_result.expected_return:>6.2%} per year              ║
║  Risk Level:          {shrink_vol['risk_level']:^20} ({comparison.shrinkage_result.volatility:.2%})  ║
║  Performance Quality: {shrink_sharpe['quality']:^20} (Sharpe: {comparison.shrinkage_result.sharpe_ratio:.2f}) ║
║  Diversification:     {shrink_div['level']:^20} ({comparison.shrinkage_dispersion['effective_n']:.1f} assets) ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY DIFFERENCES                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Risk Change:         {comparison.volatility_difference:+.2%} ({better_volatility:^20})  ║
║  Performance Change:  {comparison.sharpe_difference:+.2f} ({better_sharpe:^20})  ║
║  Diversification:     {comparison.shrinkage_dispersion['effective_n'] - comparison.sample_dispersion['effective_n']:+.1f} assets ({better_div:^20}) ║
║                                                                              ║
║  💡 RECOMMENDATION                                                           ║
║     The Robust Method typically provides more stable and reliable results.  ║
║     It reduces the impact of estimation errors and produces more balanced  ║
║     portfolio allocations. Consider using the Robust Method weights.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return summary


def create_investor_friendly_csv(comparison, optimization_type: str) -> pd.DataFrame:
    """
    Create investor-friendly CSV output.
    
    Parameters
    ----------
    comparison : ComparisonResult
        Comparison result
    optimization_type : str
        Type of optimization
    
    Returns
    -------
    pd.DataFrame
        Investor-friendly comparison DataFrame
    """
    # Create simplified comparison
    data = {
        'Metric': [
            'Expected Annual Return (Standard)',
            'Expected Annual Return (Robust)',
            'Risk Level - Volatility (Standard)',
            'Risk Level - Volatility (Robust)',
            'Risk-Adjusted Performance - Sharpe (Standard)',
            'Risk-Adjusted Performance - Sharpe (Robust)',
            'Number of Effective Assets (Standard)',
            'Number of Effective Assets (Robust)',
            'Diversification Level (Standard)',
            'Diversification Level (Robust)',
            'Change in Risk (Robust vs Standard)',
            'Change in Performance (Robust vs Standard)',
            'Change in Diversification (Robust vs Standard)'
        ],
        'Value': [
            f"{comparison.sample_result.expected_return:.2%}",
            f"{comparison.shrinkage_result.expected_return:.2%}",
            f"{comparison.sample_result.volatility:.2%}",
            f"{comparison.shrinkage_result.volatility:.2%}",
            f"{comparison.sample_result.sharpe_ratio:.2f}",
            f"{comparison.shrinkage_result.sharpe_ratio:.2f}",
            f"{comparison.sample_dispersion['effective_n']:.1f}",
            f"{comparison.shrinkage_dispersion['effective_n']:.1f}",
            interpret_diversification(comparison.sample_dispersion['effective_n'], len(comparison.sample_result.weights))['level'],
            interpret_diversification(comparison.shrinkage_dispersion['effective_n'], len(comparison.shrinkage_result.weights))['level'],
            f"{comparison.volatility_difference:+.2%}",
            f"{comparison.sharpe_difference:+.2f}",
            f"{comparison.shrinkage_dispersion['effective_n'] - comparison.sample_dispersion['effective_n']:+.1f} assets"
        ],
        'What This Means': [
            'How much your portfolio is expected to grow per year',
            'How much your portfolio is expected to grow per year (more reliable estimate)',
            'How much your portfolio price can swing up or down',
            'How much your portfolio price can swing (more stable estimate)',
            'How good your returns are relative to risk taken',
            'How good your returns are relative to risk (more accurate)',
            'How many different investments you effectively have',
            'How many different investments you effectively have (more balanced)',
            'How spread out your investments are',
            'How spread out your investments are (better balanced)',
            'Lower is better - shows if robust method reduces risk',
            'Higher is better - shows if robust method improves performance',
            'Higher is better - shows if robust method improves diversification'
        ]
    }
    
    return pd.DataFrame(data)


def create_simple_weights_csv(result, method_name: str) -> pd.DataFrame:
    """
    Create simple, investor-friendly weights CSV.
    
    Parameters
    ----------
    result : OptimizationResult
        Optimization result
    method_name : str
        Name of the method (e.g., "Robust Method - Best Returns")
    
    Returns
    -------
    pd.DataFrame
        Simple weights DataFrame
    """
    df = pd.DataFrame({
        'Stock': result.weights.index,
        'Recommended Allocation (%)': (result.weights.values * 100).round(2),
        'What This Means': [
            f"Invest {weight*100:.1f}% of your portfolio in {ticker}" 
            for ticker, weight in result.weights.items()
        ]
    })
    
    return df


def interpret_var(var_amount: float, var_percent: float, portfolio_value: float, confidence: float) -> Dict[str, Any]:
    """
    Interpret Value at Risk (VaR) in investor-friendly terms.
    
    Parameters
    ----------
    var_amount : float
        VaR in INR terms
    var_percent : float
        VaR as percentage
    portfolio_value : float
        Total portfolio value
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    Dict with interpretation and explanation
    """
    conf_pct = confidence * 100
    
    if var_percent < 0.01:
        risk_level = "Very Low"
        explanation = f"Your portfolio has very low risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    elif var_percent < 0.02:
        risk_level = "Low"
        explanation = f"Your portfolio has low risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    elif var_percent < 0.05:
        risk_level = "Moderate"
        explanation = f"Your portfolio has moderate risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    elif var_percent < 0.10:
        risk_level = "High"
        explanation = f"Your portfolio has high risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    else:
        risk_level = "Very High"
        explanation = f"Your portfolio has very high risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    
    return {
        'risk_level': risk_level,
        'explanation': explanation,
        'daily_loss': f"₹{var_amount:,.2f}",
        'daily_loss_pct': f"{var_percent:.2%}"
    }


def format_risk_summary(var_dict: Dict, cvar_dict: Dict, portfolio_value: float, confidence: float) -> str:
    """
    Create investor-friendly risk summary.
    
    Parameters
    ----------
    var_dict : Dict
        VaR results
    cvar_dict : Dict
        CVaR results
    portfolio_value : float
        Portfolio value
    confidence : float
        Confidence level
    
    Returns
    -------
    str
        Formatted risk summary
    """
    var_info = interpret_var(
        var_dict.get('var_amount', 0),
        abs(var_dict.get('var_percent', 0)),
        portfolio_value,
        confidence
    )
    
    cvar_info = interpret_var(
        cvar_dict.get('cvar_amount', 0),
        abs(cvar_dict.get('cvar_percent', 0)),
        portfolio_value,
        confidence
    )
    
    conf_pct = confidence * 100
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  PORTFOLIO RISK ASSESSMENT                                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  💰 PORTFOLIO VALUE                                             ║
║     ₹{portfolio_value:,.2f}                                              ║
║                                                                ║
║  ⚠️  DAILY RISK (Worst Case - {conf_pct:.0f}% Confidence)                    ║
║     Risk Level: {var_info['risk_level']:^20}                    ║
║     Maximum Daily Loss: {var_info['daily_loss']:^15} ({var_info['daily_loss_pct']}) ║
║     {var_info['explanation']:<58} ║
║                                                                ║
║  📉 AVERAGE LOSS (If Bad Day Occurs)                           ║
║     Average Loss on Bad Days: ₹{cvar_dict.get('cvar_amount', 0):,.2f} ({abs(cvar_dict.get('cvar_percent', 0)):.2%}) ║
║     This is the average loss you'd expect if a bad day happens. ║
║                                                                ║
║  💡 WHAT THIS MEANS                                             ║
║     • On most days ({conf_pct:.0f}% of the time), your losses won't exceed the ║
║       maximum daily loss shown above.                         ║
║     • If a bad day does occur, expect losses around the       ║
║       average loss amount.                                     ║
║     • These are estimates based on historical patterns.       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def format_scenario_result(scenario_name: str, return_val: float, volatility: float, 
                          sharpe: float, base_sharpe: float, portfolio_value: float) -> str:
    """
    Format scenario analysis result in investor-friendly way.
    
    Parameters
    ----------
    scenario_name : str
        Name of scenario
    return_val : float
        Expected return under scenario
    volatility : float
        Volatility under scenario
    sharpe : float
        Sharpe ratio under scenario
    base_sharpe : float
        Base case Sharpe ratio
    portfolio_value : float
        Portfolio value
    
    Returns
    -------
    str
        Formatted scenario result
    """
    vol_info = interpret_volatility(volatility)
    sharpe_info = interpret_sharpe_ratio(sharpe)
    sharpe_change = sharpe - base_sharpe
    
    expected_value = portfolio_value * (1 + return_val)
    value_change = expected_value - portfolio_value
    
    change_direction = "Better" if sharpe_change > 0 else "Worse" if sharpe_change < 0 else "Similar"
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  SCENARIO ANALYSIS: {scenario_name:^40}  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📈 EXPECTED PERFORMANCE                                      ║
║     Annual Return: {return_val:>6.2%} per year                            ║
║     Portfolio Value After 1 Year: ₹{expected_value:,.2f}              ║
║     Change: ₹{value_change:+,.2f} ({return_val:+.2%})                    ║
║                                                                ║
║  📊 RISK LEVEL                                                 ║
║     {vol_info['risk_level']:^20} ({volatility:.2%} volatility)        ║
║     {vol_info['explanation']:<58} ║
║                                                                ║
║  ⭐ PERFORMANCE QUALITY                                        ║
║     Sharpe Ratio: {sharpe:.2f} ({sharpe_info['quality']:^20})  ║
║     {sharpe_info['explanation']:<58} ║
║     Change from Normal: {sharpe_change:+.2f} ({change_direction:^20}) ║
║                                                                ║
║  💡 WHAT THIS MEANS                                            ║
║     This shows how your portfolio would perform if this       ║
║     market scenario were to occur. Use this to understand     ║
║     potential risks and opportunities.                        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def format_stress_test_result(scenario_name: str, result: Dict, portfolio_value: float) -> str:
    """
    Format historical stress test result in investor-friendly way.
    
    Parameters
    ----------
    scenario_name : str
        Name of historical scenario
    result : Dict
        Stress test result dictionary
    portfolio_value : float
        Portfolio value
    
    Returns
    -------
    str
        Formatted stress test result
    """
    final_value = portfolio_value * (1 + result.get('total_return', 0))
    value_change = final_value - portfolio_value
    
    vol_info = interpret_volatility(result.get('volatility', 0))
    sharpe_info = interpret_sharpe_ratio(result.get('sharpe_ratio', 0))
    
    # Determine if it was a good or bad period
    if result.get('total_return', 0) < -0.20:
        period_type = "Severe Downturn"
        period_desc = "This was a very difficult period for markets."
    elif result.get('total_return', 0) < -0.10:
        period_type = "Market Decline"
        period_desc = "This was a challenging period for markets."
    elif result.get('total_return', 0) < 0:
        period_type = "Mild Decline"
        period_desc = "This was a slightly negative period."
    elif result.get('total_return', 0) < 0.10:
        period_type = "Moderate Growth"
        period_desc = "This was a positive period for markets."
    else:
        period_type = "Strong Growth"
        period_desc = "This was an excellent period for markets."
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  HISTORICAL STRESS TEST: {scenario_name:^40}  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📅 PERIOD TYPE                                                ║
║     {period_type:^20}                                        ║
║     {period_desc:<58} ║
║     Period: {result.get('period_label', 'N/A'):<50} ║
║                                                                ║
║  💰 PORTFOLIO PERFORMANCE                                      ║
║     Starting Value: ₹{portfolio_value:,.2f}                              ║
║     Ending Value:   ₹{final_value:,.2f}                              ║
║     Total Change:   ₹{value_change:+,.2f} ({result.get('total_return', 0):+.2%})                    ║
║     Annual Return:  {result.get('annualized_return', 0):>6.2%} per year                            ║
║                                                                ║
║  📊 RISK METRICS                                              ║
║     Maximum Loss:   {result.get('max_drawdown', 0):>6.2%} (worst decline from peak)              ║
║     Volatility:     {vol_info['risk_level']:^20} ({result.get('volatility', 0):.2%})  ║
║     Performance:    {sharpe_info['quality']:^20} (Sharpe: {result.get('sharpe_ratio', 0):.2f}) ║
║                                                                ║
║  💡 WHAT THIS MEANS                                            ║
║     This shows how your portfolio would have performed        ║
║     during this historical market event. Use this to          ║
║     understand how your portfolio handles difficult periods.  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def get_scenario_menu() -> Dict[str, Dict[str, Any]]:
    """
    Get menu of available scenarios for user selection.
    
    Returns
    -------
    Dict mapping scenario numbers to scenario info
    """
    from portfolio.scenario_engine import ScenarioEngine
    from portfolio.stress_testing import StressTester
    
    scenarios = {}
    
    # Hypothetical scenarios
    hypothetical = ScenarioEngine.create_standard_scenarios()
    for i, shock in enumerate(hypothetical, 1):
        scenarios[f"H{i}"] = {
            'type': 'hypothetical',
            'name': shock.name,
            'description': f"Hypothetical scenario: {shock.name}",
            'shock': shock
        }
    
    # Historical scenarios
    historical = StressTester.get_historical_scenarios()
    for i, (name, (start, end)) in enumerate(historical.items(), 1):
        scenarios[f"S{i}"] = {
            'type': 'historical',
            'name': name,
            'description': f"Historical period: {name} ({start} to {end})",
            'dates': (start, end)
        }
    
    return scenarios

