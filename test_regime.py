#!/usr/bin/env python3
"""
Quick test of the regime intelligence fix.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from portfolio.api_m7 import get_market_regime

# Test with a few tickers
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

print("Testing regime intelligence with tickers:", tickers)
print("-" * 60)

result = get_market_regime(
    tickers=tickers,
    risk_free_rate=0.07,
    horizons=[21, 63],
    hmm_restarts=3,
    hmm_max_iter=150,
)

if result.get("error"):
    print(f"ERROR: {result['error']}")
else:
    print(f"Current Regime: {result.get('current_regime')}")
    print(f"Regime Probabilities: {result.get('regime_probabilities')}")
    print(f"Log Likelihood: {result.get('log_likelihood')}")
    print(f"Elapsed Time: {result.get('elapsed_seconds')}s")
    print("\nTransition Matrix:")
    for from_regime, transitions in result.get('transition_matrix', {}).items():
        print(f"  From {from_regime}:")
        for to_regime, prob in transitions.items():
            print(f"    → {to_regime}: {prob:.4f}")
    print(f"\nExpected Regime Durations (days):")
    for regime, days in result.get('expected_regime_duration_days', {}).items():
        print(f"  {regime}: {days:.1f} days")
