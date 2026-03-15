"""
User-friendly Milestone 7 runner.

Supports:
1. Interactive input mode (default, easiest)
2. CLI arguments for repeatable runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from milestone7.intelligence_engine import run_adaptive_intelligence

DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "WIPRO.NS",
]
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


def section(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def subsection(title: str, width: int = 35) -> None:
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def parse_horizons(horizons_text: str) -> list[int]:
    try:
        vals = [int(x.strip()) for x in horizons_text.split(",") if x.strip()]
        vals = [v for v in vals if v > 0]
        return vals or [21, 63]
    except Exception:
        return [21, 63]


def normalize_rate(value: float) -> float:
    return value / 100.0 if value > 1 else value


def normalize_tickers(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        t = item.strip().upper()
        if not t:
            continue
        if not t.endswith(".NS"):
            t = t + ".NS"
        out.append(t)
    return out


def build_sector_map(tickers: list[str]) -> dict[str, str]:
    known = {
        "RELIANCE.NS": "Energy",
        "TCS.NS": "Technology",
        "HDFCBANK.NS": "Financials",
        "INFY.NS": "Technology",
        "ITC.NS": "Consumer",
        "WIPRO.NS": "Technology",
        "ICICIBANK.NS": "Financials",
        "AXISBANK.NS": "Financials",
        "SBIN.NS": "Financials",
    }
    return {t: known.get(t, "Other") for t in tickers}


def ask(prompt: str, default: str | None = None) -> str:
    shown = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    try:
        v = input(shown).strip()
    except EOFError:
        v = ""
    if not v and default is not None:
        return default
    return v


def ask_float(prompt: str, default: float) -> float:
    raw = ask(prompt, str(default))
    try:
        return float(raw)
    except Exception:
        return float(default)


def ask_int(prompt: str, default: int) -> int:
    raw = ask(prompt, str(default))
    try:
        return int(raw)
    except Exception:
        return int(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 7 adaptive allocation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/run_milestone7.py
  python examples/run_milestone7.py --tickers "RELIANCE.NS,TCS.NS,INFY.NS" --data-source auto --period 2y --horizons 21,63
        """,
    )
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated NSE tickers")
    parser.add_argument("--portfolio-value", type=float, default=None, help="Your total portfolio amount in INR")
    parser.add_argument("--data-source", type=str, default=None, choices=["auto", "live", "synthetic"])
    parser.add_argument("--period", type=str, default=None, help="Live data period (example: 2y)")
    parser.add_argument("--synthetic-days", type=int, default=None, help="Synthetic data length in business days")
    parser.add_argument("--seed", type=int, default=None, help="Synthetic random seed")
    parser.add_argument("--rf-base", type=float, default=None, help="Risk-free rate (0.07 or 7)")
    parser.add_argument("--horizons", type=str, default=None, help="Forecast horizons in days, comma-separated")
    parser.add_argument("--hmm-restarts", type=int, default=None)
    parser.add_argument("--hmm-max-iter", type=int, default=None)
    parser.add_argument("--garch-n-sim", type=int, default=None)
    parser.add_argument("--uncertainty-n-boot", type=int, default=None)
    parser.add_argument("--verbose-engine", action="store_true", help="Show detailed internal engine logs")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt for missing values")
    return parser.parse_args()


def collect_inputs(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "tickers": ",".join(DEFAULT_TICKERS),
        "portfolio_value": 1_000_000.0,
        "data_source": "auto",
        "period": "2y",
        "synthetic_days": 756,
        "seed": 42,
        "rf_base": 0.07,
        "horizons": "21,63",
        "hmm_restarts": 3,
        "hmm_max_iter": 150,
        "garch_n_sim": 300,
        "uncertainty_n_boot": 200,
    }

    if args.non_interactive:
        if args.tickers is None:
            raise ValueError("--tickers is required in --non-interactive mode.")
        if args.portfolio_value is None:
            args.portfolio_value = defaults["portfolio_value"]
        if args.data_source is None:
            args.data_source = defaults["data_source"]
        if args.period is None:
            args.period = defaults["period"]
        if args.synthetic_days is None:
            args.synthetic_days = defaults["synthetic_days"]
        if args.seed is None:
            args.seed = defaults["seed"]
        if args.rf_base is None:
            args.rf_base = defaults["rf_base"]
        if args.horizons is None:
            args.horizons = defaults["horizons"]
        if args.hmm_restarts is None:
            args.hmm_restarts = defaults["hmm_restarts"]
        if args.hmm_max_iter is None:
            args.hmm_max_iter = defaults["hmm_max_iter"]
        if args.garch_n_sim is None:
            args.garch_n_sim = defaults["garch_n_sim"]
        if args.uncertainty_n_boot is None:
            args.uncertainty_n_boot = defaults["uncertainty_n_boot"]
        return args

    section("Milestone 7 Setup")
    print("Simple mode: enter your stocks and portfolio amount.")
    print("Press Enter to use defaults.")

    if args.tickers is None:
        args.tickers = ask("Tickers (comma-separated NSE symbols)", defaults["tickers"])
    if args.portfolio_value is None:
        args.portfolio_value = ask_float("Total portfolio amount (INR)", defaults["portfolio_value"])
    if args.data_source is None:
        args.data_source = ask("Data source (auto/live/synthetic)", defaults["data_source"]).lower()
        if args.data_source not in {"auto", "live", "synthetic"}:
            args.data_source = defaults["data_source"]
    if args.period is None:
        args.period = defaults["period"]
    if args.period not in VALID_PERIODS:
        args.period = defaults["period"]
    if args.synthetic_days is None:
        args.synthetic_days = defaults["synthetic_days"]
    if args.seed is None:
        args.seed = defaults["seed"]
    if args.rf_base is None:
        args.rf_base = defaults["rf_base"]
    if args.horizons is None:
        args.horizons = defaults["horizons"]
    if args.hmm_restarts is None:
        args.hmm_restarts = defaults["hmm_restarts"]
    if args.hmm_max_iter is None:
        args.hmm_max_iter = defaults["hmm_max_iter"]
    if args.garch_n_sim is None:
        args.garch_n_sim = defaults["garch_n_sim"]
    if args.uncertainty_n_boot is None:
        args.uncertainty_n_boot = defaults["uncertainty_n_boot"]
    return args


def generate_synthetic_regime_data(
    tickers: list[str],
    n_days: int = 756,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    regime_schedule = [
        (int(n_days * 0.40), 0.15 / 252, 0.14 / np.sqrt(252), 0.55),
        (int(n_days * 0.25), -0.08 / 252, 0.22 / np.sqrt(252), 0.65),
        (int(n_days * 0.10), -0.30 / 252, 0.40 / np.sqrt(252), 0.75),
        (n_days, 0.08 / 252, 0.18 / np.sqrt(252), 0.50),
    ]

    chunks = []
    for n_obs, mu_daily, sigma_daily, rho in regime_schedule:
        corr = rho * np.ones((n_assets, n_assets)) + (1 - rho) * np.eye(n_assets)
        cov = np.diag([sigma_daily] * n_assets) @ corr @ np.diag([sigma_daily] * n_assets)
        chol = np.linalg.cholesky(cov + 1e-8 * np.eye(n_assets))
        noise = rng.standard_normal((n_obs, n_assets))
        returns = mu_daily + (noise @ chol.T)
        chunks.append(returns)

    all_returns = np.vstack(chunks)[:n_days]
    dates = pd.bdate_range(end="2025-03-31", periods=n_days)
    return pd.DataFrame(all_returns, index=dates, columns=tickers)


def load_returns(
    tickers: list[str],
    data_source: str,
    period: str,
    synthetic_days: int,
    seed: int,
) -> pd.DataFrame:
    daily_returns = None

    if data_source in {"auto", "live"}:
        try:
            from portfolio.data_loader import get_stock_data

            print(f"Loading live market data (period={period})...")
            prices = get_stock_data(tickers, period=period, use_cache=True)
            daily_returns = np.log(prices / prices.shift(1)).dropna()
            print(f"Loaded {len(daily_returns)} live trading days.")
        except Exception as exc:
            if data_source == "live":
                raise RuntimeError(f"Live data load failed: {exc}") from exc
            print(f"Live data unavailable: {exc}")
            print("Switching to synthetic data.")

    if daily_returns is None or len(daily_returns) < 100:
        daily_returns = generate_synthetic_regime_data(
            tickers=tickers,
            n_days=synthetic_days,
            seed=seed,
        )
        print(f"Generated synthetic data: {len(daily_returns)} days x {len(tickers)} assets.")

    return daily_returns


def _risk_level_from_vol(vol_ann: float) -> str:
    if vol_ann < 0.12:
        return "Low"
    if vol_ann < 0.20:
        return "Medium"
    return "High"


def _market_message(regime: str) -> str:
    r = regime.lower()
    if "bull" in r:
        return "Market looks supportive right now."
    if "bear" in r:
        return "Market looks weak right now. Be cautious."
    if "crisis" in r:
        return "Market stress is high right now. Protect downside."
    return "Market is in a transition phase. Stay diversified."


def print_report_summary(report, portfolio_value: float) -> None:
    data = report.to_dict()

    section("Your Portfolio Guidance")
    rp = data["regime_probabilities"]
    top_prob = max(rp["current_probs"].values()) if rp["current_probs"] else 0.0
    regime_label = rp["current_regime_label"]
    vf = data["volatility_forecast"]
    risk_level = _risk_level_from_vol(vf["forecast_30d_vol"])

    print(f"1) Market Mood: {regime_label} ({top_prob:.0%} confidence)")
    print(f"   {_market_message(regime_label)}")
    print(f"2) Risk Level Next Month: {risk_level}")
    print(f"   Expected ups/downs (annualized style): {vf['forecast_30d_vol']:.2%}")

    print("3) Suggested Portfolio Split:")
    allocation = data["optimal_allocation"]
    if "weights" not in allocation:
        print("   No allocation result was produced.")
    else:
        method = allocation.get("optimization_type", "N/A")
        print(f"   Strategy used internally: {method}")
        for ticker, weight in sorted(allocation["weights"].items(), key=lambda x: -x[1]):
            rupees = portfolio_value * weight
            print(f"   - {ticker}: {weight:.2%} (about ₹{rupees:,.0f})")

    fr = data["forward_return_forecast"]["forward_distributions"]
    h_min = min(fr, key=lambda x: int(x.rstrip("d")))
    h_max = max(fr, key=lambda x: int(x.rstrip("d")))
    short = fr[h_min]["expected_return_ann"]
    medium = fr[h_max]["expected_return_ann"]
    print("4) Expected Direction:")
    print(f"   - Near term ({h_min}): {short:.2%} annualized expectation")
    print(f"   - Medium term ({h_max}): {medium:.2%} annualized expectation")

    print("5) Simple Action Plan:")
    if risk_level == "High":
        print("   - Keep position sizes controlled.")
        print("   - Avoid concentrating too much in one stock.")
    elif risk_level == "Medium":
        print("   - Stay diversified and rebalance gradually.")
    else:
        print("   - Market risk is relatively calm; follow allocation plan.")

    print(f"\nRun time: {report.elapsed_seconds:.1f}s")


def main() -> int:
    args = parse_args()
    args = collect_inputs(args)

    tickers = normalize_tickers(args.tickers)
    if len(tickers) < 2:
        print("Need at least 2 tickers.")
        return 1

    horizons = parse_horizons(args.horizons)
    rf_base = normalize_rate(float(args.rf_base))

    section("Milestone 7")
    print("Running your portfolio analysis...")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Portfolio value: ₹{float(args.portfolio_value):,.0f}")

    daily_returns = load_returns(
        tickers=tickers,
        data_source=args.data_source,
        period=args.period,
        synthetic_days=int(args.synthetic_days),
        seed=int(args.seed),
    )

    report = run_adaptive_intelligence(
        prices_or_returns=daily_returns,
        tickers=tickers,
        is_returns=True,
        horizons=horizons,
        rf_base=rf_base,
        sector_map=build_sector_map(tickers),
        hmm_restarts=int(args.hmm_restarts),
        hmm_max_iter=int(args.hmm_max_iter),
        garch_n_sim=int(args.garch_n_sim),
        uncertainty_n_boot=int(args.uncertainty_n_boot),
        quiet=not args.verbose_engine,
    )

    print_report_summary(report, portfolio_value=float(args.portfolio_value))

    out_dict = report.to_dict()
    out_path = os.path.join(os.path.dirname(__file__), "milestone7_report.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2, default=str)
    print(f"\nSaved full JSON report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
