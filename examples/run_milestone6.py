"""
Milestone 6: Encrypted P2P Virtual Trade Simulation
=====================================================

Simulate purchasing any NSE stock against your real portfolio --
cryptographically sealed, analytically complete, real holdings untouched.

Usage:
    python run_milestone6.py                             # interactive

    python run_milestone6.py \\
        --holdings "TCS.NS:20,HDFCBANK.NS:30,INFY.NS:40,ITC.NS:100" \\
        --prices   "TCS.NS:3500,HDFCBANK.NS:1600,INFY.NS:1800,ITC.NS:460" \\
        --trade-ticker WIPRO.NS --trade-qty 50 --trade-price 450

    python run_milestone6.py \\
        --holdings "RELIANCE.NS:10,TCS.NS:15,ITC.NS:100" \\
        --prices   "RELIANCE.NS:2900,TCS.NS:3500,ITC.NS:460" \\
        --trade-ticker TCS.NS --trade-qty 25 --trade-price 3500 \\
        --mc-paths 5000

Architecture (fully isolated from M3-M5):
    milestone6/
        __init__.py
        crypto_layer.py          ML-DSA keypair, SHA-3 hashing, signing
        virtual_trade_engine.py  Blended portfolio construction
        impact_analyzer.py       Risk/return delta computation
        projection_engine.py     Multivariate GBM Monte Carlo
        pipeline.py              End-to-end orchestrator
"""

import sys
import os
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup -- same pattern as M3 / M4 / M5
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from milestone6.pipeline import run_virtual_trade_simulation
from milestone6.crypto_layer import generate_keypair


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Milestone 6: Encrypted P2P Virtual Trade Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_milestone6.py

  python run_milestone6.py \\
      --holdings "TCS.NS:20,HDFCBANK.NS:30,INFY.NS:40" \\
      --prices   "TCS.NS:3500,HDFCBANK.NS:1600,INFY.NS:1800" \\
      --trade-ticker WIPRO.NS --trade-qty 50 --trade-price 450

  python run_milestone6.py \\
      --holdings "RELIANCE.NS:10,TCS.NS:15,ITC.NS:100" \\
      --prices   "RELIANCE.NS:2900,TCS.NS:3500,ITC.NS:460" \\
      --trade-ticker TCS.NS --trade-qty 25 --trade-price 3500 --mc-paths 5000
        """
    )
    parser.add_argument("--holdings",        type=str,   default=None,
        help="Holdings as TICKER:SHARES. Example: TCS.NS:20,HDFCBANK.NS:30")
    parser.add_argument("--prices",          type=str,   default=None,
        help="Prices as TICKER:PRICE. Example: TCS.NS:3500,HDFCBANK.NS:1600")
    parser.add_argument("--trade-ticker",    type=str,   default=None,
        help="NSE ticker to virtually buy. Example: WIPRO.NS")
    parser.add_argument("--trade-qty",       type=float, default=None,
        help="Number of shares to virtually buy. Example: 50")
    parser.add_argument("--trade-price",     type=float, default=None,
        help="Price per share at simulation time (INR). If omitted, fetched live from yfinance.")
    parser.add_argument("--portfolio-value", type=float, default=None,
        help="Total INR value of real portfolio. "
             "If omitted, computed from holdings x prices.")
    parser.add_argument("--risk-free-rate",  type=float, default=0.07,
        help="Annual risk-free rate (default: 0.07 = 7%%)")
    parser.add_argument("--mc-paths",        type=int,   default=1000,
        help="Monte Carlo paths per horizon (default: 1000, min: 1000)")
    parser.add_argument("--period",          type=str,   default="2y",
        choices=["6mo", "1y", "2y", "3y"],
        help="Historical data window (default: 2y)")
    parser.add_argument("--note",            type=str,   default=None,
        help="Optional annotation for the transaction record.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# NSE reference data
# ---------------------------------------------------------------------------

NSE_REFERENCE_PRICES = {
    "RELIANCE.NS":2900.0,  "TCS.NS":3500.0,     "INFY.NS":1800.0,
    "HDFCBANK.NS":1600.0,  "ICICIBANK.NS":1050.0,"WIPRO.NS":450.0,
    "AXISBANK.NS":1100.0,  "KOTAKBANK.NS":1750.0,"SBIN.NS":760.0,
    "LT.NS":3200.0,        "ITC.NS":460.0,        "HINDUNILVR.NS":2450.0,
    "SUNPHARMA.NS":1550.0, "DRREDDY.NS":5800.0,   "MARUTI.NS":12500.0,
    "TATAMOTORS.NS":850.0, "TATASTEEL.NS":130.0,  "BHARTIARTL.NS":950.0,
    "HCLTECH.NS":1350.0,   "BAJFINANCE.NS":7000.0,"TECHM.NS":150.0,
    "TITAN.NS":3300.0,     "ADANIPORTS.NS":800.0, "NESTLEIND.NS":25000.0,
    "DIVISLAB.NS":3700.0,  "CIPLA.NS":1400.0,     "BAJAJFINSV.NS":1600.0,
    "EICHERMOT.NS":4500.0, "HEROMOTOCO.NS":4800.0,
}

NSE_SECTOR_MAP = {
    "RELIANCE.NS":"Energy",    "TCS.NS":"Technology",
    "INFY.NS":"Technology",    "WIPRO.NS":"Technology",
    "HCLTECH.NS":"Technology", "TECHM.NS":"Technology",
    "HDFCBANK.NS":"Financials","ICICIBANK.NS":"Financials",
    "AXISBANK.NS":"Financials","KOTAKBANK.NS":"Financials",
    "SBIN.NS":"Financials",    "BAJFINANCE.NS":"Financials",
    "BAJAJFINSV.NS":"Financials",
    "ITC.NS":"Consumer Staples","HINDUNILVR.NS":"Consumer Staples",
    "NESTLEIND.NS":"Consumer Staples",
    "MARUTI.NS":"Consumer Discretionary",
    "TATAMOTORS.NS":"Consumer Discretionary",
    "TITAN.NS":"Consumer Discretionary",
    "EICHERMOT.NS":"Consumer Discretionary",
    "HEROMOTOCO.NS":"Consumer Discretionary",
    "TATASTEEL.NS":"Materials", "SUNPHARMA.NS":"Healthcare",
    "DRREDDY.NS":"Healthcare",  "CIPLA.NS":"Healthcare",
    "DIVISLAB.NS":"Healthcare", "BHARTIARTL.NS":"Communication",
    "LT.NS":"Industrials",      "ADANIPORTS.NS":"Industrials",
}


# ---------------------------------------------------------------------------
# Synthetic return generator
# ---------------------------------------------------------------------------

def generate_synthetic_returns(tickers, n_days=504, seed=2024):
    """
    One-factor market model calibrated to Nifty 50 sector characteristics.
    Used when real market data is unavailable.
    """
    rng = np.random.default_rng(seed)
    sector_beta = {
        "Technology":0.85,    "Financials":1.15,  "Energy":0.95,
        "Consumer Staples":0.65, "Consumer Discretionary":1.05,
        "Materials":1.20,     "Healthcare":0.70,  "Communication":0.90,
        "Industrials":1.00,   "Other":1.00,
    }
    sector_idio = {
        "Technology":0.008,   "Financials":0.012, "Energy":0.011,
        "Consumer Staples":0.007, "Consumer Discretionary":0.013,
        "Materials":0.015,    "Healthcare":0.009, "Communication":0.010,
        "Industrials":0.010,  "Other":0.011,
    }
    mkt_ret = rng.normal(0.0004, 0.010, size=n_days)
    returns = {}
    for t in tickers:
        s    = NSE_SECTOR_MAP.get(t, "Other")
        beta = sector_beta.get(s, 1.0) + rng.uniform(-0.05, 0.05)
        idio = rng.normal(0.0, sector_idio.get(s, 0.011) * rng.uniform(0.9, 1.1), n_days)
        returns[t] = beta * mkt_ret + idio
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame(returns, index=dates)


# ---------------------------------------------------------------------------
# Input parsers
# ---------------------------------------------------------------------------

def parse_holdings(s):
    """'TCS.NS:20,HDFCBANK.NS:30' -> {'TCS.NS': 20.0, 'HDFCBANK.NS': 30.0}"""
    result = {}
    for item in s.split(","):
        item = item.strip()
        if ":" not in item:
            raise ValueError(f"Invalid holding '{item}'. Use TICKER:SHARES.")
        ticker, qty = item.rsplit(":", 1)
        ticker = ticker.strip().upper()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        result[ticker] = float(qty.strip())
    return result


def fetch_live_prices_safe(tickers):
    """
    Fetch current market prices via yfinance (the same API the rest of the
    project uses).  Returns a dict {ticker: price}.  Falls back to
    NSE_REFERENCE_PRICES then Rs.1,000 for any ticker that fails.
    """
    prices = {}
    try:
        from portfolio.data_loader import fetch_live_prices
        live = fetch_live_prices(tickers)
        for t in tickers:
            p = live.get(t)
            if p and float(p) > 0:
                prices[t] = float(p)
    except Exception as e:
        print(f"  Warning: live price fetch failed ({e}), using reference prices.")

    for t in tickers:
        if t not in prices or not prices[t]:
            ref = NSE_REFERENCE_PRICES.get(t)
            if ref:
                prices[t] = ref
                print(f"  Warning: No live price for {t} -- using reference Rs.{ref:,.0f}.")
            else:
                prices[t] = 1_000.0
                print(f"  Warning: No price found for {t} -- using Rs.1,000.")
    return prices


def fetch_live_price_single(ticker):
    """
    Fetch the current market price for a single ticker via yfinance.
    Returns float price, or None if unavailable.
    """
    try:
        from portfolio.data_loader import fetch_live_prices
        live = fetch_live_prices([ticker])
        p = live.get(ticker)
        if p and float(p) > 0:
            return float(p)
    except Exception:
        pass
    return None


def parse_prices(s, holdings):
    """
    Parse an explicit prices string if provided, then fill any gaps with
    live yfinance prices for the real portfolio tickers.
    Falls back to NSE_REFERENCE_PRICES then Rs.1,000.
    """
    result = {}

    # 1. Parse any explicitly supplied prices
    if s:
        for item in s.split(","):
            item = item.strip()
            if ":" not in item:
                continue
            ticker, price = item.rsplit(":", 1)
            ticker = ticker.strip().upper()
            if not ticker.endswith(".NS"):
                ticker += ".NS"
            result[ticker] = float(price.strip())

    # 2. For holdings with no manually supplied price, fetch live from yfinance
    missing = [t for t in holdings if t not in result]
    if missing:
        print(f"  Fetching live prices from yfinance for: {', '.join(missing)}")
        live = fetch_live_prices_safe(missing)
        result.update(live)

    return result



# ---------------------------------------------------------------------------
# Section helpers (M3/M4/M5 style)
# ---------------------------------------------------------------------------

def section(title, width=80):
    print(); print("=" * width); print(title); print("=" * width)

def subsection(title, width=80):
    print(); print("-" * width); print(title); print("-" * width)


# ---------------------------------------------------------------------------
# Interactive input
# ---------------------------------------------------------------------------

def interactive_input(args):
    section("MILESTONE 6: ENCRYPTED P2P VIRTUAL TRADE SIMULATION")
    print()
    print("  Simulate any stock purchase against your real portfolio.")
    print("  Real holdings are NEVER modified -- cryptographically sealed.")
    print("  Prices are fetched LIVE from yfinance (15-20 min delay).")
    print()

    print("STEP 1: Your current real portfolio")
    print("  Format: TICKER:SHARES,TICKER:SHARES")
    print("  Example: TCS.NS:20,HDFCBANK.NS:30,INFY.NS:40,ITC.NS:100")
    print("  Press Enter for demo portfolio")
    print()
    h = input("  Holdings: ").strip()
    args.holdings = h if h else "TCS.NS:20,HDFCBANK.NS:30,INFY.NS:40,ITC.NS:100"
    if not h:
        print(f"  Using demo: {args.holdings}")

    # Parse holdings now so we can auto-fetch prices
    try:
        _holdings_preview = parse_holdings(args.holdings)
    except Exception:
        _holdings_preview = {}

    print()
    print("STEP 2: Current prices")
    print("  Press Enter to auto-fetch LIVE prices from yfinance.")
    print("  Or supply manually: TICKER:PRICE,TICKER:PRICE")
    print()
    p = input("  Prices (Enter = live fetch): ").strip()
    if p:
        args.prices = p
        print("  Using manually supplied prices.")
    else:
        args.prices = None
        if _holdings_preview:
            print(f"  Fetching live prices for {len(_holdings_preview)} tickers...")
            _live = fetch_live_prices_safe(list(_holdings_preview.keys()))
            for t, px in _live.items():
                print(f"    {t:22s}  Rs.{px:>10,.2f}")

    print()
    print("STEP 3: The virtual trade to simulate")
    print()
    t = input("  Ticker to buy (e.g. WIPRO.NS): ").strip().upper()
    args.trade_ticker = t if t else "WIPRO.NS"
    if not args.trade_ticker.endswith(".NS"):
        args.trade_ticker += ".NS"
    if not t:
        print(f"  Using demo ticker: {args.trade_ticker}")

    q = input("  Quantity (shares): ").strip()
    try:    args.trade_qty = float(q) if q else 50.0
    except: args.trade_qty = 50.0

    # Auto-fetch live price for the trade ticker
    print(f"  Fetching live price for {args.trade_ticker}...")
    live_trade_px = fetch_live_price_single(args.trade_ticker)
    if live_trade_px:
        print(f"  Live price: Rs.{live_trade_px:,.2f}")
        default_px = live_trade_px
        pr = input(f"  Price per share [live Rs.{live_trade_px:,.2f}, Enter to use]: ").strip()
    else:
        ref_px = NSE_REFERENCE_PRICES.get(args.trade_ticker, 1000.0)
        print(f"  Live fetch failed -- reference price: Rs.{ref_px:,.0f}")
        default_px = ref_px
        pr = input(f"  Price per share [Rs.{ref_px:,.0f}]: ").strip()

    try:    args.trade_price = float(pr) if pr else default_px
    except: args.trade_price = default_px

    print()
    pv = input("  Total portfolio value in Rs. (Enter to auto-compute from live prices): ").strip()
    try:    args.portfolio_value = float(pv.replace(",","").replace("Rs.","")) if pv else None
    except: args.portfolio_value = None

    print()
    rf = input(f"  Risk-free rate % [{args.risk_free_rate*100:.0f}]: ").strip()
    if rf:
        try:
            r = float(rf)
            args.risk_free_rate = r / 100 if r > 1 else r
        except: pass

    print()
    mc = input("  Monte Carlo paths/horizon [1000] (5000 = higher precision): ").strip()
    try:    args.mc_paths = max(1000, int(mc)) if mc else 1000
    except: args.mc_paths = 1000

    return args



def save_artifacts(result, impact_rows, mc_rows, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    paths = {}

    jp = os.path.join(artifacts_dir, "virtual_trade_result.json")
    with open(jp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    paths["json"] = jp

    tp = os.path.join(artifacts_dir, "transaction_record.json")
    with open(tp, "w") as f:
        json.dump(result.get("transaction_record", {}), f, indent=2, default=str)
    paths["transaction"] = tp

    if impact_rows:
        ip = os.path.join(artifacts_dir, "portfolio_impact.csv")
        pd.DataFrame(impact_rows).to_csv(ip, index=False)
        paths["impact"] = ip

    if mc_rows:
        mp = os.path.join(artifacts_dir, "monte_carlo_projection.csv")
        pd.DataFrame(mc_rows).to_csv(mp, index=False)
        paths["monte_carlo"] = mp

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.holdings is None:
        args = interactive_input(args)

    try:
        real_holdings = parse_holdings(args.holdings)
    except ValueError as e:
        print(f"Error parsing holdings: {e}"); return

    real_prices     = parse_prices(args.prices or "", real_holdings)
    trade_ticker    = args.trade_ticker.upper().strip()
    if not trade_ticker.endswith(".NS"):
        trade_ticker += ".NS"

    trade_qty   = float(args.trade_qty)
    # Auto-fetch live trade price if not supplied on the CLI
    if args.trade_price is None:
        print(f"  Fetching live price for {trade_ticker}...")
        live_px = fetch_live_price_single(trade_ticker)
        if live_px:
            args.trade_price = live_px
            print(f"  Live price: Rs.{live_px:,.2f}")
        else:
            args.trade_price = NSE_REFERENCE_PRICES.get(trade_ticker, 1_000.0)
            print(f"  Live fetch failed -- using reference Rs.{args.trade_price:,.0f}")
    trade_price = float(args.trade_price)
    rf          = args.risk_free_rate
    mc_paths    = max(1_000, args.mc_paths)
    is_new      = trade_ticker not in real_holdings

    computed_pv     = sum(real_holdings.get(t,0) * real_prices.get(t,0) for t in real_holdings)
    portfolio_value = args.portfolio_value if args.portfolio_value else computed_pv
    if portfolio_value <= 0:
        print("Error: portfolio value is zero."); return
    trade_value = trade_qty * trade_price

    # ── Header ──────────────────────────────────────────────────────────────
    section("MILESTONE 6 -- VIRTUAL TRADE SIMULATION ENGINE")
    print()
    print("  Real Portfolio:")
    for t, sh in real_holdings.items():
        px = real_prices.get(t, 0)
        v  = sh * px
        w  = v / portfolio_value if portfolio_value > 0 else 0
        print(f"    {t:22s}  {sh:>8.0f} sh  @ Rs.{px:>8,.0f}  "
              f"= Rs.{v:>12,.0f}  ({w:.1%})  [{NSE_SECTOR_MAP.get(t,'Other')}]")
    price_source = "live yfinance" if not args.prices else "manual override"
    print(f"\n  Total Real Value   : Rs.{portfolio_value:,.0f}  (prices: {price_source})")
    tag = "NEW TICKER" if is_new else f"existing: {real_holdings.get(trade_ticker,0):.0f} sh"
    print(f"\n  Virtual Trade:")
    print(f"    BUY  {trade_ticker:22s}  {trade_qty:>8.0f} sh  "
          f"@ Rs.{trade_price:>8,.0f}  = Rs.{trade_value:>12,.0f}  [{tag}]")
    print(f"\n  Risk-Free Rate    : {rf:.1%}")
    print(f"  MC Paths/Horizon  : {mc_paths:,}")

    # ── Step 1: Return data ──────────────────────────────────────────────────
    subsection("Step 1: Building Historical Return Data")
    all_tickers = list(real_holdings.keys())
    if trade_ticker not in all_tickers:
        all_tickers.append(trade_ticker)
    period_days = {"6mo":126,"1y":252,"2y":504,"3y":756}.get(args.period, 504)

    print("  Tickers:")
    for t in all_tickers:
        print(f"    {t:22s}  [{NSE_SECTOR_MAP.get(t,'Other')}]")

    daily_returns = None
    try:
        from portfolio.data_loader import get_stock_data
        from portfolio.optimizer import compute_daily_returns as _cdr
        print(f"\n  Loading real market data via yfinance (period={args.period})...")
        prices_df     = get_stock_data(all_tickers, period=args.period)
        daily_returns = _cdr(prices_df)
        missing = [t for t in all_tickers if t not in daily_returns.columns]
        if missing:
            print(f"  Synthesising returns for: {', '.join(missing)}")
            daily_returns = pd.concat(
                [daily_returns, generate_synthetic_returns(missing, len(daily_returns))],
                axis=1)
        print(f"  Real data loaded: {len(daily_returns)} trading days")
    except Exception as e:
        print(f"  Real data unavailable ({e})")
        print(f"  Using synthetic NSE-calibrated returns ({period_days} days)")
        daily_returns = generate_synthetic_returns(all_tickers, n_days=period_days)

    print(f"  Shape  : {daily_returns.shape[0]} days x {daily_returns.shape[1]} assets")
    print(f"  Period : {daily_returns.index[0].date()} -> {daily_returns.index[-1].date()}")

    # ── Step 2: Key generation ───────────────────────────────────────────────
    subsection("Step 2: Generating Cryptographic Key Pair (ML-DSA / FIPS 204)")
    print("  Generating ML-DSA-III key pair...")
    print("  Parameters: q=8,380,417  n=256  k=6  l=5  eta=4  tau=49  beta=196")
    t0      = time.time()
    keypair = generate_keypair()
    pk, _   = keypair
    print(f"  Generated in {(time.time()-t0)*1000:.0f} ms")
    print(f"  Key ID      : {pk.key_id}")
    print(f"  Fingerprint : {pk.fingerprint()}")
    print(f"  Hash family : SHA-3 / SHAKE-256 (FIPS 202)")
    print(f"  Private key : NEVER written to disk or output JSON")

    # ── Step 3: Pipeline ─────────────────────────────────────────────────────
    subsection("Step 3: Executing Virtual Trade Pipeline")
    print("  [1] Virtual portfolio construction   blending real + simulated weights")
    print("  [2] Cryptographic signing            SHA-3 hash + ML-DSA signature")
    print("  [3] Impact analysis                  delta(return / vol / Sharpe / CVaR)")
    print("  [4] Monte Carlo projection           1Y / 3Y / 5Y GBM simulation")
    print()
    t0 = time.time()
    try:
        result = run_virtual_trade_simulation(
            ticker         = trade_ticker,
            quantity       = trade_qty,
            price          = trade_price,
            real_holdings  = real_holdings,
            real_prices    = real_prices,
            daily_returns  = daily_returns,
            total_value    = portfolio_value,
            risk_free_rate = rf,
            n_mc_paths     = mc_paths,
            keypair        = keypair,
            trade_note     = args.note,
        )
    except Exception as e:
        print(f"  Pipeline error: {e}")
        import traceback; traceback.print_exc(); return

    print(f"  Complete in {time.time()-t0:.1f}s")

    # ── Step 4: Transaction Security ─────────────────────────────────────────
    subsection("Step 4: Transaction Security Report")
    sec      = result["transaction_security"]
    verified = sec["signature_verified"]
    icon     = "VERIFIED" if verified else "UNVERIFIED"
    z, zb    = sec["z_infinity_norm"], sec["acceptance_bound"]

    print(f"  Signature Verified   : {icon}")
    print(f"  Transaction ID       : {sec['tx_id']}")
    print(f"  SHA-3 Hash (payload) : {sec['tx_hash']}")
    print(f"  Public Key           : {sec['public_key_fingerprint']}")
    print(f"  Scheme               : {sec['signature_scheme']}")
    print(f"  ||z||_inf (response) : {z:,}  (bound: {zb:,})  "
          f"[{'PASS' if z < zb else 'FAIL'}]")
    print(f"  Signed at            : "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sec['signed_at']))}")
    print("\n  Payload fields:")
    for k, v in result["transaction_record"]["payload"].items():
        if k != "note":
            print(f"    {k:<34s} {str(v)[:55]}")

    # ── Step 5: Impact Analysis ──────────────────────────────────────────────
    subsection("Step 5: Portfolio Impact Analysis")
    comp  = result["portfolio_comparison"]
    real  = comp["real"]
    virt  = comp["virtual"]
    delta = result["portfolio_impact"]

    print(f"\n  {'Metric':<30}  {'Real':>10}  {'Virtual':>10}  {'Delta':>12}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*12}")

    def row(label, rv, vv, dv, fmt):
        a = "UP" if dv > 0.0001 else ("DN" if dv < -0.0001 else "--")
        print(f"  {label:<30}  {rv:>10{fmt}}  {vv:>10{fmt}}  {a} {dv:>+{fmt}}")

    row("Expected Return (p.a.)", real["expected_return"],       virt["expected_return"],
        delta["expected_return_change"], ".4f")
    row("Volatility (p.a.)",      real["volatility"],            virt["volatility"],
        delta["volatility_change"], ".4f")
    row("Sharpe Ratio",           real["sharpe_ratio"],          virt["sharpe_ratio"],
        delta["sharpe_change"], ".4f")
    row("CVaR 95% (daily)",       real["cvar_95"],               virt["cvar_95"],
        delta["cvar_change"], ".6f")
    row("Diversification Ratio",  real["diversification_ratio"], virt["diversification_ratio"],
        delta["diversification_change"], ".4f")
    row("HHI (Concentration)",    real["hhi"],                   virt["hhi"],
        delta["hhi_change"], ".4f")
    row("Effective-N",            real["effective_n"],           virt["effective_n"],
        delta["effective_n_change"], ".2f")

    print("\n  Factor Shifts:")
    fs     = result["factor_shift"]
    flabels = {
        "market_beta_delta":            "Market Beta",
        "systematic_variance_delta":    "Systematic Variance",
        "idiosyncratic_variance_delta": "Idiosyncratic Variance",
        "r_squared_delta":              "R-Squared",
        "tracking_error_delta":         "Tracking Error vs EW",
    }
    for k, lab in flabels.items():
        v = fs.get(k, 0)
        a = "UP" if v > 0 else ("DN" if v < 0 else "--")
        print(f"    {lab:<30}  {a} {v:>+.6f}")

    print("\n  Weight Changes (Real -> Virtual):")
    for t in sorted(set(list(real.get("weights",{}).keys()) +
                        list(virt.get("weights",{}).keys()))):
        rw = real.get("weights",{}).get(t, 0.0)
        vw = virt.get("weights",{}).get(t, 0.0)
        dw = vw - rw
        tag = "  <- NEW" if rw == 0 and vw > 0 else ""
        print(f"    {t:22s}  {rw:>6.1%} [{'#'*int(rw*32):<10}] -> "
              f"{vw:>6.1%} [{'#'*int(vw*32):<10}]  {dw:>+7.1%}{tag}")

    # ── Step 6: Monte Carlo ──────────────────────────────────────────────────
    subsection("Step 6: Monte Carlo Projection (Multivariate GBM)")
    mc      = result["monte_carlo_projection"]
    mc_rows = []

    print(f"  Model   : Geometric Brownian Motion, Cholesky-correlated shocks")
    print(f"  Ito corr: drift_i = mu_i - 0.5 * sigma_ii  per step")
    print(f"  Paths   : {mc['n_paths']:,} per horizon")
    print()

    hdr = (f"  {'H':>3}  {'Portfolio':<10}  {'Expected':>14}  {'Median':>14}  "
           f"{'P5 Worst':>14}  {'Downside%':>10}  {'CAGR':>8}")
    print(hdr)
    print("  " + "-"*(len(hdr)-2))

    for hl in ["1Y","3Y","5Y"]:
        hd = mc.get(hl, {})
        if not hd:
            continue
        for plabel, pkey in [("Real","real"),("Virtual","virtual")]:
            p = hd.get(pkey, {})
            if not p: continue
            ev   = p.get("expected_value",0)
            med  = p.get("median_value",0)
            p5   = p.get("p5_value",0)
            dp   = p.get("downside_prob",0)
            cagr = p.get("median_cagr",0)
            print(f"  {hl:>3}  {plabel:<10}  Rs.{ev:>11,.0f}  Rs.{med:>11,.0f}  "
                  f"Rs.{p5:>11,.0f}  {dp:>10.1%}  {cagr:>8.2%}")
            mc_rows.append({"Horizon":hl,"Portfolio":plabel,
                "Expected_INR":round(ev),"Median_INR":round(med),
                "P5_INR":round(p5),"Downside_Prob":round(dp,4),
                "Median_CAGR":round(cagr,4)})

        d     = hd.get("deltas",{})
        ev_d  = d.get("expected_value_delta",0)
        dp_d  = d.get("downside_prob_delta",0)
        cagr_d = d.get("median_cagr_delta",0)
        arr   = "+" if ev_d >= 0 else "-"
        print(f"  {'':>3}  {'DELTA':<10}  "
              f"   {arr}Rs.{abs(ev_d):>10,.0f}  {'':>14}  {'':>14}  "
              f"{dp_d:>+10.1%}  {cagr_d:>+8.2%}")
        print()

    print(f"  Best horizon : {mc['best_horizon']}")
    print("\n  Verdict:")
    for s in mc["overall_verdict"].split(". "):
        if s.strip(): print(f"    -> {s.strip()}.")

    # ── Step 7: Risk Summary ─────────────────────────────────────────────────
    subsection("Step 7: Risk Summary")
    for line in result["risk_summary"].split("\n"):
        if   line.startswith("Verdict:"):  print(f"  *** {line}")
        elif line.startswith(("UP","DN","--","Post")): print(f"      {line}")
        elif line.strip():                  print(f"  {line}")
        else:                               print()

    # ── Step 8: Save Artifacts ───────────────────────────────────────────────
    subsection("Step 8: Saving Artifacts")
    artifacts_dir = "artifacts/milestone6"
    impact_rows = []
    for metric, dv in delta.items():
        clean = metric.replace("_change","").replace("_delta","")
        impact_rows.append({
            "Metric":    metric,
            "Real":      real.get(clean),
            "Virtual":   virt.get(clean),
            "Delta":     round(dv, 8),
            "Direction": "INCREASE" if dv>0 else "DECREASE" if dv<0 else "UNCHANGED",
        })
    paths = save_artifacts(result, impact_rows, mc_rows, artifacts_dir)
    for _, path in paths.items():
        print(f"  Saved: {path}")

    # ── Final Summary ────────────────────────────────────────────────────────
    section("FINAL SUMMARY -- MILESTONE 6 RESULTS")
    print()
    print(f"  Transaction    : {'VERIFIED' if verified else 'UNVERIFIED'}")
    print(f"  TX ID          : {sec['tx_id']}")
    print()
    print(f"  Virtual Trade  : BUY {trade_qty:.0f} x {trade_ticker} "
          f"@ Rs.{trade_price:,.0f}  =  Rs.{trade_value:,.0f}")
    print(f"  Portfolio Value: Rs.{portfolio_value:,.0f}  ->  "
          f"Rs.{portfolio_value+trade_value:,.0f}  "
          f"(+{trade_value/portfolio_value:.1%})")
    print()
    print(f"  Impact:")
    print(f"    Expected Return  : {real['expected_return']:>7.2%}  ->  "
          f"{virt['expected_return']:>7.2%}  ({delta['expected_return_change']:>+.2%})")
    print(f"    Volatility       : {real['volatility']:>7.2%}  ->  "
          f"{virt['volatility']:>7.2%}  ({delta['volatility_change']:>+.2%})")
    print(f"    Sharpe Ratio     : {real['sharpe_ratio']:>7.2f}  ->  "
          f"{virt['sharpe_ratio']:>7.2f}  ({delta['sharpe_change']:>+.2f})")
    print(f"    CVaR 95% (daily) : {real['cvar_95']:>7.4f}  ->  "
          f"{virt['cvar_95']:>7.4f}  ({delta['cvar_change']:>+.4f})")
    print()
    h5 = mc.get("5Y",{})
    if h5:
        r5, v5, d5 = h5.get("real",{}), h5.get("virtual",{}), h5.get("deltas",{})
        ev_d5 = d5.get("expected_value_delta",0)
        print(f"  5-Year MC ({mc_paths:,} paths):")
        print(f"    Real     : E[V]=Rs.{r5.get('expected_value',0):>12,.0f}  "
              f"P5=Rs.{r5.get('p5_value',0):>12,.0f}  "
              f"CAGR={r5.get('median_cagr',0):.2%}")
        print(f"    Virtual  : E[V]=Rs.{v5.get('expected_value',0):>12,.0f}  "
              f"P5=Rs.{v5.get('p5_value',0):>12,.0f}  "
              f"CAGR={v5.get('median_cagr',0):.2%}")
        print(f"    Delta    : E[V] {'+' if ev_d5>=0 else '-'}Rs.{abs(ev_d5):>10,.0f}  "
              f"Best: {mc['best_horizon']}")
        print()
    verdict_short = result["risk_summary"].split("Verdict:")[-1].strip().split("\n")[0]
    print(f"  *** Verdict: {verdict_short}")
    print()
    print(f"  Saved to: {artifacts_dir}/")
    print(f"    virtual_trade_result.json   complete JSON output")
    print(f"    transaction_record.json     cryptographic audit trail")
    print(f"    portfolio_impact.csv        before/after analytics table")
    print(f"    monte_carlo_projection.csv  1Y/3Y/5Y projection table")
    print()
    print("=" * 80)
    print("Milestone 6 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()