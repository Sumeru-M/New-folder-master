"""
run_milestone7.py
=================
Interactive Regime-Switching Adaptive Allocation Engine.

You only need to enter your stock tickers and investment preferences.
Sectors and industries are looked up automatically via yfinance.

Run:
    python run_milestone7.py
"""

import sys, os, time, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Load milestone7_complete.py ───────────────────────────────────────────────
import types, importlib.util

def _load_m7():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here,                    "milestone7_complete.py"),
        os.path.join(here, "portfolio",       "milestone7_complete.py"),
        os.path.join(here, "..",              "milestone7_complete.py"),
        os.path.join(here, "..", "portfolio", "milestone7_complete.py"),
        os.path.join(here, "..", "..",        "milestone7_complete.py"),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("milestone7_complete", path)
            mod  = types.ModuleType("milestone7_complete")
            mod.__spec__ = spec
            sys.modules["milestone7_complete"] = mod
            spec.loader.exec_module(mod)
            return mod
    searched = "\n".join(f"  {os.path.normpath(p)}" for p in candidates)
    raise FileNotFoundError(
        f"milestone7_complete.py not found. Searched:\n{searched}\n"
        "Fix: place milestone7_complete.py in your portfolio/ folder."
    )

_m7 = _load_m7()
run_adaptive_intelligence = _m7.run_adaptive_intelligence


# ═════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═════════════════════════════════════════════════════════════════════════════

def banner(title):
    print("\n" + "═" * 66)
    print(f"  {title}")
    print("═" * 66)

def sub(title):
    print(f"\n  ── {title} " + "─" * max(1, 50 - len(title)))

def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"\n  {prompt}{suffix}: ").strip()
    return val if val else (str(default) if default is not None else "")

def ask_float(prompt, default=None, min_val=None, max_val=None):
    while True:
        raw = ask(prompt, default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    Please enter at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"    Please enter at most {max_val}.")
                continue
            return val
        except ValueError:
            print(f"    '{raw}' is not a valid number. Try again.")

def ask_int(prompt, default=None, min_val=1, max_val=None):
    while True:
        raw = ask(prompt, default)
        try:
            val = int(float(raw))
            if val < min_val:
                print(f"    Please enter at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"    Please enter at most {max_val}.")
                continue
            return val
        except ValueError:
            print(f"    '{raw}' is not a whole number. Try again.")

def ask_yes_no(prompt, default="y"):
    raw = ask(prompt + " (y/n)", default).lower()
    return raw.startswith("y")

def fmt_pct(x):   return f"{x*100:.2f}%"
def fmt_pct1(x):  return f"{x*100:.1f}%"


# ═════════════════════════════════════════════════════════════════════════════
# Automatic sector lookup via yfinance
# ═════════════════════════════════════════════════════════════════════════════

def lookup_sectors(tickers: list) -> dict:
    """
    For each ticker, query yfinance for the sector.
    Returns {ticker: sector_string}.
    Falls back gracefully if yfinance is unavailable or the ticker is not found.
    """
    sector_map = {}
    try:
        import yfinance as yf
    except ImportError:
        print("\n  Note: yfinance not installed — sector limits will be skipped.")
        print("  Install with:  pip install yfinance")
        return {}

    print()
    for ticker in tickers:
        try:
            info   = yf.Ticker(ticker).info
            sector = info.get("sector") or info.get("industry") or "Other"
            name   = info.get("longName", ticker)
            sector_map[ticker] = sector
            print(f"  {ticker:20s}  {name[:35]:35s}  sector: {sector}")
        except Exception:
            sector_map[ticker] = "Other"
            print(f"  {ticker:20s}  (lookup failed — assigned 'Other')")

    return sector_map


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic data generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(tickers, n_days, seed=42):
    """
    Creates realistic fake stock return data with four built-in market phases.
    Used when no real data is available.
    """
    rng = np.random.default_rng(seed)
    N   = len(tickers)
    phases = [
        (int(n_days * 0.40), +0.15/252, 0.14/np.sqrt(252), 0.55),
        (int(n_days * 0.25), -0.08/252, 0.22/np.sqrt(252), 0.65),
        (int(n_days * 0.10), -0.30/252, 0.40/np.sqrt(252), 0.75),
        (n_days,             +0.08/252, 0.18/np.sqrt(252), 0.50),
    ]
    blocks = []
    for (n_obs, mu_d, sig_d, rho) in phases:
        Corr = rho * np.ones((N, N)) + (1 - rho) * np.eye(N)
        Sig  = np.diag([sig_d]*N) @ Corr @ np.diag([sig_d]*N)
        L    = np.linalg.cholesky(Sig + 1e-8 * np.eye(N))
        z    = rng.standard_normal((n_obs, N))
        blocks.append(mu_d + z @ L.T)
    arr   = np.vstack(blocks)[:n_days]
    dates = pd.bdate_range(end="2025-03-31", periods=n_days)
    return pd.DataFrame(arr, index=dates, columns=tickers)


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Collect user inputs
# ═════════════════════════════════════════════════════════════════════════════

def collect_inputs():
    banner("MILESTONE 7 — REGIME-SWITCHING PORTFOLIO INTELLIGENCE")
    print("""
  This tool analyses the current market regime (Bull / Bear / Crisis /
  Transitional) and recommends how to allocate your portfolio given that
  regime and your personal risk preferences.

  You will be asked for:
    1. The stocks you want to analyse
    2. How much historical data to use
    3. Your risk appetite and investment goals
    4. A few technical settings (defaults are fine for most users)
    """)

    # ── Step 1: Tickers ───────────────────────────────────────────────────────
    sub("Step 1 of 4 — Your Stocks")
    print("""
  Enter the NSE tickers you want to include.
  Example: RELIANCE.NS   TCS.NS   HDFCBANK.NS
  You need at least 2. Press Enter with nothing typed when you are done.
    """)

    tickers = []
    while True:
        raw = input("  Ticker (or press Enter to finish): ").strip().upper()
        if not raw:
            if len(tickers) < 2:
                print("  Please enter at least 2 tickers.")
                continue
            break
        t = raw if "." in raw else raw + ".NS"
        if t in tickers:
            print(f"  {t} already added. Try a different ticker.")
            continue
        tickers.append(t)
        print(f"  Added: {t}  (total so far: {len(tickers)})")

    # Auto-lookup sectors
    sub("Looking up sector information automatically")
    print("  Querying yfinance for sector data...\n")
    sector_map = lookup_sectors(tickers)

    if sector_map:
        # Show what was found and ask for sector cap
        sectors_found = sorted(set(sector_map.values()))
        print(f"\n  Sectors found: {', '.join(sectors_found)}")
        apply_sector_cap = ask_yes_no(
            "Apply a maximum weight limit per sector?", "y"
        )
        if apply_sector_cap:
            sector_cap = ask_float(
                "Maximum allowed weight for any single sector\n"
                "  (e.g. 0.40 means no sector can exceed 40% of the portfolio)",
                0.40, min_val=0.10, max_val=1.0
            )
        else:
            sector_cap  = 1.0
            sector_map  = None   # don't apply any cap
    else:
        sector_cap  = 1.0
        sector_map  = None

    # ── Step 2: Data source ───────────────────────────────────────────────────
    sub("Step 2 of 4 — Historical Data")
    print("""
  The engine needs historical daily returns to detect market regimes.
  The more data you provide, the more reliable the regime detection.

  Options:
    [1]  Built-in synthetic data  (always works — good for testing)
    [2]  Load from a CSV file     (your own historical data)
    [3]  Fetch live data online   (requires internet + yfinance installed)
    """)

    data_choice   = ask("Your choice", "1")
    daily_returns = None

    if data_choice == "3":
        n_years = ask_float(
            "How many years of history to fetch? (e.g. 3)",
            3, min_val=1, max_val=10
        )
        try:
            import yfinance as yf
            print("  Fetching live data from Yahoo Finance...")
            prices = yf.download(tickers, period=f"{int(n_years)}y",
                                 progress=False, auto_adjust=True)["Close"]
            prices.columns = tickers[:len(prices.columns)]
            daily_returns  = np.log(prices / prices.shift(1)).dropna()
            print(f"  Loaded {len(daily_returns)} trading days of live data.")
        except Exception as e:
            print(f"  Could not fetch live data ({e}). Falling back to option 2.")
            data_choice = "2"

    if data_choice == "2":
        csv_path = ask(
            "Full path to your CSV file\n"
            "  (columns = ticker names, rows = one daily log-return per row)"
        ).strip()
        if csv_path and os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                available = [t for t in tickers if t in df.columns]
                if len(available) < 2:
                    print(f"  Not enough matching tickers in CSV. "
                          f"Found: {available}. Using synthetic data instead.")
                else:
                    tickers       = available
                    daily_returns = df[tickers].dropna()
                    print(f"  Loaded {len(daily_returns)} days from CSV.")
            except Exception as e:
                print(f"  Could not read CSV ({e}). Using synthetic data.")
        else:
            print("  File not found. Using synthetic data.")

    if daily_returns is None or len(daily_returns) < 100:
        n_days = ask_int(
            "How many days of synthetic data to generate?\n"
            "  252 ≈ 1 year  |  756 ≈ 3 years  |  1260 ≈ 5 years",
            756, min_val=252, max_val=3000
        )
        daily_returns = generate_synthetic_data(tickers, n_days)
        print(f"  Generated {len(daily_returns)} days × {len(tickers)} stocks.")

    # ── Step 3: Investment preferences ────────────────────────────────────────
    sub("Step 3 of 4 — Your Investment Preferences")

    print("""
  Risk appetite — this controls how aggressively the engine
  pursues returns versus protecting against losses:

    [1]  Conservative  — protect my capital first, lower returns are fine
    [2]  Balanced      — moderate risk for moderate return  (default)
    [3]  Aggressive    — maximise returns, I can handle higher risk
    """)

    risk_choice = ask("Your risk appetite", "2")
    if risk_choice == "1":
        risk_label = "Conservative"
    elif risk_choice == "3":
        risk_label = "Aggressive"
    else:
        risk_label = "Balanced"

    rf_rate = ask_float(
        "Current risk-free rate (e.g. government bond yield)\n"
        "  Enter as a decimal: 0.07 means 7%",
        0.07, min_val=0.0, max_val=0.30
    )

    print("""
  Forecast horizons — the engine will compute expected returns
  and risk at two time horizons of your choice.

  Common choices:
    21 trading days  ≈  1 calendar month
    63 trading days  ≈  1 calendar quarter
    """)

    h1 = ask_int("First horizon (trading days)", 21,  min_val=5,      max_val=252)
    h2 = ask_int("Second horizon (trading days)", 63, min_val=h1 + 1, max_val=504)

    # ── Step 4: Advanced settings ──────────────────────────────────────────────
    sub("Step 4 of 4 — Precision Settings")
    print("""
  These control how thoroughly the engine fits the regime model.
  Press Enter on each to accept the recommended defaults.
    """)

    hmm_restarts     = ask_int(
        "HMM restarts  (more = more reliable, but slower)\n"
        "  Recommended: 3",
        3, min_val=1, max_val=10
    )
    hmm_max_iter     = ask_int(
        "Max HMM iterations per restart\n"
        "  Recommended: 150",
        150, min_val=50, max_val=1000
    )
    garch_n_sim      = ask_int(
        "Bootstrap samples for volatility confidence intervals\n"
        "  Recommended: 300",
        300, min_val=50, max_val=2000
    )
    uncertainty_boot = ask_int(
        "Bootstrap samples for regime probability confidence intervals\n"
        "  Recommended: 200",
        200, min_val=50, max_val=2000
    )

    return {
        "tickers":          tickers,
        "daily_returns":    daily_returns,
        "rf_rate":          rf_rate,
        "horizons":         [h1, h2],
        "risk_label":       risk_label,
        "sector_map":       sector_map,
        "sector_cap":       sector_cap,
        "hmm_restarts":     hmm_restarts,
        "hmm_max_iter":     hmm_max_iter,
        "garch_n_sim":      garch_n_sim,
        "uncertainty_boot": uncertainty_boot,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Run the engine
# ═════════════════════════════════════════════════════════════════════════════

def run_engine(cfg):
    banner("RUNNING REGIME INTELLIGENCE ENGINE")

    n_days = len(cfg["daily_returns"])
    print(f"""
  Settings summary:
    Stocks          : {", ".join(cfg["tickers"])}
    Data            : {n_days} trading days  ({n_days/252:.1f} years)
    Risk appetite   : {cfg["risk_label"]}
    Risk-free rate  : {fmt_pct(cfg["rf_rate"])}
    Horizons        : {cfg["horizons"][0]} days  and  {cfg["horizons"][1]} days
    Sector cap      : {fmt_pct(cfg["sector_cap"]) if cfg["sector_map"] else "not applied"}
    HMM restarts    : {cfg["hmm_restarts"]}

  Please wait — fitting the regime model takes 10–30 seconds...
    """)

    t0 = time.time()
    report = run_adaptive_intelligence(
        prices_or_returns  = cfg["daily_returns"],
        tickers            = cfg["tickers"],
        is_returns         = True,
        horizons           = cfg["horizons"],
        rf_base            = cfg["rf_rate"],
        sector_map         = cfg["sector_map"],
        hmm_restarts       = cfg["hmm_restarts"],
        hmm_max_iter       = cfg["hmm_max_iter"],
        garch_n_sim        = cfg["garch_n_sim"],
        uncertainty_n_boot = cfg["uncertainty_boot"],
        quiet              = True,
    )
    print(f"  Done in {time.time() - t0:.1f} seconds.")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — Display results in plain English
# ═════════════════════════════════════════════════════════════════════════════

REGIME_DESCRIPTIONS = {
    "Low-Vol Bull": (
        "📈  BULL MARKET  — Markets are calm and trending upward.\n"
        "     Volatility is low and returns are positive on average.\n"
        "     This is the most favourable environment for equity exposure."
    ),
    "High-Vol Bear": (
        "📉  BEAR MARKET  — Markets are falling with elevated volatility.\n"
        "     Returns are negative on average. The engine will reduce risk\n"
        "     and shift toward more defensive positions."
    ),
    "Crisis": (
        "🚨  CRISIS  — Severe market stress detected.\n"
        "     Sharp drawdowns and very high volatility. The engine applies\n"
        "     maximum diversification, strict position limits, and tail-risk\n"
        "     protection. This is a strong 'risk-off' signal."
    ),
    "Transitional": (
        "🔄  TRANSITIONAL  — The market is between regimes.\n"
        "     Mixed signals — could be recovering or beginning a decline.\n"
        "     The engine uses balanced, moderately cautious parameters."
    ),
}

def display_results(report, cfg):
    d = report.to_dict()

    # ── 1. Current regime ─────────────────────────────────────────────────────
    banner("WHAT REGIME IS THE MARKET IN RIGHT NOW?")
    rp     = d["regime_probabilities"]
    regime = rp["current_regime_label"]
    print(f"\n  {REGIME_DESCRIPTIONS.get(regime, regime)}\n")

    print("  Probability of each regime (how confident the model is):")
    for name, prob in rp["current_probs"].items():
        bar = "█" * int(prob * 30)
        print(f"    {name:20s}  {fmt_pct1(prob):6s}  {bar}")

    print(f"\n  Model fit score (log-likelihood): {rp['log_likelihood']:.1f}")
    print("  This is a measure of how well the model explains your data.")

    # ── 2. Duration and forward probability ───────────────────────────────────
    banner("HOW LONG IS THIS REGIME LIKELY TO LAST?")
    tm  = d["transition_matrix"]
    dur = tm["expected_duration_days"]
    print()
    for name, days in dur.items():
        if days > 1e7:
            print(f"  {name:20s}  extremely persistent (model is very certain "
                  f"about this regime)")
        else:
            print(f"  {name:20s}  expected to last ~{days:.0f} trading days  "
                  f"({days/21:.1f} months)")

    print(f"\n  Estimated time for the market to fully shift to a new character:")
    mt = tm['mixing_time_days']
    print(f"    ~{mt} trading days  ({mt/252:.1f} years)")

    print(f"\n  Probability of still being in the {regime} regime over time:")
    fwd = tm["forward_probabilities"]
    for h_str, probs in fwd.items():
        p   = probs.get(regime, 0.0)
        bar = "█" * int(p * 25)
        print(f"    In {h_str:5s}  {fmt_pct1(p):6s}  {bar}")

    # ── 3. Volatility ─────────────────────────────────────────────────────────
    banner("HOW VOLATILE IS YOUR PORTFOLIO EXPECTED TO BE?")
    vf = d["volatility_forecast"]
    cv = vf["current_vol_ann"]
    daily_swing = cv * 100000 / np.sqrt(252)
    print(f"""
  Current portfolio volatility : {fmt_pct(cv)}  per year
  30-day forecast              : {fmt_pct(vf['forecast_30d_vol'])}  per year

  In plain terms: if your portfolio is worth ₹1,00,000, a daily move of
  roughly ₹{daily_swing:,.0f} up or down is within the normal range.

  Volatility forecast with 95% confidence range:
    Days   Forecast       Likely range
    ────   ────────       ────────────""")
    for h, vals in vf["forecast_path"].items():
        v  = vals["vol_ann"]
        lo = vals["ci_lower_95"]
        hi = vals["ci_upper_95"]
        print(f"    {h:>4}   {fmt_pct(v):10s}   {fmt_pct(lo)}  to  {fmt_pct(hi)}")

    # ── 4. Return forecast ────────────────────────────────────────────────────
    h1, h2 = cfg["horizons"]
    banner(f"WHAT RETURNS CAN YOU EXPECT OVER THE NEXT {h1} AND {h2} DAYS?")
    frf = d["forward_return_forecast"]["forward_distributions"]

    for h_str, fd in frf.items():
        h_days   = int(h_str.replace("d", ""))
        h_months = h_days / 21
        exp_r    = fd["expected_return_ann"]
        vol      = fd["annualised_vol"]
        var95    = fd["var_95_percent"]
        cvar95   = fd["cvar_95_percent"]
        p10      = fd["tail_prob_minus10pct"]
        skew     = fd["skewness"]

        direction = "gain" if exp_r >= 0 else "loss"
        skew_desc = (
            "left-skewed — losses tend to be larger than gains"
            if skew < -0.1 else
            "right-skewed — gains tend to be larger than losses"
            if skew > 0.1 else
            "roughly symmetric"
        )
        print(f"""
  Over the next {h_days} trading days  (~{h_months:.1f} months):
  ┌─────────────────────────────────────────────────────────┐
  │  Expected return      : {fmt_pct(exp_r):>8s} per year  ({direction})    │
  │  Expected volatility  : {fmt_pct(vol):>8s} per year               │
  │  Value at Risk (95%)  : {fmt_pct(var95):>8s}                        │
  │    → In 5% of scenarios you could lose MORE than this   │
  │  Cond. VaR (95%)      : {fmt_pct(cvar95):>8s}                        │
  │    → When things go badly, average loss is this much    │
  │  Chance of losing >10%: {fmt_pct1(p10):>6s}                          │
  │  Return skew          : {skew:>+6.2f}  ({skew_desc[:30]})  │
  └─────────────────────────────────────────────────────────┘""")

    # ── 5. Allocation recommendation ─────────────────────────────────────────
    banner("RECOMMENDED PORTFOLIO ALLOCATION")
    oa = d["optimal_allocation"]
    ap = d["adaptive_parameter_shift"]
    params = ap["adapted_parameters"]

    print(f"""
  Your risk profile   : {cfg["risk_label"]}
  Current regime      : {regime}
  Optimisation method : {oa.get('optimization_type', 'N/A')}

  How the regime has changed the engine's settings:""")
    for note in ap["parameter_shift_notes"]:
        print(f"    • {note}")

    print(f"""
  Key parameters used:
    Return weight      (λ_return) : {params['lam_return']:.3f}
      Higher = engine values returns more
    Variance penalty   (λ_vol)    : {params['lam_vol']:.3f}
      Higher = engine penalises volatility more
    Tail-risk penalty  (λ_cvar)   : {params['lam_cvar']:.3f}
      Higher = engine penalises extreme losses more
    Max weight per stock           : {fmt_pct(params['max_weight'])}
    Target portfolio volatility    : {fmt_pct(params['target_vol_ann'])}
    Position scale factor          : {params['position_scale']:.3f}
      Your effective exposure is {fmt_pct1(params['position_scale'])} of full size
      because current market vol exceeds your target vol""")

    if "weights" in oa:
        sharpe = oa.get('sharpe_ratio', 0)
        sharpe_desc = (
            "excellent" if sharpe > 1.0 else
            "good"      if sharpe > 0.5 else
            "fair"      if sharpe > 0.0 else
            "negative — the regime is unfavourable"
        )
        print(f"""
  Portfolio performance estimates:
    Sharpe ratio      : {sharpe:.3f}  ({sharpe_desc})
    Portfolio vol     : {fmt_pct(oa.get('volatility_ann', 0))}
    CVaR 95%          : {fmt_pct(oa.get('cvar_95', 0))}

  Recommended weights:
  ─────────────────────────────────────────────""")
        weights = sorted(oa["weights"].items(), key=lambda x: -x[1])
        for ticker, w in weights:
            bar = "█" * int(w * 50)
            sector = (cfg["sector_map"] or {}).get(ticker, "")
            sector_str = f"  ({sector})" if sector else ""
            print(f"    {ticker:22s}  {fmt_pct1(w):6s}  {bar}{sector_str}")
    else:
        print("\n  Optimizer not available — equal risk parity weights applied.")

    # ── 6. Uncertainty ────────────────────────────────────────────────────────
    banner("HOW CONFIDENT IS THE MODEL IN THESE ESTIMATES?")
    um  = d["uncertainty_metrics"]
    ess = um["effective_sample_size"]

    quality = (
        "Good — sufficient data for reliable estimates."
        if ess >= 200 else
        "Limited — treat estimates with some caution."
    )
    print(f"""
  Effective data used : {ess} days  ({ess/252:.1f} years of information)
  Data quality        : {quality}

  Regime probability confidence intervals:
  (The true probability lies within these bounds 95% of the time)
  ─────────────────────────────────────────────""")
    for state, ci in um["regime_probability_ci_95"].items():
        hw   = ci["half_width"]
        cert = "very certain" if hw < 0.05 else "fairly certain" if hw < 0.15 else "uncertain"
        print(f"    {state:20s}  {fmt_pct1(ci['mean']):6s}  "
              f"± {fmt_pct1(hw)}  ({cert})")

    print(f"\n  Expected return confidence intervals:")
    for h_str, fc in um["forecast_return_ci_95"].items():
        print(f"    {h_str}: {fmt_pct(fc['mean_ann'])} per year  "
              f"[range: {fmt_pct(fc['lower_95'])} to {fmt_pct(fc['upper_95'])}]")

    print(f"\n  Which stocks are most sensitive to regime changes?")
    print(f"  (High = weight changes a lot when the regime shifts)")
    ws     = um["weight_regime_sensitivity"]
    top_ws = sorted(ws.items(), key=lambda x: -x[1]["regime_sensitivity_index"])
    for ticker, v in top_ws:
        rsi  = v["regime_sensitivity_index"]
        bar  = "█" * min(int(rsi * 200), 30)
        sens = "high" if rsi > 0.05 else "moderate" if rsi > 0.01 else "low"
        print(f"    {ticker:22s}  {bar}  ({sens})")

    print(f"\n  Total engine runtime: {report.elapsed_seconds:.1f} seconds")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║     MILESTONE 7 — REGIME-SWITCHING ALLOCATION ENGINE        ║")
    print("  ║     Adaptive Portfolio Intelligence · NSE Edition           ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    try:
        cfg    = collect_inputs()
        report = run_engine(cfg)
        display_results(report, cfg)

        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "milestone7_report.json"
        )
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\n  Full detailed report saved to: {out_path}\n")
        print("  ✅  Milestone 7 complete.\n")

    except KeyboardInterrupt:
        print("\n\n  Exited by user.\n")


if __name__ == "__main__":
    main()
