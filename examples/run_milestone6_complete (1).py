"""
run_milestone6_complete.py
==========================
Interactive Virtual P2P Trade Simulator with Security Testing.

Flow:
  PART 1 — You describe your portfolio and the trade you want to simulate.
           The system runs the virtual trade and shows you the impact.

  PART 2 — You choose which attack scenarios to test against your
           transaction, and the two security engines analyse each one.

Run:
    python run_milestone6_complete.py
"""

import sys, os, time, logging, json

# Silence all INFO-level log chatter — only real warnings will show
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("immune_bayesian").setLevel(logging.CRITICAL)

# ── Locate and load milestone6_complete.py ────────────────────────────────────
import types, importlib.util

def _find_and_load():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [here,
               os.path.join(here, "portfolio"),
               os.path.join(here, ".."),
               os.path.join(here, "..", "portfolio"),
               os.path.join(here, "..", ".."),
               os.path.join(here, "..", "src")]:
        candidate = os.path.normpath(os.path.join(folder, "milestone6_complete.py"))
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("milestone6_complete", candidate)
            mod  = types.ModuleType("milestone6_complete")
            mod.__spec__ = spec
            sys.modules["milestone6_complete"] = mod
            sys.path.insert(0, os.path.dirname(candidate))
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Cannot find milestone6_complete.py. "
        "Place it in the same folder as this runner and try again."
    )

_mod = _find_and_load()

run_virtual_trade_simulation = _mod.run_virtual_trade_simulation
generate_keypair             = _mod.generate_keypair
build_transaction_object     = _mod.build_transaction_object
SecurityEngine               = _mod.SecurityEngine
BayesianSecurityPipeline     = _mod.BayesianSecurityPipeline

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═════════════════════════════════════════════════════════════════════════════

def banner(title):
    print("\n" + "═" * 66)
    print(f"  {title}")
    print("═" * 66)

def sub(title):
    print(f"\n  ── {title} ──")

def ask(prompt, default=None):
    """Ask a question. If user just presses Enter, use default."""
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else val

def ask_float(prompt, default=None, min_val=None):
    while True:
        raw = ask(prompt, default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    ✗  Please enter a value of at least {min_val}.")
                continue
            return val
        except ValueError:
            print(f"    ✗  '{raw}' is not a number. Try again.")

def ask_int(prompt, default=None, min_val=1):
    while True:
        raw = ask(prompt, default)
        try:
            val = int(float(raw))
            if val < min_val:
                print(f"    ✗  Please enter at least {min_val}.")
                continue
            return val
        except ValueError:
            print(f"    ✗  '{raw}' is not a whole number. Try again.")

def ask_yes_no(prompt, default="y"):
    raw = ask(prompt + " (y/n)", default).lower()
    return raw.startswith("y")

def fmt_inr(x):
    """Format a number as INR with commas."""
    return f"₹{x:,.2f}"

def fmt_pct(x):
    """Format a decimal as a percentage."""
    return f"{x*100:.2f}%"

def pct_change_arrow(x):
    if x > 0.0001:  return f"+{fmt_pct(x)} ↑"
    if x < -0.0001: return f"{fmt_pct(x)} ↓"
    return "no change"


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic historical returns (used when user has no CSV)
# ═════════════════════════════════════════════════════════════════════════════

def make_synthetic_returns(tickers, n_days=504, seed=42):
    rng  = np.random.default_rng(seed)
    data = rng.normal(0.0003, 0.014, size=(n_days, len(tickers)))
    return pd.DataFrame(data, columns=tickers)


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Virtual Trade Setup (user input)
# ═════════════════════════════════════════════════════════════════════════════

def collect_portfolio():
    """Ask the user to describe their current portfolio."""
    banner("PART 1 — YOUR CURRENT PORTFOLIO")
    print("""
  We need to know what stocks you currently hold and at what prices,
  so we can show you how your proposed trade would change things.
  
  You can enter as many holdings as you like.
  When you are done adding stocks, just press Enter on the ticker field.
    """)

    holdings = {}
    prices   = {}

    print("  Enter your current holdings one by one.")
    print("  Example:  Ticker = RELIANCE.NS  |  Shares = 10  |  Current price = 2850")
    print()

    while True:
        ticker_raw = input("  Stock ticker (or press Enter to finish): ").strip().upper()
        if not ticker_raw:
            if not holdings:
                print("  You need at least one stock. Please add a holding.")
                continue
            break
        # Normalise — add .NS if user forgot
        ticker = ticker_raw if "." in ticker_raw else ticker_raw + ".NS"
        shares = ask_float(f"  How many shares of {ticker} do you hold?", min_val=0.01)
        price  = ask_float(f"  Current price of {ticker} (₹)?", min_val=0.01)
        holdings[ticker] = shares
        prices[ticker]   = price
        value = shares * price
        print(f"  Added: {ticker}  —  {shares} shares × {fmt_inr(price)} = {fmt_inr(value)}")
        print()

    total = sum(holdings[t] * prices[t] for t in holdings)
    print(f"\n  Your portfolio has {len(holdings)} stock(s).")
    print(f"  Total portfolio value: {fmt_inr(total)}")

    return holdings, prices, total


def collect_trade(holdings, prices):
    """Ask the user which trade they want to simulate."""
    banner("THE TRADE YOU WANT TO TEST")
    print("""
  Now tell us about the trade you are thinking of making.
  This is a VIRTUAL trade — no real money moves.
  We will show you exactly how it would affect your portfolio.
    """)

    ticker_raw = ask("Which stock do you want to buy? (e.g. INFY.NS)").strip().upper()
    ticker     = ticker_raw if "." in ticker_raw else ticker_raw + ".NS"
    quantity   = ask_float(f"How many shares of {ticker} do you want to buy?", min_val=1)
    price      = ask_float(f"At what price per share (₹)?", min_val=0.01)
    trade_cost = quantity * price
    print(f"\n  Trade cost: {quantity} × {fmt_inr(price)} = {fmt_inr(trade_cost)}")

    return ticker, quantity, price


def run_trade_and_show_results(holdings, prices, total_value, ticker, quantity, price):
    """Run the virtual trade simulation and print plain-English results."""
    banner("RUNNING YOUR VIRTUAL TRADE")
    print("  Please wait — running simulation and Monte Carlo projections...\n")

    all_tickers = list(set(list(holdings.keys()) + [ticker]))
    returns_df  = make_synthetic_returns(all_tickers)

    try:
        result = run_virtual_trade_simulation(
            ticker        = ticker,
            quantity      = quantity,
            price         = price,
            real_holdings = holdings,
            real_prices   = prices,
            daily_returns = returns_df,
            total_value   = total_value,
            n_mc_paths    = 1_000,
        )
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        return None

    # ── Transaction receipt ───────────────────────────────────────────────────
    sub("Transaction Receipt")
    tx = result.get("transaction_record", {})
    print(f"  Transaction ID  : {tx.get('tx_id', 'N/A')}")
    print(f"  Signature       : {tx.get('signature',{}).get('scheme','N/A')}")
    verified = tx.get("verification_status", False)
    print(f"  Verified        : {'✓ Yes — this transaction is authentic' if verified else '✗ No — signature check failed'}")

    # ── Portfolio impact ──────────────────────────────────────────────────────
    sub("How This Trade Changes Your Portfolio")
    imp = result.get("portfolio_impact", {})
    ret_chg  = imp.get("expected_return_change", 0)
    vol_chg  = imp.get("volatility_change", 0)
    shp_chg  = imp.get("sharpe_change", 0)
    cvar_chg = imp.get("cvar_change", 0)
    div_chg  = imp.get("diversification_change", 0)

    print(f"  Expected annual return : {pct_change_arrow(ret_chg)}")
    print(f"  Portfolio volatility   : {pct_change_arrow(vol_chg)}  (lower is safer)")
    print(f"  Sharpe ratio           : {shp_chg:+.3f}  (higher = better risk-adjusted return)")
    print(f"  Tail risk (CVaR 95%)   : {pct_change_arrow(cvar_chg)}  (lower is safer)")
    print(f"  Diversification score  : {pct_change_arrow(div_chg)}  (higher = more spread out)")

    # ── Monte Carlo verdict ───────────────────────────────────────────────────
    sub("Long-Term Projection (Monte Carlo — 1,000 simulated futures)")
    mc   = result.get("monte_carlo_projection", {})
    best = mc.get("best_horizon", "?")
    verdict = mc.get("overall_verdict", "No verdict available.")
    print(f"  Best horizon for this trade : {best}")
    print(f"\n  Verdict: {verdict}")

    for horizon in ["1Y", "3Y", "5Y"]:
        h = mc.get(horizon)
        if not h:
            continue
        real_v = h.get("real",    {}).get("expected_value", 0)
        virt_v = h.get("virtual", {}).get("expected_value", 0)
        delta  = h.get("deltas",  {}).get("expected_value_delta", 0)
        sign   = "+" if delta >= 0 else ""
        print(f"  {horizon}  |  Without trade: {fmt_inr(real_v)}"
              f"  →  With trade: {fmt_inr(virt_v)}"
              f"  ({sign}{fmt_inr(delta)})")

    # ── Risk summary ──────────────────────────────────────────────────────────
    sub("Risk Summary")
    print(f"  {result.get('risk_summary', 'No summary available.')}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Security Testing on the actual transaction
# ═════════════════════════════════════════════════════════════════════════════

ATTACK_MENU = {
    "1": {
        "name": "Signature Forgery",
        "desc": "Someone tampers with the transaction and the signature no longer matches.",
        "tx_override": {"verification_status": False},
        "entropy": 0.72,
    },
    "2": {
        "name": "Weak Hash / Poor Randomness",
        "desc": "The transaction hash looks suspiciously non-random — a sign of a weak random number generator.",
        "hash_override": "00" * 32,
        "entropy": 0.003,
    },
    "3": {
        "name": "Replay Attack",
        "desc": "The exact same transaction is submitted again — someone is trying to duplicate your trade.",
        "replay": True,
        "entropy": 0.72,
    },
    "4": {
        "name": "Burst / High-Frequency Attack",
        "desc": "Many copies of the transaction are fired in rapid succession to overwhelm the system.",
        "burst": True,
        "entropy": 0.72,
    },
    "5": {
        "name": "Combined Attack (Signature Forgery + Weak Hash)",
        "desc": "The worst case — both the signature is invalid AND the hash entropy is suspicious.",
        "tx_override": {"verification_status": False},
        "hash_override": "00" * 32,
        "entropy": 0.003,
    },
}

def print_attack_menu():
    print("""
  Choose which attack scenario(s) you want to test.
  The security engines will analyse your actual transaction under each attack.

  Available attacks:
    """)
    for key, atk in ATTACK_MENU.items():
        print(f"  [{key}]  {atk['name']}")
        print(f"       {atk['desc']}\n")
    print("  [A]  Run all attacks")
    print("  [0]  Skip security testing and exit")


def build_attack_tx(original_tx, attack_def, burst_index=0):
    """
    Take the real transaction from the simulation and corrupt it
    according to the chosen attack definition.
    """
    tx = dict(original_tx)

    # Apply any field overrides
    for k, v in attack_def.get("tx_override", {}).items():
        tx[k] = v

    # Apply hash override
    if "hash_override" in attack_def:
        tx["sha3_hash"] = attack_def["hash_override"]

    # Inject pre-computed entropy score so Bayesian engine uses it
    tx["_entropy_composite_score"] = attack_def.get("entropy", 0.72)

    # For burst: slightly different tx_id each time so replay cache fires
    if attack_def.get("burst"):
        tx["tx_id"]      = f"BURST_{burst_index:03d}_{original_tx.get('tx_id','')[:8]}"
        tx["signed_at"]  = time.time() + burst_index * 0.05

    # For replay: keep same tx_id so replay cache catches it
    if attack_def.get("replay"):
        tx["signed_at"]  = time.time() + 0.1   # submitted slightly later

    return tx


def run_security_test(original_tx, attack_key, attack_def, sec_engine, bay_engine):
    """Run one attack scenario through both security engines and print results."""
    name = attack_def["name"]
    print(f"\n  Testing: {name}")
    print(f"  {attack_def['desc']}")
    print()

    is_burst = attack_def.get("burst", False)
    n_burst  = 6 if is_burst else 1

    pqc_results = []
    bay_results = []

    for i in range(n_burst):
        atk_tx = build_attack_tx(original_tx, attack_def, burst_index=i)

        # First run through replay to seed memory, then run attack
        if attack_def.get("replay") and i == 0:
            # Feed the original first so replay cache has seen it
            sec_engine.process_transaction_security(dict(original_tx))
            bay_engine.process_transaction_security(dict(original_tx))

        pqc_r = sec_engine.process_transaction_security(atk_tx)
        bay_r = bay_engine.process_transaction_security(atk_tx)
        pqc_results.append(pqc_r)
        bay_results.append(bay_r)

    # ── PQC results ───────────────────────────────────────────────────────────
    print("  ┌── PQC Immune Defense ───────────────────────────────────────────┐")
    for i, r in enumerate(pqc_results):
        threat_word = {
            "low":    "🟢 LOW    — transaction looks fine",
            "medium": "🟡 MEDIUM — suspicious, flagged for review",
            "high":   "🔴 HIGH   — serious threat detected",
        }.get(r.threat_level, r.threat_level)

        label = f"  Attempt {i+1}" if is_burst else "  Result "
        print(f"  │  {label}  Threat level : {threat_word}")
        print(f"  │           Hash quality : {r.entropy_score:.4f}  "
              f"({'normal' if r.entropy_score > 0.8 else 'SUSPICIOUS — very low randomness'})")
        print(f"  │           Status       : {r.quarantine_status}")
        if r.quarantine_status == "QUARANTINED":
            print(f"  │  ⚠  This transaction has been BLOCKED and set aside.")
            print(f"  │     Actions taken: {', '.join(r.actions_triggered)}")
        if r.key_rotation:
            print(f"  │  🔑  The system has rotated to a new cryptographic key.")

    print("  └─────────────────────────────────────────────────────────────────┘")

    # ── Bayesian results ──────────────────────────────────────────────────────
    ICONS = {"SAFE":"🟢","MONITOR":"🟡","ELEVATED_RISK":"🟠","CRITICAL_THREAT":"🔴"}
    PLAIN = {
        "SAFE":           "SAFE — no threat detected",
        "MONITOR":        "MONITOR — keeping an eye on this",
        "ELEVATED_RISK":  "ELEVATED RISK — extra checks required",
        "CRITICAL_THREAT":"CRITICAL THREAT — blocked immediately",
    }

    print("  ┌── Bayesian Immune Defense ──────────────────────────────────────┐")
    for i, r in enumerate(bay_results):
        icon  = ICONS.get(r.threat_level, "⚪")
        plain = PLAIN.get(r.threat_level, r.threat_level)
        prob  = r.posterior_probability
        label = f"  Attempt {i+1}" if is_burst else "  Result "

        print(f"  │  {label}  {icon} {plain}")
        print(f"  │           Threat probability : {prob*100:.1f}%  "
              f"({'near certain' if prob > 0.9 else 'likely' if prob > 0.5 else 'low' if prob < 0.2 else 'possible'})")
        if r.memory_boosted_prior:
            print(f"  │  🧠  The system recognised this pattern from a previous attack.")
            print(f"        It responded faster because it has seen this before.")
        if r.quarantine_status == "QUARANTINED":
            print(f"  │  ⛔  Transaction QUARANTINED — will not be processed.")
        if r.key_rotation_signal:
            print(f"  │  🔑  Key rotation signalled — a new key pair will be generated.")

    print("  └─────────────────────────────────────────────────────────────────┘")


def run_security_section(original_tx):
    """Interactive security testing loop."""
    banner("PART 2 — SECURITY TESTING")
    print("""
  Your virtual trade produced a cryptographically signed transaction.
  Now you can simulate different types of attacks on that transaction
  to see how the two security engines respond.

  This helps you understand what the system protects against.
    """)

    sec = SecurityEngine()
    bay = BayesianSecurityPipeline(base_prior=0.05)

    # Warm both engines with the real benign transaction first
    sec.process_transaction_security(dict(original_tx))
    bay.process_transaction_security(dict(original_tx))
    print("  ✓ Both security engines have been primed with your real transaction.\n")

    while True:
        print_attack_menu()
        choice = input("  Your choice (1–5, A for all, 0 to exit): ").strip().upper()

        if choice == "0":
            print("\n  Skipping security tests. Goodbye!\n")
            break

        if choice == "A":
            keys_to_run = list(ATTACK_MENU.keys())
        elif choice in ATTACK_MENU:
            keys_to_run = [choice]
        else:
            print("  ✗ Invalid choice. Please enter 1–5, A, or 0.")
            continue

        for key in keys_to_run:
            run_security_test(
                original_tx  = original_tx,
                attack_key   = key,
                attack_def   = ATTACK_MENU[key],
                sec_engine   = sec,
                bay_engine   = bay,
            )

        # ── Session summary ───────────────────────────────────────────────────
        banner("SECURITY SESSION SUMMARY")
        ps = sec.system_status()
        bs = bay.system_status()

        total_pqc   = ps["transactions_processed"]
        quarantine_pqc = ps["response_engine"]["total_quarantined"]
        rotations   = ps["key_system"]["total_rotations"]
        total_bay   = bs["transactions_processed"]
        quarantine_bay = bs["quarantine_ledger"]["size"]
        patterns    = bs["memory_summary"]["total_patterns"]

        print(f"  Transactions checked     : {total_pqc} (PQC)  /  {total_bay} (Bayesian)")
        print(f"  Transactions quarantined : {quarantine_pqc} (PQC)  /  {quarantine_bay} (Bayesian)")
        print(f"  Key rotations triggered  : {rotations}")
        print(f"  Threat patterns learned  : {patterns}  "
              f"(the system now recognises these attack styles)")

        again = ask_yes_no("\n  Would you like to test another attack?", default="y")
        if not again:
            print("\n  All done. Your transaction and security results have been shown above.\n")
            break


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║        VIRTUAL P2P TRADE SIMULATOR  +  SECURITY TESTER      ║")
    print("  ║        Milestone 6  —  Post-Quantum Secure Trading          ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print("""
  Welcome! This tool has two parts:

  PART 1 — You describe your portfolio and the trade you want to test.
           The system simulates the trade and shows you the impact on
           your returns, risk, and long-term projected value.

  PART 2 — You pick attack scenarios to test against your transaction.
           Two independent security engines (PQC Immune Defense and
           Bayesian Threat Scoring) analyse and respond to each attack.

  Press Ctrl+C at any time to exit.
    """)

    try:
        # ── Part 1: collect inputs and run trade ──────────────────────────────
        holdings, prices, total_value = collect_portfolio()
        ticker, quantity, price       = collect_trade(holdings, prices)
        result = run_trade_and_show_results(
            holdings, prices, total_value, ticker, quantity, price
        )

        if result is None:
            print("  Trade simulation failed. Exiting.\n")
            return

        # ── Part 2: security testing on the actual transaction ────────────────
        original_tx = result.get("transaction_record", {})
        if not original_tx:
            print("  No transaction record found. Cannot run security tests.\n")
            return

        do_security = ask_yes_no(
            "\n  Would you like to run security attack tests on this transaction?",
            default="y"
        )
        if do_security:
            run_security_section(original_tx)
        else:
            print("\n  Security testing skipped. Goodbye!\n")

    except KeyboardInterrupt:
        print("\n\n  Exited by user.\n")


if __name__ == "__main__":
    main()
