"""
Cleanup script to remove Python bytecode caches and optionally clear data cache CSVs.

Usage:
    python clean_cache.py                # remove __pycache__ dirs
    python clean_cache.py --clear-data-cache  # also clear data_cache/*.csv
    python clean_cache.py --clear-data-cache --yes  # skip confirmation
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def remove_pycache(root: Path) -> int:
    count = 0
    for path in root.rglob("__pycache__"):
        try:
            shutil.rmtree(path)
            count += 1
        except Exception as e:
            print(f"Warning: could not remove {path}: {e}")
    return count


def clear_data_cache(root: Path) -> int:
    cache_dir = root / "data_cache"
    if not cache_dir.exists():
        return 0
    removed = 0
    for f in cache_dir.glob("*.csv"):
        try:
            f.unlink()
            removed += 1
        except Exception as e:
            print(f"Warning: could not remove {f}: {e}")
    return removed


def main():
    parser = argparse.ArgumentParser(description="Cleanup caches.")
    parser.add_argument(
        "--clear-data-cache",
        action="store_true",
        help="Also clear CSV files under data_cache/ (yfinance downloads).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation when clearing data cache.",
    )
    args = parser.parse_args()

    pycache_removed = remove_pycache(ROOT)
    print(f"Removed {pycache_removed} __pycache__ directorie(s).")

    if args.clear_data_cache:
        if not args.yes:
            resp = input("Clear data_cache/*.csv? (y/n): ").strip().lower()
            if resp not in {"y", "yes"}:
                print("Skipped clearing data_cache.")
                return
        removed = clear_data_cache(ROOT)
        print(f"Removed {removed} file(s) from data_cache/.")


if __name__ == "__main__":
    main()
