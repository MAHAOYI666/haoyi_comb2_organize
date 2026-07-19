from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[1]
if str(EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALS_ROOT))

from comb_eval.pnl import summarize_pnl_with_benchmark

from _common import frame_to_text, select_columns, write_text

PNL_KEY_COLUMNS = [
    "pnl_m",
    "ret_pct",
    "tvr_pct",
    "ir",
    "sharpe",
    "dd_pct",
    "longonly_pnl_m",
    "longonly_ret_pct",
    "longonly_tvr_pct",
    "longonly_ir",
    "longonly_sharpe",
    "margin",
    "fitness",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a daily pnl dump.")
    parser.add_argument("path", help="daily pnl path")
    parser.add_argument("--pnlzz500", help="Optional benchmark pnl path")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    parser.add_argument("--summary-output", help="Optional path to write summary table")
    args = parser.parse_args()

    result = summarize_pnl_with_benchmark(args.path, args.pnlzz500, start=args.start, end=args.end)
    text = frame_to_text(select_columns(result.table, PNL_KEY_COLUMNS))
    print(text)
    if args.summary_output:
        write_text(frame_to_text(result.table) + "\n", args.summary_output)


if __name__ == "__main__":
    main()
