from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comb_eval.ic import summarize_ic

from _common import frame_to_text, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a runsim daily IC dump.")
    parser.add_argument("path", help="runsim dump path")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    parser.add_argument("--normalize-names", action="store_true", help="Map ic/5dic to 1d_IC/5d_IC style names")
    parser.add_argument("--summary-output", help="Optional path to write summary table")
    args = parser.parse_args()

    result = summarize_ic(args.path, start=args.start, end=args.end, normalize_names=args.normalize_names)
    text = frame_to_text(result.table)
    print(text)
    if args.summary_output:
        write_text(text + "\n", args.summary_output)


if __name__ == "__main__":
    main()
