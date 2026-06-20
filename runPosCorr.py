#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "comb2-metrics"))

from comb2_metrics import pos_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="comb-pos-corr",
        description="Calculate mean daily position correlation for two parquet position files.",
        epilog="example: comb-pos-corr pos_a.parquet pos_b.parquet",
    )
    parser.add_argument("pos1", help="First position parquet path")
    parser.add_argument("pos2", help="Second position parquet path")
    parser.add_argument("--min-valid", type=int, default=1000, help="Minimum valid cross-section count per day")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for label, path in (("pos1", args.pos1), ("pos2", args.pos2)):
        if not Path(path).expanduser().is_file():
            print(f"comb-pos-corr: {label} file not found: {path}", file=sys.stderr)
            return 2
    corr = pos_corr(args.pos1, args.pos2, min_valid=args.min_valid)
    print(corr.mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
