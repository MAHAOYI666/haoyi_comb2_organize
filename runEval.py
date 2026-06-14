#!/root/autodl/python310fs/bin/python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ORGANIZE_ROOT = Path(__file__).resolve().parent
for local_path in (ORGANIZE_ROOT, ORGANIZE_ROOT / "evals"):
    text_path = str(local_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from comb_eval.report import check_config_outputs, config_eval_to_text, run_config_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing comb2-organize config output.")
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")
    parser.add_argument("--report-dir", help="Directory for eval artifacts; defaults to <output_root>/eval_report")
    parser.add_argument("--plot-output", help="Path for the signal analysis long image")
    parser.add_argument("--pnlzz500", help="Optional benchmark pnl path")
    parser.add_argument("--label", help="1d forward-return label path for PNL/IC/decile calculation")
    parser.add_argument("--label-5d", help="5d forward-return label path for IC calculation")
    parser.add_argument("--label-is-table", action="store_true", help="Read --label as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--label-5d-is-table", action="store_true", help="Read --label-5d as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--label-df-type", default="true", help="df_type passed to Memmaper2.load for label paths")
    parser.add_argument("--booksize", type=float, help="Booksize for decile backtests")
    parser.add_argument("--tradecost-ratio", type=float, help="Cost multiplier; cost = tradevalue * 0.003 * ratio")
    parser.add_argument("--skip-deciles", action="store_true", help="Skip 10-group backtests")
    parser.add_argument("--skip-exposure", action="store_true", help="Skip Barra exposure analysis")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config_flag or args.config
    if not config_path:
        print("[EVAL] missing config.xml; pass a positional config or --config", file=sys.stderr)
        return 2

    check = check_config_outputs(config_path)
    if not check.ok:
        print("[EVAL] 输出文件不齐")
        print(f"[config] {check.config_path}")
        print(f"[output_root] {check.output_root}")
        print("[missing]")
        for item in check.missing:
            print(f"- {item.name}: {item.path} ({item.reason})")
        return 2

    print("[EVAL] 输出文件齐全")
    for item in check.files:
        print(f"- {item.name}: {item.path}")

    result = run_config_evaluation(
        config_path,
        report_dir=args.report_dir,
        plot_path=args.plot_output,
        pnlzz500_path=args.pnlzz500,
        label_path=args.label,
        label_5d_path=args.label_5d,
        label_is_table=args.label_is_table,
        label_5d_is_table=args.label_5d_is_table,
        label_df_type=_parse_df_type(args.label_df_type),
        booksize=args.booksize,
        tradecost_ratio=args.tradecost_ratio,
        start=args.start,
        end=args.end,
        skip_deciles=args.skip_deciles,
        skip_exposure=args.skip_exposure,
    )
    print(config_eval_to_text(result))
    return 0


def _parse_df_type(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    raise SystemExit(main())
