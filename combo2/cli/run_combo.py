from __future__ import annotations
import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from .. import config as organize_config_module
from ..runtime import (
    Node, ExperimentRunner, TeeStream, build_backtest_node, build_strategy_file,
    calculate_alpha_ic, dump_alpha_analysis, load_combo_base_class,
    configure_torch_threads, run_loaded_config,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runCombo",
        description="Run a comb2 experiment from an XML config.",
        epilog="example: runCombo config.xml",
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")
    return parser.parse_args()


def resolve_config_arg(args: argparse.Namespace, prog: str = "runCombo") -> str | None:
    config_path = args.config_flag or args.config
    if not config_path:
        print(f"{prog}: missing config file; pass config.xml or --config config.xml", file=sys.stderr)
        return None
    resolved = Path(config_path).expanduser()
    if not resolved.is_file():
        print(f"{prog}: config file not found: {config_path}", file=sys.stderr)
        return None
    return str(resolved)


def main() -> int:
    args = parse_args()
    config_path = resolve_config_arg(args)
    if config_path is None:
        return 2
    organize_config = organize_config_module.load_config(config_path)
    log_path = Path(organize_config["combo"]["output"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            return run_loaded_config(organize_config, config_path)

