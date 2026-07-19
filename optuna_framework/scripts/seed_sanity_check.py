"""Manual seed sanity check runner for detailed configs."""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.config_renderer import render_config
from optuna_framework.paths import build_named_run_paths
from optuna_framework.runner import build_run_command, run_inference
from optuna_framework.scripts._script_common import add_common_config_args, add_plan_check_arg, ensure_plan_for_args, load_config_from_args, print_command
from optuna_framework.search_space import ConfigDrivenAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two same-seed baseline tuning-period checks.")
    add_common_config_args(parser)
    add_plan_check_arg(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--seed", type=int, default=42, help="Seed to check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    ensure_plan_for_args(config, args, dry_run=args.dry_run)
    adapter = ConfigDrivenAdapter(config)
    params = adapter.baseline_params()
    run_paths_list = []
    for repeat in (1, 2):
        run_paths = build_named_run_paths(
            config.study_root,
            f"seed_sanity/seed_{args.seed}_repeat_{repeat}",
            kind="seed_sanity",
            run_window=config.tuning_run_window,
            score_window=config.scoring_window,
            snaptime=f"seed_sanity_seed_{args.seed}_repeat_{repeat}",
        )
        run_paths_list.append(run_paths)

    if args.dry_run:
        print(f"[DRY-RUN] seed sanity study_root={config.study_root} seed={args.seed}")
        for idx, run_paths in enumerate(run_paths_list, start=1):
            print_command(f"[DRY-RUN] seed_sanity/repeat_{idx}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    metrics = []
    for run_paths in run_paths_list:
        render_config(config, run_paths, adapter, params, {"combo.model.seed": args.seed})
        metrics.append(run_inference(run_paths))
    diff = abs(metrics[0].sharpe_idx - metrics[1].sharpe_idx)
    print(f"repeat_1_sharpe_idx={metrics[0].sharpe_idx:.8f}")
    print(f"repeat_2_sharpe_idx={metrics[1].sharpe_idx:.8f}")
    print(f"abs_diff={diff:.8f}")
    if diff > 1e-4:
        print("Warning: same-seed sharpe_idx diff is above 1e-4; manual review recommended.")


if __name__ == "__main__":
    main()
