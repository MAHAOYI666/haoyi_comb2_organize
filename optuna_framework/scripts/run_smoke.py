"""Manual smoke-study runner for detailed configs."""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.paths import build_trial_run_paths
from optuna_framework.runner import build_run_command
from optuna_framework.scripts._script_common import add_common_config_args, add_plan_check_arg, ensure_plan_for_args, load_config_from_args, print_command
from optuna_framework.scripts.run_study import make_callback, make_objective, storage_url
from optuna_framework.search_space import ConfigDrivenAdapter
from optuna_framework.study_utils import completed_history_count, create_study, maybe_enqueue_baseline, optimize_study, write_study_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke Optuna study.")
    add_common_config_args(parser)
    add_plan_check_arg(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print planned smoke setup without running optimize")
    parser.add_argument("--n-trials", type=int, default=10, help="Smoke trial count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    ensure_plan_for_args(config, args, dry_run=args.dry_run)
    adapter = ConfigDrivenAdapter(config)
    if args.dry_run:
        print(f"[DRY-RUN] smoke study_root={config.study_root}")
        print(f"[DRY-RUN] storage={storage_url(config.study_root, 'study_smoke.db')}")
        print("[DRY-RUN] sampler=TPESampler(n_startup_trials=2, multivariate=True, group=True, seed=42)")
        print("[DRY-RUN] pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1)")
        print(f"[DRY-RUN] run_window={config.tuning_run_window[0]}-{config.tuning_run_window[1]}")
        print(f"[DRY-RUN] scoring_window={config.scoring_window[0]}-{config.scoring_window[1]}")
        run_paths = build_trial_run_paths(config.study_root, 0, config.tuning_run_window, config.scoring_window)
        print_command("[DRY-RUN] smoke/trial_00000", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    config.study_root.mkdir(parents=True, exist_ok=True)
    study = create_study(config.optuna_name + "_smoke", storage_url(config.study_root, "study_smoke.db"), smoke=True)
    maybe_enqueue_baseline(study, adapter.baseline_params())
    remaining = max(0, int(args.n_trials) - completed_history_count(study))
    optimize_study(study, make_objective(config), remaining, callbacks=[make_callback(config.study_root)])
    write_study_reports(study, config.study_root)


if __name__ == "__main__":
    main()
