"""Manual seed sanity check runner."""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.config_renderer import render_config
from optuna_framework.paths import build_named_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_inference
from optuna_framework.scripts._script_common import adapter_for_name, print_command
from optuna_framework.studies.eg_torch_v1 import (
    ADAPTER_NAME,
    BASELINE_CONFIG_PATH,
    FIXED_OVERRIDES,
    SCORING_WINDOW,
    STUDY_NAME,
    TUNING_RUN_WINDOW,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run two same-seed baseline tuning-period checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    parser.add_argument("--seed", type=int, default=42, help="Seed to check")
    return parser.parse_args()


def main() -> None:
    """Run or print the same-seed sanity check."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_NAME)
    adapter = adapter_for_name(ADAPTER_NAME)
    params = adapter.baseline_params()
    run_paths_list = []
    for repeat in (1, 2):
        run_paths = build_named_run_paths(
            study_root,
            f"seed_sanity/seed_{args.seed}_repeat_{repeat}",
            kind="seed_sanity",
            run_window=TUNING_RUN_WINDOW,
            score_window=SCORING_WINDOW,
            snaptime=f"seed_sanity_seed_{args.seed}_repeat_{repeat}",
        )
        run_paths_list.append(run_paths)

    if args.dry_run:
        print(f"[DRY-RUN] seed sanity study_root={study_root} seed={args.seed}")
        for idx, run_paths in enumerate(run_paths_list, start=1):
            print_command(f"[DRY-RUN] seed_sanity/repeat_{idx}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    metrics = []
    for run_paths in run_paths_list:
        overrides = {**FIXED_OVERRIDES, "combo.model.seed": args.seed}
        render_config(BASELINE_CONFIG_PATH, run_paths, adapter, params, overrides)
        metrics.append(run_inference(run_paths))
    diff = abs(metrics[0].sharpe_idx - metrics[1].sharpe_idx)
    print(f"repeat_1_sharpe_idx={metrics[0].sharpe_idx:.8f}")
    print(f"repeat_2_sharpe_idx={metrics[1].sharpe_idx:.8f}")
    print(f"abs_diff={diff:.8f}")
    if diff > 1e-4:
        print("Warning: same-seed sharpe_idx diff is above 1e-4; manual review recommended.")


if __name__ == "__main__":
    main()

