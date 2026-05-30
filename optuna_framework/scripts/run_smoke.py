"""Manual smoke-study runner with aggressive sampler/pruner startup settings."""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.paths import build_trial_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command
from optuna_framework.scripts._script_common import adapter_for_name, print_command
from optuna_framework.scripts.run_study import OPTUNA_STUDY_NAME, make_callback, make_objective, storage_url
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC
from optuna_framework.study_utils import completed_history_count, create_study, maybe_enqueue_baseline, optimize_study, write_study_reports


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run a 10-trial smoke Optuna study.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned smoke setup without running optimize")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    parser.add_argument("--n-trials", type=int, default=10, help="Smoke trial count")
    return parser.parse_args()


def main() -> None:
    """Run or print smoke-study setup."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_SPEC.name)
    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    if args.dry_run:
        print(f"[DRY-RUN] smoke study_root={study_root}")
        print(f"[DRY-RUN] storage={storage_url(study_root, 'study_smoke.db')}")
        print("[DRY-RUN] sampler=TPESampler(n_startup_trials=2, multivariate=True, group=True, seed=42)")
        print("[DRY-RUN] pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1)")
        for segment in STUDY_SPEC.tuning_segments:
            run_paths = build_trial_run_paths(study_root, 0, segment)
            print_command(f"[DRY-RUN] smoke/trial_00000/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    study_root.mkdir(parents=True, exist_ok=True)
    study = create_study(OPTUNA_STUDY_NAME + "_smoke", storage_url(study_root, "study_smoke.db"), smoke=True)
    maybe_enqueue_baseline(study, adapter.baseline_params())
    remaining = max(0, int(args.n_trials) - completed_history_count(study))
    optimize_study(study, make_objective(study_root), remaining, callbacks=[make_callback(study_root)])
    write_study_reports(study, study_root)


if __name__ == "__main__":
    main()
