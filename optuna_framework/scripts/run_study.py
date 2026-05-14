"""Manual Phase A Optuna study runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import final_objective, load_baseline_thresholds, require_tuning_period_baseline
from optuna_framework.config_renderer import render_config
from optuna_framework.paths import build_trial_run_paths, resolve_study_root
from optuna_framework.runner import SegmentRunError
from optuna_framework.runner import build_run_command, run_segment
from optuna_framework.scripts._script_common import adapter_for_name, print_command
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC
from optuna_framework.study_utils import (
    append_resource_metric,
    cleanup_bad_trial_artifacts,
    completed_history_count,
    create_study,
    export_optuna_visualizations,
    maybe_enqueue_baseline,
    optimize_study,
    write_study_reports,
)
from optuna_framework.trial_meta import init_trial_meta, update_segment, update_trial_state


OPTUNA_STUDY_NAME = "eg_torch_v1"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase A Optuna search.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned study setup without running optimize")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    parser.add_argument("--n-trials", type=int, default=STUDY_SPEC.n_trials, help="Override trial count")
    parser.add_argument("--cleanup-bad-trials", action="store_true", help="Remove heavyweight artifacts for rejected trials")
    return parser.parse_args()


def storage_url(study_root: Path, filename: str = "study.db") -> str:
    """Return a sqlite storage URL for a study root."""

    db_path = (Path(study_root) / filename).resolve()
    return "sqlite:///" + str(db_path).replace("\\", "/")


def make_objective(study_root: Path, cleanup_bad_trials: bool = False) -> Any:
    """Create the Optuna objective closure."""

    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    thresholds = load_baseline_thresholds(threshold_path)
    require_tuning_period_baseline(thresholds)

    def objective(trial: Any) -> float:
        try:
            from optuna import TrialPruned
        except ImportError as exc:
            from optuna_framework.study_utils import missing_dependency_message

            raise RuntimeError(missing_dependency_message("optuna")) from exc

        params = adapter.suggest_params(trial)
        materialized = adapter.materialize_params(params)
        trial_dir = study_root / "trials" / f"trial_{trial.number:05d}"
        init_trial_meta(trial_dir, trial.number, materialized, [segment.name for segment in STUDY_SPEC.tuning_segments])
        segment_metrics = []
        try:
            for step, segment in enumerate(STUDY_SPEC.tuning_segments, start=1):
                run_paths = build_trial_run_paths(study_root, trial.number, segment)
                render_config(STUDY_SPEC.baseline_config_path, run_paths, adapter, params, STUDY_SPEC.fixed_overrides)
                metrics = run_segment(run_paths)
                segment_metrics.append(metrics)
                update_segment(trial_dir, segment.name, "complete", metrics.to_dict())
                score = metrics.sharpe_idx
                trial.report(score, step=step)
                if trial.should_prune():
                    update_trial_state(trial_dir, "pruned", objective=score)
                    raise TrialPruned()
            objective_value, hard_filter_triggered = final_objective(segment_metrics, thresholds["hard_filter"])
            update_trial_state(
                trial_dir,
                "complete",
                objective=objective_value,
                hard_filter_triggered=hard_filter_triggered,
            )
            if cleanup_bad_trials and hard_filter_triggered:
                cleanup_bad_trial_artifacts(trial_dir)
            return objective_value
        except SegmentRunError:
            update_trial_state(trial_dir, "failed")
            raise

    return objective


def make_callback(study_root: Path) -> Any:
    """Create a lightweight reporting/resource callback."""

    def callback(study: Any, trial: Any) -> None:
        append_resource_metric(study_root, trial)
        write_study_reports(study, study_root)
        failed = sum(1 for item in study.trials if str(item.state).endswith("FAIL"))
        if failed > 5:
            print(f"Warning: failed trial count is {failed}; manual review recommended.")

    return callback


def run_phase_a(args: argparse.Namespace) -> None:
    """Run the formal Phase A study."""

    study_root = resolve_study_root(args.study_root, STUDY_SPEC.name)
    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    if args.dry_run:
        print(f"[DRY-RUN] Phase A study_root={study_root}")
        print(f"[DRY-RUN] storage={storage_url(study_root)}")
        print(f"[DRY-RUN] n_trials={args.n_trials} n_jobs=1")
        for segment in STUDY_SPEC.tuning_segments:
            run_paths = build_trial_run_paths(study_root, 0, segment)
            print_command(f"[DRY-RUN] trial_00000/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    study_root.mkdir(parents=True, exist_ok=True)
    study = create_study(OPTUNA_STUDY_NAME, storage_url(study_root), smoke=False)
    maybe_enqueue_baseline(study, adapter.baseline_params())
    remaining = max(0, int(args.n_trials) - completed_history_count(study))
    optimize_study(
        study,
        make_objective(study_root, cleanup_bad_trials=args.cleanup_bad_trials),
        remaining,
        callbacks=[make_callback(study_root)],
    )
    write_study_reports(study, study_root)
    export_optuna_visualizations(study, study_root)


def main() -> None:
    """Script entry point."""

    run_phase_a(parse_args())


if __name__ == "__main__":
    main()
