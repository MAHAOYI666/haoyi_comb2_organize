"""Manual Phase A Optuna study runner."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import final_objective, load_baseline_thresholds, running_score
from optuna_framework.config_renderer import render_config
from optuna_framework.gpu_allocation import GpuAllocator
from optuna_framework.paths import build_trial_run_paths, resolve_study_root
from optuna_framework.runner import SegmentRunError
from optuna_framework.runner import build_run_command, run_segment
from optuna_framework.scripts._script_common import add_study_args, load_study_and_adapter, print_command
from optuna_framework.status import log_status
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase A Optuna search.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned study setup without running optimize")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    parser.add_argument("--n-trials", type=int, default=None, help="Override trial count")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of parallel Optuna trials")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids or devices, for example: 0,1 or cuda:0,cuda:1")
    parser.add_argument("--startup-trials", type=int, default=None, help="Override TPESampler/MedianPruner startup trials")
    parser.add_argument("--cleanup-bad-trials", action="store_true", help="Remove heavyweight artifacts for rejected trials")
    add_study_args(parser)
    return parser.parse_args()


def storage_url(study_root: Path, filename: str = "study.db") -> str:
    """Return a sqlite storage URL for a study root."""

    db_path = (Path(study_root) / filename).resolve()
    return "sqlite:///" + str(db_path).replace("\\", "/")


def make_objective(
    study_root: Path,
    study_spec: Any,
    adapter: Any,
    cleanup_bad_trials: bool = False,
    gpu_allocator: Any = None,
) -> Any:
    """Create the Optuna objective closure."""

    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    thresholds = load_baseline_thresholds(threshold_path)

    def objective(trial: Any) -> float:
        try:
            from optuna import TrialPruned
        except ImportError as exc:
            from optuna_framework.study_utils import missing_dependency_message

            raise RuntimeError(missing_dependency_message("optuna")) from exc

        params = adapter.suggest_params(trial)
        materialized = adapter.materialize_params(params)
        trial_dir = study_root / "trials" / f"trial_{trial.number:05d}"
        init_trial_meta(trial_dir, trial.number, materialized, [segment.name for segment in study_spec.tuning_segments])
        log_status(f"trial={trial.number:05d} start", f"params={_format_params(materialized)}", f"dir={trial_dir}")
        segment_metrics = []
        sharpes = []
        gpu_lease = gpu_allocator.acquire() if gpu_allocator is not None else None
        fixed_overrides = dict(study_spec.fixed_overrides)
        if gpu_lease is not None:
            fixed_overrides["combo.model.device"] = gpu_lease.device
            log_status(f"trial={trial.number:05d} gpu={gpu_lease.device}")
        try:
            for step, segment in enumerate(study_spec.tuning_segments, start=1):
                run_paths = build_trial_run_paths(study_root, trial.number, segment)
                render_config(study_spec.baseline_config_path, run_paths, adapter, params, fixed_overrides)
                metrics = run_segment(run_paths)
                segment_metrics.append(metrics)
                sharpes.append(metrics.sharpe_idx)
                update_segment(trial_dir, segment.name, "complete", metrics.to_dict())
                score = running_score(sharpes)
                trial.report(score, step=step)
                log_status(
                    f"trial={trial.number:05d} segment={segment.name} progress",
                    f"step={step}/{len(study_spec.tuning_segments)}",
                    f"score={score:.6g}",
                    f"sharpe={metrics.sharpe_idx:.6g}",
                )
                if trial.should_prune():
                    update_trial_state(trial_dir, "pruned", objective=score)
                    log_status(f"trial={trial.number:05d} pruned", f"score={score:.6g}", f"step={step}")
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
            log_status(
                f"trial={trial.number:05d} done",
                f"objective={objective_value:.6g}",
                f"hard_filter={str(hard_filter_triggered).lower()}",
            )
            return objective_value
        except SegmentRunError as exc:
            update_trial_state(trial_dir, "failed")
            log_status(f"trial={trial.number:05d} failed", str(exc).splitlines()[0])
            raise
        finally:
            if gpu_lease is not None:
                gpu_lease.release()
                log_status(f"trial={trial.number:05d} gpu_released={gpu_lease.device}")

    return objective


def make_callback(study_root: Path) -> Any:
    """Create a lightweight reporting/resource callback."""

    lock = threading.Lock()

    def callback(study: Any, trial: Any) -> None:
        with lock:
            append_resource_metric(study_root, trial)
            write_study_reports(study, study_root)
            completed = completed_history_count(study)
            log_status(
                "progress",
                f"completed={completed}",
                f"last_trial={getattr(trial, 'number', '')}",
                f"last_state={str(getattr(trial, 'state', '')).split('.')[-1]}",
                f"best={_best_value_text(study)}",
            )
            failed = sum(1 for item in study.trials if str(item.state).endswith("FAIL"))
            if failed > 5:
                print(f"Warning: failed trial count is {failed}; manual review recommended.")

    return callback


def run_phase_a(args: argparse.Namespace) -> None:
    """Run the formal Phase A study."""

    study_spec, adapter, _plugin = load_study_and_adapter(args)
    study_root = resolve_study_root(args.study_root, study_spec.name)
    n_trials = study_spec.n_trials if args.n_trials is None else int(args.n_trials)
    devices = _parse_devices(args.gpus)
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else (len(devices) if devices else 1)
    if args.dry_run:
        print(f"[DRY-RUN] Phase A study_root={study_root}")
        print(f"[DRY-RUN] storage={storage_url(study_root)}")
        print(f"[DRY-RUN] n_trials={n_trials} n_jobs={n_jobs}")
        print(f"[DRY-RUN] startup_trials={8 if args.startup_trials is None else int(args.startup_trials)}")
        if devices:
            print(f"[DRY-RUN] devices={','.join(devices)}")
        for segment in study_spec.tuning_segments:
            run_paths = build_trial_run_paths(study_root, 0, segment)
            print_command(f"[DRY-RUN] trial_00000/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    study_root.mkdir(parents=True, exist_ok=True)
    gpu_allocator = GpuAllocator(devices) if devices else None
    optuna_study_name = study_spec.adapter_name or study_spec.name
    study = create_study(optuna_study_name, storage_url(study_root), smoke=False, n_startup_trials=args.startup_trials)
    maybe_enqueue_baseline(study, adapter.baseline_params())
    remaining = max(0, n_trials - completed_history_count(study))
    log_status(
        "study start",
        f"name={optuna_study_name}",
        f"study_root={study_root}",
        f"target_trials={n_trials}",
        f"remaining={remaining}",
        f"n_jobs={n_jobs}",
        f"devices={','.join(devices) if devices else 'default'}",
    )
    optimize_study(
        study,
        make_objective(study_root, study_spec, adapter, cleanup_bad_trials=args.cleanup_bad_trials, gpu_allocator=gpu_allocator),
        remaining,
        callbacks=[make_callback(study_root)],
        n_jobs=n_jobs,
    )
    write_study_reports(study, study_root)
    export_optuna_visualizations(study, study_root)
    log_status("study done", f"reports={study_root / 'reports'}")


def main() -> None:
    """Script entry point."""

    run_phase_a(parse_args())


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item.startswith("cuda:"):
            devices.append(item)
        else:
            devices.append(f"cuda:{int(item)}")
    return tuple(devices)


def _format_params(params: dict[str, Any], max_items: int = 12) -> str:
    items = list(params.items())
    shown = ", ".join(f"{key}={_short_value(value)}" for key, value in items[:max_items])
    if len(items) > max_items:
        shown += f", ...(+{len(items) - max_items})"
    return "{" + shown + "}"


def _short_value(value: Any, max_chars: int = 80) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _best_value_text(study: Any) -> str:
    try:
        best = study.best_value
    except Exception:
        return "none"
    return f"{best:.6g}"


if __name__ == "__main__":
    main()
