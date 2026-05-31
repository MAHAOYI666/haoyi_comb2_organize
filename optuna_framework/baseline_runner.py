"""Baseline evaluation runner shared by config and CLI entry points."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optuna_framework.aggregators import build_baseline_thresholds
from optuna_framework.config_renderer import render_config
from optuna_framework.gpu_allocation import GpuAllocator
from optuna_framework.metrics_parser import SegmentMetrics, parse_full_period
from optuna_framework.paths import build_baseline_run_paths
from optuna_framework.runner import build_run_command, command_to_string, run_segment
from optuna_framework.specs import SegmentSpec
from optuna_framework.status import log_status


@dataclass(frozen=True)
class BaselineSegmentResult:
    """Metrics produced by one baseline segment run."""

    segment: SegmentSpec
    metrics: SegmentMetrics
    full_by_year: dict[str, SegmentMetrics] | None = None


def run_baseline_evaluations(
    study_spec: Any,
    adapter: Any,
    study_root: Path,
    dry_run: bool = False,
    n_jobs: int = 1,
    devices: tuple[str, ...] = (),
) -> None:
    """Run baseline segments and write ``baseline_thresholds.json``."""

    params = adapter.baseline_params()
    segments = study_spec.baseline_segments()
    worker_count = _worker_count(n_jobs, devices)

    if dry_run:
        print(f"[DRY-RUN] baseline study_root={study_root}")
        print(f"[DRY-RUN] baseline n_jobs={worker_count}")
        if devices:
            print(f"[DRY-RUN] baseline devices={','.join(devices)}")
        for segment in segments:
            run_paths = build_baseline_run_paths(study_root, segment)
            print(f"[DRY-RUN] baseline/{segment.name} config={run_paths.config_path}")
            print(f"[DRY-RUN] baseline/{segment.name} cmd={command_to_string(build_run_command(run_paths.config_path))}")
        return

    study_root.mkdir(parents=True, exist_ok=True)
    gpu_allocator = GpuAllocator(devices) if devices else None
    log_status(
        "baseline start",
        f"study_root={study_root}",
        f"segments={len(segments)}",
        f"n_jobs={worker_count}",
        f"devices={','.join(devices) if devices else 'default'}",
    )
    results = _run_segments_parallel(
        study_spec,
        adapter,
        study_root,
        params,
        segments,
        worker_count,
        gpu_allocator,
    )
    _write_thresholds(study_root, results)
    log_status("baseline done", f"thresholds={study_root / 'baseline' / 'baseline_thresholds.json'}")


def _worker_count(n_jobs: int, devices: tuple[str, ...]) -> int:
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    if devices:
        return min(int(n_jobs), len(devices))
    return 1


def _run_segments_parallel(
    study_spec: Any,
    adapter: Any,
    study_root: Path,
    params: dict[str, Any],
    segments: tuple[SegmentSpec, ...],
    worker_count: int,
    gpu_allocator: GpuAllocator | None,
) -> list[BaselineSegmentResult]:
    if worker_count == 1:
        results = []
        for index, segment in enumerate(segments, start=1):
            results.append(_run_one_segment(study_spec, adapter, study_root, params, segment, gpu_allocator))
            log_status("baseline progress", f"completed={index}/{len(segments)}")
        return results

    results: list[BaselineSegmentResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_run_one_segment, study_spec, adapter, study_root, params, segment, gpu_allocator)
            for segment in segments
        ]
        for future in as_completed(futures):
            results.append(future.result())
            log_status("baseline progress", f"completed={len(results)}/{len(segments)}")
    return results


def _run_one_segment(
    study_spec: Any,
    adapter: Any,
    study_root: Path,
    params: dict[str, Any],
    segment: SegmentSpec,
    gpu_allocator: GpuAllocator | None,
) -> BaselineSegmentResult:
    gpu_lease = gpu_allocator.acquire() if gpu_allocator is not None else None
    fixed_overrides = dict(study_spec.fixed_overrides)
    if gpu_lease is not None:
        fixed_overrides["combo.model.device"] = gpu_lease.device
        log_status(f"baseline/{segment.name} gpu={gpu_lease.device}")
    try:
        run_paths = build_baseline_run_paths(study_root, segment)
        render_config(study_spec.baseline_config_path, run_paths, adapter, params, fixed_overrides)
        log_status(f"baseline/{segment.name} rendered", f"config={run_paths.config_path}")
        metrics = run_segment(run_paths)
        if segment.name == "full_period":
            return BaselineSegmentResult(segment, metrics, parse_full_period(run_paths.pnl_summary_path))
        return BaselineSegmentResult(segment, metrics)
    finally:
        if gpu_lease is not None:
            gpu_lease.release()
            log_status(f"baseline/{segment.name} gpu_released={gpu_lease.device}")


def _write_thresholds(study_root: Path, results: list[BaselineSegmentResult]) -> None:
    by_segment = {}
    full_metrics = None
    full_by_year = {}
    for result in results:
        if result.segment.name == "full_period":
            full_by_year = result.full_by_year or {}
            full_metrics = full_by_year.get("full")
        else:
            by_segment[result.segment.name] = result.metrics

    if full_metrics is None:
        raise RuntimeError("full_period baseline metrics were not produced")
    thresholds = build_baseline_thresholds(by_segment, full_metrics, full_by_year)
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline_thresholds={threshold_path}")
