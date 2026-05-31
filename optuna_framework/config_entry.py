"""Config-driven Optuna entry point for ``runCombo.py config.xml``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from optuna_framework.aggregators import build_baseline_thresholds
from optuna_framework.config_renderer import render_config
from optuna_framework.gpu_allocation import GpuAllocator
from optuna_framework.metrics_parser import parse_full_period
from optuna_framework.paths import build_baseline_run_paths, build_trial_run_paths
from optuna_framework.plugin_loader import load_plugin
from optuna_framework.runner import build_run_command, command_to_string, run_segment
from optuna_framework.scripts.run_study import make_callback, make_objective, storage_url
from optuna_framework.study_utils import (
    completed_history_count,
    create_study,
    export_optuna_visualizations,
    maybe_enqueue_baseline,
    optimize_study,
    write_study_reports,
)


@dataclass(frozen=True)
class ConfigOptunaRun:
    """Optuna settings parsed from one XML config."""

    config_path: Path
    enabled: bool
    plugin_path: Path
    study_root: Path
    mode: str
    n_trials: int | None
    smoke_trials: int
    startup_trials: int | None
    smoke_startup_trials: int | None
    n_jobs: int
    devices: tuple[str, ...]
    cleanup_bad_trials: bool
    dry_run: bool


def is_optuna_enabled(config_path: str | Path | None) -> bool:
    """Return whether the XML config requests Optuna orchestration."""

    if config_path is None:
        return False
    optuna_element = _read_optuna_element(Path(config_path))
    if optuna_element is None:
        return False
    return _parse_bool(optuna_element.get("enabled", "false"))


def run_optuna_from_config(config_path: str | Path) -> bool:
    """Run the config-declared Optuna workflow.

    Returns ``True`` when the config was an Optuna config and normal combo
    execution should be skipped.
    """

    run_config = load_optuna_run_config(config_path)
    if not run_config.enabled:
        return False

    loaded = load_plugin(plugin=run_config.plugin_path)
    study_spec = replace(
        loaded.study_spec,
        baseline_config_path=run_config.config_path,
        plugin_path=run_config.plugin_path,
    )
    if run_config.n_trials is not None:
        study_spec = replace(study_spec, n_trials=run_config.n_trials)

    print(
        "[OPTUNA]",
        f"mode={run_config.mode}",
        f"plugin={run_config.plugin_path}",
        f"study_root={run_config.study_root}",
        f"n_jobs={run_config.n_jobs}",
        f"devices={','.join(run_config.devices) if run_config.devices else 'config-default'}",
    )

    if run_config.mode == "baseline":
        _run_baseline(study_spec, loaded.adapter, run_config.study_root, dry_run=run_config.dry_run)
    elif run_config.mode == "smoke":
        _run_smoke(
            study_spec,
            loaded.adapter,
            run_config.study_root,
            run_config.smoke_trials,
            dry_run=run_config.dry_run,
            n_jobs=run_config.n_jobs,
            devices=run_config.devices,
            startup_trials=run_config.smoke_startup_trials,
        )
    elif run_config.mode == "study":
        _run_study(
            study_spec,
            loaded.adapter,
            run_config.study_root,
            cleanup_bad_trials=run_config.cleanup_bad_trials,
            dry_run=run_config.dry_run,
            n_jobs=run_config.n_jobs,
            devices=run_config.devices,
            startup_trials=run_config.startup_trials,
        )
    elif run_config.mode == "auto":
        threshold_path = run_config.study_root / "baseline" / "baseline_thresholds.json"
        if not threshold_path.exists() or run_config.dry_run:
            _run_baseline(study_spec, loaded.adapter, run_config.study_root, dry_run=run_config.dry_run)
        _run_study(
            study_spec,
            loaded.adapter,
            run_config.study_root,
            cleanup_bad_trials=run_config.cleanup_bad_trials,
            dry_run=run_config.dry_run,
            n_jobs=run_config.n_jobs,
            devices=run_config.devices,
            startup_trials=run_config.startup_trials,
        )
    else:
        raise ValueError(f"unsupported optuna mode: {run_config.mode}")
    return True


def load_optuna_run_config(config_path: str | Path) -> ConfigOptunaRun:
    """Parse and resolve the ``<optuna>`` element from an XML config."""

    config_path = Path(config_path).expanduser().resolve()
    root = ET.parse(config_path).getroot()
    optuna_element = root.find("optuna")
    if optuna_element is None:
        model_dir = _model_dir_from_config(root, config_path)
        plugin_path = model_dir / "optuna_plugin.py"
        return ConfigOptunaRun(
            config_path=config_path,
            enabled=False,
            plugin_path=plugin_path.resolve(),
            study_root=(config_path.parent / "optuna_runs").resolve(),
            mode="auto",
            n_trials=None,
            smoke_trials=3,
            startup_trials=None,
            smoke_startup_trials=None,
            n_jobs=1,
            devices=(),
            cleanup_bad_trials=False,
            dry_run=False,
        )

    enabled = _parse_bool(optuna_element.get("enabled", "false"))
    mode = _normalize_mode(optuna_element.get("mode", "auto"))
    model_dir = _model_dir_from_config(root, config_path)
    raw_plugin = (
        optuna_element.get("path")
        or optuna_element.get("plugin")
        or optuna_element.get("plugin_path")
        or "optuna_plugin.py"
    )
    plugin_path = _resolve_model_relative_path(raw_plugin, model_dir)
    raw_study_root = optuna_element.get("study_root")
    if raw_study_root:
        study_root = _resolve_config_relative_path(raw_study_root, config_path.parent)
    else:
        study_root = (config_path.parent / "optuna_runs").resolve()
    n_trials = _parse_optional_int(optuna_element.get("n_trials"))
    smoke_trials = _parse_optional_int(optuna_element.get("smoke_trials")) or 3
    startup_trials = _parse_optional_int(optuna_element.get("startup_trials"))
    smoke_startup_trials = _parse_optional_int(optuna_element.get("smoke_startup_trials"))
    devices = _parse_devices(optuna_element.get("devices") or optuna_element.get("gpus") or "")
    n_jobs = _parse_optional_int(optuna_element.get("n_jobs") or optuna_element.get("max_parallel"))
    if n_jobs is None:
        n_jobs = len(devices) if devices else 1
    cleanup_bad_trials = _parse_bool(optuna_element.get("cleanup_bad_trials", "false"))
    dry_run = _parse_bool(optuna_element.get("dry_run", "false"))
    return ConfigOptunaRun(
        config_path=config_path,
        enabled=enabled,
        plugin_path=plugin_path,
        study_root=study_root,
        mode=mode,
        n_trials=n_trials,
        smoke_trials=smoke_trials,
        startup_trials=startup_trials,
        smoke_startup_trials=smoke_startup_trials,
        n_jobs=n_jobs,
        devices=devices,
        cleanup_bad_trials=cleanup_bad_trials,
        dry_run=dry_run,
    )


def _run_baseline(study_spec: Any, adapter: Any, study_root: Path, dry_run: bool = False) -> None:
    params = adapter.baseline_params()
    if dry_run:
        print(f"[DRY-RUN] baseline study_root={study_root}")
        for segment in study_spec.baseline_segments():
            run_paths = build_baseline_run_paths(study_root, segment)
            _print_command(f"[DRY-RUN] baseline/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    by_segment = {}
    full_metrics = None
    full_by_year = {}
    for segment in study_spec.baseline_segments():
        run_paths = build_baseline_run_paths(study_root, segment)
        render_config(study_spec.baseline_config_path, run_paths, adapter, params, study_spec.fixed_overrides)
        metrics = run_segment(run_paths)
        if segment.name == "full_period":
            full_by_year = parse_full_period(run_paths.pnl_summary_path)
            full_metrics = full_by_year["full"]
        else:
            by_segment[segment.name] = metrics

    if full_metrics is None:
        raise RuntimeError("full_period baseline metrics were not produced")
    thresholds = build_baseline_thresholds(by_segment, full_metrics, full_by_year)
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(_json_dumps(thresholds), encoding="utf-8")
    print(f"baseline_thresholds={threshold_path}")


def _run_study(
    study_spec: Any,
    adapter: Any,
    study_root: Path,
    cleanup_bad_trials: bool = False,
    dry_run: bool = False,
    n_jobs: int = 1,
    devices: tuple[str, ...] = (),
    startup_trials: int | None = None,
) -> None:
    if dry_run:
        print(f"[DRY-RUN] Phase A study_root={study_root}")
        print(f"[DRY-RUN] storage={storage_url(study_root)}")
        print(f"[DRY-RUN] n_trials={study_spec.n_trials} n_jobs={int(n_jobs)}")
        print(f"[DRY-RUN] startup_trials={8 if startup_trials is None else int(startup_trials)}")
        if devices:
            print(f"[DRY-RUN] devices={','.join(devices)}")
        for segment in study_spec.tuning_segments:
            run_paths = build_trial_run_paths(study_root, 0, segment)
            _print_command(f"[DRY-RUN] trial_00000/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    study_root.mkdir(parents=True, exist_ok=True)
    gpu_allocator = GpuAllocator(devices) if devices else None
    optuna_study_name = study_spec.adapter_name or study_spec.name
    study = create_study(optuna_study_name, storage_url(study_root), smoke=False, n_startup_trials=startup_trials)
    maybe_enqueue_baseline(study, adapter.baseline_params())
    remaining = max(0, study_spec.n_trials - completed_history_count(study))
    optimize_study(
        study,
        make_objective(study_root, study_spec, adapter, cleanup_bad_trials=cleanup_bad_trials, gpu_allocator=gpu_allocator),
        remaining,
        callbacks=[make_callback(study_root)],
        n_jobs=int(n_jobs),
    )
    write_study_reports(study, study_root)
    export_optuna_visualizations(study, study_root)


def _run_smoke(
    study_spec: Any,
    adapter: Any,
    study_root: Path,
    n_trials: int,
    dry_run: bool = False,
    n_jobs: int = 1,
    devices: tuple[str, ...] = (),
    startup_trials: int | None = None,
) -> None:
    resolved_startup_trials = 2 if startup_trials is None else int(startup_trials)
    if dry_run:
        print(f"[DRY-RUN] smoke study_root={study_root}")
        print(f"[DRY-RUN] storage={storage_url(study_root, 'study_smoke.db')}")
        print(f"[DRY-RUN] sampler=TPESampler(n_startup_trials={resolved_startup_trials}, multivariate=True, group=True, seed=42)")
        print(f"[DRY-RUN] n_trials={int(n_trials)} n_jobs={int(n_jobs)}")
        if devices:
            print(f"[DRY-RUN] devices={','.join(devices)}")
        for segment in study_spec.tuning_segments:
            run_paths = build_trial_run_paths(study_root, 0, segment)
            _print_command(f"[DRY-RUN] smoke/trial_00000/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    study_root.mkdir(parents=True, exist_ok=True)
    gpu_allocator = GpuAllocator(devices) if devices else None
    optuna_study_name = (study_spec.adapter_name or study_spec.name) + "_smoke"
    study = create_study(optuna_study_name, storage_url(study_root, "study_smoke.db"), smoke=True, n_startup_trials=startup_trials)
    maybe_enqueue_baseline(study, adapter.baseline_params())
    remaining = max(0, int(n_trials) - completed_history_count(study))
    optimize_study(
        study,
        make_objective(study_root, study_spec, adapter, gpu_allocator=gpu_allocator),
        remaining,
        callbacks=[make_callback(study_root)],
        n_jobs=int(n_jobs),
    )
    write_study_reports(study, study_root)


def _read_optuna_element(config_path: Path) -> ET.Element | None:
    if not config_path.exists() or config_path.suffix.lower() != ".xml":
        return None
    root = ET.parse(config_path).getroot()
    if root.tag != "config":
        return None
    return root.find("optuna")


def _model_dir_from_config(root: ET.Element, config_path: Path) -> Path:
    paths = root.find("./combo/paths")
    model_path = paths.get("model_path") if paths is not None else None
    if not model_path:
        return config_path.parent
    resolved_model_path = Path(model_path).expanduser()
    if not resolved_model_path.is_absolute():
        resolved_model_path = config_path.parent / resolved_model_path
    return resolved_model_path.resolve().parent


def _resolve_model_relative_path(raw_path: str, model_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = model_dir / path
    return path.resolve()


def _resolve_config_relative_path(raw_path: str, config_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "baseline_then_study": "auto",
        "baseline_study": "auto",
        "search": "study",
        "phase_a": "study",
    }
    return aliases.get(normalized, normalized)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


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


def _print_command(prefix: str, config_path: Path, cmd: list[str]) -> None:
    print(f"{prefix} config={config_path}")
    print(f"{prefix} cmd={command_to_string(cmd)}")


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
