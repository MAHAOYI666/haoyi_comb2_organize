"""Manual Phase B candidate stability runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import load_baseline_thresholds
from optuna_framework.config_renderer import render_config
from optuna_framework.paths import build_named_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_segment
from optuna_framework.scripts._script_common import add_study_args, fixture_thresholds_path, load_study_and_adapter, print_command


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase B seed stability checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    add_study_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run or print Phase B commands."""

    args = parse_args()
    study_spec, adapter, _plugin = load_study_and_adapter(args)
    study_root = resolve_study_root(args.study_root, study_spec.name)
    thresholds = _load_thresholds(study_root, args.dry_run)
    candidates = _load_candidates(study_root, args.dry_run, adapter)
    seeds = _phase_seeds(study_spec, adapter)
    if args.dry_run:
        print(f"[DRY-RUN] Phase B study_root={study_root}")
        print(f"[DRY-RUN] candidates={len(candidates)} seeds={list(seeds)}")
        for cand_idx, _params in enumerate(candidates, start=1):
            for seed in seeds:
                for segment in study_spec.tuning_segments:
                    segment_dir = study_root / "phase_b" / f"candidate_{cand_idx:02d}" / f"seed_{seed}" / segment.name
                    run_paths = build_named_run_paths(
                        study_root,
                        segment_dir,
                        segment,
                        snaptime=f"phase_b_c{cand_idx:02d}_seed_{seed}_{segment.name}",
                        kind="phase_b",
                    )
                    print_command(
                        f"[DRY-RUN] phase_b/candidate_{cand_idx:02d}/seed_{seed}/{segment.name}",
                        run_paths.config_path,
                        build_run_command(run_paths.config_path),
                    )
        return

    results = []
    survivors = []
    for cand_idx, params in enumerate(candidates, start=1):
        candidate_rows = []
        eliminated_reasons = []
        seed_mean_sharpes = []
        for seed in seeds:
            seed_metrics = []
            for segment in study_spec.tuning_segments:
                segment_dir = study_root / "phase_b" / f"candidate_{cand_idx:02d}" / f"seed_{seed}" / segment.name
                run_paths = build_named_run_paths(
                    study_root,
                    segment_dir,
                    segment,
                    snaptime=f"phase_b_c{cand_idx:02d}_seed_{seed}_{segment.name}",
                    kind="phase_b",
                )
                run_paths.segment_dir.mkdir(parents=True, exist_ok=True)
                seeded_model_path = adapter.write_seeded_model(run_paths.segment_dir, study_spec.baseline_config_path, "phase_b")
                overrides = {**study_spec.fixed_overrides, **adapter.seeded_overrides(seed, seeded_model_path)}
                render_config(study_spec.baseline_config_path, run_paths, adapter, params, overrides)
                metric = run_segment(run_paths)
                seed_metrics.append(metric)
                candidate_rows.append({"seed": seed, "segment": segment.name, **metric.to_dict()})
                segment_limits = thresholds["hard_filter_by_segment"][segment.name]
                if metric.sharpe_idx < segment_limits["min_sharpe"]:
                    eliminated_reasons.append(f"seed {seed} {segment.name} sharpe below baseline-0.3")
                if metric.dd_li > segment_limits["max_dd"]:
                    eliminated_reasons.append(f"seed {seed} {segment.name} dd_li above baseline*1.3")
            seed_mean_sharpes.append(float(np.mean([metric.sharpe_idx for metric in seed_metrics])))
        if np.std(seed_mean_sharpes) > 0.3:
            eliminated_reasons.append("three-seed mean sharpe std > 0.3")
        eliminated = bool(eliminated_reasons)
        payload = {
            "candidate": f"candidate_{cand_idx:02d}",
            "params": params,
            "seeds": list(seeds),
            "eliminated": eliminated,
            "reasons": sorted(set(eliminated_reasons)),
            "metrics": candidate_rows,
        }
        results.append(payload)
        if not eliminated:
            survivors.append(payload)
    phase_dir = study_root / "phase_b"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (phase_dir / "survivors.json").write_text(json.dumps(survivors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"phase_b_results={phase_dir / 'results.json'}")
    print(f"phase_b_survivors={phase_dir / 'survivors.json'}")


def _phase_seeds(study_spec: Any, adapter: Any) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in getattr(study_spec, "phase_b_seeds", ()) or adapter.phase_seeds())
    if not seeds:
        seeds = adapter.phase_seeds()
    return seeds


def _load_thresholds(study_root: Path, dry_run: bool) -> dict[str, Any]:
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    if threshold_path.exists():
        return load_baseline_thresholds(threshold_path)
    if dry_run:
        fixture = fixture_thresholds_path()
        print(f"[DRY-RUN] using fixture thresholds={fixture}")
        return load_baseline_thresholds(fixture)
    return load_baseline_thresholds(threshold_path)


def _load_candidates(study_root: Path, dry_run: bool, adapter: Any) -> list[dict[str, Any]]:
    top10_path = study_root / "reports" / "top10.csv"
    if top10_path.exists():
        df = pd.read_csv(top10_path)
        candidates = []
        for _, row in df.head(6).iterrows():
            params = {key.removeprefix("param_"): row[key] for key in row.index if key.startswith("param_")}
            if not params:
                continue
            candidates.append(_normalize_params(params, adapter))
        distinct = _select_distinct(candidates, limit=3)
        if len(distinct) >= 1:
            return distinct[:3]
    if not dry_run:
        raise FileNotFoundError(f"Phase A top10.csv not found or empty: {top10_path}")
    baseline = adapter.baseline_params()
    variant1 = {**baseline, "lr": 5e-6, "dropout": 0.4}
    variant2 = {**baseline, "hiddenSize": 384, "fcSize": 128}
    return [baseline, variant1, variant2]


def _select_distinct(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(candidate != existing for existing in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _normalize_params(params: dict[str, Any], adapter: Any) -> dict[str, Any]:
    baseline = adapter.baseline_params()
    normalized = {}
    for key, default in baseline.items():
        value = params.get(key, default)
        if isinstance(default, int) and not isinstance(default, bool):
            normalized[key] = int(value)
        elif isinstance(default, float):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


if __name__ == "__main__":
    main()
