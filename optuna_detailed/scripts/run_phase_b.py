"""Manual Phase B candidate stability runner for detailed configs."""

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

from optuna_detailed.aggregators import load_baseline_thresholds, require_tuning_period_baseline
from optuna_detailed.config_renderer import render_config
from optuna_detailed.metrics_parser import WindowMetrics
from optuna_detailed.paths import build_named_run_paths
from optuna_detailed.runner import build_run_command, run_inference
from optuna_detailed.scripts._script_common import add_common_config_args, add_plan_check_arg, ensure_plan_for_args, fixture_thresholds_path, load_config_from_args, print_command
from optuna_detailed.search_space import ConfigDrivenAdapter
from optuna_detailed.study_config import PhaseBConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase B seed stability checks.")
    add_common_config_args(parser)
    add_plan_check_arg(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    ensure_plan_for_args(config, args, dry_run=args.dry_run)
    adapter = ConfigDrivenAdapter(config)
    thresholds = _load_thresholds(config.study_root, args.dry_run)
    tuning_baseline = None if args.dry_run else require_tuning_period_baseline(thresholds)
    candidates = _load_candidates(config, adapter, args.dry_run)
    seeds = config.phase_b.seeds
    if args.dry_run:
        print(f"[DRY-RUN] Phase B study_root={config.study_root}")
        print(f"[DRY-RUN] candidates={len(candidates)} seeds={list(seeds)}")
        print(f"[DRY-RUN] run_window={config.tuning_run_window[0]}-{config.tuning_run_window[1]}")
        print(f"[DRY-RUN] scoring_window={config.scoring_window[0]}-{config.scoring_window[1]}")
        for cand_idx, _params in enumerate(candidates, start=1):
            for seed in seeds:
                subdir = f"phase_b/candidate_{cand_idx:02d}/seed_{seed}"
                run_paths = build_named_run_paths(
                    config.study_root,
                    subdir,
                    kind="phase_b",
                    run_window=config.tuning_run_window,
                    score_window=config.scoring_window,
                    snaptime=f"phase_b_c{cand_idx:02d}_seed_{seed}",
                )
                print_command(f"[DRY-RUN] {subdir}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    results = []
    survivors = []
    for cand_idx, params in enumerate(candidates, start=1):
        candidate_rows = []
        seed_metrics = []
        for seed in seeds:
            subdir = f"phase_b/candidate_{cand_idx:02d}/seed_{seed}"
            run_paths = build_named_run_paths(
                config.study_root,
                subdir,
                kind="phase_b",
                run_window=config.tuning_run_window,
                score_window=config.scoring_window,
                snaptime=f"phase_b_c{cand_idx:02d}_seed_{seed}",
            )
            render_config(config, run_paths, adapter, params, _seeded_overrides(seed))
            metric = run_inference(run_paths)
            seed_metrics.append((seed, metric))
            candidate_rows.append({"seed": seed, **metric.to_dict()})
        eliminated_reasons = _phase_b_rejection_reasons(seed_metrics, tuning_baseline, config.phase_b)
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
    phase_dir = config.study_root / "phase_b"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (phase_dir / "survivors.json").write_text(json.dumps(survivors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"phase_b_results={phase_dir / 'results.json'}")
    print(f"phase_b_survivors={phase_dir / 'survivors.json'}")


def _load_thresholds(study_root: Path, dry_run: bool) -> dict[str, Any]:
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    if threshold_path.exists():
        return load_baseline_thresholds(threshold_path)
    if dry_run:
        fixture = fixture_thresholds_path()
        print(f"[DRY-RUN] using fixture thresholds={fixture}")
        return load_baseline_thresholds(fixture)
    return load_baseline_thresholds(threshold_path)


def _load_candidates(config: Any, adapter: ConfigDrivenAdapter, dry_run: bool) -> list[dict[str, Any]]:
    top10_path = config.study_root / "reports" / "top10.csv"
    if top10_path.exists():
        df = pd.read_csv(top10_path)
        candidates = []
        for _, row in df.head(config.phase_b.candidate_scan_top_n).iterrows():
            params = {key.removeprefix("param_"): row[key] for key in row.index if key.startswith("param_")}
            if not params:
                continue
            candidates.append(adapter.normalize_params(params))
        distinct = _select_distinct(candidates, limit=config.phase_b.candidate_limit)
        if len(distinct) >= 1:
            return distinct[: config.phase_b.candidate_limit]
    if not dry_run:
        raise FileNotFoundError(f"Phase A top10.csv not found or empty: {top10_path}")
    return _dry_run_candidates(config, adapter)


def _select_distinct(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(candidate != existing for existing in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _dry_run_candidates(config: Any, adapter: ConfigDrivenAdapter) -> list[dict[str, Any]]:
    baseline = adapter.baseline_params()
    candidates = [baseline]
    for spec in config.params:
        if len(candidates) >= config.phase_b.candidate_limit:
            break
        variant = dict(baseline)
        if spec.param_type == "categorical":
            for choice in adapter._typed_choices(spec):
                if choice != baseline[spec.name]:
                    variant[spec.name] = choice
                    break
        elif spec.param_type == "float":
            low, high = float(spec.low), float(spec.high)
            candidate = min(high, max(low, float(baseline[spec.name]) * 2.0))
            if candidate == baseline[spec.name]:
                candidate = (low + high) / 2.0
            variant[spec.name] = candidate
        elif spec.param_type == "int":
            variant[spec.name] = int((int(float(spec.low)) + int(float(spec.high))) // 2)
        if variant != baseline and variant not in candidates:
            candidates.append(variant)
    return candidates[: config.phase_b.candidate_limit]


def _seeded_overrides(seed: int) -> dict[str, Any]:
    return {"combo.model.seed": int(seed)}


def _phase_b_rejection_reasons(
    seed_metrics: list[tuple[int, WindowMetrics]],
    baseline: dict[str, Any],
    phase_b: PhaseBConfig | None = None,
) -> list[str]:
    cfg = phase_b or PhaseBConfig(seeds=(42, 43, 44))
    reasons = []
    baseline_sharpe = float(baseline["sharpe_idx"])
    baseline_dd_li = float(baseline["dd_li"])
    sharpes = []
    for seed, metric in seed_metrics:
        sharpes.append(metric.sharpe_idx)
        if metric.sharpe_idx < baseline_sharpe - cfg.sharpe_margin:
            reasons.append(f"seed {seed} scoring-window sharpe_idx below baseline-{cfg.sharpe_margin:g}")
        if metric.dd_li > baseline_dd_li * cfg.dd_multiplier:
            reasons.append(f"seed {seed} scoring-window dd_li above baseline*{cfg.dd_multiplier:g}")
    if len(sharpes) == len(cfg.seeds) and float(np.std(sharpes)) > cfg.sharpe_std_max:
        reasons.append(f"three-seed scoring-window sharpe_idx std > {cfg.sharpe_std_max:g}")
    return sorted(set(reasons))


if __name__ == "__main__":
    main()
