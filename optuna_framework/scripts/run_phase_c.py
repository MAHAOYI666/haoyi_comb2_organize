"""Manual Phase C full-period holdout validation runner for detailed configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import load_baseline_thresholds
from optuna_framework.config_renderer import render_config
from optuna_framework.metrics_parser import parse_full_period
from optuna_framework.paths import build_named_run_paths
from optuna_framework.runner import build_run_command, run_inference
from optuna_framework.scripts._script_common import add_common_config_args, add_plan_check_arg, ensure_plan_for_args, fixture_thresholds_path, load_config_from_args, print_command
from optuna_framework.search_space import ConfigDrivenAdapter
from optuna_framework.study_config import PhaseBConfig, PhaseCConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase C full-period validation.")
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
    candidates = _load_survivors(config, adapter, args.dry_run)
    if args.dry_run:
        print(f"[DRY-RUN] Phase C study_root={config.study_root}")
        print(f"[DRY-RUN] candidates={len(candidates)} seeds={list(config.phase_b.seeds)}")
        print(f"[DRY-RUN] run_window={config.full_run_window[0]}-{config.full_run_window[1]}")
        for candidate in candidates:
            for seed in _candidate_seeds(candidate, config.phase_b):
                subdir = f"phase_c/{candidate['candidate']}/seed_{seed}"
                run_paths = build_named_run_paths(
                    config.study_root,
                    subdir,
                    kind="phase_c",
                    run_window=config.full_run_window,
                    snaptime=f"phase_c_{candidate['candidate']}_seed_{seed}",
                )
                print_command(f"[DRY-RUN] {subdir}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    results = []
    for candidate in candidates:
        seeds = _candidate_seeds(candidate, config.phase_b)
        seed_results = []
        for seed in seeds:
            subdir = f"phase_c/{candidate['candidate']}/seed_{seed}"
            run_paths = build_named_run_paths(
                config.study_root,
                subdir,
                kind="phase_c",
                run_window=config.full_run_window,
                snaptime=f"phase_c_{candidate['candidate']}_seed_{seed}",
            )
            render_config(config, run_paths, adapter, candidate["params"], _seeded_overrides(seed))
            run_inference(run_paths)
            parsed = parse_full_period(run_paths.pnl_summary_path)
            seed_results.append({"seed": seed, "metrics": {key: value.to_dict() for key, value in parsed.items()}})
        accepted, reasons = _accept_candidate(seed_results, thresholds, config.phase_c)
        results.append({**candidate, "seeds": list(seeds), "accepted": accepted, "reasons": reasons, "seed_results": seed_results})
    phase_dir = config.study_root / "phase_c"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"phase_c_results={phase_dir / 'results.json'}")


def _accept_candidate(
    seed_results: list[dict[str, Any]],
    thresholds: dict,
    phase_c: PhaseCConfig | None = None,
) -> tuple[bool, list[str]]:
    cfg = phase_c or PhaseCConfig(holdout_years=("2020", "2024"), tuning_years=("2021", "2022", "2023"))
    reasons = []
    holdout_baseline = thresholds["by_segment"]
    full_baseline = thresholds["full_period"]
    full_sharpes = []
    tuning_better_counts = []
    for seed_result in seed_results:
        metrics = seed_result["metrics"]
        full_sharpes.append(metrics["full"]["sharpe_idx"])
        for year in cfg.holdout_years:
            baseline_key = _holdout_key(holdout_baseline, year)
            if metrics[year]["sharpe_idx"] < holdout_baseline[baseline_key]["sharpe_idx"] - cfg.holdout_sharpe_margin:
                reasons.append(f"seed {seed_result['seed']} holdout_{year} sharpe below threshold")
        if metrics["full"]["dd_li"] > full_baseline["dd_li"] * cfg.full_dd_multiplier:
            reasons.append(f"seed {seed_result['seed']} full dd_li above threshold")
        better = 0
        for year in cfg.tuning_years:
            base_year = full_baseline["by_year"][year]["sharpe_idx"]
            if metrics[year]["sharpe_idx"] > base_year:
                better += 1
        tuning_better_counts.append(better)
    if min(tuning_better_counts) < cfg.min_tuning_years_better:
        reasons.append("fewer than two tuning years beat baseline for at least one seed")
    if float(np.std(full_sharpes)) >= cfg.full_sharpe_std_max:
        reasons.append(f"full-period sharpe std across seeds >= {cfg.full_sharpe_std_max:g}")
    return len(reasons) == 0, sorted(set(reasons))


def _load_thresholds(study_root: Path, dry_run: bool) -> dict[str, Any]:
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    if threshold_path.exists():
        return load_baseline_thresholds(threshold_path)
    if dry_run:
        fixture = fixture_thresholds_path()
        print(f"[DRY-RUN] using fixture thresholds={fixture}")
        return load_baseline_thresholds(fixture)
    return load_baseline_thresholds(threshold_path)


def _load_survivors(config: Any, adapter: ConfigDrivenAdapter, dry_run: bool) -> list[dict[str, Any]]:
    survivors_path = config.study_root / "phase_b" / "survivors.json"
    if survivors_path.exists():
        return json.loads(survivors_path.read_text(encoding="utf-8"))
    if not dry_run:
        raise FileNotFoundError(f"Phase B survivors not found: {survivors_path}")
    return [{"candidate": "candidate_01", "params": adapter.baseline_params(), "seeds": list(config.phase_b.seeds)}]


def _candidate_seeds(candidate: dict[str, Any], phase_b: PhaseBConfig | None = None) -> tuple[int, ...]:
    cfg = phase_b or PhaseBConfig(seeds=(42, 43, 44))
    raw_seeds = candidate.get("seeds")
    if raw_seeds is None:
        raw_seeds = sorted({int(row["seed"]) for row in candidate.get("metrics", []) if "seed" in row})
    if not raw_seeds:
        raw_seeds = cfg.seeds
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(seeds) != len(cfg.seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"Phase C expects {len(cfg.seeds)} distinct Phase B seeds, got: {seeds}")
    return seeds


def _seeded_overrides(seed: int) -> dict[str, Any]:
    return {"combo.model.seed": int(seed)}


def _holdout_key(holdout_baseline: dict[str, Any], year: str) -> str:
    candidates = [f"holdout_{year}"]
    if year == "2024":
        candidates.insert(0, "holdout_2024h1")
    for candidate in candidates:
        if candidate in holdout_baseline:
            return candidate
    raise KeyError(f"baseline thresholds missing holdout year {year}; checked {candidates}")


if __name__ == "__main__":
    main()
