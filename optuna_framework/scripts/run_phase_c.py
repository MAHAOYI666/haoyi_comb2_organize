"""Manual Phase C full-period holdout validation runner."""

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
from optuna_framework.paths import build_named_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_segment
from optuna_framework.scripts._script_common import adapter_for_name, fixture_thresholds_path, print_command
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase C full-period validation.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    return parser.parse_args()


def main() -> None:
    """Run or print Phase C commands."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_SPEC.name)
    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    thresholds = _load_thresholds(study_root, args.dry_run)
    candidates = _load_survivors(study_root, args.dry_run)
    full_segment = STUDY_SPEC.segment_by_name("full_period")
    if args.dry_run:
        print(f"[DRY-RUN] Phase C study_root={study_root}")
        print(f"[DRY-RUN] candidates={len(candidates)} seeds={list(SEEDS)}")
        for cand_idx, candidate in enumerate(candidates, start=1):
            for seed in _candidate_seeds(candidate):
                segment_dir = study_root / "phase_c" / candidate["candidate"] / f"seed_{seed}" / full_segment.name
                run_paths = build_named_run_paths(
                    study_root,
                    segment_dir,
                    full_segment,
                    snaptime=f"phase_c_{candidate['candidate']}_seed_{seed}_{full_segment.name}",
                    kind="phase_c",
                )
                print_command(
                    f"[DRY-RUN] phase_c/{candidate['candidate']}/seed_{seed}/full_period",
                    run_paths.config_path,
                    build_run_command(run_paths.config_path),
                )
        return

    results = []
    for candidate in candidates:
        seeds = _candidate_seeds(candidate)
        seed_results = []
        for seed in seeds:
            segment_dir = study_root / "phase_c" / candidate["candidate"] / f"seed_{seed}" / full_segment.name
            run_paths = build_named_run_paths(
                study_root,
                segment_dir,
                full_segment,
                snaptime=f"phase_c_{candidate['candidate']}_seed_{seed}_{full_segment.name}",
                kind="phase_c",
            )
            overrides = _seeded_overrides(seed)
            render_config(STUDY_SPEC.baseline_config_path, run_paths, adapter, candidate["params"], overrides)
            run_segment(run_paths)
            parsed = parse_full_period(run_paths.pnl_summary_path)
            seed_results.append({"seed": seed, "metrics": {key: value.to_dict() for key, value in parsed.items()}})
        accepted, reasons = _accept_candidate(seed_results, thresholds)
        results.append({**candidate, "seeds": list(seeds), "accepted": accepted, "reasons": reasons, "seed_results": seed_results})
    phase_dir = study_root / "phase_c"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"phase_c_results={phase_dir / 'results.json'}")


def _accept_candidate(seed_results: list[dict[str, Any]], thresholds: dict) -> tuple[bool, list[str]]:
    reasons = []
    baseline = thresholds["by_segment"]
    full_baseline = thresholds["full_period"]
    full_sharpes = []
    tuning_better_counts = []
    for seed_result in seed_results:
        metrics = seed_result["metrics"]
        full_sharpes.append(metrics["full"]["sharpe_idx"])
        if metrics["2020"]["sharpe_idx"] < baseline["holdout_2020"]["sharpe_idx"] - 0.3:
            reasons.append(f"seed {seed_result['seed']} holdout_2020 sharpe below threshold")
        if metrics["2024"]["sharpe_idx"] < baseline["holdout_2024h1"]["sharpe_idx"] - 0.3:
            reasons.append(f"seed {seed_result['seed']} holdout_2024h1 sharpe below threshold")
        if metrics["full"]["dd_li"] > full_baseline["dd_li"] * 1.3:
            reasons.append(f"seed {seed_result['seed']} full dd_li above threshold")
        better = 0
        for year in ("2021", "2022", "2023"):
            base_year = full_baseline["by_year"][year]["sharpe_idx"]
            if metrics[year]["sharpe_idx"] > base_year:
                better += 1
        tuning_better_counts.append(better)
    if min(tuning_better_counts) < 2:
        reasons.append("fewer than two tuning years beat baseline for at least one seed")
    if float(np.std(full_sharpes)) >= 0.25:
        reasons.append("full-period sharpe std across seeds >= 0.25")
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


def _load_survivors(study_root: Path, dry_run: bool) -> list[dict[str, Any]]:
    survivors_path = study_root / "phase_b" / "survivors.json"
    if survivors_path.exists():
        survivors = json.loads(survivors_path.read_text(encoding="utf-8"))
        return survivors
    if not dry_run:
        raise FileNotFoundError(f"Phase B survivors not found: {survivors_path}")
    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    return [{"candidate": "candidate_01", "params": adapter.baseline_params(), "seeds": list(SEEDS)}]


def _candidate_seeds(candidate: dict[str, Any]) -> tuple[int, ...]:
    raw_seeds = candidate.get("seeds")
    if raw_seeds is None:
        raw_seeds = sorted({int(row["seed"]) for row in candidate.get("metrics", []) if "seed" in row})
    if not raw_seeds:
        raw_seeds = SEEDS
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"Phase C expects exactly three distinct Phase B seeds, got: {seeds}")
    return seeds


def _seeded_overrides(seed: int) -> dict[str, Any]:
    return {
        **STUDY_SPEC.fixed_overrides,
        "combo.model.seed": int(seed),
    }


if __name__ == "__main__":
    main()
