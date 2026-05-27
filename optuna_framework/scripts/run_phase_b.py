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

from optuna_framework.aggregators import load_baseline_thresholds, require_tuning_period_baseline
from optuna_framework.config_renderer import render_config
from optuna_framework.metrics_parser import WindowMetrics
from optuna_framework.paths import build_named_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_inference
from optuna_framework.scripts._script_common import adapter_for_name, fixture_thresholds_path, print_command
from optuna_framework.studies.eg_torch_v1 import (
    ADAPTER_NAME,
    BASELINE_CONFIG_PATH,
    FIXED_OVERRIDES,
    SCORING_WINDOW,
    STUDY_NAME,
    TUNING_RUN_WINDOW,
)


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase B seed stability checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    return parser.parse_args()


def main() -> None:
    """Run or print Phase B commands."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_NAME)
    adapter = adapter_for_name(ADAPTER_NAME)
    thresholds = _load_thresholds(study_root, args.dry_run)
    tuning_baseline = None if args.dry_run else require_tuning_period_baseline(thresholds)
    candidates = _load_candidates(study_root, args.dry_run)
    if args.dry_run:
        print(f"[DRY-RUN] Phase B study_root={study_root}")
        print(f"[DRY-RUN] candidates={len(candidates)} seeds={list(SEEDS)}")
        print(f"[DRY-RUN] run_window={TUNING_RUN_WINDOW[0]}-{TUNING_RUN_WINDOW[1]}")
        print(f"[DRY-RUN] scoring_window={SCORING_WINDOW[0]}-{SCORING_WINDOW[1]}")
        for cand_idx, _params in enumerate(candidates, start=1):
            for seed in SEEDS:
                subdir = f"phase_b/candidate_{cand_idx:02d}/seed_{seed}"
                run_paths = build_named_run_paths(
                    study_root,
                    subdir,
                    kind="phase_b",
                    run_window=TUNING_RUN_WINDOW,
                    score_window=SCORING_WINDOW,
                    snaptime=f"phase_b_c{cand_idx:02d}_seed_{seed}",
                )
                print_command(f"[DRY-RUN] {subdir}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    results = []
    survivors = []
    for cand_idx, params in enumerate(candidates, start=1):
        candidate_rows = []
        seed_metrics = []
        for seed in SEEDS:
            subdir = f"phase_b/candidate_{cand_idx:02d}/seed_{seed}"
            run_paths = build_named_run_paths(
                study_root,
                subdir,
                kind="phase_b",
                run_window=TUNING_RUN_WINDOW,
                score_window=SCORING_WINDOW,
                snaptime=f"phase_b_c{cand_idx:02d}_seed_{seed}",
            )
            render_config(BASELINE_CONFIG_PATH, run_paths, adapter, params, _seeded_overrides(seed))
            metric = run_inference(run_paths)
            seed_metrics.append((seed, metric))
            candidate_rows.append({"seed": seed, **metric.to_dict()})
        eliminated_reasons = _phase_b_rejection_reasons(seed_metrics, tuning_baseline)
        eliminated = bool(eliminated_reasons)
        payload = {
            "candidate": f"candidate_{cand_idx:02d}",
            "params": params,
            "seeds": list(SEEDS),
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


def _load_thresholds(study_root: Path, dry_run: bool) -> dict[str, Any]:
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    if threshold_path.exists():
        return load_baseline_thresholds(threshold_path)
    if dry_run:
        fixture = fixture_thresholds_path()
        print(f"[DRY-RUN] using fixture thresholds={fixture}")
        return load_baseline_thresholds(fixture)
    return load_baseline_thresholds(threshold_path)


def _load_candidates(study_root: Path, dry_run: bool) -> list[dict[str, Any]]:
    top10_path = study_root / "reports" / "top10.csv"
    if top10_path.exists():
        df = pd.read_csv(top10_path)
        candidates = []
        for _, row in df.head(6).iterrows():
            params = {key.removeprefix("param_"): row[key] for key in row.index if key.startswith("param_")}
            if not params:
                continue
            candidates.append(_normalize_params(params))
        distinct = _select_distinct(candidates, limit=3)
        if len(distinct) >= 1:
            return distinct[:3]
    if not dry_run:
        raise FileNotFoundError(f"Phase A top10.csv not found or empty: {top10_path}")
    adapter = adapter_for_name(ADAPTER_NAME)
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


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    adapter = adapter_for_name(ADAPTER_NAME)
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


def _seeded_overrides(seed: int) -> dict[str, Any]:
    return {
        **FIXED_OVERRIDES,
        "combo.model.seed": int(seed),
    }


def _phase_b_rejection_reasons(seed_metrics: list[tuple[int, WindowMetrics]], baseline: dict[str, Any]) -> list[str]:
    reasons = []
    baseline_sharpe = float(baseline["sharpe_idx"])
    baseline_dd_li = float(baseline["dd_li"])
    sharpes = []
    for seed, metric in seed_metrics:
        sharpes.append(metric.sharpe_idx)
        if metric.sharpe_idx < baseline_sharpe - 0.15:
            reasons.append(f"seed {seed} scoring-window sharpe_idx below baseline-0.15")
        if metric.dd_li > baseline_dd_li * 1.15:
            reasons.append(f"seed {seed} scoring-window dd_li above baseline*1.15")
    if len(sharpes) == len(SEEDS) and float(np.std(sharpes)) > 0.15:
        reasons.append("three-seed scoring-window sharpe_idx std > 0.15")
    return sorted(set(reasons))


if __name__ == "__main__":
    main()
