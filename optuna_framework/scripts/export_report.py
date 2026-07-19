"""Export a markdown report from completed detailed workflow artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.scripts._script_common import add_common_config_args, add_plan_check_arg, ensure_plan_for_args, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detailed Optuna workflow REPORT.md.")
    add_common_config_args(parser)
    add_plan_check_arg(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print report path without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    ensure_plan_for_args(config, args, dry_run=args.dry_run)
    report_path = config.study_root / "REPORT.md"
    if args.dry_run:
        print(f"[DRY-RUN] report_path={report_path}")
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report(config), encoding="utf-8")
    print(f"report={report_path}")


def _build_report(config) -> str:
    baseline = _read_json(config.study_root / "baseline" / "baseline_thresholds.json")
    phase_b = _read_json(config.study_root / "phase_b" / "results.json")
    phase_c = _read_json(config.study_root / "phase_c" / "results.json")
    top10 = _read_csv(config.study_root / "reports" / "top10.csv")
    trials = _read_csv(config.study_root / "reports" / "trials.csv")
    title = f"Optuna {config.study_name} report"
    if phase_c is not None and not any(item.get("accepted") for item in phase_c):
        title = "no improvement found"
    lines = [f"# {title}", ""]
    lines.append("## Experiment")
    lines.append(f"- study: {config.study_name}")
    lines.append(f"- optuna_name: {config.optuna_name}")
    lines.append(f"- config: {config.config_path}")
    lines.append(f"- baseline config: {config.baseline_config_path}")
    lines.append(f"- fixed_overrides: `{config.fixed_overrides}`")
    lines.append("")
    lines.append("## Windows")
    lines.append(f"- baseline and Phase C run: {config.full_run_window[0]}-{config.full_run_window[1]}")
    lines.append(f"- Phase A/B run: {config.tuning_run_window[0]}-{config.tuning_run_window[1]}")
    lines.append(f"- Phase A/B scoring: {config.scoring_window[0]}-{config.scoring_window[1]}")
    lines.append("")
    lines.append("## Baseline")
    if baseline:
        tuning = baseline.get("tuning_period", {})
        if tuning:
            lines.append(f"- tuning_period: sharpe_idx={tuning.get('sharpe_idx')}, dd_li={tuning.get('dd_li')}, days={tuning.get('days')}")
        for name, metric in baseline.get("by_segment", {}).items():
            lines.append(f"- {name}: sharpe_idx={metric['sharpe_idx']}, dd_li={metric['dd_li']}, days={metric['days']}")
        full = baseline.get("full_period", {})
        if full:
            lines.append(f"- full_period: sharpe_idx={full.get('sharpe_idx')}, dd_li={full.get('dd_li')}, days={full.get('days')}")
    else:
        lines.append("- baseline_thresholds.json missing")
    lines.append("")
    lines.append("## Phase A")
    if trials is not None and not trials.empty:
        state_counts = trials["state"].value_counts().to_dict() if "state" in trials else {}
        lines.append(f"- state counts: `{state_counts}`")
    else:
        lines.append("- trials.csv missing")
    lines.append("")
    lines.append("## Top 10")
    if top10 is not None and not top10.empty:
        lines.append(top10.to_markdown(index=False))
    else:
        lines.append("- top10.csv missing")
    lines.append("")
    lines.append("## Phase B")
    if phase_b:
        for item in phase_b:
            lines.append(f"- {item['candidate']}: eliminated={item.get('eliminated')} reasons={item.get('reasons')}")
    else:
        lines.append("- Phase B results missing")
    lines.append("")
    lines.append("## Phase C")
    if phase_c:
        for item in phase_c:
            lines.append(f"- {item['candidate']}: accepted={item.get('accepted')} reasons={item.get('reasons')}")
    else:
        lines.append("- Phase C results missing")
    lines.append("")
    lines.append("## Recommendation")
    accepted = [item for item in phase_c or [] if item.get("accepted")]
    if accepted:
        lines.append(f"- recommended: {accepted[0]['candidate']}")
        lines.append(f"- params: `{accepted[0].get('params')}`")
    else:
        lines.append("- no improvement found; review failed holdout or stability criteria before changing production config")
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
