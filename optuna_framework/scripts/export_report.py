"""Export a markdown report from completed manual workflow artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.paths import resolve_study_root
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Export Optuna workflow REPORT.md.")
    parser.add_argument("--dry-run", action="store_true", help="Print report path without writing")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    return parser.parse_args()


def main() -> None:
    """Export or print the report target."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_SPEC.name)
    report_path = study_root / "REPORT.md"
    if args.dry_run:
        print(f"[DRY-RUN] report_path={report_path}")
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report(study_root), encoding="utf-8")
    print(f"report={report_path}")


def _build_report(study_root: Path) -> str:
    baseline = _read_json(study_root / "baseline" / "baseline_thresholds.json")
    phase_b = _read_json(study_root / "phase_b" / "results.json")
    phase_c = _read_json(study_root / "phase_c" / "results.json")
    top10 = _read_csv(study_root / "reports" / "top10.csv")
    trials = _read_csv(study_root / "reports" / "trials.csv")
    title = "Optuna eg_torch_v1 report"
    if phase_c is not None and not any(item.get("accepted") for item in phase_c):
        title = "no improvement found"
    lines = [f"# {title}", ""]
    lines.append("## Experiment")
    lines.append(f"- study: {STUDY_SPEC.name}")
    lines.append(f"- adapter: {STUDY_SPEC.adapter_name}")
    lines.append(f"- baseline config: {STUDY_SPEC.baseline_config_path}")
    lines.append(f"- fixed_overrides: `{STUDY_SPEC.fixed_overrides}`")
    lines.append("")
    lines.append("## Segments")
    for segment in STUDY_SPEC.baseline_segments():
        lines.append(f"- {segment.name}: {segment.role} {segment.start_ds}-{segment.end_ds}")
    lines.append("")
    lines.append("## Baseline")
    if baseline:
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
