"""Render one baseline-equivalent seg01 config to ``/tmp/dry_run``."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.config_renderer import OPTUNA_RUNTIME_PATCH_KEYS, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_baseline_run_paths
from optuna_framework.scripts._script_common import add_study_args, load_study_and_adapter


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Render one baseline-equivalent Optuna config.")
    parser.add_argument("--dry-root", default="/tmp/dry_run", help="Directory for the rendered dry-run config")
    add_study_args(parser)
    return parser.parse_args()


def main() -> None:
    """Render a dry-run config and verify XML differences are whitelisted."""

    args = parse_args()
    study_spec, adapter, _plugin = load_study_and_adapter(args)
    segment = study_spec.segment_by_name("seg01")
    dry_root = Path(args.dry_root).expanduser().resolve()
    run_paths = build_baseline_run_paths(dry_root, segment)
    params = adapter.baseline_params()
    render_config(
        baseline_config_path=study_spec.baseline_config_path,
        run_paths=run_paths,
        adapter=adapter,
        params=params,
        fixed_overrides=study_spec.fixed_overrides,
    )
    diffs = structured_xml_diff(study_spec.baseline_config_path, run_paths.config_path)
    assert_only_allowed_diffs(diffs, OPTUNA_RUNTIME_PATCH_KEYS)
    print(f"rendered_config={run_paths.config_path}")
    print("xml_diff_allowed=true")
    for diff in diffs:
        print(f"diff {diff.key}: {diff.before!r} -> {diff.after!r}")


if __name__ == "__main__":
    main()
