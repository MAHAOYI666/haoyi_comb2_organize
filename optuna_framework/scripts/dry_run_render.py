"""Render one baseline-equivalent seg01 config to ``/tmp/dry_run``."""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.config_renderer import OPTUNA_RUNTIME_PATCH_KEYS, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_baseline_run_paths
from optuna_framework.scripts._script_common import adapter_for_name
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC


def main() -> None:
    """Render a dry-run config and verify XML differences are whitelisted."""

    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    segment = STUDY_SPEC.segment_by_name("seg01")
    dry_root = Path("/tmp/dry_run").resolve()
    run_paths = build_baseline_run_paths(dry_root, segment)
    params = adapter.baseline_params()
    render_config(
        baseline_config_path=STUDY_SPEC.baseline_config_path,
        run_paths=run_paths,
        adapter=adapter,
        params=params,
        fixed_overrides=STUDY_SPEC.fixed_overrides,
    )
    diffs = structured_xml_diff(STUDY_SPEC.baseline_config_path, run_paths.config_path)
    assert_only_allowed_diffs(diffs, OPTUNA_RUNTIME_PATCH_KEYS)
    print(f"rendered_config={run_paths.config_path}")
    print("xml_diff_allowed=true")
    for diff in diffs:
        print(f"diff {diff.key}: {diff.before!r} -> {diff.after!r}")


if __name__ == "__main__":
    main()

