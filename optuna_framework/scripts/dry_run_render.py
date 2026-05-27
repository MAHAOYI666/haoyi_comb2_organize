"""Render one baseline-equivalent full-window config under ``optuna_runs``."""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.config_renderer import OPTUNA_RUNTIME_PATCH_KEYS, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_baseline_run_paths, get_repo_root
from optuna_framework.scripts._script_common import adapter_for_name
from optuna_framework.studies.eg_torch_v1 import ADAPTER_NAME, BASELINE_CONFIG_PATH, FIXED_OVERRIDES, FULL_RUN_WINDOW, SCORING_WINDOW


def main() -> None:
    """Render a dry-run config and verify XML differences are whitelisted."""

    adapter = adapter_for_name(ADAPTER_NAME)
    dry_root = (get_repo_root() / "optuna_runs" / "dry_run_render").resolve()
    run_paths = build_baseline_run_paths(dry_root, FULL_RUN_WINDOW, SCORING_WINDOW)
    params = adapter.baseline_params()
    render_config(
        baseline_config_path=BASELINE_CONFIG_PATH,
        run_paths=run_paths,
        adapter=adapter,
        params=params,
        fixed_overrides=FIXED_OVERRIDES,
    )
    diffs = structured_xml_diff(BASELINE_CONFIG_PATH, run_paths.config_path)
    assert_only_allowed_diffs(diffs, OPTUNA_RUNTIME_PATCH_KEYS)
    print(f"rendered_config={run_paths.config_path}")
    print(f"run_window={FULL_RUN_WINDOW[0]}-{FULL_RUN_WINDOW[1]}")
    print(f"scoring_window={SCORING_WINDOW[0]}-{SCORING_WINDOW[1]}")
    print("xml_diff_allowed=true")
    for diff in diffs:
        print(f"diff {diff.key}: {diff.before!r} -> {diff.after!r}")


if __name__ == "__main__":
    main()

