"""Render one baseline-equivalent detailed config under ``optuna_runs``."""

from __future__ import annotations

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

import argparse

from optuna_framework.config_renderer import OPTUNA_RUNTIME_PATCH_KEYS, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_baseline_run_paths, get_repo_root
from optuna_framework.scripts._script_common import add_common_config_args, ensure_plan_for_args, load_config_from_args
from optuna_framework.search_space import ConfigDrivenAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a detailed dry-run config.")
    add_common_config_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    ensure_plan_for_args(config, args, dry_run=True)
    adapter = ConfigDrivenAdapter(config)
    dry_root = (get_repo_root() / "optuna_runs" / "dry_run_render_detailed").resolve()
    run_paths = build_baseline_run_paths(dry_root, config.full_run_window, config.scoring_window)
    params = adapter.baseline_params()
    render_config(config, run_paths, adapter, params)
    diffs = structured_xml_diff(config.baseline_config_path, run_paths.config_path)
    allowed_keys = set(OPTUNA_RUNTIME_PATCH_KEYS)
    allowed_keys.update(_override_diff_key(path) for path in config.fixed_overrides)
    assert_only_allowed_diffs(diffs, allowed_keys)
    print(f"rendered_config={run_paths.config_path}")
    print(f"run_window={config.full_run_window[0]}-{config.full_run_window[1]}")
    print(f"scoring_window={config.scoring_window[0]}-{config.scoring_window[1]}")
    print("xml_diff_allowed=true")
    for diff in diffs:
        print(f"diff {diff.key}: {diff.before!r} -> {diff.after!r}")


def _override_diff_key(dotted_path: str) -> str:
    parts = dotted_path.split(".")
    return "config." + ".".join(parts[:-1]) + f".@{parts[-1]}"


if __name__ == "__main__":
    main()
