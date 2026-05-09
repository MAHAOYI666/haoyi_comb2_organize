"""Path construction tests."""

from __future__ import annotations

from optuna_framework.paths import build_trial_run_paths, get_repo_root
from optuna_framework.specs import SegmentSpec


def test_get_repo_root_points_to_project() -> None:
    root = get_repo_root()
    assert (root / "runCombo.py").exists()
    assert (root / "optuna_framework").exists()


def test_trial_paths_are_unique_and_absolute(tmp_path) -> None:
    seg01 = SegmentSpec("seg01", "tuning", 20210104, 20211231)
    seg02 = SegmentSpec("seg02", "tuning", 20220104, 20221230)
    paths_a = build_trial_run_paths(tmp_path, 1, seg01)
    paths_b = build_trial_run_paths(tmp_path, 1, seg02)
    paths_c = build_trial_run_paths(tmp_path, 2, seg01)
    assert paths_a.output_root != paths_b.output_root
    assert paths_a.output_root != paths_c.output_root
    assert paths_a.checkpoint_root != paths_c.checkpoint_root
    assert paths_a.snaptime == "trial_00001_seg01"
    assert paths_a.config_path.is_absolute()
    assert paths_a.output_root.is_absolute()
    assert paths_a.checkpoint_root.is_absolute()

