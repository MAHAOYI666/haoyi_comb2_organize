"""Path construction tests."""

from __future__ import annotations

from optuna_framework.paths import build_baseline_run_paths, build_named_run_paths, build_trial_run_paths, get_repo_root


def test_get_repo_root_points_to_project() -> None:
    root = get_repo_root()
    assert (root / "runCombo.py").exists()
    assert (root / "optuna_framework").exists()


def test_new_run_paths_are_absolute_and_follow_layout(tmp_path) -> None:
    baseline = build_baseline_run_paths(tmp_path, (20200102, 20240628), (20210104, 20231229))
    trial_1 = build_trial_run_paths(tmp_path, 1, (20200102, 20231229), (20210104, 20231229))
    trial_2 = build_trial_run_paths(tmp_path, 2, (20200102, 20231229), (20210104, 20231229))
    phase_b = build_named_run_paths(
        tmp_path,
        "phase_b/candidate_01/seed_42",
        kind="phase_b",
        run_window=(20200102, 20231229),
        score_window=(20210104, 20231229),
        snaptime="phase_b_c01_seed_42",
    )

    assert baseline.run_dir == (tmp_path / "baseline" / "full_run").resolve()
    assert baseline.snaptime == "baseline_full_run"
    assert trial_1.run_dir == (tmp_path / "trials" / "trial_00001").resolve()
    assert trial_1.trial_dir == trial_1.run_dir
    assert trial_1.output_root != trial_2.output_root
    assert trial_1.checkpoint_root != trial_2.checkpoint_root
    assert trial_1.snaptime == "trial_00001"
    assert phase_b.run_dir == (tmp_path / "phase_b" / "candidate_01" / "seed_42").resolve()
    assert phase_b.snaptime == "phase_b_c01_seed_42"
    assert phase_b.run_window == (20200102, 20231229)
    assert phase_b.score_window == (20210104, 20231229)
    assert phase_b.config_path.is_absolute()
    assert phase_b.output_root.is_absolute()
    assert phase_b.checkpoint_root.is_absolute()
