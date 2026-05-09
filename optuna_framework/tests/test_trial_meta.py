"""trial_meta state tests."""

from __future__ import annotations

from optuna_framework.trial_meta import init_trial_meta, read_trial_meta, update_segment, update_trial_state


def test_trial_meta_state_machine(tmp_path) -> None:
    trial_dir = tmp_path / "trial_00012"
    init_trial_meta(trial_dir, 12, {"lr": 1e-6}, ["seg01", "seg02"])
    meta = read_trial_meta(trial_dir)
    assert meta["trial_number"] == 12
    assert meta["state"] == "running"
    assert meta["segments"]["seg01"]["state"] == "skipped"
    update_segment(trial_dir, "seg01", "complete", {"sharpe_idx": 1.2, "dd_li": 0.1, "days": 240})
    meta = update_trial_state(trial_dir, "complete", objective=1.0, hard_filter_triggered=False)
    assert meta["state"] == "complete"
    assert meta["ended_at"] is not None
    assert meta["segments"]["seg01"]["sharpe_idx"] == 1.2
    assert (trial_dir / "trial_meta.json").exists()
    assert not (trial_dir / "trial_meta.json.tmp").exists()

