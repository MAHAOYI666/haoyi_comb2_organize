from pathlib import Path

from optuna_framework.runner import _format_failed_run_message, _tail_text
from optuna_framework.specs import RunPaths, SegmentSpec


def _run_paths(tmp_path: Path) -> RunPaths:
    segment = SegmentSpec("seg01", "tuning", 20210104, 20211231)
    segment_dir = tmp_path / "baseline" / "seg01"
    return RunPaths(
        study_root=tmp_path,
        segment_dir=segment_dir,
        config_path=segment_dir / "config.xml",
        output_root=segment_dir / "output",
        checkpoint_root=segment_dir / "checkpoints",
        pnl_summary_path=segment_dir / "output" / "backtest" / "pnl_summary.csv",
        stdout_path=segment_dir / "run.stdout.log",
        stderr_path=segment_dir / "run.stderr.log",
        params_path=segment_dir / "params.json",
        resolved_meta_path=segment_dir / "resolved_meta.json",
        snaptime="baseline_seg01",
        segment=segment,
        kind="baseline",
    )


def test_tail_text_returns_bounded_file_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "run.stderr.log"
    log_path.write_text("\n".join(f"line {idx}" for idx in range(60)), encoding="utf-8")

    tail = _tail_text(log_path, max_lines=3)

    assert tail == "line 57\nline 58\nline 59"


def test_format_failed_run_message_includes_paths_and_log_tails(tmp_path: Path) -> None:
    run_paths = _run_paths(tmp_path)
    run_paths.segment_dir.mkdir(parents=True)
    run_paths.stderr_path.write_text("stderr detail\n", encoding="utf-8")
    run_paths.stdout_path.write_text("stdout detail\n", encoding="utf-8")

    message = _format_failed_run_message(run_paths, ["python3", "runCombo.py", str(run_paths.config_path)], 1)

    assert "runCombo failed for seg01 with returncode=1" in message
    assert f"config: {run_paths.config_path}" in message
    assert f"stdout_log: {run_paths.stdout_path}" in message
    assert f"stderr_log: {run_paths.stderr_path}" in message
    assert "stderr_tail:\nstderr detail" in message
    assert "stdout_tail:\nstdout detail" in message
