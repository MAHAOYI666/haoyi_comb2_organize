from pathlib import Path

from optuna_framework.runner import _format_failed_run_message, _tail_text
from optuna_framework.specs import RunPaths


def _run_paths(tmp_path: Path) -> RunPaths:
    run_dir = tmp_path / "baseline" / "full_run"
    return RunPaths(
        study_root=tmp_path,
        run_dir=run_dir,
        config_path=run_dir / "config.xml",
        output_root=run_dir / "output",
        checkpoint_root=run_dir / "checkpoints",
        pnl_summary_path=run_dir / "output" / "backtest" / "pnl_summary.csv",
        stdout_path=run_dir / "run.stdout.log",
        stderr_path=run_dir / "run.stderr.log",
        params_path=run_dir / "params.json",
        resolved_meta_path=run_dir / "resolved_meta.json",
        snaptime="baseline_full_run",
        kind="baseline",
        trial_number=None,
        run_start_ds=20200102,
        run_end_ds=20240628,
        score_start_ds=20210104,
        score_end_ds=20231229,
    )


def test_tail_text_returns_bounded_file_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "run.stderr.log"
    log_path.write_text("\n".join(f"line {idx}" for idx in range(60)), encoding="utf-8")

    tail = _tail_text(log_path, max_lines=3)

    assert tail == "line 57\nline 58\nline 59"


def test_format_failed_run_message_includes_paths_and_log_tails(tmp_path: Path) -> None:
    run_paths = _run_paths(tmp_path)
    run_paths.run_dir.mkdir(parents=True)
    run_paths.stderr_path.write_text("stderr detail\n", encoding="utf-8")
    run_paths.stdout_path.write_text("stdout detail\n", encoding="utf-8")

    message = _format_failed_run_message(run_paths, [".\\.venv\\Scripts\\python.exe", "runCombo.py", str(run_paths.config_path)], 1)

    assert f"runCombo failed for {run_paths.run_dir} with returncode=1" in message
    assert f"config: {run_paths.config_path}" in message
    assert f"stdout_log: {run_paths.stdout_path}" in message
    assert f"stderr_log: {run_paths.stderr_path}" in message
    assert "stderr_tail:\nstderr detail" in message
    assert "stdout_tail:\nstdout detail" in message
