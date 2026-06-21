from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "evals"), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_cli_in(cwd: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "evals"), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_run_combo_help_and_missing_config():
    help_proc = run_cli("runCombo.py", "-h")
    assert help_proc.returncode == 0
    assert "Run a comb2 experiment" in help_proc.stdout

    missing_proc = run_cli("runCombo.py")
    assert missing_proc.returncode == 2
    assert "missing config file" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr


def test_run_eval_help_and_missing_config():
    help_proc = run_cli("runEval.py", "-h")
    assert help_proc.returncode == 0
    assert "Evaluate comb2 signals" in help_proc.stdout

    missing_proc = run_cli("runEval.py")
    assert missing_proc.returncode == 2
    assert "missing config.xml" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr


def test_run_eval_specialized_modes(tmp_path):
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    left = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
        index=dates,
        columns=["000001", "000002", "000003"],
    )
    right = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 2.0, 1.0]],
        index=dates,
        columns=left.columns,
    )
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    left.to_parquet(left_path)
    right.to_parquet(right_path)

    corr_proc = run_cli("runEval.py", "--corr", str(left_path), str(right_path), "--min-valid", "2", "--top-pct", "50")
    assert corr_proc.returncode == 0
    assert "avg_corr" in corr_proc.stdout
    assert "avg_long_overlap" in corr_proc.stdout

    daily_ic = pd.DataFrame(
        {
            "ic": [0.01, 0.02, 0.03],
            "5dic": [0.02, 0.03, 0.04],
            "rankic": [0.03, 0.04, 0.05],
            "percic": [0.04, 0.05, 0.06],
            "coverage": [1.0, 1.0, 1.0],
        },
        index=dates,
    )
    daily_ic_path = tmp_path / "daily_ic.csv"
    daily_ic.to_csv(daily_ic_path)
    sim_proc = run_cli("runEval.py", "--sim", str(daily_ic_path), "--input-is-ic", "--normalize-names")
    assert sim_proc.returncode == 0
    assert "1d_IC.avg" in sim_proc.stdout

    daily_pnl = pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 80.0],
            "long": [10000.0, 10000.0, 10000.0],
            "short": [-10000.0, -10000.0, -10000.0],
            "sh_hld": [20000.0, 20000.0, 20000.0],
            "sh_trd": [1000.0, 1000.0, 1000.0],
            "n_long": [2, 2, 2],
            "n_short": [1, 1, 1],
            "longonly_pnl": [60.0, 20.0, 40.0],
        },
        index=dates,
    )
    daily_pnl_path = tmp_path / "daily_pnl.csv"
    daily_pnl.to_csv(daily_pnl_path)
    pnl_proc = run_cli("runEval.py", "--pnl", str(daily_pnl_path), "--input-is-pnl")
    assert pnl_proc.returncode == 0
    assert "ret_pct" in pnl_proc.stdout

    new_pnl = daily_pnl.copy()
    new_pnl["longonly_pnl"] = [80.0, 50.0, 90.0]
    new_pnl_path = tmp_path / "new_daily_pnl.csv"
    new_pnl.to_csv(new_pnl_path)
    va_proc = run_cli("runEval.py", "--va", str(daily_pnl_path), str(new_pnl_path), "--weights", "0.1,0.2")
    assert va_proc.returncode == 0
    assert "0.10" in va_proc.stdout


def test_combo_runner_help_and_missing_config():
    help_proc = run_cli("comboRunner.py", "-h")
    assert help_proc.returncode == 0
    assert "standard runCombo pipeline" in help_proc.stdout

    missing_proc = run_cli("comboRunner.py")
    assert missing_proc.returncode == 2
    assert "config.xml not found" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr


def test_ablation_help_and_missing_config():
    help_proc = run_cli("runAblationByZero.py", "-h")
    assert help_proc.returncode == 0
    assert "zero-ablation" in help_proc.stdout

    missing_proc = run_cli("runAblationByZero.py")
    assert missing_proc.returncode == 2
    assert "missing config file" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr


def test_pos_corr_help_and_missing_inputs():
    help_proc = run_cli("runPosCorr.py", "-h")
    assert help_proc.returncode == 0
    assert "position correlation" in help_proc.stdout

    missing_proc = run_cli("runPosCorr.py")
    assert missing_proc.returncode == 2
    assert "required: pos1, pos2" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr

    bad_path_proc = run_cli("runPosCorr.py", "missing-a.parquet", "missing-b.parquet")
    assert bad_path_proc.returncode == 2
    assert "pos1 file not found" in bad_path_proc.stderr
    assert "Traceback" not in bad_path_proc.stderr


def test_comb_eval_missing_command_and_input():
    missing_proc = run_cli("-m", "comb_eval.cli")
    assert missing_proc.returncode == 2
    assert "missing command" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr

    missing_input_proc = run_cli("-m", "comb_eval.cli", "eval")
    assert missing_input_proc.returncode == 2
    assert "missing input" in missing_input_proc.stderr
    assert "Traceback" not in missing_input_proc.stderr


def test_combo_hello_world_creates_editable_starter_files(tmp_path):
    help_proc = run_cli("comboHelloWorld.py", "-h")
    assert help_proc.returncode == 0
    assert "combo-hello-world" in help_proc.stdout

    cancelled = run_cli_in(tmp_path, str(REPO_ROOT / "comboHelloWorld.py"), input_text="n\n")
    assert cancelled.returncode == 1
    assert not (tmp_path / "Model.py").exists()

    created = run_cli_in(tmp_path, str(REPO_ROOT / "comboHelloWorld.py"), "-y")
    assert created.returncode == 0
    assert (tmp_path / "Model.py").is_file()
    assert (tmp_path / "config.xml").is_file()
    assert (tmp_path / "config.human").is_file()

    model_text = (tmp_path / "Model.py").read_text(encoding="utf-8")
    assert "class ResearchModel" in model_text
    assert "class AlphaStrategy" not in model_text
    assert "adaptive_hidden_size" in model_text

    config_text = (tmp_path / "config.xml").read_text(encoding="utf-8")
    assert 'model_path="Model.py"' in config_text
    assert 'trainDelay="0"' in config_text
    assert 'hidden_size=' not in config_text
    assert 'fc_size=' not in config_text
    assert 'path="example_factor"' in config_text
    assert 'path="label1d"' in config_text
    root = ET.fromstring(config_text)
    assert root.find("./strategy").get("path") is None

    from config import DEFAULT_CONFIG, load_config

    parsed = load_config(str(tmp_path / "config.xml"))
    assert parsed["combo"]["paths"]["model_path"] == str((tmp_path / "Model.py").resolve())
    assert parsed["strategy"]["path"] == DEFAULT_CONFIG["strategy"]["path"]
    assert Path(parsed["strategy"]["path"]).is_file()
    assert parsed["strategy"]["path"].endswith("comb2_pcmaster/default_strategy.py")
    assert parsed["combo"]["runtime"]["trainDelay"] == 0
    assert len(parsed["combo"]["loader"]["data_items"]) == 2

    overwrite = run_cli_in(tmp_path, str(REPO_ROOT / "comboHelloWorld.py"), "-y")
    assert overwrite.returncode == 2
    assert "refusing to overwrite" in overwrite.stderr
