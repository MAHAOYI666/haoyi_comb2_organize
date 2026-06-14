from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


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
    assert "Evaluate an existing" in help_proc.stdout

    missing_proc = run_cli("runEval.py")
    assert missing_proc.returncode == 2
    assert "missing config.xml" in missing_proc.stderr
    assert "Traceback" not in missing_proc.stderr


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
