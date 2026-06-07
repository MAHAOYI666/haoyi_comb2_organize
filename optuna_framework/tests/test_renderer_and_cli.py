from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from optuna_framework.config_renderer import OPTUNA_RUNTIME_PATCH_KEYS, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_trial_run_paths, get_repo_root
from optuna_framework.search_space import ConfigDrivenAdapter
from optuna_framework.study_config import load_study_config


def test_render_baseline_params_writes_paths_and_fixed_override(tmp_path) -> None:
    config = load_study_config(study_root_override=tmp_path / "study")
    adapter = ConfigDrivenAdapter(config)
    run_paths = build_trial_run_paths(config.study_root, 7, config.tuning_run_window, config.scoring_window)

    materialized = render_config(config, run_paths, adapter, adapter.baseline_params())
    diffs = structured_xml_diff(config.baseline_config_path, run_paths.config_path)
    assert_only_allowed_diffs(diffs, OPTUNA_RUNTIME_PATCH_KEYS)

    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./strategy").get("start_ds") == "20200102"
    assert root.find("./strategy").get("end_ds") == "20231229"
    assert root.find("./combo/runtime").get("snaptime") == "trial_00007"
    assert root.find("./combo/paths").get("model_path") == str((get_repo_root() / "eg-torch" / "model.py").resolve())
    assert root.find("./combo/output").get("enable_alpha_analysis") == "false"
    assert root.find("./constants").get("output_root") == str(run_paths.output_root)
    assert materialized["scheduler_step_size"] == 10
    assert root.find("./combo/model").get("scheduler_step_size") == "10"


def test_cli_dry_run_does_not_launch_runcombo(tmp_path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "optuna_framework/scripts/run_baseline.py",
            "--config",
            "optuna_framework/config.xml",
            "--study-root",
            str(tmp_path / "study"),
            "--dry-run",
        ],
        cwd=get_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    assert "[DRY-RUN] baseline" in proc.stdout
    assert "runCombo.py" in proc.stdout
    assert not (tmp_path / "study" / "baseline" / "full_run" / "run.stdout.log").exists()


def test_dry_run_render_cli_outputs_plan_and_rendered_config(tmp_path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "optuna_framework/scripts/dry_run_render.py",
            "--config",
            "optuna_framework/config.xml",
            "--study-root",
            str(tmp_path / "study"),
        ],
        cwd=get_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    assert "config_plan=" in proc.stdout
    assert "rendered_config=" in proc.stdout
    assert "xml_diff_allowed=true" in proc.stdout
    assert (tmp_path / "study" / "config_plan.txt").exists()
