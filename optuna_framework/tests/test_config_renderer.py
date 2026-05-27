"""XML config rendering tests."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

from optuna_framework.config_renderer import PATH_PATCH_KEYS, apply_fixed_override, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_trial_run_paths, get_repo_root
from optuna_framework.scripts._script_common import adapter_for_name
from optuna_framework.studies.eg_torch_v1 import ADAPTER_NAME, BASELINE_CONFIG_PATH, FIXED_OVERRIDES, SCORING_WINDOW, TUNING_RUN_WINDOW


def test_render_baseline_params_only_changes_path_whitelist(tmp_path) -> None:
    adapter = adapter_for_name(ADAPTER_NAME)
    run_paths = build_trial_run_paths(tmp_path, 7, TUNING_RUN_WINDOW, SCORING_WINDOW)
    materialized = render_config(BASELINE_CONFIG_PATH, run_paths, adapter, adapter.baseline_params())
    diffs = structured_xml_diff(BASELINE_CONFIG_PATH, run_paths.config_path)
    assert_only_allowed_diffs(diffs, PATH_PATCH_KEYS)
    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./strategy").get("start_ds") == "20200102"
    assert root.find("./strategy").get("end_ds") == "20231229"
    assert root.find("./combo/runtime").get("snaptime") == "trial_00007"
    assert root.find("./combo/paths").get("model_path") == str((get_repo_root() / "eg-torch" / "model.py").resolve())
    assert root.find("./strategy").get("path") == str((get_repo_root() / "alpha_strategy.py").resolve())
    assert root.find("./constants").get("output_root") == str(run_paths.output_root)
    assert root.find("./constants").get("checkpoint_root") == str(run_paths.checkpoint_root)
    meta = json.loads(run_paths.resolved_meta_path.read_text(encoding="utf-8"))
    assert meta["run_window"] == {"start_ds": 20200102, "end_ds": 20231229}
    assert meta["score_window"] == {"start_ds": 20210104, "end_ds": 20231229}
    assert "segment" not in meta
    assert "role" not in meta
    assert materialized["scheduler_step_size"] == 10
    assert json.loads(run_paths.params_path.read_text(encoding="utf-8"))["scheduler_step_size"] == 10


def test_fixed_overrides_take_effect(tmp_path) -> None:
    adapter = adapter_for_name(ADAPTER_NAME)
    run_paths = build_trial_run_paths(tmp_path, 8, TUNING_RUN_WINDOW, SCORING_WINDOW)
    render_config(
        BASELINE_CONFIG_PATH,
        run_paths,
        adapter,
        adapter.baseline_params(),
        fixed_overrides={"combo.model.device": "cuda"},
    )
    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./combo/model").get("device") == "cuda"
    meta = json.loads(run_paths.resolved_meta_path.read_text(encoding="utf-8"))
    assert meta["fixed_overrides"] == {"combo.model.device": "cuda"}


def test_apply_fixed_override_rejects_missing_element() -> None:
    root = ET.parse(get_repo_root() / "eg-torch" / "config.xml").getroot()
    try:
        apply_fixed_override(root, "combo.missing.device", "cuda")
    except ValueError as exc:
        assert "missing element" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_fixed_override_can_create_combo_output_element(tmp_path) -> None:
    adapter = adapter_for_name(ADAPTER_NAME)
    run_paths = build_trial_run_paths(tmp_path, 9, TUNING_RUN_WINDOW, SCORING_WINDOW)

    render_config(
        BASELINE_CONFIG_PATH,
        run_paths,
        adapter,
        adapter.baseline_params(),
        fixed_overrides=FIXED_OVERRIDES,
    )

    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./combo/output").get("enable_alpha_analysis") == "false"
