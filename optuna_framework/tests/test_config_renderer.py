"""XML config rendering tests."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

from optuna_framework.config_renderer import PATH_PATCH_KEYS, apply_fixed_override, assert_only_allowed_diffs, render_config, structured_xml_diff
from optuna_framework.paths import build_trial_run_paths, get_repo_root
from optuna_framework.scripts._script_common import load_study_and_adapter


class Args:
    model_dir = None
    plugin = None


class LgbmArgs:
    model_dir = "eg-lgbm"
    plugin = None


class TcnArgs:
    model_dir = None
    plugin = "/root/autodl/0530.tcn/optuna_plugin.py"


def _study_and_adapter():
    study_spec, adapter, _plugin = load_study_and_adapter(Args())
    return study_spec, adapter


def _lgbm_study_and_adapter():
    study_spec, adapter, _plugin = load_study_and_adapter(LgbmArgs())
    return study_spec, adapter


def _tcn_study_and_adapter():
    study_spec, adapter, _plugin = load_study_and_adapter(TcnArgs())
    return study_spec, adapter


def test_render_baseline_params_only_changes_path_whitelist(tmp_path) -> None:
    study_spec, adapter = _study_and_adapter()
    segment = study_spec.segment_by_name("seg01")
    run_paths = build_trial_run_paths(tmp_path, 7, segment)
    materialized = render_config(study_spec.baseline_config_path, run_paths, adapter, adapter.baseline_params())
    diffs = structured_xml_diff(study_spec.baseline_config_path, run_paths.config_path)
    assert_only_allowed_diffs(diffs, PATH_PATCH_KEYS)
    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./strategy").get("start_ds") == "20210104"
    assert root.find("./strategy").get("end_ds") == "20211231"
    assert root.find("./combo/runtime").get("snaptime") == "trial_00007_seg01"
    assert root.find("./combo/paths").get("model_path") == str((get_repo_root() / "eg-torch" / "model.py").resolve())
    assert root.find("./strategy").get("path") == str((get_repo_root() / "alpha_strategy.py").resolve())
    assert root.find("./constants").get("output_root") == str(run_paths.output_root)
    assert root.find("./constants").get("checkpoint_root") == str(run_paths.checkpoint_root)
    assert materialized["scheduler_step_size"] == 10
    assert json.loads(run_paths.params_path.read_text(encoding="utf-8"))["scheduler_step_size"] == 10


def test_fixed_overrides_take_effect(tmp_path) -> None:
    study_spec, adapter = _study_and_adapter()
    segment = study_spec.segment_by_name("seg01")
    run_paths = build_trial_run_paths(tmp_path, 8, segment)
    render_config(
        study_spec.baseline_config_path,
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
    study_spec, adapter = _study_and_adapter()
    segment = study_spec.segment_by_name("seg01")
    run_paths = build_trial_run_paths(tmp_path, 9, segment)

    render_config(
        study_spec.baseline_config_path,
        run_paths,
        adapter,
        adapter.baseline_params(),
        fixed_overrides={"combo.output.enable_alpha_analysis": False},
    )

    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./combo/output").get("enable_alpha_analysis") == "false"


def test_render_lgbm_params_keeps_generic_path_patch_behavior(tmp_path) -> None:
    study_spec, adapter = _lgbm_study_and_adapter()
    segment = study_spec.segment_by_name("seg01")
    run_paths = build_trial_run_paths(tmp_path, 10, segment)
    materialized = render_config(study_spec.baseline_config_path, run_paths, adapter, adapter.baseline_params())
    diffs = structured_xml_diff(study_spec.baseline_config_path, run_paths.config_path)
    assert_only_allowed_diffs(diffs, PATH_PATCH_KEYS)
    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./combo/paths").get("model_path") == str((get_repo_root() / "eg-lgbm" / "model.py").resolve())
    assert root.find("./combo/runtime").get("snaptime") == "trial_00010_seg01"
    assert float(root.find("./combo/model").get("lr")) == 0.05
    assert int(root.find("./combo/model").get("num_leaves")) == 31
    assert materialized["bagging_freq"] == 1


def test_render_external_tcn_plugin_keeps_generic_path_patch_behavior(tmp_path) -> None:
    study_spec, adapter = _tcn_study_and_adapter()
    segment = study_spec.segment_by_name("seg01")
    run_paths = build_trial_run_paths(tmp_path, 11, segment)
    materialized = render_config(study_spec.baseline_config_path, run_paths, adapter, adapter.baseline_params())
    diffs = structured_xml_diff(study_spec.baseline_config_path, run_paths.config_path)
    assert_only_allowed_diffs(diffs, PATH_PATCH_KEYS)
    root = ET.parse(run_paths.config_path).getroot()
    assert root.find("./combo/paths").get("model_path") == "/root/autodl/0530.tcn/Model.py"
    assert root.find("./combo/runtime").get("snaptime") == "trial_00011_seg01"
    assert float(root.find("./combo/model").get("lr")) == 2e-5
    assert int(root.find("./combo/model").get("hiddenSize")) == 256
    assert materialized["scheduler_step_size"] == 10
