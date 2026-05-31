"""Optuna plugin loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from optuna_framework.plugin_loader import load_plugin, resolve_plugin_path


def test_default_plugin_loads_eg_torch() -> None:
    loaded = load_plugin()
    assert loaded.path.name == "optuna_plugin.py"
    assert loaded.path.parent.name == "eg-torch"
    assert loaded.study_spec.name == "study_eg_torch_v1"
    assert loaded.adapter.name == "eg_torch_v1"


def test_model_dir_and_plugin_resolve_same_path() -> None:
    by_model = resolve_plugin_path(model_dir="eg-torch")
    by_plugin = resolve_plugin_path(plugin="eg-torch/optuna_plugin.py")
    assert by_model == by_plugin


def test_lgbm_plugin_loads_from_model_dir() -> None:
    loaded = load_plugin(model_dir="eg-lgbm")
    assert loaded.path.name == "optuna_plugin.py"
    assert loaded.path.parent.name == "eg-lgbm"
    assert loaded.study_spec.name == "study_eg_lgbm_v1"
    assert loaded.adapter.name == "eg_lgbm_v1"


def test_external_0530_plugin_loads_by_absolute_path() -> None:
    loaded = load_plugin(plugin="/root/autodl/0530.tcn/optuna_plugin.py")
    assert loaded.path.name == "optuna_plugin.py"
    assert loaded.study_spec.name == "study_0530_tcn_v1"
    assert loaded.adapter.name == "tcn_0530_v1"


def test_loader_rejects_missing_plugin_contract(tmp_path) -> None:
    plugin = tmp_path / "optuna_plugin.py"
    plugin.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="STUDY_SPEC"):
        load_plugin(plugin=plugin)


def test_loader_rejects_both_model_dir_and_plugin() -> None:
    with pytest.raises(ValueError, match="either model_dir or plugin"):
        resolve_plugin_path(model_dir="eg-torch", plugin="eg-torch/optuna_plugin.py")
