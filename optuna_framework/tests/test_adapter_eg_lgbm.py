"""eg-lgbm adapter tests."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from optuna_framework.paths import get_repo_root
from optuna_framework.plugin_loader import load_plugin


class FakeTrial:
    """Minimal Optuna-like trial for deterministic suggestions."""

    def suggest_float(self, name, low, high, log=False):
        assert low < high
        if name == "lr":
            assert log is True
            return 0.02
        if name == "feature_fraction":
            return 0.7
        if name == "bagging_fraction":
            return 0.9
        raise AssertionError(name)

    def suggest_categorical(self, name, choices):
        assert choices
        return choices[0]

    def suggest_int(self, name, low, high):
        assert name == "epochs"
        assert (low, high) == (50, 300)
        return 120


def test_suggest_params_ranges() -> None:
    adapter = load_plugin(model_dir="eg-lgbm").adapter
    params = adapter.suggest_params(FakeTrial())
    assert params["lr"] == 0.02
    assert params["epochs"] == 120
    assert params["num_leaves"] in adapter.num_leaves_choices
    assert params["feature_fraction"] == 0.7
    assert params["bagging_fraction"] == 0.9
    assert params["bagging_freq"] in adapter.bagging_freq_choices
    assert params["min_data_in_leaf"] in adapter.min_data_in_leaf_choices


def test_baseline_materializes_to_original_model_fields() -> None:
    adapter = load_plugin(model_dir="eg-lgbm").adapter
    root = ET.parse(get_repo_root() / "eg-lgbm" / "config.xml").getroot()
    materialized = adapter.apply_params_to_xml(root, adapter.baseline_params())
    model = root.find("./combo/model")
    assert model is not None
    assert float(model.get("lr")) == 0.05
    assert int(model.get("epochs")) == 100
    assert int(model.get("num_leaves")) == 31
    assert float(model.get("feature_fraction")) == 0.8
    assert float(model.get("bagging_fraction")) == 0.8
    assert int(model.get("bagging_freq")) == 1
    assert int(model.get("min_data_in_leaf")) == 100
    assert materialized == adapter.baseline_params()


def test_apply_params_to_xml_writes_lgbm_attrs_only() -> None:
    adapter = load_plugin(model_dir="eg-lgbm").adapter
    root = ET.parse(get_repo_root() / "eg-lgbm" / "config.xml").getroot()
    original_hidden = root.find("./combo/model").get("hiddenSize")
    original_device = root.find("./combo/model").get("device")
    params = {
        "lr": 0.01,
        "epochs": 150,
        "num_leaves": 63,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 3,
        "min_data_in_leaf": 50,
    }
    adapter.apply_params_to_xml(root, params)
    model = root.find("./combo/model")
    assert model is not None
    assert float(model.get("lr")) == 0.01
    assert int(model.get("epochs")) == 150
    assert int(model.get("num_leaves")) == 63
    assert float(model.get("feature_fraction")) == 0.65
    assert float(model.get("bagging_fraction")) == 0.75
    assert int(model.get("bagging_freq")) == 3
    assert int(model.get("min_data_in_leaf")) == 50
    assert model.get("hiddenSize") == original_hidden
    assert model.get("device") == original_device


def test_default_seed_hooks_are_sufficient() -> None:
    adapter = load_plugin(model_dir="eg-lgbm").adapter
    assert adapter.phase_seeds() == (42, 43, 44)
    assert adapter.write_seeded_model(get_repo_root(), get_repo_root() / "eg-lgbm" / "config.xml", "phase_b") is None
    assert adapter.seeded_overrides(43) == {"combo.model.seed": 43}
